
import streamlit as st
from load_pdf import load_and_split_pdf
from embed_store import embed_and_store
from chatbot import load_chatbot
from langchain_community.vectorstores import FAISS
from utils import get_embeddings
from summarizer import summarizer_ui
from quiz_generator import quiz_generator_ui
from bullet_points import bullet_points_ui
from knowledge_graph import knowledge_graph_ui
from dotenv import load_dotenv
import os
import hashlib
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

load_dotenv()

st.set_page_config(page_title="StudyBuddy AI", page_icon="📚", layout="wide")

# Minimal CSS for polished look
st.markdown(
    """
    <style>
    .sb-badge {display:inline-block; padding:4px 10px; border-radius:999px; background:#1F2937; color:#E5E7EB; margin-right:8px; font-size:12px;}
    .sb-card {padding:14px 16px; border-radius:12px; background:#111827; box-shadow: 0 1px 2px rgba(0,0,0,0.25);}
    .sb-gap {margin-top: 12px;}
    /* Chat bubble styles */
    [data-testid="stChatMessage"] .stMarkdown {width: 100%;}
    [data-testid="stChatMessage"] div div p {margin-bottom: 0.5rem;}
    .stChatMessage.user {background: #0F172A; border-radius: 12px; padding: 10px 12px;}
    .stChatMessage.assistant {background: #111827; border-radius: 12px; padding: 10px 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("StudyBuddy AI")
st.caption("Upload a PDF to chat, summarize, generate quizzes, and extract bullet points—grounded to your document.")

with st.sidebar:
    st.header("Setup")
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    st.divider()
    with st.expander("Advanced: Retrieval Settings", expanded=False):
        k = st.slider("Top-K Chunks", 2, 10, 6)
        fetch_k = st.slider("Fetch-K (MMR)", 10, 60, 30, step=5)
        st.caption("Higher K may improve recall but costs more tokens.")
        show_context_debug = st.checkbox("Show retrieved context under answers", value=False)
    st.divider()
    st.subheader("About")
    st.caption("StudyBuddy helps you chat with your PDFs using Azure OpenAI and FAISS. All answers are grounded to the uploaded document.")

if uploaded_files:
    all_chunks = []
    per_file_counts = []
    has_any_text = False
    sha = hashlib.sha256()

    st.info("Parsing and indexing document(s)…")
    with st.spinner("Extracting text and building/using vector index…"):
        for uploaded_file in uploaded_files:
            # Persist temp
            with open("temp.pdf", "wb") as f:
                b = uploaded_file.read()
                f.write(b)
            sha.update(b)
            # Parse
            chunks = load_and_split_pdf("temp.pdf")
            if chunks:
                has_any_text = True
                # Tag source metadata
                for d in chunks:
                    d.metadata = getattr(d, "metadata", {}) or {}
                    d.metadata["source"] = getattr(uploaded_file, "name", "document.pdf")
                all_chunks.extend(chunks)
                per_file_counts.append((getattr(uploaded_file, "name", "document.pdf"), len(chunks)))
            else:
                per_file_counts.append((getattr(uploaded_file, "name", "document.pdf"), 0))
            # Cleanup temp
            try:
                os.remove("temp.pdf")
            except Exception:
                pass

        if not has_any_text:
            st.error("No text could be extracted from the uploaded PDFs. Please upload different files.")
            st.stop()

        # Keep chunks in session for feature modules
        st.session_state["chunks"] = all_chunks

        # Combined index path based on all files' bytes
        file_hash = sha.hexdigest()[:16]
        index_dir = Path("faiss_index") / file_hash
        st.session_state["file_hash"] = file_hash
        st.session_state["index_dir"] = str(index_dir)

        embeddings = get_embeddings()
        if (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists():
            vectorstore = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
        else:
            index_dir.parent.mkdir(parents=True, exist_ok=True)
            vectorstore = embed_and_store(all_chunks, save_path=str(index_dir))

    st.success("Documents processed successfully!")

    # Create retriever (MMR for diversity)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k})

    # Load LLM
    llm = load_chatbot()

    # Sidebar document info
    with st.sidebar:
        st.subheader("Document Info")
        st.caption(f"Index: {file_hash}")
        total_chunks = len(st.session_state.get("chunks", []))
        st.caption(f"Total Chunks: {total_chunks}")
        for fname, cnt in per_file_counts:
            st.caption(f"{fname}: {cnt} chunks")
        # Actions
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
    if st.button("Re-index Documents", use_container_width=True):
        # Force rebuild the FAISS index for these document(s)
        try:
            embeddings = get_embeddings()
            Path(st.session_state["index_dir"]).mkdir(parents=True, exist_ok=True)
            _ = embed_and_store(st.session_state.get("chunks", []), save_path=st.session_state["index_dir"])
            st.success("Re-indexed successfully.")
        except Exception as e:
            st.error(f"Failed to re-index: {e}")

    tabs = st.tabs(["💬 Chat", "🧾 Summarize", "📝 Quiz", "📋 Bullet Points", "🕸️ Knowledge Graph"])
    with tabs[0]:
        pass  # Chat rendered below
    with tabs[1]:
        summarizer_ui(llm, retriever)
    with tabs[2]:
        quiz_generator_ui(llm, retriever)
    with tabs[3]:
        bullet_points_ui(llm, retriever)
    with tabs[4]:
        knowledge_graph_ui(llm, retriever)

    # Build history-aware retriever (rewrites follow-up questions using chat history)
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that rewrites the user's latest question into a standalone query using the conversation history. Keep it concise and specific."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=rephrase_prompt,
    )

    # Answering chain that sees both context chunks and chat history
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an AI assistant helping with questions about a document the user has uploaded.\n"
            "Use ONLY the provided context from the document. If the context is insufficient, say you don't know.\n"
            "Do not mention the file name. Be friendly, concise, and accurate."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Question: {input}"),
        ("system", "Context to use for answering:\n{context}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # --- Chat-like interface ---
    with tabs[0]:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history or a welcome prompt
        if st.session_state.messages:
            for msg in st.session_state.messages:
                avatar = "🧑" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])
        else:
            st.info("Ask a question about your document to get started.")

        # User input box at the bottom
        user_input = st.chat_input("Ask me something about the document:")

        if user_input:
            # Add user message to history and display it immediately
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_input)

            # Convert chat history to LangChain Message objects (excluding the last user message)
            chat_history = []
            for m in st.session_state.messages[:-1]:
                if m["role"] == "user":
                    chat_history.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    chat_history.append(AIMessage(content=m["content"]))

            # Get answer from the history-aware RAG chain
            import time
            t0 = time.perf_counter()
            result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            answer = result.get("answer", "I couldn't produce an answer.")

            # Add assistant message to history and display it immediately
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(answer)
                st.caption(f"Response time: {elapsed_ms} ms")

                # Optional: show retrieved context for debugging/education
                if 'show_context_debug' in locals() and show_context_debug:
                    ctx = result.get("context", [])
                    if ctx:
                        with st.expander("Show retrieved context"):
                            for i, d in enumerate(ctx, 1):
                                source = getattr(getattr(d, 'metadata', {}), 'get', lambda k, default=None: None)("source") if hasattr(d, 'metadata') else None
                                content = getattr(d, 'page_content', str(d))
                                snippet = (content[:600] + '…') if len(content) > 600 else content
                                header = f"**Context {i}{' — ' + source if source else ''}:**"
                                st.markdown(f"{header}\n\n{snippet}")
            st.stop()  # Prevents duplicate display on rerun
