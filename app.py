
import streamlit as st
from load_pdf import load_and_split_pdf
from embed_store import embed_and_store
from chatbot import load_chatbot
from langchain_community.vectorstores import FAISS
from utils import get_embeddings
from summarizer import summarizer_ui
from quiz_generator import quiz_generator_ui
from bullet_points import bullet_points_ui
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
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    st.divider()
    with st.expander("Advanced: Retrieval Settings", expanded=False):
        k = st.slider("Top-K Chunks", 2, 10, 6)
        fetch_k = st.slider("Fetch-K (MMR)", 10, 60, 30, step=5)
        st.caption("Higher K may improve recall but costs more tokens.")
        show_context_debug = st.checkbox("Show retrieved context under answers", value=False)
    st.divider()
    st.subheader("About")
    st.caption("StudyBuddy helps you chat with your PDFs using Azure OpenAI and FAISS. All answers are grounded to the uploaded document.")

if uploaded_file:
    # Persist the upload to a temp file
    with open("temp.pdf", "wb") as f:
        file_bytes_uploaded = uploaded_file.read()
        f.write(file_bytes_uploaded)

    # Compute a stable hash to cache embeddings/index per unique file
    with open("temp.pdf", "rb") as f:
        file_bytes = f.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
    index_dir = Path("faiss_index") / file_hash
    # Persist file meta for later actions
    st.session_state["file_hash"] = file_hash
    st.session_state["index_dir"] = str(index_dir)
    st.session_state["file_name"] = getattr(uploaded_file, "name", "document.pdf")
    st.session_state["file_bytes"] = file_bytes

    st.info("Parsing and indexing document…")
    with st.spinner("Extracting text and building/using vector index…"):
        # Parse
        chunks = load_and_split_pdf("temp.pdf")
        if not chunks:
            st.error("No text could be extracted from the uploaded PDF. Please upload a different file.")
            st.stop()
        # Keep chunks in session for feature modules
        st.session_state["chunks"] = chunks

        # Prepare embeddings
        embeddings = get_embeddings()

        # Build or load FAISS index for this file hash
        if (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists():
            # Local-only trusted index; allow_dangerous_deserialization is safe here
            vectorstore = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
        else:
            # Ensure parent dir exists
            index_dir.parent.mkdir(parents=True, exist_ok=True)
            vectorstore = embed_and_store(chunks, save_path=str(index_dir))

        # Cleanup temporary file
        try:
            os.remove("temp.pdf")
        except Exception:
            pass

    st.success("Document processed successfully!")

    # Create retriever (MMR for diversity)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k})

    # Load LLM
    llm = load_chatbot()

    # Sidebar document info
    with st.sidebar:
        st.subheader("Document Info")
        st.caption(f"File: {st.session_state.get('file_name', 'document.pdf')}")
        st.caption(f"Index: {file_hash}")
        chunk_count = len(st.session_state.get("chunks", []))
        st.caption(f"Chunks: {chunk_count}")
        # Actions
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
        if st.button("Re-index Document", use_container_width=True):
            # Force rebuild the FAISS index for this file
            try:
                # Write temp again from stored bytes
                with open("temp.pdf", "wb") as f:
                    f.write(st.session_state.get("file_bytes", b""))
                # Re-parse
                chunks = load_and_split_pdf("temp.pdf")
                st.session_state["chunks"] = chunks
                embeddings = get_embeddings()
                Path(st.session_state["index_dir"]).mkdir(parents=True, exist_ok=True)
                _ = embed_and_store(chunks, save_path=st.session_state["index_dir"])
                st.success("Re-indexed successfully.")
            except Exception as e:
                st.error(f"Failed to re-index: {e}")

    tabs = st.tabs(["💬 Chat", "🧾 Summarize", "📝 Quiz", "📋 Bullet Points"])
    with tabs[0]:
        pass  # Chat rendered below
    with tabs[1]:
        summarizer_ui(llm, retriever)
    with tabs[2]:
        quiz_generator_ui(llm, retriever)
    with tabs[3]:
        bullet_points_ui(llm, retriever)

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
            result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
            answer = result.get("answer", "I couldn't produce an answer.")

            # Add assistant message to history and display it immediately
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(answer)

                # Optional: show retrieved context for debugging/education
                if 'show_context_debug' in locals() and show_context_debug:
                    ctx = result.get("context", [])
                    if ctx:
                        with st.expander("Show retrieved context"):
                            for i, d in enumerate(ctx, 1):
                                content = getattr(d, 'page_content', str(d))
                                snippet = (content[:600] + '…') if len(content) > 600 else content
                                st.markdown(f"**Context {i}:**\n\n{snippet}")
            st.stop()  # Prevents duplicate display on rerun
