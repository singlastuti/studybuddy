
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

st.set_page_config(page_title="StudyBuddy AI", layout="wide")
st.title("📚 AI Study Buddy")

uploaded_file = st.file_uploader("Upload your study material (PDF)", type="pdf")

if uploaded_file:
    # Persist the upload to a temp file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Compute a stable hash to cache embeddings/index per unique file
    with open("temp.pdf", "rb") as f:
        file_bytes = f.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
    index_dir = Path("faiss_index") / file_hash

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
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 30})

    # Load LLM
    llm = load_chatbot()

    summarizer_ui(llm, retriever)
    quiz_generator_ui(llm, retriever)
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
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input box at the bottom
    user_input = st.chat_input("Ask me something about the document:")

    if user_input:
        # Add user message to history and display it immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
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
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.stop()  # Prevents duplicate display on rerun
