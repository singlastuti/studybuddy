
import streamlit as st
from load_pdf import load_and_split_pdf
from embed_store import embed_and_store
from chatbot import load_chatbot
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from utils import get_embeddings
from summarizer import summarizer_ui
from quiz_generator import quiz_generator_ui
from bullet_points import bullet_points_ui
from dotenv import load_dotenv
import os
import hashlib
from pathlib import Path

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

    # Custom prompt template to give context about the uploaded document
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an AI assistant helping with questions about a document the user has uploaded.\n"
            "Use ONLY the provided context from the document. If the context is insufficient, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:\n"
            "Note: Do not mention the file name (temp.pdf) in your response. Be friendly, concise, and accurate."
        ),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": custom_prompt, "document_variable_name": "context"},
    )

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

        # Get answer from the chain
        result = qa_chain.invoke({"query": user_input})
        answer = result["result"]

        # Add assistant message to history and display it immediately
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.stop()  # Prevents duplicate display on rerun
