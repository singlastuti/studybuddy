
import streamlit as st
from load_pdf import load_and_split_pdf
from embed_store import embed_and_store
from chatbot import load_chatbot
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from utils import get_embeddings
import os

st.set_page_config(page_title="StudyBuddy AI", layout="wide")
st.title("📚 AI Study Buddy")

uploaded_file = st.file_uploader("Upload your study material (PDF)", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.info("Parsing and embedding document...")

    chunks = load_and_split_pdf("temp.pdf")
    if not chunks:
        st.error("No text could be extracted from the uploaded PDF. Please upload a different file.")
        st.stop()
    embed_and_store(chunks)

    st.success("Document processed successfully!")

    

    # Load vectorstore and create retriever
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    # Load LLM
    llm = load_chatbot()

        # --- Summariser UI ---
    with st.expander("Summarise the uploaded document"):
        summary_length = st.radio(
            "Select summary length:",
            ("Concise", "Medium", "Detailed"),
            horizontal=True
        )
        if st.button("Summarise Document"):
            # Map summary length to instructions
            length_map = {
                "Concise": "in not more than2-3 sentences",
                "Medium": "in a short paragraph",
                "Detailed": "in a detailed summary, explaining each key point in detail"
            }
            summary_instruction = (
                f"Summarise the uploaded document {length_map[summary_length]}. "
                "Use only the document content."
            )
            # Use the retriever to get all chunks (or just concatenate all text)
            all_text = " ".join([doc.page_content for doc in retriever.vectorstore.docstore._dict.values()])
            # Ask the LLM to summarise
            summary_prompt = (
                f"{summary_instruction}\n\nDocument:\n{all_text}\n\nSummary:"
            )
            summary_result = llm.invoke(summary_prompt)
            # If summary_result is a dict or has a 'content' or 'text' attribute, extract it:
            if isinstance(summary_result, dict) and "content" in summary_result:
                summary_text = summary_result["content"]
            elif hasattr(summary_result, "content"):
                summary_text = summary_result.content
            elif hasattr(summary_result, "text"):
                summary_text = summary_result.text
            else:
                summary_text = str(summary_result)

            st.info(summary_text)

    # Custom prompt template to give context about the uploaded document
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an AI assistant helping with questions about a document the user has uploaded. "
            "The document is titled 'temp.pdf'. You can use the following context to answer the question.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            "Note that you should not use the name temp.pdf in your response, it is just a storage name for the uploaded document."
            "Be friendly and concise in your responses, but also thorough. "
        )
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": custom_prompt}
    )

        # --- Chat-like interface ---
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
