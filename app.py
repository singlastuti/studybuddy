
import streamlit as st
from load_pdf import load_and_split_pdf
from embed_store import embed_and_store
from chatbot import load_chatbot
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
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
    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT").strip(),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT").strip(),
        api_key=os.getenv("OPENAI_API_KEY").strip(),
        api_version=os.getenv("OPENAI_API_VERSION").strip(),
        chunk_size=1000
    )
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    # Load LLM
    llm = load_chatbot()


    # Custom prompt template to give context about the uploaded document
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an AI assistant helping with questions about a document the user has uploaded. "
            "The document is titled 'temp.pdf'. Use ONLY the following context to answer the question.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": custom_prompt}
    )

    user_input = st.text_input("Ask me something about the document:")
    if user_input:
        result = qa_chain.invoke({"query": user_input})
        st.write(result["result"])
