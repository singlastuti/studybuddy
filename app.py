import streamlit as st
from load_pdf import load_and_split_pdf
from embed_store import embed_and_store
from chatbot import load_chatbot

st.set_page_config(page_title="StudyBuddy AI", layout="wide")
st.title("📚 AI Study Buddy")

uploaded_file = st.file_uploader("Upload your study material (PDF)", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.info("Parsing and embedding document...")
    chunks = load_and_split_pdf("temp.pdf")
    embed_and_store(chunks)

    st.success("Document processed successfully!")

    qa_chain = load_chatbot()

    user_input = st.text_input("Ask me something about the document:")
    if user_input:
        result = qa_chain.run(user_input)
        st.write(result)
