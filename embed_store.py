import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def embed_and_store(chunks, save_path="faiss_index"):
    os.environ["OPENAI_API_KEY"] = "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"

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)

    return vectorstore
