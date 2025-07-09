from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

def embed_and_store(chunks):
    print("DEPLOYMENT:", os.getenv("AZURE_EMBEDDING_DEPLOYMENT"))
    print("ENDPOINT:", os.getenv("AZURE_OPENAI_ENDPOINT"))
    print("KEY:", os.getenv("OPENAI_API_KEY"))
    print("VERSION:", os.getenv("OPENAI_API_VERSION"))
    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT").strip(),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT").strip(),
        api_key=os.getenv("OPENAI_API_KEY").strip(),
        api_version=os.getenv("OPENAI_API_VERSION").strip(),
        chunk_size=1000
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore