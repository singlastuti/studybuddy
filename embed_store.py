from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from utils import get_embeddings

load_dotenv()

def embed_and_store(chunks, save_path: str = "faiss_index"):
    """Embed chunks and persist a FAISS index locally.

    Returns the vectorstore.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    return vectorstore