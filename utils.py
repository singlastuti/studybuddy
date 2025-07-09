from langchain_openai import AzureOpenAIEmbeddings
import os

def get_embeddings():
    return AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT").strip(),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT").strip(),
        api_key=os.getenv("OPENAI_API_KEY").strip(),
        api_version=os.getenv("OPENAI_API_VERSION").strip(),
        chunk_size=1000
    )