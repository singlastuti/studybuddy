from langchain_openai import AzureOpenAIEmbeddings
import os

def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val or not val.strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val.strip()

def get_embeddings():
    # Streamlit cache is optional; avoid importing streamlit at module import time
    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _cached():
            return AzureOpenAIEmbeddings(
                model=_require_env("AZURE_EMBEDDING_DEPLOYMENT"),
                azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
                api_key=_require_env("OPENAI_API_KEY"),
                api_version=_require_env("OPENAI_API_VERSION"),
                chunk_size=1000,
            )

        return _cached()
    except Exception:
        # Fallback without caching (e.g., during non-Streamlit runs)
        return AzureOpenAIEmbeddings(
            model=_require_env("AZURE_EMBEDDING_DEPLOYMENT"),
            azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_require_env("OPENAI_API_KEY"),
            api_version=_require_env("OPENAI_API_VERSION"),
            chunk_size=1000,
        )