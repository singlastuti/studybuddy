from langchain_community.vectorstores import FAISS
import os
import time
from dotenv import load_dotenv
from utils import get_embeddings

load_dotenv()

def embed_and_store(chunks, save_path: str = "faiss_index"):
    """Embed chunks and persist a FAISS index locally with rate-limit backoff.

    - Batches documents to reduce per-call load (default 16; override EMBED_BATCH_SIZE).
    - Retries on 429 with exponential backoff up to a cap (default 6 attempts).

    Returns the vectorstore.
    """
    if not chunks:
        raise ValueError("No chunks to embed.")

    embeddings = get_embeddings()
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "16"))
    max_attempts = int(os.getenv("EMBED_MAX_ATTEMPTS", "6"))
    base_sleep = float(os.getenv("EMBED_BASE_SLEEP", "2"))  # seconds

    vectorstore = None
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        attempt = 0
        while True:
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                break  # batch succeeded
            except Exception as e:
                # Heuristic: back off on rate limit errors (429) or quota mentions
                msg = str(e).lower()
                is_rate = "429" in msg or "rate limit" in msg or "throttle" in msg
                attempt += 1
                if is_rate and attempt < max_attempts:
                    sleep_s = min(base_sleep * (2 ** (attempt - 1)), 60)
                    time.sleep(sleep_s)
                    continue
                # Out of retries or other exception: re-raise
                raise

    # Persist after all batches
    if vectorstore is None:
        raise RuntimeError("Failed to create vectorstore; no batches were embedded.")
    vectorstore.save_local(save_path)
    return vectorstore