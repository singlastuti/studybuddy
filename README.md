# StudyBuddy

Streamlit app to study PDFs with Azure OpenAI + FAISS.

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env` and set values:

```bash
cp .env .env.local  # optional; ensure .env is not committed
```

Required env vars:
- OPENAI_API_KEY
- AZURE_OPENAI_ENDPOINT
- AZURE_EMBEDDING_DEPLOYMENT
- AZURE_CHAT_DEPLOYMENT
- OPENAI_API_VERSION

## Run
```bash
streamlit run app.py
```

## Notes
- FAISS indexes are cached per-file using a SHA-256 hash and stored under `faiss_index/<hash>/`.
- `.gitignore` excludes `.env`, FAISS indexes, and temp files.
- Summarization, quiz, and bullet points use a chunked map-reduce approach to avoid token overflows.
