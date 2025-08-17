# Copilot Instructions for StudyBuddy

## Project Overview
StudyBuddy is a Streamlit-based AI assistant for studying with PDFs. It enables users to upload a PDF, then interact with it via chat, summarization, quiz generation, and bullet-point extraction. The backend leverages Azure OpenAI for both embeddings and chat, and FAISS for vector search.

## Architecture & Data Flow
- **PDF Upload & Parsing**: User uploads a PDF via Streamlit UI. `load_pdf.py` loads and splits the document into text chunks.
- **Embedding & Storage**: `embed_store.py` (using `get_embeddings()` from `utils.py`) embeds chunks with Azure OpenAI and stores them in a local FAISS index (`faiss_index/`).
- **Retrieval**: On each question, the FAISS vectorstore is loaded and used to retrieve relevant chunks for the LLM.
- **LLM Usage**: `chatbot.py` loads the Azure OpenAI chat model. The main app (`app.py`) uses this for chat, summarization, quiz, and bullet-point features.
- **UI Features**: Each feature (chat, summarizer, quiz, bullet points) is modularized in its own script and imported into `app.py`.

## Key Files & Patterns
- `app.py`: Main Streamlit app. Orchestrates all features and UI. Uses session state for chat history.
- `load_pdf.py`: PDF loading and chunking logic.
- `embed_store.py`: Embeds and stores document chunks in FAISS.
- `utils.py`: Provides `get_embeddings()` for consistent embedding model config.
- `chatbot.py`: Loads the Azure OpenAI chat model.
- `summarizer.py`, `quiz_generator.py`, `bullet_points.py`: Modular UI components for each feature.
- `.env`: Stores Azure OpenAI credentials and deployment info. Loaded via `dotenv`.
- `faiss_index/`: Stores FAISS index and metadata for retrieval.

## Developer Workflows
- **Run the app**: `streamlit run app.py`
- **Install dependencies**: Use `pip install -r requirements.txt` (if present) or install main packages manually.
- **Environment**: Always activate the Python virtual environment (`venv`) before running or installing.
- **Add new features**: Create a new script (e.g., `my_feature.py`), define a `*_ui(llm, retriever)` function, and import/call it in `app.py` after LLM/retriever are initialized.

## Project-Specific Conventions
- All LLM and embedding configs are sourced from `.env` via `os.getenv` and `load_dotenv()`.
- Use `get_embeddings()` from `utils.py` everywhere to ensure consistent embedding model parameters.
- For retrieval, always load FAISS with `allow_dangerous_deserialization=True` (trusted local index only).
- UI features are implemented as Streamlit expanders or chat sections for modularity and clarity.
- Quiz and summarizer features use the LLM directly with custom prompts, and answers are shown in expanders for interactivity.

## Integration & External Dependencies
- **Azure OpenAI**: Used for both embeddings and chat. Requires correct deployment names and endpoint in `.env`.
- **FAISS**: Used for vector search. Index is stored locally and loaded as needed.
- **Streamlit**: All UI and user interaction.
- **PyMuPDF**: For PDF parsing.
- **python-dotenv**: For environment variable management.

## Examples
- To add a new feature, create `my_feature.py`:
  ```python
  def my_feature_ui(llm, retriever):
      with st.expander("My Feature"):
          # ...
  ```
  Then in `app.py`:
  ```python
  from my_feature import my_feature_ui
  my_feature_ui(llm, retriever)
  ```
- To update credentials, edit `.env` and restart the app.

---
For questions about project structure or adding features, see `app.py` for integration patterns. For embedding/chat config, see `utils.py` and `.env`.
