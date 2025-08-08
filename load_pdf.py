from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

def load_and_split_pdf(path, chunk_size: int = 1000, chunk_overlap: int = 100):
    """Load a PDF from disk and split into text chunks.

    Returns an empty list if parsing fails or no text is found.
    """
    try:
        loader = PyMuPDFLoader(path)
        documents = loader.load()
        if not documents:
            logger.warning("No text extracted from PDF: %s", path)
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        return chunks
    except Exception as e:
        logger.exception("Failed to load/split PDF %s: %s", path, e)
        return []
