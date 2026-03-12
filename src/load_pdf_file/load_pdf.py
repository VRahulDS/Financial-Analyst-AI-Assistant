from langchain_community.document_loaders import PyPDFLoader
import logging

logger = logging.getLogger(__name__)

def load_pdf(file_path):
    logger.info("loading the PDF file")
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logger.info(f"Completed loading the PDF with {len(docs)} documents")
        return docs
    except Exception as e:
        logger.error(f"Unable to load the PDF: {e}")



