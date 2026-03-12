from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
logger = logging.getLogger(__name__)

def chunk_document(file_content, chunk_size=800, overlap_size=100):

    if not file_content:
        logger.info("chunk failed as the content is empty")
        return []

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size
        )

        chunks = splitter.split_text(file_content)
        logger.info("chunks are created")
        return chunks

    except Exception as e:
        logger.error(f"unable to create chunks: {e}")
        return []