import logging
import torch
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def embed_document(docs, model):

    if len(docs) == 0:
        logger.info("embeds are empty")
        return []

    try:
        embeddings = model.embed_documents(docs)
        logger.info(f"Created embeddings for {len(docs)} documents")
        return embeddings

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return []