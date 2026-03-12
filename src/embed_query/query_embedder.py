import logging

logger = logging.getLogger(__name__)


def embed_query(query, model):
    if not query or query.strip() == "":
        logger.warning("Query is empty")
        return []

    try:
        embedding = model.embed_query(query)
        logger.info("Query embedding created successfully")
        return embedding

    except Exception as e:
        logger.error(f"Query embedding failed: {e}")
        return []