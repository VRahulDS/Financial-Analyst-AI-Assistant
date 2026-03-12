import logging
logger = logging.getLogger(__name__)

def retrieve_doc(query_embedding, collection, n_results=4):
    relevant_results = {
        "ids": [],
        "documents": [],
        "distances": [],
    }

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )

    logger.info(f"Retrieving top {n_results} documents...")

    return results["documents"][0]