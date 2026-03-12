import logging

logger = logging.getLogger(__name__)


def store_vectors(chunks, embeddings, collection, file_name=None):

    if len(chunks) != len(embeddings):
        logger.error("Mismatch between chunks and embeddings")
        return

    if len(chunks) == 0 or len(embeddings) == 0:
        logger.info("Chunks or embeddings are empty. Nothing to store.")
        return

    try:
        next_id = collection.count()
        ids = [f"document_{i}" for i in range(next_id, next_id + len(chunks))]
        collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=chunks
        )

        logger.info(f"Added {len(chunks)} chunks to vector DB from file: {file_name}")

    except Exception as e:
        logger.error(f"Failed ingesting the documents: {e}")