import chromadb
from paths import DB_DIR
import logging
logger = logging.getLogger(__name__)

def initialize_DB(reset=False):
    
    try:
        logger.info("setting up the DB")
        client = chromadb.PersistentClient(path=DB_DIR)
        if reset:
            try:
                client.delete_collection("publications")
            except:
                pass

        collection = client.get_or_create_collection(
            name="publications"
        )
        return collection
    except Exception as e:
        logger.error("DB initialization failed: {e}") 