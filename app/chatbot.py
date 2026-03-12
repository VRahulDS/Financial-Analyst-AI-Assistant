import uuid
from src.response_generator.generate_response import generate_response
from src.sessions.session_store import session_store

def start_session() -> str:
    """
    Creates a new session ID.
    """
    return str(uuid.uuid4())

def chat(session_id: str, query: str, collection, embed_model, prompt_config, app_config):
    """
    Main chat function called from frontend.
    """
    response = generate_response(
        session_id=session_id,
        query=query,
        collection=collection,
        embed_model=embed_model,
        prompt_config=prompt_config,
        app_config=app_config
    )
    return response