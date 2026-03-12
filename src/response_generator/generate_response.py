import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_classic.memory import ConversationSummaryMemory
from src.embed_query.query_embedder import embed_query
from src.retrieve_documents.doc_retriever import retrieve_doc
from src.sessions.session_store import session_store
from src.prompts.prompt_builder import build_prompt_from_config
from paths import ENV_FPATH
from dotenv import load_dotenv
import os

load_dotenv(ENV_FPATH)

logger = logging.getLogger(__name__)


def generate_response(
        session_id: str,
        query: str,
        collection,
        embed_model,
        prompt_config: dict,
        app_config: dict
) -> str:

    session = session_store.get_session(session_id)

    if "memory" not in session:
        session["memory"] = ConversationSummaryMemory(
            llm=ChatGroq(model=app_config["llm"], temperature=0, api_key=os.getenv('GROQ_API_KEY')),
            max_token_limit=app_config["memory_strategies"]["summarization_max_tokens"],
            input_key="input",
            memory_key="chat_history",
        )

    memory = session["memory"]

    query_embedding = embed_query(query, embed_model)
    n_results = app_config["vectordb"]["n_results"]
    docs = retrieve_doc(query_embedding, collection, n_results)
    context = "\n\n".join(docs)

    logger.info(f"Retrieved {len(docs)} document chunks for session '{session_id}'.")

    prompt_text = build_prompt_from_config(
        config=prompt_config["finance_analyst_rag_prompt"],
        input_data=context,
        app_config=app_config,
    )

    memory_summary = memory.load_memory_variables({}).get("chat_history", "")

    final_prompt = (
        f"{prompt_text}\n\n"
        f"Conversation Summary:\n{memory_summary}\n\n"
        f"User Question:\n{query}"
    )

    llm = ChatGroq(model=app_config["llm"], temperature=0, api_key=os.getenv('GROQ_API_KEY'))
    response = llm.invoke([HumanMessage(content=final_prompt)])

    logger.info(f"LLM response generated for session '{session_id}'.")

    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(response.content)

    return response.content