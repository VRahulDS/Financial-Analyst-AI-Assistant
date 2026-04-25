"""
src/ragas_eval/ragas_pipeline.py
=================================
RAGAS Evaluation Pipeline for the Financial Analyst Assistant.

HOW IT FITS IN YOUR PROJECT:
─────────────────────────────
Your existing RAG pipeline flow is:

  PDF  →  load_pdf  →  chunk_document  →  embed_document  →  ChromaDB
                                                                  ↓
  User Query  →  embed_query  →  retrieve_doc  →  build_prompt  →  ChatGroq  →  Response

RAGAS sits AFTER the full pipeline and measures the quality of each step:

  ┌─────────────────────────────────────────────────────────┐
  │  For each eval question:                                 │
  │                                                          │
  │  1. embed_query(question)         ← your existing fn     │
  │  2. retrieve_doc(embedding)       ← your existing fn     │
  │  3. generate_response(question)   ← your existing fn     │
  │                                                          │
  │  RAGAS then scores:                                      │
  │  - Faithfulness    : answer vs retrieved chunks          │
  │  - Answer Relevancy: answer vs question                  │
  │  - Context Precision: retrieved chunks vs question       │
  │  - Context Recall  : retrieved chunks vs ground_truth    │
  └─────────────────────────────────────────────────────────┘

WHAT YOU NEED TO PROVIDE:
  - eval_samples  : list of {"question": ..., "ground_truth": ...}
  - collection    : your ChromaDB collection (already populated)
  - embed_model   : your HuggingFaceEmbeddings model
  - prompt_config : your loaded prompt_config.yaml dict
  - app_config    : your loaded config.yaml dict
  - groq_api_key  : your Groq API key string
"""

import logging
import pandas as pd
from typing import Optional

# ── LLM (same one your app already uses) ────────────────────────────────────
from langchain_groq import ChatGroq

# ── RAGAS core ───────────────────────────────────────────────────────────────
# evaluate() is the main entry point — it takes a HuggingFace Dataset
# and a list of metric objects, runs all metrics, returns a Result object.
from ragas import evaluate

# ── RAGAS metrics ─────────────────────────────────────────────────────────────
# Each metric is a singleton object. You pass the ones you want to evaluate().
#
# faithfulness      — Does the answer only contain claims supported by context?
#                     Score 0–1. Low score → hallucination risk.
#
# answer_relevancy  — Is the answer relevant to the question?
#                     Score 0–1. Low score → answer drifted from question.
#
# context_precision — Are the top-k retrieved chunks actually useful?
#                     Score 0–1. Low score → retrieval is noisy.
#
# context_recall    — Do the retrieved chunks cover the ground truth answer?
#                     Score 0–1. Low score → relevant info missing from DB
#                     or n_results is too low in config.yaml.
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# ── RAGAS wrappers ────────────────────────────────────────────────────────────
# RAGAS has its own LLM and Embeddings interfaces.
# These wrappers let you pass your existing LangChain objects directly
# instead of setting up separate RAGAS-specific clients.
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ── HuggingFace Datasets ──────────────────────────────────────────────────────
# RAGAS expects input as a HuggingFace Dataset with these exact column names:
#   "question"     : the user question (str)
#   "answer"       : the generated answer from your RAG pipeline (str)
#   "contexts"     : list of retrieved document chunks (list[str])
#   "ground_truth" : reference answer for context_recall metric (str)
from datasets import Dataset

# ── Your existing pipeline functions ─────────────────────────────────────────
# These are the SAME functions your app.py already uses.
# RAGAS evaluation just calls them in a loop with eval questions.
from src.embed_query.query_embedder import embed_query        # turns question → vector
from src.retrieve_documents.doc_retriever import retrieve_doc # vector → top-k chunks
from src.response_generator.generate_response import generate_response  # full RAG response
from src.sessions.session_store import session_store          # session memory manager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT EVALUATION QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
# These are shown in the RAGAS Eval tab UI as the starting dataset.
# Users can edit/replace them in the Streamlit data editor.
#
# FORMAT RULES:
#   "question"     → a realistic question a user would ask the chatbot
#   "ground_truth" → the ideal answer (used ONLY by context_recall metric)
#                    It does NOT need to be exact — approximate is fine.
#                    If you skip context_recall, ground_truth is not used.
DEFAULT_EVAL_QUESTIONS = [
    {
        "question": "What is the total revenue reported in the document?",
        "ground_truth": "The document contains the total revenue figure for the reported period.",
    },
    {
        "question": "What are the key risk factors mentioned?",
        "ground_truth": "The document outlines the major risk factors affecting the company.",
    },
    {
        "question": "What is the net profit or net income for the period?",
        "ground_truth": "The document states the net profit or net income for the period.",
    },
    {
        "question": "What is the earnings per share (EPS)?",
        "ground_truth": "The document provides the EPS figure for the reporting period.",
    },
    {
        "question": "What are the main business segments or divisions?",
        "ground_truth": "The document describes the company's main business segments or divisions.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL FUNCTION — Build the RAGAS Dataset
# ─────────────────────────────────────────────────────────────────────────────
def _build_ragas_dataset(
    eval_samples: list[dict],   # [{"question": ..., "ground_truth": ...}, ...]
    collection,                  # your ChromaDB collection object
    embed_model,                 # your HuggingFaceEmbeddings object
    prompt_config: dict,         # loaded from prompt_config.yaml
    app_config: dict,            # loaded from config.yaml
    groq_api_key: str,           # GROQ_API_KEY from .env
) -> Dataset:
    """
    For each eval question, this function:
      1. Embeds the question using YOUR embed_query() function
      2. Retrieves top-k chunks using YOUR retrieve_doc() function
      3. Generates an answer using YOUR generate_response() function
      4. Collects question + answer + contexts + ground_truth into a dict

    Then converts that dict into a HuggingFace Dataset, which is the
    format RAGAS expects.

    WHY A SEPARATE SESSION PER SAMPLE?
    ────────────────────────────────────
    Your generate_response() uses session memory (ConversationSummaryMemory).
    If we reused the same session, conversation history from Q1 would bleed
    into Q2, skewing the scores.
    By using a fresh session ID per sample (e.g. "ragas_eval_0")
    and clearing it from session_store.sessions before use, each question
    is evaluated independently — just like a fresh chat would be.
    """

    # These 4 lists will become the 4 columns of the RAGAS Dataset.
    # They must all have the same length after the loop.
    rows = {
        "question":     [],   # str       — the eval question
        "answer":       [],   # str       — your RAG pipeline's generated answer
        "contexts":     [],   # list[str] — top-k retrieved document chunks
        "ground_truth": [],   # str       — reference answer (for context_recall)
    }

    for i, sample in enumerate(eval_samples):
        question     = sample["question"].strip()
        ground_truth = sample.get("ground_truth", "").strip()

        logger.info(f"[RAGAS] Sample {i+1}/{len(eval_samples)}: {question[:60]}…")

        # ── Step A: Embed the question ──────────────────────────────────────
        # This calls src/embed_query/query_embedder.py → embed_query()
        # which runs: model.embed_query(question)
        # Returns: list[float]  e.g. [0.023, -0.14, 0.87, ...]  (384 dimensions)
        query_embedding = embed_query(question, embed_model)

        # ── Step B: Retrieve relevant document chunks ───────────────────────
        # This calls src/retrieve_documents/doc_retriever.py → retrieve_doc()
        # which runs: collection.query(query_embeddings=[...], n_results=5)
        # Returns: list[str]  e.g. ["Eaton net sales were $27.4B...", "Segment margin..."]
        #
        # n_results comes from your config.yaml → vectordb.n_results (default 5)
        # To improve context_recall, increase this to 7 or 10.
        n_results = app_config["vectordb"]["n_results"]
        retrieved_chunks = retrieve_doc(query_embedding, collection, n_results)

        # ── Step C: Generate an answer via the full RAG pipeline ────────────
        # This calls src/response_generator/generate_response.py → generate_response()
        # which builds the full prompt (with context + memory) and calls ChatGroq.
        # Returns: str  e.g. "Eaton's total net sales for 2025 were $27.4 billion..."
        #
        # IMPORTANT: use a unique session_id per sample.
        # If you reuse session_id, the ConversationSummaryMemory from sample 0
        # will be included in the prompt for sample 1, making scores unreliable.
        session_id = f"ragas_eval_{i}"

        # Clear any leftover session from a previous eval run in the same app session.
        # session_store.sessions is the dict inside SessionStore.__init__()
        # in src/sessions/session_store.py
        session_store.sessions.pop(session_id, None)

        answer = generate_response(
            session_id=session_id,
            query=question,
            collection=collection,
            embed_model=embed_model,
            prompt_config=prompt_config,
            app_config=app_config,
        )

        # ── Step D: Append all 4 values to their respective lists ──────────
        rows["question"].append(question)
        rows["answer"].append(answer)
        rows["contexts"].append(retrieved_chunks)  # MUST be list[str], not str
        rows["ground_truth"].append(ground_truth)

    # Convert dict-of-lists → HuggingFace Dataset
    # This is the only format ragas.evaluate() accepts as input.
    #
    # Result looks like:
    # Dataset({
    #     features: ['question', 'answer', 'contexts', 'ground_truth'],
    #     num_rows: 5
    # })
    return Dataset.from_dict(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION — Run full RAGAS evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_ragas_evaluation(
    eval_samples: list[dict],
    collection,
    embed_model,
    prompt_config: dict,
    app_config: dict,
    groq_api_key: str,
    metrics: Optional[list] = None,
) -> pd.DataFrame:
    """
    Main entry point called from the RAGAS Eval tab in app.py.

    Full internal flow:
      1. Wrap your LangChain LLM + embeddings into RAGAS-compatible wrappers
      2. Assign them to each metric object (metrics need LLM to score)
      3. Call _build_ragas_dataset() → runs your full RAG pipeline per question
      4. Call ragas.evaluate() → runs all metrics on the dataset
      5. Format results as a pandas DataFrame with an average summary row

    Parameters:
      eval_samples  — list of dicts: [{"question": ..., "ground_truth": ...}]
      collection    — ChromaDB collection (must already be populated)
      embed_model   — HuggingFaceEmbeddings (same as used in app.py)
      prompt_config — dict loaded from prompt_config.yaml
      app_config    — dict loaded from config.yaml
      groq_api_key  — string from os.getenv("GROQ_API_KEY")
      metrics       — list of RAGAS metric objects; defaults to all 4

    Returns:
      pd.DataFrame with columns:
        question | answer | faithfulness | answer_relevancy |
        context_precision | context_recall
      Plus a final "⬛ AVERAGE" row.
    """

    # Default to all 4 metrics if none are specified
    if metrics is None:
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # ── Wrap your LLM for RAGAS ──────────────────────────────────────────────
    # RAGAS needs an LLM internally to compute scores.
    # Example: for faithfulness, RAGAS asks the LLM:
    #   "Given these contexts: [...], is this claim: '...' supported? Answer Yes or No."
    #
    # LangchainLLMWrapper makes ChatGroq compatible with RAGAS's internal interface.
    # We use the same model from config.yaml → llm ("llama-3.1-8b-instant")
    # so no new API key or model setup is needed.
    ragas_llm = LangchainLLMWrapper(
        ChatGroq(
            model=app_config["llm"],  # from config.yaml, e.g. "llama-3.1-8b-instant"
            temperature=0,             # must be 0 — RAGAS scoring needs deterministic output
            api_key=groq_api_key,
        )
    )

    # ── Wrap your embeddings for RAGAS ───────────────────────────────────────
    # answer_relevancy metric computes cosine similarity between:
    #   - the embedding of the question
    #   - the embedding of the generated answer
    # LangchainEmbeddingsWrapper makes your HuggingFaceEmbeddings (all-MiniLM-L6-v2)
    # compatible with RAGAS's internal embeddings interface.
    ragas_embeddings = LangchainEmbeddingsWrapper(embed_model)

    # ── Assign LLM and embeddings to each metric object ──────────────────────
    # RAGAS metric objects (faithfulness, etc.) are stateful singletons.
    # You must set .llm and .embeddings on them before calling evaluate().
    # Each metric uses them differently:
    #
    #   faithfulness      → LLM: "Is claim X supported by context Y?"
    #   answer_relevancy  → embeddings: cosine_sim(question, answer)
    #   context_precision → LLM: "Is chunk X useful for answering question Y?"
    #   context_recall    → LLM: "Does chunk X contain info from ground_truth Y?"
    for metric in metrics:
        metric.llm = ragas_llm
        # Not all metrics use embeddings (faithfulness doesn't),
        # so we only set it if the metric has an embeddings attribute.
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    # ── Build the evaluation dataset ─────────────────────────────────────────
    # This runs your full RAG pipeline once per eval question.
    # Time estimate: ~5–15 seconds per question depending on Groq latency.
    logger.info(f"[RAGAS] Building dataset for {len(eval_samples)} samples…")
    dataset = _build_ragas_dataset(
        eval_samples=eval_samples,
        collection=collection,
        embed_model=embed_model,
        prompt_config=prompt_config,
        app_config=app_config,
        groq_api_key=groq_api_key,
    )

    # ── Run RAGAS scoring ─────────────────────────────────────────────────────
    # ragas.evaluate() internally:
    #   1. Iterates over each row in the dataset
    #   2. For each row, runs every selected metric
    #   3. Each metric makes 1–5 LLM calls to score that row
    #   4. Aggregates scores per row
    #
    # Total LLM calls ≈ num_samples × num_metrics × avg_calls_per_metric
    # e.g. 5 samples × 4 metrics × ~3 calls = ~60 LLM calls to Groq
    logger.info("[RAGAS] Running scoring…")
    result = evaluate(dataset=dataset, metrics=metrics)

    # ── Convert to DataFrame ──────────────────────────────────────────────────
    # result.to_pandas() column names vary by RAGAS version:
    #   older (≤0.1.x) : "question", "answer", "contexts", "ground_truth", <metric cols>
    #   newer (≥0.2.x) : "user_input", "response", "retrieved_contexts",
    #                     "reference", <metric cols>
    # We handle both by normalising to a common set of names first.
    scores_df: pd.DataFrame = result.to_pandas()

    logger.info(f"[RAGAS] Raw result columns: {list(scores_df.columns)}")

    # ── Normalise column names across RAGAS versions ──────────────────────────
    rename_map = {
        # newer → our standard names
        "user_input":          "question",
        "response":            "answer",
        "retrieved_contexts":  "contexts",
        "reference":           "ground_truth",
    }
    scores_df = scores_df.rename(columns=rename_map)

    # ── Identify metric columns (everything that is not metadata) ─────────────
    non_metric = {"question", "answer", "contexts", "ground_truth"}
    metric_cols = [c for c in scores_df.columns if c not in non_metric]

    # ── Build display DataFrame with only the columns that actually exist ──────
    display_cols = []
    if "question" in scores_df.columns:
        display_cols.append("question")
    if "answer" in scores_df.columns:
        display_cols.append("answer")
    display_cols += metric_cols

    scores_df = scores_df[display_cols]

    # ── Append an AVERAGE summary row ─────────────────────────────────────────
    avg_row: dict = {}
    if "question" in display_cols:
        avg_row["question"] = "⬛ AVERAGE"
    if "answer" in display_cols:
        avg_row["answer"] = "—"
    for col in metric_cols:
        avg_row[col] = round(pd.to_numeric(scores_df[col], errors="coerce").mean(), 4)

    scores_df = pd.concat(
        [scores_df, pd.DataFrame([avg_row])],
        ignore_index=True
    )

    logger.info("[RAGAS] Evaluation complete.")
    return scores_df