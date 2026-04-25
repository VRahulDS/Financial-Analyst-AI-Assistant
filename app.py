"""
app.py — Main Streamlit application
=====================================
CHANGES MADE FOR RAGAS (search for "# ← RAGAS" to find every change):
  1. Line ~14 : Added import for run_ragas_evaluation and DEFAULT_EVAL_QUESTIONS
  2. Line ~228: Added "🧪 RAGAS Eval" as third tab
  3. Line ~360: Added the full RAGAS Eval tab block at the bottom

Everything else is UNCHANGED from your original app.py.
"""

import uuid
import streamlit as st
import yaml
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

from src.logging.logger import setup_logger
from src.file_uploader.upload_document import upload_document
from src.load_pdf_file.load_pdf import load_pdf
from src.chunk_document.document_chunker import chunk_document
from src.db_setup.initialize_DB import initialize_DB
from src.embed_document.doc_embedder import embed_document
from src.response_generator.generate_response import generate_response
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH

# ← RAGAS: import the evaluation function and default Q&A pairs
# run_ragas_evaluation() is the main function called when user clicks "Run"
# DEFAULT_EVAL_QUESTIONS is the starting dataset shown in the data editor table
from src.ragas_eval.ragas_pipeline import run_ragas_evaluation, DEFAULT_EVAL_QUESTIONS

# ── Bootstrap ──────────────────────────────────────────────────────────────
setup_logger()

# ── Load config ────────────────────────────────────────────────────────────
with open(APP_CONFIG_FPATH, "r") as f:
    app_config = yaml.safe_load(f)

with open(PROMPT_CONFIG_FPATH, "r") as f:
    prompt_config = yaml.safe_load(f)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Finance Analyst AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #161b22; border-radius: 12px; padding: 4px; border: 1px solid #30363d; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #8b949e; font-weight: 500; padding: 8px 24px; font-family: 'DM Sans', sans-serif; }
.stTabs [aria-selected="true"] { background: #21262d !important; color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }
.uploadedFile { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 10px !important; }
.stButton > button { background: linear-gradient(135deg, #1f6feb, #388bfd); color: white; border: none; border-radius: 8px; padding: 10px 28px; font-family: 'DM Sans', sans-serif; font-weight: 500; font-size: 15px; transition: all 0.2s ease; }
.stButton > button:hover { background: linear-gradient(135deg, #388bfd, #58a6ff); transform: translateY(-1px); box-shadow: 0 4px 15px rgba(31,111,235,0.4); }
.stChatMessage { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 12px !important; margin-bottom: 12px !important; }
.stChatInputContainer { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 12px !important; }
.stChatFloatingInputContainer { position: fixed !important; bottom: 0 !important; padding-bottom: 16px !important; background: #0d1117 !important; z-index: 999 !important; }
section[data-testid="stChatMessageContainer"], .stChatMessageContainer { padding-bottom: 100px !important; }
.stAlert { border-radius: 10px !important; border: none !important; }
.status-ready { background: #0d2818; border: 1px solid #238636; border-radius: 10px; padding: 14px 18px; color: #3fb950; font-weight: 500; margin-bottom: 16px; }
.status-not-ready { background: #1c1700; border: 1px solid #9e6a03; border-radius: 10px; padding: 14px 18px; color: #e3b341; font-weight: 500; margin-bottom: 16px; }
.main-header { font-family: 'DM Serif Display', serif; font-size: 2.4rem; color: #e6edf3; margin-bottom: 4px; }
.sub-header { color: #8b949e; font-size: 1rem; margin-bottom: 32px; }
.metric-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px 20px; text-align: center; }
.metric-value { font-size: 1.8rem; font-weight: 600; color: #58a6ff; font-family: 'DM Serif Display', serif; }
.metric-label { font-size: 0.82rem; color: #8b949e; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "embed_model" not in st.session_state:
    st.session_state.embed_model = None
if "ingest_stats" not in st.session_state:
    st.session_state.ingest_stats = {}

# ── Always reconnect to the persistent ChromaDB on every rerun ────────────
if "collection" not in st.session_state:
    st.session_state.collection = initialize_DB(reset=False)

# Mark DB as ready if the collection already has documents from a previous run
if "db_ready" not in st.session_state:
    try:
        existing_count = st.session_state.collection.count()
        st.session_state.db_ready = existing_count > 0
        if st.session_state.db_ready and not st.session_state.ingest_stats:
            st.session_state.ingest_stats = {
                "pages": "?",
                "chunks": existing_count,
                "file": "Previously ingested document",
            }
    except Exception:
        st.session_state.db_ready = False


@st.cache_resource(show_spinner=False)
def load_embed_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📊 Finance Analyst AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">RAG-powered insights from your financial reports</div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────
# ← RAGAS: added "🧪  RAGAS Eval" as the third tab
tab_ingest, tab_chat, tab_eval = st.tabs(["📁  Upload & Ingest", "💬  Chat", "🧪  RAGAS Eval"])


# ══════════════════════════════════════════════════════════════════════════
#  TAB 1 – UPLOAD & INGEST  (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.markdown("### Upload a Financial Report")
    st.markdown("Upload a PDF report. Click **Ingest Document** to chunk, embed, and store it in the vector database.")

    col_up, col_info = st.columns([2, 1], gap="large")

    with col_up:
        file_path = upload_document()
        ingest_clicked = st.button("⚡ Ingest Document", disabled=(file_path is None))

        if ingest_clicked and file_path:
            progress_bar = st.progress(0, text="Starting ingestion…")

            progress_bar.progress(10, text="📄 Loading PDF…")
            docs = load_pdf(str(file_path))
            raw_text = "\n".join([doc.page_content for doc in docs])

            progress_bar.progress(30, text="✂️ Chunking document…")
            chunks = chunk_document(raw_text)

            progress_bar.progress(50, text="🧠 Loading embedding model…")
            if st.session_state.embed_model is None:
                st.session_state.embed_model = load_embed_model()
            embed_model = st.session_state.embed_model

            progress_bar.progress(65, text="🔢 Embedding chunks…")
            embeddings = embed_document(chunks, embed_model)

            progress_bar.progress(80, text="🗄️ Storing in vector database…")
            collection = initialize_DB(reset=True)
            ids = [str(i) for i in range(len(chunks))]
            collection.add(documents=chunks, embeddings=embeddings, ids=ids)
            st.session_state.collection = collection

            progress_bar.progress(100, text="✅ Ingestion complete!")
            st.session_state.db_ready = True
            st.session_state.chat_history = []
            st.session_state.ingest_stats = {
                "pages": len(docs),
                "chunks": len(chunks),
                "file": Path(file_path).name,
            }
            st.success("Document successfully ingested! Switch to the **Chat** tab to ask questions.")

    with col_info:
        st.markdown("#### Vector DB Status")
        if st.session_state.db_ready:
            stats = st.session_state.ingest_stats
            st.markdown('<div class="status-ready">🟢 &nbsp;Database ready</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:10px">
                <div class="metric-value">{stats.get('pages', '–')}</div>
                <div class="metric-label">Pages loaded</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats.get('chunks', '–')}</div>
                <div class="metric-label">Chunks stored</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"📄 {stats.get('file', '')}")
        else:
            st.markdown('<div class="status-not-ready">🟡 &nbsp;No document ingested yet</div>', unsafe_allow_html=True)
            st.caption("Upload and ingest a PDF to enable the chat.")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 2 – CHAT  (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not st.session_state.db_ready:
        st.info("⬅️ Please upload and ingest a financial report in the **Upload & Ingest** tab first.")
    else:
        if st.session_state.embed_model is None:
            with st.spinner("Loading embedding model…"):
                st.session_state.embed_model = load_embed_model()

        st.markdown("### Ask anything about your financial report")
        st.caption(f"Document: **{st.session_state.ingest_stats.get('file', '')}** &nbsp;|&nbsp; Session: `{st.session_state.session_id[:8]}…`")

        chat_container = st.container(height=520, border=False)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if user_input := st.chat_input("Ask a question about the financial report…"):
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)

            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Analysing report…"):
                        response = generate_response(
                            session_id=st.session_state.session_id,
                            query=user_input,
                            collection=st.session_state.collection,
                            embed_model=st.session_state.embed_model,
                            prompt_config=prompt_config,
                            app_config=app_config,
                        )
                    st.markdown(response)

            st.session_state.chat_history.append({"role": "assistant", "content": response})


# ══════════════════════════════════════════════════════════════════════════
#  TAB 3 – RAGAS EVALUATION   ← RAGAS: this entire block is new
# ══════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### 🧪 RAGAS Evaluation Pipeline")
    st.markdown(
        "Evaluate your RAG pipeline on **Faithfulness**, **Answer Relevancy**, "
        "**Context Precision**, and **Context Recall** using the RAGAS framework."
    )

    # Guard: DB must be populated before evaluation runs
    if not st.session_state.db_ready:
        st.info("⬅️ Please upload and ingest a financial report in the **Upload & Ingest** tab first.")
    else:

        # ── SECTION 1: Metric Selector ─────────────────────────────────────
        # User picks which of the 4 RAGAS metrics to run.
        # Unchecking a metric skips it and reduces LLM calls / run time.
        #
        # Metric explanations:
        #   Faithfulness      → "Did the answer come from the context, or was it hallucinated?"
        #   Answer Relevancy  → "Did the answer actually address the question asked?"
        #   Context Precision → "Were the retrieved chunks relevant to the question?"
        #   Context Recall    → "Did the retrieved chunks contain everything in ground_truth?"
        st.markdown("#### ① Select Metrics")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        use_faithfulness      = col_m1.checkbox("Faithfulness",       value=True,
            help="Does the answer only use info from the retrieved context?")
        use_answer_relevancy  = col_m2.checkbox("Answer Relevancy",   value=True,
            help="Is the answer relevant to the question?")
        use_context_precision = col_m3.checkbox("Context Precision",  value=True,
            help="Are the retrieved chunks actually useful for the question?")
        use_context_recall    = col_m4.checkbox("Context Recall",     value=True,
            help="Do the retrieved chunks cover the ground truth answer?")

        # ── SECTION 2: Q&A Sample Editor ──────────────────────────────────
        # st.data_editor shows an editable table.
        # Pre-filled with DEFAULT_EVAL_QUESTIONS from ragas_pipeline.py.
        # To use Eaton-specific questions, paste from eaton_2025_ragas_eval.csv.
        st.markdown("#### ② Edit Evaluation Questions & Ground Truths")
        st.caption(
            "Each row needs a **question** and a **ground_truth** (reference answer). "
            "Ground truth can be approximate — RAGAS uses it only to judge context recall."
        )

        import pandas as pd

        # Build initial DataFrame from the default questions list
        # (defined at top of ragas_pipeline.py as DEFAULT_EVAL_QUESTIONS)
        default_df = pd.DataFrame(DEFAULT_EVAL_QUESTIONS)

        # Render editable table — returns a new DataFrame with user's edits
        edited_df = st.data_editor(
            default_df,
            num_rows="dynamic",       # allows adding/deleting rows
            use_container_width=True,
            column_config={
                "question":     st.column_config.TextColumn("Question",     width="large"),
                "ground_truth": st.column_config.TextColumn("Ground Truth", width="large"),
            },
            key="ragas_eval_table",
        )

        # ── SECTION 3: Run Button ──────────────────────────────────────────
        st.markdown("#### ③ Run Evaluation")
        run_eval = st.button(
            "▶️ Run RAGAS Evaluation",
            type="primary",
            disabled=edited_df.empty,
        )

        if run_eval:
            import os
            from dotenv import load_dotenv
            from paths import ENV_FPATH

            # Load GROQ_API_KEY from .env
            # This is the same key used by generate_response() in generate_response.py
            load_dotenv(ENV_FPATH)
            groq_key = os.getenv("GROQ_API_KEY", "")

            if not groq_key:
                st.error("GROQ_API_KEY not found in .env — please add it.")
            else:
                # Ensure the embedding model is loaded
                if st.session_state.embed_model is None:
                    with st.spinner("Loading embedding model…"):
                        st.session_state.embed_model = load_embed_model()

                # Build list of RAGAS metric objects based on checkbox state
                # These are the same singleton objects imported in ragas_pipeline.py
                from ragas.metrics import (
                    faithfulness      as _faithfulness,
                    answer_relevancy  as _answer_relevancy,
                    context_precision as _context_precision,
                    context_recall    as _context_recall,
                )

                selected_metrics = []
                if use_faithfulness:      selected_metrics.append(_faithfulness)
                if use_answer_relevancy:  selected_metrics.append(_answer_relevancy)
                if use_context_precision: selected_metrics.append(_context_precision)
                if use_context_recall:    selected_metrics.append(_context_recall)

                if not selected_metrics:
                    st.warning("Please select at least one metric.")
                else:
                    # Convert edited DataFrame → list of dicts for ragas_pipeline.py
                    # Drop any rows with an empty question field
                    eval_samples = (
                        edited_df[["question", "ground_truth"]]
                        .dropna(subset=["question"])
                        .query("question.str.strip() != ''")
                        .to_dict("records")
                    )

                    with st.spinner(
                        f"Running RAGAS on {len(eval_samples)} samples — "
                        "this may take 1–3 minutes…"
                    ):
                        try:
                            # ── MAIN CALL to ragas_pipeline.py ─────────────
                            # Passes your ChromaDB collection, embed model,
                            # configs, and API key to run_ragas_evaluation().
                            #
                            # Internally it calls (per sample):
                            #   embed_query()       → src/embed_query/query_embedder.py
                            #   retrieve_doc()      → src/retrieve_documents/doc_retriever.py
                            #   generate_response() → src/response_generator/generate_response.py
                            #
                            # Then passes everything to ragas.evaluate() for scoring.
                            results_df = run_ragas_evaluation(
                                eval_samples=eval_samples,
                                collection=st.session_state.collection,
                                embed_model=st.session_state.embed_model,
                                prompt_config=prompt_config,
                                app_config=app_config,
                                groq_api_key=groq_key,
                                metrics=selected_metrics,
                            )
                            # Save results in session_state so they survive reruns
                            st.session_state["ragas_results"] = results_df

                        except Exception as e:
                            st.error(f"Evaluation failed: {e}")
                            st.exception(e)

        # ── SECTION 4: Results Display ─────────────────────────────────────
        # Only shown after a successful evaluation run.
        # Persists in session_state["ragas_results"] until next run.
        if "ragas_results" in st.session_state:
            results_df = st.session_state["ragas_results"]
            st.markdown("#### 📊 Evaluation Results")

            # Metric columns = everything except the metadata columns
            non_metric = {"question", "answer", "user_input", "response",
                          "contexts", "retrieved_contexts", "ground_truth", "reference"}
            metric_cols = [c for c in results_df.columns if c not in non_metric]

            # Detect which column holds the "question" text (varies by RAGAS version)
            question_col = "question" if "question" in results_df.columns else "user_input"

            # ── Average score cards ────────────────────────────────────────
            avg_row = results_df[results_df[question_col] == "⬛ AVERAGE"]
            if not avg_row.empty:
                summary_cols = st.columns(len(metric_cols))
                for i, col_name in enumerate(metric_cols):
                    val = float(avg_row.iloc[0][col_name])
                    color = (
                        "#3fb950" if val >= 0.7 else
                        "#e3b341" if val >= 0.4 else
                        "#f85149"
                    )
                    summary_cols[i].markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value" style="color:{color}">{val:.2f}</div>'
                        f'<div class="metric-label">{col_name.replace("_"," ").title()}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── Per-question detail table ──────────────────────────────────
            detail_df = (
                results_df[results_df[question_col] != "⬛ AVERAGE"]
                .reset_index(drop=True)
            )
            st.dataframe(detail_df, use_container_width=True)

            # ── Download button ────────────────────────────────────────────
            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results CSV",
                data=csv_bytes,
                file_name="ragas_evaluation_results.csv",
                mime="text/csv",
            )