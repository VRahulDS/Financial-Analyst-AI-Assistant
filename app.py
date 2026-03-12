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

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #161b22;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #30363d;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #8b949e;
    font-weight: 500;
    padding: 8px 24px;
    font-family: 'DM Sans', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
}

/* Upload area */
.uploadedFile {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 15px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(31,111,235,0.4);
}

/* Chat messages */
.stChatMessage {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
}

/* Chat input */
.stChatInputContainer {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
}

/* Pin chat input to the very bottom of the viewport */
.stChatFloatingInputContainer {
    position: fixed !important;
    bottom: 0 !important;
    padding-bottom: 16px !important;
    background: #0d1117 !important;
    z-index: 999 !important;
}

/* Add bottom padding so last message isn't hidden behind the fixed input */
section[data-testid="stChatMessageContainer"],
.stChatMessageContainer {
    padding-bottom: 100px !important;
}

/* Info / success boxes */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}

/* Status box */
.status-ready {
    background: #0d2818;
    border: 1px solid #238636;
    border-radius: 10px;
    padding: 14px 18px;
    color: #3fb950;
    font-weight: 500;
    margin-bottom: 16px;
}
.status-not-ready {
    background: #1c1700;
    border: 1px solid #9e6a03;
    border-radius: 10px;
    padding: 14px 18px;
    color: #e3b341;
    font-weight: 500;
    margin-bottom: 16px;
}

/* Header */
.main-header {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #e6edf3;
    margin-bottom: 4px;
}
.sub-header {
    color: #8b949e;
    font-size: 1rem;
    margin-bottom: 32px;
}

/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 600;
    color: #58a6ff;
    font-family: 'DM Serif Display', serif;
}
.metric-label {
    font-size: 0.82rem;
    color: #8b949e;
    margin-top: 2px;
}
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
# initialize_DB(reset=False) calls get_or_create_collection — safe to call
# every rerun; it never deletes data unless reset=True.
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
tab_ingest, tab_chat = st.tabs(["📁  Upload & Ingest", "💬  Chat"])


# ══════════════════════════════════════════════════════════════════════════
#  TAB 1 – UPLOAD & INGEST
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

            # Step 1 – Load PDF
            progress_bar.progress(10, text="📄 Loading PDF…")
            docs = load_pdf(str(file_path))
            raw_text = "\n".join([doc.page_content for doc in docs])

            # Step 2 – Chunk
            progress_bar.progress(30, text="✂️ Chunking document…")
            chunks = chunk_document(raw_text)

            # Step 3 – Load embedding model
            progress_bar.progress(50, text="🧠 Loading embedding model…")
            if st.session_state.embed_model is None:
                st.session_state.embed_model = load_embed_model()
            embed_model = st.session_state.embed_model

            # Step 4 – Embed chunks
            progress_bar.progress(65, text="🔢 Embedding chunks…")
            embeddings = embed_document(chunks, embed_model)

            # Step 5 – Store in ChromaDB (reset=True wipes old doc, starts fresh)
            progress_bar.progress(80, text="🗄️ Storing in vector database…")
            collection = initialize_DB(reset=True)   # ← only place reset=True is used
            ids = [str(i) for i in range(len(chunks))]
            collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=ids,
            )
            # Re-assign so the rest of the session uses the fresh collection
            st.session_state.collection = collection

            # Done
            progress_bar.progress(100, text="✅ Ingestion complete!")
            st.session_state.db_ready = True
            st.session_state.chat_history = []   # reset chat for new doc
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
#  TAB 2 – CHAT
# ══════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not st.session_state.db_ready:
        st.info("⬅️ Please upload and ingest a financial report in the **Upload & Ingest** tab first.")
    else:
        # Lazy-load embedding model if not already loaded
        if st.session_state.embed_model is None:
            with st.spinner("Loading embedding model…"):
                st.session_state.embed_model = load_embed_model()

        st.markdown("### Ask anything about your financial report")
        st.caption(f"Document: **{st.session_state.ingest_stats.get('file', '')}** &nbsp;|&nbsp; Session: `{st.session_state.session_id[:8]}…`")

        # Scrollable message history container — input floats below this
        chat_container = st.container(height=520, border=False)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat input — Streamlit renders this outside/after the container,
        # so it always sits at the bottom of the page
        if user_input := st.chat_input("Ask a question about the financial report…"):
            # Append to history then re-render inside the container
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)

            # Generate response
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