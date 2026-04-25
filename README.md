# Finance Analyst AI
### RAG-Powered Financial Document Chatbot

> Submitted to: **ReadyTensor Agentic AI Essentials**  
> Built with: LangChain · ChromaDB · Groq LLM · HuggingFace Embeddings · Streamlit

---

## Overview

Finance Analyst AI is an agentic Retrieval-Augmented Generation (RAG) chatbot that answers questions about financial reports. Upload a PDF — annual report, 10-K, earnings release, or any financial document — and the system ingests, chunks, embeds, and stores it in a persistent vector database. The chat interface then allows natural language Q&A grounded entirely in the uploaded document.

The project demonstrates core agentic patterns: **retrieval, grounded generation, tool use, and conversational memory**.

---

## Architecture & Pipeline

The system is composed of six modular pipeline stages:

| Stage | Description |
|---|---|
| **1. Upload** | User uploads a PDF via the Streamlit UI. Saved to the raw files directory. |
| **2. Load** | `PyPDFLoader` extracts text from all pages of the PDF. |
| **3. Chunk** | `RecursiveCharacterTextSplitter` splits text into 800-token chunks with 100-token overlap. |
| **4. Embed** | HuggingFace `all-MiniLM-L6-v2` converts chunks to dense vector embeddings. |
| **5. Store** | Embeddings and chunks are persisted in ChromaDB (`PersistentClient`) on disk. |
| **6. Retrieve & Generate** | At query time, the user question is embedded, top-N chunks are retrieved, and the Groq LLM generates a grounded answer with conversation memory. |

The retrieval and generation pipeline is evaluated using structured metrics to ensure grounded and relevant responses.

---

## Key Features

- **Persistent vector store** — ChromaDB survives app restarts; no need to re-ingest on every run
- **Smart reset** — uploading a new PDF wipes the old collection and rebuilds from scratch
- **Conversation memory** — `ConversationSummaryMemory` keeps a rolling summary across turns
- **Session isolation** — each browser session gets its own session ID and memory context
- **Grounded answers only** — the LLM is instructed never to use knowledge outside the document
- **Progress feedback** — step-by-step progress bar during ingestion
- **Auto-detect existing DB** — if a document was previously ingested, the Chat tab is ready immediately on restart

---

## Evaluation

The system was evaluated using the **RAGAS framework** to measure retrieval and generation quality.

### Metrics

| Metric | Score |
|---|---|
| **Faithfulness** | 0.70 |
| **Context Recall** | 0.97 |
| **Context Precision** | 0.40 |

### Insights

- High **context recall (0.97)** indicates strong coverage of relevant information
- Moderate **faithfulness (0.70)** suggests responses are mostly grounded in retrieved context
- Lower **context precision (0.40)** indicates retrieval of some irrelevant chunks, introducing noise

These results show that while the system retrieves relevant information effectively, improving retrieval precision can further enhance answer quality.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| LLM | Groq (`llama-3.1-8b-instant`) | Fast inference for answer generation |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Semantic chunk & query embeddings |
| Vector DB | ChromaDB (`PersistentClient`) | Persistent similarity search |
| Memory | LangChain `ConversationSummaryMemory` | Rolling multi-turn context |
| PDF Loader | LangChain `PyPDFLoader` | Text extraction from PDFs |
| Chunker | `RecursiveCharacterTextSplitter` | Overlap-aware text splitting |
| Frontend | Streamlit | Two-tab chat + ingest UI |
| Config | YAML (`config.yaml`) | Centralised model & prompt settings |

---

## Project Structure

```
financial-analyst-assistant/
├── app.py                        # Streamlit entry point
├── config/                       # LLM, memory, prompt, vectordb config
├── pyproject.toml                # Dependencies
└── src/
    ├── chunk_document/           # RecursiveCharacterTextSplitter wrapper
    ├── embed_document/           # Batch document embedder
    ├── embed_query/              # Single query embedder
    ├── generate_response/        # Core RAG + memory pipeline
    ├── load_pdf/                 # PyPDFLoader wrapper
    ├── logger/                   # Logging setup
    ├── prompts/                  # Prompt builder utilities
    ├── retrieve_documents/       # ChromaDB retriever
    ├── sessions/                 # Session store (per-user memory)
    └── upload_document/          # Streamlit file upload handler
    └── db_setup/                 # Setting up the DB
data/
├── raw_files/                    # Uploaded PDF files
DB/                               # Chroma DB
.venv                             # Virtual environment
paths.py                          # Paths for all the necessary files and directories
pyproject.toml
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- A Groq API key — free tier available at https://console.groq.com

### Step-by-step

**1. Clone the repository**
```bash
git clone https://github.com/VRahulDS/Financial-Analyst-AI-Assistant.git
cd financial-analyst-assistant
```

**2. Install dependencies**
```bash
pip install -e .
```

**3. Set your Groq API key**
```env
GROQ_API_KEY="your_key_here"
```

**4. Launch the app**
```bash
streamlit run app.py
```

---

## How to Use the Application

### Tab 1 — Upload & Ingest

1. Upload a PDF financial report
2. Click **Ingest Document**
3. Wait for processing (Load → Chunk → Embed → Store)
4. Switch to Chat tab

### Tab 2 — Chat

- Ask questions about the document
- Follow-up questions supported via memory

---

## Inputs & Outputs

### Input

- PDF financial report
- Natural language queries

### Output

- Grounded answers based on document
- Structured markdown responses
- Fallback when answer not found

---

## Limitations

- Works best with **text-based PDFs**
- Performance degrades for **scanned/image-based documents** (no OCR support)
- Multi-column or complex layouts may affect text extraction quality
- Retrieval precision may introduce irrelevant context in some responses

---

## ReadyTensor Submission Notes

This project demonstrates:

- Retrieval-Augmented Generation
- Vector database usage
- Agentic conversational memory
- Prompt engineering
- Modular pipeline design

---

*Finance Analyst AI · ReadyTensor Agentic AI Essentials · 2026*

---

## Author

### Vasala Rahul
##### Data Scientist