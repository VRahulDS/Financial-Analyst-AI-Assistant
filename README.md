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
    |__ db_setup/                 # Setting up the DB
data/
    ├── raw_files/                # Uploaded PDF files
DB/                               # Chroma DB
.venv                             # Virtual environment
paths.py                          # Paths for all the necessary files and directories
pyproject.toml
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- A Groq API key — free tier available at [console.groq.com](https://console.groq.com)

### Step-by-step

**1. Clone the repository**
```bash
git clone https://github.com/VRahulDS/Financial-Analyst-AI-Assistant.git
cd financial-analyst-assistant
```

**2. Install dependencies from `pyproject.toml`**
```bash
pip install -e .
```


**3. Set your Groq API key**
# Windows(.env file)
```.env
GROQ_API_KEY="your_key_here"
```

**4. Launch the app**
```bash
streamlit run app.py
```

---

## How to Use the Application

### Tab 1 — Upload & Ingest

Use this tab to load a financial document into the system.

1. Click **Browse files** and select a PDF financial report
2. Click the **Ingest Document** button
3. Watch the progress bar advance through five stages: Load → Chunk → Embed Model → Embed → Store
4. Once complete, the right panel shows page count and chunk count
5. Switch to the **Chat** tab — the database is now ready

> **Tip:** If you restart the app without uploading a new file, the previously ingested document is automatically detected and the Chat tab is ready immediately.

### Tab 2 — Chat

1. Type a question about the financial report in the input box at the bottom
2. Press **Enter** or click the send button
3. The system retrieves the most relevant document chunks and generates a grounded answer
4. Follow-up questions are supported — the chatbot maintains a rolling conversation summary

### Example Questions

- What was the total revenue for FY2023?
- How did operating expenses change year over year?
- What are the key risk factors mentioned in the report?
- Summarise the CEO's message to shareholders
- What is the company's cash position at end of period?
- Which segments contributed most to revenue growth?

### Switching to a New Document

Go back to the **Upload & Ingest** tab, upload the new PDF, and click **Ingest Document**. The old collection is automatically deleted and replaced. Chat history is also reset.

---

## Configuration Reference (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `llm` | `llama-3.1-8b-instant` | Groq model name |
| `vectordb.n_results` | `5` | Number of chunks retrieved per query |
| `memory_strategies.summarization_max_tokens` | `1000` | Max tokens for memory summary |
| `memory_strategies.trimming_window_size` | `6` | Number of turns before summarisation |

---

## Inputs & Outputs

### Input
- A PDF financial report (any length) — uploaded via the UI
- Natural language questions typed in the chat interface

### Output
- Markdown-formatted answers grounded in the uploaded document
- Bullet-pointed financial insights where appropriate
- `"The question is not answerable given the provided documents."` — if the answer cannot be found

### What the Model Will NOT Do
- Answer using outside knowledge — all answers are strictly document-grounded
- Provide speculative investment advice
- Fabricate financial figures or metrics

---

## ReadyTensor Submission Notes

This project was built for the **ReadyTensor Agentic AI Essentials** programme and demonstrates the following concepts:

| Concept | Implementation |
|---|---|
| Retrieval-Augmented Generation | ChromaDB vector retrieval grounding every LLM response |
| Vector Databases | ChromaDB `PersistentClient` with semantic similarity search |
| Agentic Memory | `ConversationSummaryMemory` for stateful multi-turn dialogue |
| Prompt Engineering | YAML-driven prompt construction with role, constraints, tone, and output format |
| Reasoning Strategies | Configurable CoT / ReAct / Self-Ask patterns in `config.yaml` |
| Modular Pipeline Design | Each stage (load, chunk, embed, store, retrieve, generate) is an independent, testable module |

---

*Finance Analyst AI · ReadyTensor Agentic AI Essentials · 2026*

Author
### Vasala Rahul
##### Data Scientist
