# RAG It - Multimodal Multi-Format Document Search Engine

[![Build Status](https://img.shields.io/badge/verification-passed-green.svg)](scripts/verify_rag.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)

**RAG It** is a modular, high-performance hybrid Retrieval-Augmented Generation (RAG) search application. It integrates local vector search (powered by **ChromaDB** and local **Hugging Face** Sentence Transformers) with keyword search (powered by **BM25**) to parse, index, and query various document formats with high groundedness and source citations.

---

## 🚀 Key Features

* **Multi-Format Ingestion:** Dynamic extraction and indexing support for:
  * PDF (`.pdf`) - extracts layout text, structured tables, and visual figures.
  * Word (`.docx`) - parses paragraph flows and tabular grids sequentially.
  * Excel (`.xlsx`, `.xls`) & CSV (`.csv`) - transforms grid sheets into Markdown.
  * Markdown (`.md`) & Plain Text (`.txt`) - chunks sections dynamically by header hierarchies.
* **Row-Splitting Table Chunking:** Automatically segments massive spreadsheets and CSVs into bite-sized **50-row chunks**, prepending the column header to each sub-table to maintain structural context without exceeding LLM context windows.
* **Free Local Embeddings:** Uses the Hugging Face `all-MiniLM-L6-v2` model locally on CPU/GPU for 100% free offline indexing with zero API token costs.
* **Hybrid Retriever & Rank Fusion:** Combines semantic and keyword matches using query decomposition and **Reciprocal Rank Fusion (RRF)** for extremely accurate retrieval scores.
* **Table-Aware Context Enhancement:** Automatically retrieves adjacent text chunks on pages containing tables when a table is queried, preventing table rows from being evaluated in isolation.
* **Global Query Boosting:** Uses the LLM to identify requests for *all* figures or *all* tables, and overrides localized search thresholds to fetch all matching assets.
* **Automatic Image Launcher:** Automatically parses image citation coordinates from answers and launches the cropped image file in the default Windows photo viewer.
* **Code Grounding Disclaimer (Rule 7):** Requires the LLM to check if requested programming scripts exist in the document context, outputting a clear disclaimer before generating a hypothetical code block.
* **Collection Isolation:** Automatically separates ChromaDB vector collections by appending the dimension size (e.g. `collection_384` vs. `collection_1536`), preventing vector dimension crashes when switching embedding pipelines.
* **Evaluation Scorecard:** Features an automated groundedness benchmark suite with an execution scorecard reporting factual accuracy, recall, and response latency.

---

## 🧠 Deep Model Architectures

The system utilizes a dual-model hybrid architecture that separates local representation learning from remote generative inference:

### 1. Representation & Embedding Model (Local/Offline)
* **Model:** `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face)
* **Architecture: MiniLM (Mini Language Model)**
  * MiniLM is a distilled transformer architecture designed for high-efficiency embedding generation while retaining 99% of BERT's representation quality.
  * It employs a **6-layer Transformer encoder** with **12 self-attention heads** and a hidden layer dimension of **384**.
  * Chunks of text are tokenized and projected into a **384-dimensional dense vector space** where semantic similarities are modeled via cosine proximity.
  * Runs completely locally on CPU or GPU (CUDA), allowing 100% free vector indexing of document sheets and text blocks.

### 2. Synthesis & Generative Model (Remote/Online)
* **Model:** `gpt-4o` (OpenAI API)
* **Architecture: Generative Pre-trained Transformer 4o (Omni)**
  * A state-of-the-art multimodal transformer capable of understanding text, tables, and images.
  * **Multimodal Visual Parser:** Extracts and summarizes structural tables and figure crops visually, translating them into layout-aware Markdown context blocks.
  * **Cognitive Inference:** Rewrites conversational queries, decomposes complex questions, classifies user intents, and synthesizes citation-grounded final answers.

---

## 📁 Project Structure

```text
rag-it/
│
├── multimodal_rag/               # Core Python Package
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Settings, folder directories, and token trackers
│   ├── document_processor.py     # Parser for PDF, Word, Excel, CSV, TXT, and MD files
│   ├── embeddings.py             # Free local HuggingFace & OpenAI API embedders
│   ├── vector_store.py           # ChromaDB Vector Store indexing and query managers
│   ├── keyword_index.py          # BM25 keyword index matching manager
│   ├── retriever.py              # Hybrid retriever using RRF rank-fusing
│   └── generator.py              # Query intent classifier, rewriting, and synthesizers
│
├── scripts/                      # Executable scripts folder
│   ├── cli.py                    # Primary Command-line Interface
│   ├── run.py                    # Executable CLI wrapper
│   ├── verify_rag.py             # Automated pipeline integration verification tests
│   └── evaluate_rag.py           # Evaluation QA benchmark suite (Groundedness Scorecard)
│
├── data/                         # User data (ignored by Git)
│   ├── documents/                # Cached documents folder
│   ├── vector_db/                # Chroma database files
│   ├── cache/                    # Keyword index dumps & media crops
│   └── .gitkeep                  # Preserves folder in Git
│
├── docs/                         # System documentation
│   ├── architecture.md           # Architecture data flow diagrams
│   └── usage.md                  # Settings and command usage guides
│
├── examples/                     # Examples folder
│   ├── sample_queries.txt        # Sample search queries list
│   └── sample_documents/         # Sample PDF & CSV files
│
├── requirements.txt              # Project third-party dependencies
├── README.md                     # Project README manual
├── LICENSE                       # MIT License
├── .gitignore                    # Excludes caches and databases from Git
├── .env.example                  # Environment keys template
├── rag_it.bat                    # Windows shortcut batch file to run CLI
└── launch_chat.bat               # Windows shortcut batch file to start interactive chat
```

---

## 🛠️ Setup & Installation

1. **Install System Dependencies:**
   Make sure you have Python 3.10+ installed.

2. **Install Package Dependencies:**
   Run the following commands to install required libraries:
   ```bash
   pip install -r requirements.txt
   pip install pandas openpyxl python-docx tabulate
   ```

3. **Configure OpenAI API Key:**
   Configure your OpenAI key (used only for text generation synthesis and evaluation):
   ```bash
   rag_it config
   ```
   *(Securely prompts for the key. Alternatively, run `rag_it config --key "YOUR-KEY"` or copy `.env.example` to `.env`)*

---

## 💻 CLI Commands

Run commands via the Windows shortcut `rag_it` or by executing `python scripts/run.py`:

### 🕹️ Primary Control Panel (Recommended)
Launch the interactive Control Panel to manage all services in one visual menu:
```bash
rag_it start
```
*(Or simply run `rag_it` without arguments. This opens a menu with options to Start Chat, Ingest Documents, Set API Key, View Stats, Delete Indexes, Evaluate RAG, or Reset Databases).*

### 💬 Direct Subcommands
For power users and scripting, you can also run commands directly:

* **Start Chat directly**:
  ```bash
  rag_it chat
  ```
* **Ingest Documents**:
  ```bash
  rag_it ingest "path/to/document_or_folder"
  ```
  *(Add `--force` to overwrite existing indexes)*
* **Configure OpenAI API Key securely**:
  ```bash
  rag_it config
  ```
* **Run Single Query**:
  ```bash
  rag_it query "Your question here"
  ```
* **List Ingested Documents**:
  ```bash
  rag_it list
  ```
* **Wipe specific Document Index**:
  ```bash
  rag_it delete "document_name.pdf"
  ```
* **Run Evaluation Benchmark**:
  ```bash
  rag_it evaluate
  ```
* **Wipe all databases**:
  ```bash
  rag_it reset
  ```

---

## 📝 Example Queries

Test these sample queries on the default `biosensors.pdf` document:
* `"What are the metrics of the application?"` (or type `/stats` in the chat)
* `"Summarize every figure in biosensors.pdf in one paragraph"`
* `"What is the difference between Table 2 and Table 3?"`
* `"Generate a python script to simulate the biosensor behavior described in the document"` (Tests the grounding code generation rule)

---

## 🔧 Troubleshooting

* **ChromaDB Vector Dimension Errors:**
  * If you switch between local embeddings (384 dimensions) and OpenAI embeddings (1536 dimensions), you may encounter dimension errors on old indexes. The system automatically isolates these using named collections (`collection_384` and `collection_1536`), but if issues persist, run:
    ```bash
    rag_it reset
    ```
* **Script Execution Permissions on Windows:**
  * If you get a Windows script execution warning, open PowerShell as Administrator and run:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```

---

## 🔮 Future Improvements

1. **Secure Serializations:** Swap python's default `pickle` serialization in the BM25 indexer for a secure layout format like JSON or SQLite to eliminate security vulnerabilities.
2. **Local OCR Fallback:** Integrate a local OCR package (like `pytesseract`) to allow local diagram descriptions when running completely offline.
3. **Implement REST API Endpoint:** Build out the FastAPI web server wrapper using the unused dependencies listed in `requirements.txt`.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
