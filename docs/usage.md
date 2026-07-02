# RAG It - Usage Guide

This guide details how to setup, configure, and operate the **RAG It** search engine.

## ⚙️ Configuration

Set your OpenAI API Key (used for answer synthesis and automated evaluation):
```bash
rag_it config --key "YOUR-OPENAI-KEY"
```
The key is stored locally in `settings.json` in the root folder.

---

## 📂 Document Ingestion

Place documents in `data/documents/` or point directly to any folder/file:

### 1. Ingest a local directory:
```bash
rag_it ingest "C:\path\to\your\folder"
```

### 2. Ingest an individual file:
```bash
rag_it ingest "C:\path\to\document.pdf"
```

*Supported formats: `.pdf`, `.docx`, `.xlsx`, `.xls`, `.csv`, `.txt`, `.md`*

---

## 💬 Interactive Chat Console

Launch the chat interface:
```bash
rag_it start
```

### Active Console Commands:
* `/start` - opens the document selector/ingestion wizard to change the active search scope.
* `/stats` - views session performance metrics (turn count, average latency, total prompt and completion tokens used).
* `/evaluate` - triggers the automated QA benchmark suite.
* `/clear` - clears the conversational context memory history.
* `/decomp` - toggles semantic query decomposition (useful for complex comparative queries).
* `/open` - toggles automatic opening of cited table/figure crops on your machine.
* `/help` - views a list of all commands.
* `/exit` - closes the chat console.
