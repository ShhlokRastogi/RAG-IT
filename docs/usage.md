# RAG It - Usage Guide

This guide details how to setup, configure, and operate the **RAG It** search engine.

## 🕹️ Primary Control Panel (Recommended)

The easiest way to run the software is by launching the interactive Control Panel:
```bash
rag_it start
```
*(Or simply run `rag_it` without arguments. This opens a visual text-based menu listing all available services: Chat, Ingest, Config, Database Stats, Delete, Evaluate, and Reset).*

---

## ⚙️ Configuration

Set your OpenAI API Key (used for answer synthesis and automated evaluation):

### 1. Secure Prompt (Recommended)
Run the config command without any flags to enter your API key securely without it appearing in your shell history:
```bash
rag_it config
```

### 2. Direct Input
Alternatively, pass your key directly as a flag:
```bash
rag_it config --key "YOUR-OPENAI-KEY"
```
The key is stored locally in `settings.json` in the root folder.

---

## 📂 Document Ingestion

You can ingest files directly through the Control Panel option `[2]`, or use the command line directly:

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

You can start chatting via the Control Panel option `[1]`, or launch the chat session directly via:
```bash
rag_it chat
```

### Active Console Commands (Slash Commands):
* `/start`    - opens the document selector/ingestion wizard to change the active search scope.
* `/key`      - dynamically prompts for and updates the OpenAI API Key in the active session.
* `/stats`    - views session performance scorecard metrics (turn count, average latency, total prompt/completion tokens used).
* `/evaluate` - triggers the automated QA benchmark evaluation suite.
* `/clear`    - clears the conversational context memory history.
* `/decomp`   - toggles semantic query decomposition (useful for complex comparative queries).
* `/open`     - toggles automatic opening of cited table/figure crops on your machine.
* `/help`     - views a list of all commands.
* `/exit`     - closes the chat console.
