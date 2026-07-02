import sys
import os
import argparse
from pathlib import Path
import json
import re

# Add current workspace directory to python search path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from multimodal_rag.config import get_openai_api_key, save_openai_api_key, MEDIA_DIR
from multimodal_rag.embeddings import EmbeddingPipeline
from multimodal_rag.document_processor import DocumentProcessor
from multimodal_rag.vector_store import VectorStoreManager
from multimodal_rag.keyword_index import BM25IndexManager
from multimodal_rag.retriever import HybridRetriever
from multimodal_rag.generator import ResponseGenerator

# Enable virtual terminal processing on Windows for ANSI colors
if os.name == 'nt':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

# Terminal Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_MAGENTA = "\033[35m"

# Dynamic verbose logging control
VERBOSE_LOGS = False

import builtins
_original_print = builtins.print

def custom_print(*args, **kwargs):
    if args and isinstance(args[0], str):
        # Suppress RAG pipeline framework print statements unless verbose logs is toggled ON
        is_pipeline_log = any(args[0].startswith(prefix) for prefix in [
            "[Retriever]", "[Embedding]", "[VectorStore]", "[BM25]", "[Generator]"
        ])
        if is_pipeline_log and not VERBOSE_LOGS:
            return
    _original_print(*args, **kwargs)

builtins.print = custom_print

def format_cli_markdown(text: str) -> str:
    """Format markdown text with ANSI escapes for clean, readable terminal display."""
    if not text:
        return ""
        
    lines = text.split("\n")
    formatted_lines = []
    in_table = False
    table_rows = []
    
    for line in lines:
        # 1. Handle Tables
        if line.strip().startswith("|"):
            in_table = True
            cells = [c.strip() for c in line.split("|")[1:-1]]
            table_rows.append(cells)
            continue
        elif in_table:
            in_table = False
            if table_rows:
                filtered_rows = []
                for row in table_rows:
                    is_divider = all(all(ch in "-:" for ch in cell) for cell in row if cell)
                    if not is_divider:
                        filtered_rows.append(row)
                
                if filtered_rows:
                    col_widths = [0] * len(filtered_rows[0])
                    for row in filtered_rows:
                        for idx, cell in enumerate(row):
                            if idx < len(col_widths):
                                col_widths[idx] = max(col_widths[idx], len(cell))
                    
                    border = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
                    formatted_lines.append(f"{C_CYAN}{border}{C_RESET}")
                    
                    for r_idx, row in enumerate(filtered_rows):
                        padded_cells = []
                        for idx, cell in enumerate(row):
                            if idx < len(col_widths):
                                padded_cells.append(cell.ljust(col_widths[idx]))
                        
                        row_str = "| " + " | ".join(padded_cells) + " |"
                        if r_idx == 0:
                            formatted_lines.append(f"{C_GREEN}{C_BOLD}{row_str}{C_RESET}")
                            formatted_lines.append(f"{C_CYAN}{border}{C_RESET}")
                        else:
                            formatted_lines.append(row_str)
                    formatted_lines.append(f"{C_CYAN}{border}{C_RESET}")
                table_rows = []
        
        # 2. Format Bold text **word**
        line = re.sub(r'\*\*(.*?)\*\*', f'{C_GREEN}{C_BOLD}\\1{C_RESET}', line)
        
        # 3. Format Inline Citations [Table: xxx, Page: yyy]
        line = re.sub(r'(\[(?:Image|Table):\s*[^,]+,\s*Page:\s*\d+[^\]]*\])', f'{C_YELLOW}\\1{C_RESET}', line)
        
        # 4. Clean list bullets
        if line.strip().startswith("- "):
            line = "  • " + line.strip()[2:]
        elif line.strip().startswith("* "):
            line = "  • " + line.strip()[2:]
            
        # 5. Clean headers
        if line.strip().startswith("### "):
            line = f"\n{C_BOLD}{C_CYAN}--- {line.strip()[4:]} ---{C_RESET}"
        elif line.strip().startswith("## "):
            line = f"\n{C_BOLD}{C_CYAN}=== {line.strip()[3:]} ==={C_RESET}"
        elif line.strip().startswith("# "):
            line = f"\n{C_BOLD}{C_MAGENTA}=== {line.strip()[2:]} ==={C_RESET}"
            
        formatted_lines.append(line)
        
    if in_table and table_rows:
        filtered_rows = []
        for row in table_rows:
            is_divider = all(all(ch in "-:" for ch in cell) for cell in row if cell)
            if not is_divider:
                filtered_rows.append(row)
        if filtered_rows:
            col_widths = [0] * len(filtered_rows[0])
            for row in filtered_rows:
                for idx, cell in enumerate(row):
                    if idx < len(col_widths):
                        col_widths[idx] = max(col_widths[idx], len(cell))
            border = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
            formatted_lines.append(f"{C_CYAN}{border}{C_RESET}")
            for r_idx, row in enumerate(filtered_rows):
                padded_cells = []
                for idx, cell in enumerate(row):
                    if idx < len(col_widths):
                        padded_cells.append(cell.ljust(col_widths[idx]))
                row_str = "| " + " | ".join(padded_cells) + " |"
                if r_idx == 0:
                    formatted_lines.append(f"{C_GREEN}{C_BOLD}{row_str}{C_RESET}")
                    formatted_lines.append(f"{C_CYAN}{border}{C_RESET}")
                else:
                    formatted_lines.append(row_str)
            formatted_lines.append(f"{C_CYAN}{border}{C_RESET}")
            
    return "\n".join(formatted_lines)

def print_header(title: str):
    print(f"\n{C_BOLD}{C_CYAN}=== {title} ==={C_RESET}")

def open_visual_file(image_filename: str):
    """Opens a cited visual file in the default Windows image viewer."""
    image_path = MEDIA_DIR / image_filename
    if image_path.exists():
        print(f"{C_CYAN}[System] Opening visual asset: {image_filename}...{C_RESET}")
        try:
            os.startfile(str(image_path))
        except Exception as e:
            print(f"{C_RED}[Error] Failed to open image: {e}{C_RESET}")
    else:
        print(f"{C_YELLOW}[Warning] Visual file not found locally: {image_path}{C_RESET}")

def parse_citations(text: str) -> list[str]:
    """Extracts image/table filenames from inline citation strings."""
    # Matches patterns like [Image: doc.pdf, Page: 5, Image: filename.png] or similar
    pattern = r'\[(?:Image|Table):\s*[^,]+,\s*Page:\s*\d+,\s*(?:Image|Table):\s*([^\]]+)\]'
    matches = re.findall(pattern, text)
    return [m.strip() for m in matches if m.strip()]

def execute_query(query: str, history: list[dict], decomp: bool, top_k: int, doc_filter: str = None) -> tuple[str, list[dict]]:
    """Runs the full RAG query pipeline (rewriting, decomposition, retrieval, generation)."""
    # 1. Initialize pipelines
    em = EmbeddingPipeline()
    vs = VectorStoreManager(em.dimension)
    bm = BM25IndexManager()
    retriever = HybridRetriever(em, vs, bm)
    generator = ResponseGenerator()

    # 2. History-aware query rewriting
    standalone_query = query
    if history:
        standalone_query = generator.rewrite_query(query, history)

    # 3. Query Decomposition
    sub_queries = [standalone_query]
    if decomp:
        sub_queries = generator.decompose_query(standalone_query)

    # 4. Multi-query retrieval
    # Classify query intent using LLM for robust detection of global figure/table requests
    intent = generator.classify_intent(standalone_query)
    
    filters = {}
    if doc_filter:
        filters["document_name"] = doc_filter

    sub_query_results = []
    for sq in sub_queries:
        results = retriever.retrieve(sq, top_k=top_k, filters=filters, intent=intent)
        sub_query_results.append(results)

    # Interleave results from multiple subqueries (Round-Robin Rank Fusion)
    all_retrieved = []
    seen_ids = set()
    max_len = max(len(r) for r in sub_query_results) if sub_query_results else 0
    
    for i in range(max_len):
        for results in sub_query_results:
            if i < len(results):
                item = results[i]
                if item["id"] not in seen_ids:
                    all_retrieved.append(item)
                    seen_ids.add(item["id"])

    # Limit total merged context chunks to a generous general limit of 40
    merged_retrieved = all_retrieved[:40]

    # 5. Answer Generation
    generation_result = generator.generate_answer(standalone_query, merged_retrieved)
    return generation_result["answer"], generation_result["sources"]

def main():
    global VERBOSE_LOGS
    parser = argparse.ArgumentParser(description="RAG It Multimodal RAG CLI Interface")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # config command
    config_p = subparsers.add_parser("config", help="Configure OpenAI API Key")
    config_p.add_argument("--key", required=True, help="OpenAI API Key string")

    # ingest command
    ingest_p = subparsers.add_parser("ingest", help="Ingest a document file (PDF, Word, Excel, TXT, MD)")
    ingest_p.add_argument("file_path", help="Path to local document file or folder")
    ingest_p.add_argument("--force", action="store_true", help="Force re-ingestion even if document already exists")

    # list command
    subparsers.add_parser("list", help="List all ingested documents and statistics")

    # delete command
    delete_p = subparsers.add_parser("delete", help="Wipe index for a specific document")
    delete_p.add_argument("doc_name", help="Name of document to delete")

    # reset command
    subparsers.add_parser("reset", help="Wipe all vector and keyword search indices")

    # evaluate command
    subparsers.add_parser("evaluate", help="Run the automated QA groundedness evaluation benchmark suite")

    # query command
    query_p = subparsers.add_parser("query", help="Run a single question query")
    query_p.add_argument("query_text", help="Question text")
    query_p.add_argument("--no-decomp", action="store_true", help="Turn off query decomposition")
    query_p.add_argument("--no-open", action="store_true", help="Turn off auto-opening cited images")
    query_p.add_argument("--verbose", "-v", action="store_true", help="Show detailed system execution logs")
    query_p.add_argument("--doc", help="Filter search scope to specific document")

    # start command
    chat_p = subparsers.add_parser("start", help="Start an interactive chat session")
    chat_p.add_argument("--no-decomp", action="store_true", help="Bypass query decomposition by default")
    chat_p.add_argument("--no-open", action="store_true", help="Bypass auto-opening cited images by default")
    chat_p.add_argument("--verbose", "-v", action="store_true", help="Show detailed system execution logs by default")
    chat_p.add_argument("--doc", help="Filter chat search scope to specific document")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # --- 1. CONFIG COMMAND ---
    if args.command == "config":
        print_header("Configuring API Key")
        try:
            save_openai_api_key(args.key)
            print(f"{C_GREEN}[Success] API Key saved successfully to settings file.{C_RESET}")
        except Exception as e:
            print(f"{C_RED}[Error] Failed to save key: {e}{C_RESET}")

    # --- 2. INGEST COMMAND ---
    elif args.command == "ingest":
        print_header("Ingesting Document(s)")
        input_path = Path(args.file_path)
        if not input_path.exists():
            print(f"{C_RED}[Error] Path does not exist: {input_path}{C_RESET}")
            return

        SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt", ".md"}
        ingest_files = []
        if input_path.is_file():
            if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                ingest_files.append(input_path)
            else:
                print(f"{C_RED}[Error] Unsupported file format: {input_path.suffix}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}{C_RESET}")
                return
        elif input_path.is_dir():
            ingest_files = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])
            if not ingest_files:
                print(f"{C_YELLOW}[Warning] No supported document files found in directory: {input_path}{C_RESET}")
                return
            print(f"{C_CYAN}[System] Found {len(ingest_files)} document files in directory: {input_path}{C_RESET}")

        print(f"{C_CYAN}[System] Initializing pipelines...{C_RESET}")
        em = EmbeddingPipeline()
        vs = VectorStoreManager(em.dimension)
        bm = BM25IndexManager()
        doc_parser = DocumentProcessor()

        # Reset token tracker
        from multimodal_rag.config import TokenTracker
        TokenTracker.reset()

        ingested_count = 0
        skipped_count = 0

        # Check duplicate
        docs = vs.list_documents()

        for idx, doc_path in enumerate(ingest_files):
            print(f"\n{C_CYAN}--- Processing [{idx+1}/{len(ingest_files)}] {doc_path.name} ---{C_RESET}")
            if doc_path.name in docs and not args.force:
                print(f"{C_YELLOW}[Notice] Document '{doc_path.name}' is already ingested. Skipping.{C_RESET}")
                skipped_count += 1
                continue

            if doc_path.name in docs and args.force:
                print(f"{C_CYAN}[System] Force flag enabled. Wiping existing index for '{doc_path.name}'...{C_RESET}")
                vs.delete_document(doc_path.name)
                bm.delete_document(doc_path.name)

            print(f"{C_CYAN}[System] Parsing document structures (pages, headers, tables, images)...{C_RESET}")
            try:
                chunks = doc_parser.process_file(str(doc_path))
                if not chunks:
                    print(f"{C_YELLOW}[Warning] No chunks extracted from document.{C_RESET}")
                    continue

                print(f"{C_CYAN}[System] Chunking complete. Generating embeddings for {len(chunks)} elements...{C_RESET}")
                contents = [c["content"] for c in chunks]
                embeddings_list = em.embed_documents(contents)

                print(f"{C_CYAN}[System] Writing vectors to ChromaDB collection...{C_RESET}")
                vs.add_chunks(chunks, embeddings_list)

                print(f"{C_CYAN}[System] Writing terms to BM25 index...{C_RESET}")
                bm.add_chunks(chunks)

                print(f"{C_GREEN}[Success] Ingested '{doc_path.name}' successfully!{C_RESET}")
                ingested_count += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"{C_RED}[Error] Ingestion failed for '{doc_path.name}': {e}{C_RESET}")

        print(f"\n{C_GREEN}=== Ingestion Batch Complete ==={C_RESET}")
        print(f"  - Total Files Found:     {len(ingest_files)}")
        print(f"  - Successfully Indexed:  {ingested_count}")
        print(f"  - Skipped (Duplicates):  {skipped_count}")
        print(f"{C_CYAN}--------------------------------------------------{C_RESET}")
        print(f"{C_CYAN}[Cumulative Ingestion OpenAI Token Usage]{C_RESET}")
        print(f"  - Embedding Prompt Tokens:  {TokenTracker.embedding_tokens:,}")
        print(f"  - Chat VLM (Table/Image):   {TokenTracker.prompt_tokens:,} prompt | {TokenTracker.completion_tokens:,} completion")
        print(f"  - Total Ingestion Tokens:   {TokenTracker.embedding_tokens + TokenTracker.prompt_tokens + TokenTracker.completion_tokens:,}")
        print(f"{C_CYAN}--------------------------------------------------{C_RESET}")

    # --- 3. LIST COMMAND ---
    elif args.command == "list":
        print_header("Index Database Overview")
        try:
            em = EmbeddingPipeline()
            vs = VectorStoreManager(em.dimension)
            stats = vs.get_stats()
            docs = vs.list_documents()

            print(f"Total Chunks: {C_BOLD}{stats.get('total_chunks', 0)}{C_RESET}")
            print(f"Text Chunks:  {stats.get('text_count', 0)}")
            print(f"Table Chunks: {stats.get('table_count', 0)}")
            print(f"Image Chunks: {stats.get('image_count', 0)}")
            print("-" * 30)
            print("Ingested Documents:")
            if docs:
                for doc in docs:
                    print(f" - {C_GREEN}{doc}{C_RESET}")
            else:
                print(" (No documents ingested yet)")
        except Exception as e:
            print(f"{C_RED}[Error] Failed to fetch list: {e}{C_RESET}")

    # --- 4. DELETE COMMAND ---
    elif args.command == "delete":
        print_header(f"Deleting Document: {args.doc_name}")
        try:
            em = EmbeddingPipeline()
            vs = VectorStoreManager(em.dimension)
            bm = BM25IndexManager()

            vs.delete_document(args.doc_name)
            bm.delete_document(args.doc_name)
            print(f"{C_GREEN}[Success] Document '{args.doc_name}' has been successfully wiped from indexes.{C_RESET}")
        except Exception as e:
            print(f"{C_RED}[Error] Delete failed: {e}{C_RESET}")

    # --- 5. RESET COMMAND ---
    elif args.command == "reset":
        print_header("CRITICAL: Resetting Databases")
        confirm = input(f"{C_RED}Are you absolutely sure you want to delete all vectors and search indexes? (y/N): {C_RESET}").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            return

        try:
            em = EmbeddingPipeline()
            vs = VectorStoreManager(em.dimension)
            bm = BM25IndexManager()

            # Delete all documents one by one
            docs = vs.list_documents()
            for doc in docs:
                vs.delete_document(doc)
                bm.delete_document(doc)

            vs.reset_db()
            bm.reset_db()
            print(f"{C_GREEN}[Success] Wiped vector database and keyword search index successfully.{C_RESET}")
        except Exception as e:
            print(f"{C_RED}[Error] Reset failed: {e}{C_RESET}")

    # --- 5.5 EVALUATE COMMAND ---
    elif args.command == "evaluate":
        import evaluate_rag
        evaluate_rag.main()

    # --- 6. QUERY COMMAND ---
    elif args.command == "query":
        print_header("Executing RAG Query")
        if args.verbose:
            VERBOSE_LOGS = True
        decomp = not args.no_decomp
        open_v = not args.no_open
        print(f"Active Toggles: Decomposition={decomp}, Auto-Open Cited Visuals={open_v}, Verbose Logs={VERBOSE_LOGS}")
        print(f"Query: {C_BOLD}{args.query_text}{C_RESET}")

        try:
            from multimodal_rag.config import TokenTracker
            TokenTracker.reset()
            import time
            start_t = time.time()
            answer, sources = execute_query(args.query_text, [], decomp, 10, args.doc)
            latency = time.time() - start_t
            print(f"\n{C_GREEN}{C_BOLD}Answer:{C_RESET}\n{format_cli_markdown(answer)}\n")
            
            # Parse cited pages
            cited_pages = sorted(list(set(int(page) for doc, page in re.findall(r'\[(?:Image|Table):\s*([^,]+),\s*Page:\s*(\d+)[^\]]*\]', answer))))
            
            # Print single-query metrics
            print(f"{C_CYAN}--------------------------------------------------{C_RESET}")
            print(f"{C_CYAN}[Query Metrics]{C_RESET}")
            print(f"  - Latency:             {latency:.2f} seconds")
            print(f"  - Context Size:        {len(sources)} retrieved chunks")
            if cited_pages:
                print(f"  - Cited Pages:         {', '.join(str(p) for p in cited_pages)}")
            print(f"  - OpenAI Token Usage:  {TokenTracker.prompt_tokens + TokenTracker.embedding_tokens:,} prompt | {TokenTracker.completion_tokens:,} completion")
            print(f"{C_CYAN}--------------------------------------------------{C_RESET}")
            
            # Citation handling
            cited_media = parse_citations(answer)
            if cited_media:
                print(f"{C_YELLOW}Cited Visuals in Response: {cited_media}{C_RESET}")
                if open_v:
                    for img in cited_media:
                        open_visual_file(img)
        except Exception as e:
            print(f"{C_RED}[Error] Query execution failed: {e}{C_RESET}")

    # --- 7. START COMMAND ---
    elif args.command == "start":
        print_header("RAG It Interactive RAG Setup Wizard")
        if args.verbose:
            VERBOSE_LOGS = True
            
        import time
        decomp = not args.no_decomp
        open_v = not args.no_open
        history = []
        
        # Initialize Embedding/Vector/BM25 pipelines to fetch ingested docs
        em = EmbeddingPipeline()
        vs = VectorStoreManager(em.dimension)
        bm = BM25IndexManager()
        
        ingested_docs = vs.list_documents()
        selected_doc = args.doc
        
        if not selected_doc:
            print(f"{C_CYAN}Select a document scope for this chat session:{C_RESET}")
            print(f"  [0] {C_GREEN}Use all ingested documents (global search){C_RESET}")
            for idx, doc in enumerate(ingested_docs):
                print(f"  [{idx+1}] {doc}")
            new_idx = len(ingested_docs) + 1
            print(f"  [{new_idx}] {C_YELLOW}Ingest a new document (PDF, Excel, Word, TXT, MD)...{C_RESET}")

            while True:
                try:
                    choice = input(f"\nSelect option (0-{new_idx}): ").strip()
                    if not choice:
                        choice = "0"
                    choice_val = int(choice)
                    if choice_val == 0:
                        selected_doc = None
                        print(f"{C_GREEN}[Scope] Global search enabled.{C_RESET}")
                        break
                    elif 1 <= choice_val <= len(ingested_docs):
                        selected_doc = ingested_docs[choice_val - 1]
                        print(f"{C_GREEN}[Scope] Scoped query filter set to: '{selected_doc}'{C_RESET}")
                        break
                    elif choice_val == new_idx:
                        new_path = input(f"{C_YELLOW}Enter path to new file or directory: {C_RESET}").strip()
                        if not new_path:
                            print(f"{C_RED}[Error] Path cannot be empty.{C_RESET}")
                            continue
                        p_path = Path(new_path)
                        if not p_path.exists():
                            print(f"{C_RED}[Error] Path does not exist: {new_path}{C_RESET}")
                            continue

                        SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt", ".md"}
                        ingest_files = []
                        if p_path.is_file():
                            if p_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                                ingest_files.append(p_path)
                            else:
                                print(f"{C_RED}[Error] Unsupported file format.{C_RESET}")
                                continue
                        elif p_path.is_dir():
                            ingest_files = sorted([p for p in p_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])

                        if not ingest_files:
                            print(f"{C_RED}[Error] No supported document files found in directory: {p_path}{C_RESET}")
                            continue

                        print(f"{C_CYAN}[System] Ingesting {len(ingest_files)} document(s)...{C_RESET}")
                        doc_parser = DocumentProcessor()

                        # Reset tokens
                        from multimodal_rag.config import TokenTracker
                        TokenTracker.reset()

                        for pf in ingest_files:
                            if pf.name in ingested_docs:
                                print(f"{C_YELLOW}[Notice] Document '{pf.name}' is already ingested. Skipping.{C_RESET}")
                                continue
                            print(f"{C_CYAN}[System] Processing '{pf.name}'...{C_RESET}")
                            try:
                                chunks = doc_parser.process_file(str(pf))
                                if not chunks:
                                    continue
                                contents = [c["content"] for c in chunks]
                                embeddings_list = em.embed_documents(contents)
                                vs.add_chunks(chunks, embeddings_list)
                                bm.add_chunks(chunks)
                                print(f"{C_GREEN}[Success] Indexed '{pf.name}'{C_RESET}")
                            except Exception as e:
                                print(f"{C_RED}[Error] Failed to index '{pf.name}': {e}{C_RESET}")

                        selected_doc = ingest_files[0].name if len(ingest_files) == 1 else None
                        if len(ingest_files) > 1:
                            print(f"{C_GREEN}[Scope] Global search enabled (using all ingested files).{C_RESET}")
                        else:
                            print(f"{C_GREEN}[Scope] Scoped query filter set to: '{selected_doc}'{C_RESET}")
                        break
                    else:
                        print(f"{C_RED}Invalid option. Please enter 0 to {new_idx}.{C_RESET}")
                except ValueError:
                    print(f"{C_RED}Please enter a valid choice.{C_RESET}")
                except (KeyboardInterrupt, EOFError):
                    print("\nSetup wizard aborted. Exiting.")
                    return
        else:
            print(f"{C_GREEN}[Scope] Scoped query filter pre-set to: '{selected_doc}'{C_RESET}")
            
        # Session metrics tracking
        session_queries = 0
        session_total_latency = 0.0
        session_pages_cited = set()
        session_docs_cited = set()
        session_start_time = time.time()
        session_prompt_tokens = 0
        session_completion_tokens = 0
        
        # Display controls
        print(f"\n{C_CYAN}Controls:{C_RESET}")
        print("  - Type your question and hit Enter.")
        print("  - Type /exit or /quit to end the chat.")
        print(f"  - Type /decomp to toggle Query Decomposition. (Currently: {C_BOLD}{'ON' if decomp else 'OFF'}{C_RESET})")
        print(f"  - Type /open to toggle Auto-Opening Cited Images. (Currently: {C_BOLD}{'ON' if open_v else 'OFF'}{C_RESET})")
        print(f"  - Type /verbose to toggle detailed system logs. (Currently: {C_BOLD}{'ON' if VERBOSE_LOGS else 'OFF'}{C_RESET})")
        print("  - Type /stats to view your current session metrics scorecard.")
        print("  - Type /clear to wipe the active session memory/history.")
        print("  - Type /help to see this menu.")
        print("-" * 60)

        while True:
            try:
                user_input = input(f"\n{C_BOLD}User > {C_RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print_header("Session Scorecard & Metrics")
                session_duration = time.time() - session_start_time
                avg_latency = session_total_latency / session_queries if session_queries > 0 else 0.0
                print(f"  - Total Chat Turns:    {session_queries}")
                print(f"  - Avg Response Time:   {avg_latency:.2f} seconds")
                if session_docs_cited:
                    print(f"  - Documents Cited:     {', '.join(session_docs_cited)}")
                if session_pages_cited:
                    print(f"  - Pages Cited:         {', '.join(str(p) for p in sorted(list(session_pages_cited)))}")
                print(f"  - Session Duration:    {session_duration:.1f} seconds")
                print(f"  - Total Session Tokens:{session_prompt_tokens:,} prompt | {session_completion_tokens:,} completion")
                print(f"{C_CYAN}=================================================={C_RESET}")
                print("\nExiting chat. Goodbye!")
                break

            if not user_input:
                continue

            # Command checks
            if user_input.lower() in ["/exit", "/quit"]:
                print_header("Session Scorecard & Metrics")
                session_duration = time.time() - session_start_time
                avg_latency = session_total_latency / session_queries if session_queries > 0 else 0.0
                print(f"  - Total Chat Turns:    {session_queries}")
                print(f"  - Avg Response Time:   {avg_latency:.2f} seconds")
                if session_docs_cited:
                    print(f"  - Documents Cited:     {', '.join(session_docs_cited)}")
                if session_pages_cited:
                    print(f"  - Pages Cited:         {', '.join(str(p) for p in sorted(list(session_pages_cited)))}")
                print(f"  - Session Duration:    {session_duration:.1f} seconds")
                print(f"  - Total Session Tokens:{session_prompt_tokens:,} prompt | {session_completion_tokens:,} completion")
                print(f"{C_CYAN}=================================================={C_RESET}")
                print("Ending chat. Goodbye!")
                break
            elif user_input.lower() == "/decomp":
                decomp = not decomp
                print(f"{C_YELLOW}[Toggle] Query Decomposition is now {'ON' if decomp else 'OFF'}{C_RESET}")
                continue
            elif user_input.lower() == "/open":
                open_v = not open_v
                print(f"{C_YELLOW}[Toggle] Auto-Opening of Visuals is now {'ON' if open_v else 'OFF'}{C_RESET}")
                continue
            elif user_input.lower() == "/verbose":
                VERBOSE_LOGS = not VERBOSE_LOGS
                print(f"{C_YELLOW}[Toggle] Detailed system logging is now {'ON' if VERBOSE_LOGS else 'OFF'}{C_RESET}")
                continue
            elif user_input.lower() == "/start":
                print_header("RAG It Interactive Scope Selector")
                ingested_docs = vs.list_documents()
                print(f"{C_CYAN}Select a document scope for the chat session:{C_RESET}")
                print(f"  [0] {C_GREEN}Use all ingested documents (global search){C_RESET}")
                for idx, doc in enumerate(ingested_docs):
                    print(f"  [{idx+1}] {doc}")
                new_idx = len(ingested_docs) + 1
                print(f"  [{new_idx}] {C_YELLOW}Ingest a new document (PDF, Excel, Word, CSV, TXT, MD)...{C_RESET}")
                
                while True:
                    try:
                        choice = input(f"\nSelect option (0-{new_idx}): ").strip()
                        if not choice:
                            choice = "0"
                        choice_val = int(choice)
                        if choice_val == 0:
                            selected_doc = None
                            print(f"{C_GREEN}[Scope] Global search enabled.{C_RESET}")
                            break
                        elif 1 <= choice_val <= len(ingested_docs):
                            selected_doc = ingested_docs[choice_val - 1]
                            print(f"{C_GREEN}[Scope] Scoped query filter set to: '{selected_doc}'{C_RESET}")
                            break
                        elif choice_val == new_idx:
                            new_path = input(f"{C_YELLOW}Enter path to new file or directory: {C_RESET}").strip()
                            if not new_path:
                                print(f"{C_RED}[Error] Path cannot be empty.{C_RESET}")
                                continue
                            p_path = Path(new_path)
                            if not p_path.exists():
                                print(f"{C_RED}[Error] Path does not exist: {new_path}{C_RESET}")
                                continue
                            
                            SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt", ".md"}
                            ingest_files = []
                            if p_path.is_file():
                                if p_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                                    ingest_files.append(p_path)
                                else:
                                    print(f"{C_RED}[Error] Unsupported file format.{C_RESET}")
                                    continue
                            elif p_path.is_dir():
                                ingest_files = sorted([p for p in p_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])
                                
                            if not ingest_files:
                                print(f"{C_RED}[Error] No supported document files found in directory: {p_path}{C_RESET}")
                                continue
                            
                            print(f"{C_CYAN}[System] Ingesting {len(ingest_files)} document(s)...{C_RESET}")
                            doc_parser = DocumentProcessor()
                            
                            # Reset tokens
                            from multimodal_rag.config import TokenTracker
                            TokenTracker.reset()
                            
                            for pf in ingest_files:
                                if pf.name in ingested_docs:
                                    print(f"{C_YELLOW}[Notice] Document '{pf.name}' is already ingested. Skipping.{C_RESET}")
                                    continue
                                print(f"{C_CYAN}[System] Processing '{pf.name}'...{C_RESET}")
                                try:
                                    chunks = doc_parser.process_file(str(pf))
                                    if not chunks:
                                        continue
                                    contents = [c["content"] for c in chunks]
                                    embeddings_list = em.embed_documents(contents)
                                    vs.add_chunks(chunks, embeddings_list)
                                    bm.add_chunks(chunks)
                                    print(f"{C_GREEN}[Success] Indexed '{pf.name}'{C_RESET}")
                                except Exception as e:
                                    print(f"{C_RED}[Error] Failed to index '{pf.name}': {e}{C_RESET}")
                            
                            selected_doc = ingest_files[0].name if len(ingest_files) == 1 else None
                            if len(ingest_files) > 1:
                                print(f"{C_GREEN}[Scope] Global search enabled (using all ingested files).{C_RESET}")
                            else:
                                print(f"{C_GREEN}[Scope] Scoped query filter set to: '{selected_doc}'{C_RESET}")
                            break
                        else:
                            print(f"{C_RED}Invalid option. Please enter 0 to {new_idx}.{C_RESET}")
                    except ValueError:
                        print(f"{C_RED}Please enter a valid choice.{C_RESET}")
                    except (KeyboardInterrupt, EOFError):
                        print("\nScope switch cancelled.")
                        break
                continue
            elif user_input.lower() == "/stats":
                print_header("Current Session Scorecard & Metrics")
                session_duration = time.time() - session_start_time
                avg_latency = session_total_latency / session_queries if session_queries > 0 else 0.0
                print(f"  - Total Chat Turns:    {session_queries}")
                print(f"  - Avg Response Time:   {avg_latency:.2f} seconds")
                if session_docs_cited:
                    print(f"  - Documents Cited:     {', '.join(session_docs_cited)}")
                if session_pages_cited:
                    print(f"  - Pages Cited:         {', '.join(str(p) for p in sorted(list(session_pages_cited)))}")
                print(f"  - Session Duration:    {session_duration:.1f} seconds")
                print(f"  - Total Session Tokens:{session_prompt_tokens:,} prompt | {session_completion_tokens:,} completion")
                print(f"{C_CYAN}=================================================={C_RESET}")
                continue
            elif user_input.lower() == "/evaluate":
                import evaluate_rag
                evaluate_rag.main()
                continue
            elif user_input.lower() == "/clear":
                history = []
                print(f"{C_CYAN}[System] Session memory cleared.{C_RESET}")
                continue
            elif user_input.lower() == "/help":
                print(f"{C_CYAN}Available Commands:{C_RESET}")
                print("  /exit, /quit - Close loop")
                print("  /decomp      - Toggle query decomposition")
                print("  /open        - Toggle auto-opening cited images")
                print("  /verbose     - Toggle detailed pipeline logs")
                print("  /start       - Switch active document scope or ingest new file")
                print("  /stats       - View current session scorecard metrics")
                print("  /evaluate    - Run the automated QA groundedness evaluation benchmark suite")
                print("  /clear       - Clear active memory history")
                continue
            elif user_input.startswith("/"):
                print(f"{C_RED}Unknown command: {user_input}. Type /help for options.{C_RESET}")
                continue

            # Execute RAG query
            print(f"{C_CYAN}[RAG] Processing question...{C_RESET}")
            try:
                from multimodal_rag.config import TokenTracker
                TokenTracker.reset()
                start_t = time.time()
                answer, sources = execute_query(user_input, history, decomp, 10, selected_doc)
                latency = time.time() - start_t
                
                # Update session stats
                session_queries += 1
                session_total_latency += latency
                session_prompt_tokens += (TokenTracker.prompt_tokens + TokenTracker.embedding_tokens)
                session_completion_tokens += TokenTracker.completion_tokens
                for doc, page in re.findall(r'\[(?:Image|Table):\s*([^,]+),\s*Page:\s*(\d+)[^\]]*\]', answer):
                    session_docs_cited.add(doc.strip())
                    session_pages_cited.add(int(page.strip()))
                
                print(f"\n{C_GREEN}{C_BOLD}RAG It >{C_RESET}\n{format_cli_markdown(answer)}")
                
                # Print single-query metrics right there
                print(f"\n{C_CYAN}  [Query Latency: {latency:.2f}s | Context Size: {len(sources)} chunks | Tokens: {TokenTracker.prompt_tokens + TokenTracker.embedding_tokens}p + {TokenTracker.completion_tokens}c]{C_RESET}")
                
                # Append to active session history
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": answer})

                # Visual citations check
                cited_media = parse_citations(answer)
                if cited_media:
                    print(f"\n{C_YELLOW}[Citations] Cited media: {cited_media}{C_RESET}")
                    if open_v:
                        for img in cited_media:
                            # Prompt before opening so it doesn't steal focus unexpectedly
                            confirm = input(f"{C_CYAN}[System] Auto-open active. Launch visual '{img}'? (Y/n): {C_RESET}").strip().lower()
                            if confirm != 'n':
                                open_visual_file(img)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"{C_RED}[Error] Failed to process query: {e}{C_RESET}")

if __name__ == "__main__":
    main()
