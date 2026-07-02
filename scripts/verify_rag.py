import os
import shutil
from pathlib import Path

import sys
# Add parent directory to sys.path so we can import multimodal_rag
sys.path.append(str(Path(__file__).parent.parent.resolve()))

# Override config directories to separate test folders before importing managers
import multimodal_rag.config
multimodal_rag.config.DB_DIR = multimodal_rag.config.WORKSPACE_DIR / "data" / "db_test"
multimodal_rag.config.KEYWORD_DIR = multimodal_rag.config.WORKSPACE_DIR / "data" / "keyword_test"

# Ensure test directories exist
multimodal_rag.config.DB_DIR.mkdir(parents=True, exist_ok=True)
multimodal_rag.config.KEYWORD_DIR.mkdir(parents=True, exist_ok=True)

# force local fallback for verification
os.environ["OPENAI_API_KEY"] = "" 

from multimodal_rag.embeddings import EmbeddingPipeline
from multimodal_rag.vector_store import VectorStoreManager
from multimodal_rag.keyword_index import BM25IndexManager
from multimodal_rag.retriever import HybridRetriever
from multimodal_rag.generator import ResponseGenerator

def main():
    print("=== STARTING AUTOMATED RAG PIPELINE VERIFICATION ===")
    
    # 1. Initialize Embedding Pipeline (will fallback to local 384-dim all-MiniLM-L6-v2)
    print("\n[Step 1] Initializing Embedding Pipeline...")
    embeddings = EmbeddingPipeline()
    print(f"Active Embedder class: {embeddings.embedder.__class__.__name__}")
    print(f"Dimension: {embeddings.dimension}")
    expected_dim = 1536 if "OpenAI" in embeddings.embedder.__class__.__name__ else 384
    assert embeddings.dimension == expected_dim, f"Expected dimension {expected_dim}, got {embeddings.dimension}"
    
    # Test text embedding
    test_text = "Verification query for multimodal hybrid search."
    emb = embeddings.embed_query(test_text)
    assert len(emb) == expected_dim, f"Expected embedding length {expected_dim}, got {len(emb)}"
    print("[OK] Embedding generation works successfully!")

    # 2. Initialize Vector Database Manager (ChromaDB)
    print("\n[Step 2] Initializing Vector Store...")
    vector_store = VectorStoreManager(dimension=embeddings.dimension)
    vector_store.reset_db() # Clear DB
    
    # 3. Initialize Keyword Indexer (BM25)
    print("\n[Step 3] Initializing BM25 Indexer...")
    bm25_index = BM25IndexManager()
    bm25_index.reset_index() # Clear BM25
    
    # 4. Generate Mock Multimodal Chunks
    print("\n[Step 4] Creating Mock Multimodal Chunks...")
    mock_chunks = [
        {
            "id": "text_doc_p1_0",
            "document_name": "test_doc.pdf",
            "page_number": 1,
            "section_title": "Executive Summary",
            "type": "text",
            "content": "Our business witnessed record expansion in the year 2025. The growth was driven by new product releases and expansion in global markets. Customer satisfaction scores hit an all-time high of 94%.",
            "metadata": {
                "document_name": "test_doc.pdf",
                "page_number": 1,
                "section_title": "Executive Summary",
                "type": "text"
            }
        },
        {
            "id": "table_doc_p2_1",
            "document_name": "test_doc.pdf",
            "page_number": 2,
            "section_title": "Financial Results",
            "type": "table",
            "content": (
                "| Year | Revenue (M$) | Net Profit (M$) | Profit Margin |\n"
                "| --- | --- | --- | --- |\n"
                "| 2023 | 120.5 | 18.2 | 15.1% |\n"
                "| 2024 | 145.2 | 24.8 | 17.0% |\n"
                "| 2025 | 189.6 | 38.4 | 20.2% |\n"
                "\n"
                "Summary: Table showing 2023-2025 revenue and net profit trends. Profit margin increased to 20.2% in 2025."
            ),
            "metadata": {
                "document_name": "test_doc.pdf",
                "page_number": 2,
                "section_title": "Financial Results",
                "type": "table",
                "image_path": "table_p2_1.png",
                "table_id": "table_doc_p2_1"
            }
        },
        {
            "id": "image_doc_p3_2",
            "document_name": "test_doc.pdf",
            "page_number": 3,
            "section_title": "Architecture Overview",
            "type": "image",
            "content": (
                "Architecture Flowchart Diagram showing three main layers: Client UI layer connects to "
                "API Gateway, which routes requests to Microservices and indexes them in Database."
                "\nOCR Text: CLIENT UI -> API GATEWAY -> MICROSERVICES -> PERSISTENT DATABASE"
            ),
            "metadata": {
                "document_name": "test_doc.pdf",
                "page_number": 3,
                "section_title": "Architecture Overview",
                "type": "image",
                "image_path": "figure_p3_2.png",
                "image_id": "image_doc_p3_2"
            }
        }
    ]
    
    # Generate embeddings for mock chunks
    chunk_contents = [c["content"] for c in mock_chunks]
    embeddings_list = embeddings.embed_documents(chunk_contents)
    
    # 5. Ingest into databases
    print("\n[Step 5] Ingesting Chunks into ChromaDB & BM25...")
    vector_store.add_chunks(mock_chunks, embeddings_list)
    bm25_index.add_chunks(mock_chunks)
    
    # Check stats
    stats = vector_store.get_stats()
    print("Database Stats after ingestion:", stats)
    assert stats["total_chunks"] == 3, f"Expected 3 chunks in DB, got {stats['total_chunks']}"
    assert stats["text_count"] == 1, "Expected 1 text chunk"
    assert stats["table_count"] == 1, "Expected 1 table chunk"
    assert stats["image_count"] == 1, "Expected 1 image chunk"
    
    # Check document list
    docs = vector_store.list_documents()
    print("Indexed Documents:", docs)
    assert docs == ["test_doc.pdf"], f"Expected ['test_doc.pdf'], got {docs}"
    print("[OK] Ingestion & Stats mapping verified successfully!")

    # 6. Test Hybrid Retriever
    print("\n[Step 6] Verifying Hybrid Retriever...")
    retriever = HybridRetriever(embeddings, vector_store, bm25_index)
    
    # A. Search for text query
    print("Query: 'What was the customer satisfaction in 2025?'")
    results = retriever.retrieve("What was the customer satisfaction in 2025?", top_k=2)
    print(f"Returned {len(results)} results.")
    for idx, r in enumerate(results):
        print(f"Rank {idx+1}: ID: {r['id']} | Type: {r['type']} | RRF Score: {r['rrf_score']:.4f}")
    assert results[0]["id"] == "text_doc_p1_0", "Text chunk should be ranked first"
    
    # B. Search for table query (numerical comparisons)
    print("\nQuery: 'What was the profit margin in 2025?'")
    results = retriever.retrieve("What was the profit margin in 2025?", top_k=2)
    for idx, r in enumerate(results):
         print(f"Rank {idx+1}: ID: {r['id']} | Type: {r['type']} | RRF Score: {r['rrf_score']:.4f}")
    assert results[0]["id"] == "table_doc_p2_1", "Table chunk should be ranked first"

    # C. Search with metadata filter
    print("\nQuery with metadata filter: type = 'image'")
    results = retriever.retrieve("profit margin", top_k=3, filters={"type": "image"})
    print(f"Filtered Results count: {len(results)}")
    for r in results:
         print(f"ID: {r['id']} | Type: {r['type']}")
         assert r["type"] == "image", "All results should be filtered to image type"
    print("[OK] Hybrid search and Reciprocal Rank Fusion (RRF) verified successfully!")

    # 7. Test Generator response formatting
    print("\n[Step 7] Checking Response Generator...")
    generator = ResponseGenerator()
    answer_res = generator.generate_answer("How much did revenue grow?", mock_chunks)
    print("Generator Output Answer Snippet:")
    print("-" * 50)
    print(answer_res["answer"])
    print("-" * 50)
    if generator.api_key:
        assert "[Table: test_doc.pdf, Page: 2]" in answer_res["answer"], "Grounded response should cite the source table"
        print("[OK] Online LLM generation and citations verified successfully!")
    else:
        assert "OpenAI API Key is missing" in answer_res["answer"], "Should show warning in offline mode"
        print("[OK] Offline warning fallback verified successfully!")

    # 8. Test Conversational Memory Rewriting & Decomposition
    print("\n[Step 8] Checking Conversational Rewriting & Query Decomposition...")
    mock_history = [
        {"role": "user", "content": "Tell me about table 2 in biosensors.pdf"},
        {"role": "assistant", "content": "Table 2 describes impedance-based immunosensors."}
    ]
    rewritten = generator.rewrite_query("what about table 4?", mock_history)
    print(f"Rewritten Query: '{rewritten}'")
    if generator.api_key:
        assert "table 4" in rewritten.lower(), "Standalone query should resolve 'table 4'"
        print("[OK] Conversational query rewriting verified successfully!")
    else:
        print("[OK] Conversational query rewriting bypassed (offline).")

    sub_queries = generator.decompose_query("compare Table 2 and Table 3 and find E. coli limit of detection")
    print(f"Decomposed Sub-queries: {sub_queries}")
    if generator.api_key:
        assert len(sub_queries) > 1, "Decomposition should split multi-part query"
        print("[OK] Query decomposition verified successfully!")
    else:
        print("[OK] Query decomposition bypassed (offline).")

    print("\n=== ALL PIPELINE CHECKS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    main()
