import sys
import time
from pathlib import Path

# Add current workspace directory to python search path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from multimodal_rag.embeddings import EmbeddingPipeline
from multimodal_rag.vector_store import VectorStoreManager
from multimodal_rag.keyword_index import BM25IndexManager
from multimodal_rag.retriever import HybridRetriever
from multimodal_rag.generator import ResponseGenerator

def main():
    print("=== RAG IT RAG EVALUATION SUITE ===")
    
    # 1. Initialize Pipeline
    print("[System] Initializing RAG pipelines...")
    em = EmbeddingPipeline()
    vs = VectorStoreManager(em.dimension)
    bm = BM25IndexManager()
    retriever = HybridRetriever(em, vs, bm)
    generator = ResponseGenerator()
    
    # 2. Check if biosensors.pdf is ingested
    docs = vs.list_documents()
    target_doc = "biosensors.pdf"
    if target_doc not in docs:
        print(f"\n[Warning] Target document '{target_doc}' is not ingested yet.")
        print(f"Please ingest it first by running: python scripts/run.py ingest data/documents/{target_doc}")
        return

    # 3. Define QA Ground Truth Benchmark
    benchmark = [
        {
            "query": "What is the limit of detection for Chlorpyrifos in Table 1?",
            "expected_page": 5,
            "expected_keywords": ["7", "10", "ng/mL"],
            "description": "Enzymatic biosensors limit"
        },
        {
            "query": "Which disease is CA125 associated with in Table 2?",
            "expected_page": 7,
            "expected_keywords": ["ovarian", "cancer"],
            "description": "CA125 cancer biomarker"
        },
        {
            "query": "What is the limit of detection for Salmonella enterica in Table 3?",
            "expected_page": 8,
            "expected_keywords": ["10", "cfu", "mL"],
            "description": "Aptasensor salmonella limit"
        },
        {
            "query": "Which immobilization matrix is used for Staphylococcus aureus in Table 4?",
            "expected_page": 10,
            "expected_keywords": ["cellulose", "MWCNT"],
            "description": "Whole-cell staphylococcus matrix"
        }
    ]

    print(f"\n[System] Starting evaluation of {len(benchmark)} benchmark queries...\n")
    
    total_queries = len(benchmark)
    retrieval_successes = 0
    keyword_matches = 0
    citation_successes = 0
    total_time = 0.0
    
    for idx, test in enumerate(benchmark):
        print(f"Query {idx+1}/{total_queries}: '{test['query']}' ({test['description']})")
        start_time = time.time()
        
        # 1. Retrieve chunks (using top_k=10 and doc filter)
        retrieved_chunks = retriever.retrieve(
            test["query"], 
            top_k=10, 
            filters={"document_name": target_doc}
        )
        
        # Calculate Retrieval Recall (check if the expected page was fetched)
        pages_retrieved = [c["metadata"]["page_number"] for c in retrieved_chunks]
        retrieval_ok = test["expected_page"] in pages_retrieved
        if retrieval_ok:
            retrieval_successes += 1
            print("  - [Retrieval]: SUCCESS (Expected page retrieved)")
        else:
            print(f"  - [Retrieval]: FAILED (Expected page {test['expected_page']}, Got {pages_retrieved})")
            
        # 2. Generate grounded answer
        answer_res = generator.generate_answer(test["query"], retrieved_chunks)
        answer = answer_res["answer"]
        latency = time.time() - start_time
        total_time += latency
        
        # Calculate Groundedness (Keyword coverage checking)
        missing_keywords = [kw for kw in test["expected_keywords"] if kw.lower() not in answer.lower()]
        keyword_ok = len(missing_keywords) == 0
        if keyword_ok:
            keyword_matches += 1
            print("  - [Groundedness]: SUCCESS (All key facts covered)")
        else:
            print(f"  - [Groundedness]: FAILED (Missing keywords: {missing_keywords})")
            
        # Calculate Citation Accuracy (check if the expected page is cited in the text)
        expected_citation = f"Page: {test['expected_page']}"
        citation_ok = expected_citation in answer
        if citation_ok:
            citation_successes += 1
            print("  - [Citation]: SUCCESS (Correct source page cited)")
        else:
            print(f"  - [Citation]: FAILED (Expected citation '{expected_citation}')")
            
        print(f"  - [Latency]: {latency:.2f} seconds\n")
        
    # 4. Generate Scorecard
    print("=" * 45)
    print("           EVALUATION SCORECARD")
    print("=" * 45)
    print(f"Total Queries Evaluated:    {total_queries}")
    print(f"Retrieval Recall@10:       {(retrieval_successes/total_queries)*100:.1f}%")
    print(f"Factual Groundedness Rate:  {(keyword_matches/total_queries)*100:.1f}%")
    print(f"Citation Accuracy Rate:    {(citation_successes/total_queries)*100:.1f}%")
    print(f"Average Response Latency:   {total_time/total_queries:.2f} seconds")
    print("=" * 45)
    
if __name__ == "__main__":
    main()
