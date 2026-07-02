from multimodal_rag.embeddings import EmbeddingPipeline
from multimodal_rag.vector_store import VectorStoreManager
from multimodal_rag.keyword_index import BM25IndexManager

class HybridRetriever:
    """Combines dense vector search and sparse keyword search (BM25) using Reciprocal Rank Fusion (RRF)."""
    def __init__(self, embeddings: EmbeddingPipeline, vector_store: VectorStoreManager, bm25_index: BM25IndexManager):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.k_rrf = 60 # standard constant for Reciprocal Rank Fusion

    def retrieve(self, query_text: str, top_k=5, filters: dict = None, intent: dict = None) -> list[dict]:
        """
        Performs hybrid retrieval and returns the top_k merged results.
        filters: optional dictionary of metadata filters, e.g. {"document_name": "annual_report.pdf"}
        """
        # 1. Fetch dense vector (semantic) results
        query_embedding = self.embeddings.embed_query(query_text)
        semantic_results = []
        if query_embedding:
            # Fetch double the top_k elements to allow RRF to fuse a richer candidate set
            semantic_results = self.vector_store.search(query_embedding, top_k=top_k * 3, filters=filters)
            
        # 2. Fetch sparse keyword (BM25) results
        keyword_results = self.bm25_index.search(query_text, top_k=top_k * 3, filters=filters)
        
        # If both are empty, return empty list
        if not semantic_results and not keyword_results:
            return []
            
        # 3. Apply Reciprocal Rank Fusion (RRF)
        # Create a dictionary to hold chunk information and RRF scores
        rrf_scores = {}
        chunk_lookup = {} # Maps chunk ID to chunk data
        
        # Helper to process scoring list
        def add_rrf_scores(results_list):
            for rank, item in enumerate(results_list):
                chunk_id = item["id"]
                # 1-indexed rank
                rank_score = 1.0 / (self.k_rrf + (rank + 1))
                
                if chunk_id in rrf_scores:
                    rrf_scores[chunk_id] += rank_score
                else:
                    rrf_scores[chunk_id] = rank_score
                    chunk_lookup[chunk_id] = item
                    
        add_rrf_scores(semantic_results)
        add_rrf_scores(keyword_results)
        
        # Sort chunks by RRF score in descending order
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Construct output results
        merged_results = []
        for cid in sorted_ids[:top_k]:
            original_item = chunk_lookup[cid]
            # Copy to prevent mutation issues
            merged_item = {
                "id": original_item["id"],
                "content": original_item["content"],
                "metadata": original_item["metadata"].copy() if "metadata" in original_item else {},
                "rrf_score": rrf_scores[cid],
                # Retain visual indicators
                "type": original_item["metadata"].get("type", "text"),
                "image_path": original_item["metadata"].get("image_path", "")
            }
            merged_results.append(merged_item)
            
        # --- Table-Aware Context Enhancement ---
        query_lower = query_text.lower()
        needs_tables = any(w in query_lower for w in ["table", "tabel", "tabl"])
        
        if needs_tables and self.bm25_index.chunks:
            import re
            table_pattern = re.compile(r'\bTable\s*(\d+|[IVXLC]+)\b', re.IGNORECASE)
            
            # Find all (document_name, page_number) pairs that contain table headers/identifiers
            table_pages = set()
            for chunk in self.bm25_index.chunks:
                is_table = (chunk["type"] == "table") or bool(table_pattern.search(chunk["content"]))
                if is_table:
                    table_pages.add((chunk["document_name"], chunk["page_number"]))
            
            seen_ids = set([r["id"] for r in merged_results])
            extra_chunks = []
            
            for chunk in self.bm25_index.chunks:
                if chunk["id"] in seen_ids:
                    continue
                if filters and filters.get("document_name") and chunk["document_name"] != filters["document_name"]:
                    continue
                
                # If this chunk belongs to a page containing a table
                if (chunk["document_name"], chunk["page_number"]) in table_pages:
                    extra_chunks.append({
                        "id": chunk["id"],
                        "content": chunk["content"],
                        "metadata": {
                            "document_name": chunk["document_name"],
                            "page_number": chunk["page_number"],
                            "section_title": chunk["section_title"],
                            "type": chunk["type"],
                            "image_path": chunk.get("image_path", ""),
                            "table_id": chunk.get("table_id", "")
                        },
                        "rrf_score": 0.0,
                        "type": chunk["type"],
                        "image_path": chunk.get("image_path", "")
                    })
            
            # Append missing table page chunks
            merged_results.extend(extra_chunks)
            print(f"[Retriever] Table-Aware Page Enhancement: Appended {len(extra_chunks)} missing page chunks to context.")
            
        # --- Global Figure/Table Context Expansion ---
        wants_all_images = intent.get("wants_all_images", False) if intent else False
        wants_all_tables = intent.get("wants_all_tables", False) if intent else False
        
        extra_global_chunks = []
        seen_ids = set([r["id"] for r in merged_results])
        
        if (wants_all_images or wants_all_tables) and self.bm25_index.chunks:
            for chunk in self.bm25_index.chunks:
                if chunk["id"] in seen_ids:
                    continue
                if filters and filters.get("document_name") and chunk["document_name"] != filters["document_name"]:
                    continue
                
                match_image = wants_all_images and chunk["type"] == "image"
                match_table = wants_all_tables and chunk["type"] == "table"
                
                if match_image or match_table:
                    extra_global_chunks.append({
                        "id": chunk["id"],
                        "content": chunk["content"],
                        "metadata": {
                            "document_name": chunk["document_name"],
                            "page_number": chunk["page_number"],
                            "section_title": chunk["section_title"],
                            "type": chunk["type"],
                            "image_path": chunk.get("image_path", ""),
                            "table_id": chunk.get("table_id", "")
                        },
                        "rrf_score": 0.0,
                        "type": chunk["type"],
                        "image_path": chunk.get("image_path", "")
                    })
            if extra_global_chunks:
                merged_results.extend(extra_global_chunks)
                print(f"[Retriever] Global Query Boost: Appended {len(extra_global_chunks)} structural chunks to context.")
            
        print(f"[Retriever] Hybrid search completed. Fused {len(semantic_results)} semantic and {len(keyword_results)} keyword results. Total context chunks: {len(merged_results)}")
        return merged_results
