import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from multimodal_rag.config import KEYWORD_DIR

def simple_tokenize(text: str) -> list[str]:
    """Helper to lowercase and split text into words."""
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())

class BM25IndexManager:
    """Manages keyword indexing and BM25 retrieval for hybrid search."""
    def __init__(self):
        self.index_path = KEYWORD_DIR / "bm25_index.pkl"
        self.chunks = []
        self.bm25 = None
        self.load_index()

    def load_index(self):
        """Loads index from disk if it exists."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.chunks = data.get("chunks", [])
                    corpus = data.get("corpus", [])
                    
                    if corpus:
                        self.bm25 = BM25Okapi(corpus)
                    else:
                        self.bm25 = None
                print(f"[BM25] Loaded index with {len(self.chunks)} chunks from disk.")
            except Exception as e:
                print(f"[BM25] Failed to load index from disk: {e}. Starting fresh.")
                self.chunks = []
                self.bm25 = None
        else:
            self.chunks = []
            self.bm25 = None

    def save_index(self):
        """Saves current index state to disk."""
        try:
            corpus = [simple_tokenize(c["content"]) for c in self.chunks]
            data = {
                "chunks": self.chunks,
                "corpus": corpus
            }
            with open(self.index_path, "wb") as f:
                pickle.dump(data, f)
            print(f"[BM25] Saved index with {len(self.chunks)} chunks to disk.")
        except Exception as e:
            print(f"[BM25] Failed to save index: {e}")

    def add_chunks(self, new_chunks: list[dict]):
        """Adds new chunks to the corpus, rebuilds the index, and saves it."""
        if not new_chunks:
            return
            
        # Clean chunks to store only essential fields to prevent heavy pickling
        cleaned_chunks = []
        for c in new_chunks:
            cleaned = {
                "id": c["id"],
                "content": c["content"],
                "document_name": c["document_name"],
                "page_number": c["page_number"],
                "section_title": c["section_title"],
                "type": c["type"]
            }
            if c["type"] == "image" and c["metadata"].get("image_path"):
                cleaned["image_path"] = c["metadata"]["image_path"]
            elif c["type"] == "table" and c["metadata"].get("image_path"):
                cleaned["image_path"] = c["metadata"]["image_path"]
                cleaned["table_id"] = c["metadata"].get("table_id", "")
            
            cleaned_chunks.append(cleaned)

        self.chunks.extend(cleaned_chunks)
        corpus = [simple_tokenize(c["content"]) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)
        self.save_index()

    def delete_document(self, document_name: str):
        """Removes all chunks matching the document name, rebuilds index, and saves."""
        original_count = len(self.chunks)
        self.chunks = [c for c in self.chunks if c["document_name"] != document_name]
        
        if len(self.chunks) != original_count:
            if self.chunks:
                corpus = [simple_tokenize(c["content"]) for c in self.chunks]
                self.bm25 = BM25Okapi(corpus)
            else:
                self.bm25 = None
            self.save_index()
            print(f"[BM25] Deleted document '{document_name}' and rebuilt index.")

    def search(self, query_text: str, top_k=5, filters: dict = None) -> list[dict]:
        """
        Searches the BM25 index and ranks documents.
        Supports filtering on metadata e.g. {"document_name": "x.pdf", "type": "table"}
        """
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = simple_tokenize(query_text)
        if not tokenized_query:
            return []

        # Get scores for all documents in the corpus
        scores = self.bm25.get_scores(tokenized_query)
        
        # Zip, filter, and sort
        scored_chunks = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            
            # Apply metadata filters
            matched = True
            if filters:
                for fk, fv in filters.items():
                    if fv is not None and chunk.get(fk) != fv:
                        matched = False
                        break
            
            if matched and score > 0:
                scored_chunks.append({
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
                    "score": float(score)
                })
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def reset_index(self):
        """Clears the index from memory and disk."""
        self.chunks = []
        self.bm25 = None
        if self.index_path.exists():
            try:
                self.index_path.unlink()
                print("[BM25] Reset complete (deleted index file).")
            except Exception as e:
                print(f"[BM25] Error deleting index file: {e}")
