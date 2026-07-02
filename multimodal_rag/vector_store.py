import chromadb
from pathlib import Path
from multimodal_rag.config import DB_DIR

class VectorStoreManager:
    """Manages persistent indexing and semantic retrieval in ChromaDB."""
    def __init__(self, dimension: int):
        self.db_path = DB_DIR
        self.dimension = dimension
        print(f"[VectorStore] Connecting to ChromaDB at: {self.db_path}...")
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # We append the dimension to the collection name to isolate index schemas
        self.collection_name = f"multimodal_rag_collection_{self.dimension}"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,  # Higher value improves search accuracy during index build
                "hnsw:search_ef": 50,         # Higher value improves recall accuracy during queries
                "hnsw:M": 16                  # Max connections per node
            }
        )
        print(f"[VectorStore] Collection active: {self.collection_name}")

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        """Indexes a list of parsed chunks and their precomputed embeddings."""
        if not chunks:
            return
        
        ids = [c["id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        
        # Flatten metadata: Chroma only supports flat primitive types (str, int, float, bool)
        metadatas = []
        for c in chunks:
            meta = {
                "document_name": c["document_name"],
                "page_number": int(c["page_number"]),
                "section_title": c["section_title"],
                "type": c["type"]
            }
            # Add table/image specific keys
            if c["type"] == "image" and c["metadata"].get("image_path"):
                meta["image_path"] = c["metadata"]["image_path"]
            elif c["type"] == "table" and c["metadata"].get("image_path"):
                meta["image_path"] = c["metadata"]["image_path"]
                meta["table_id"] = c["metadata"].get("table_id", "")
                
            metadatas.append(meta)
            
        # Add to Chroma collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"[VectorStore] Indexed {len(chunks)} chunks into {self.collection_name}.")

    def search(self, query_embedding: list[float], top_k=5, filters: dict = None) -> list[dict]:
        """
        Searches Chroma DB for matches using query embeddings.
        Supports filter criteria e.g. {"document_name": "report.pdf", "type": "table"}
        """
        chroma_filter = {}
        if filters:
            # Build metadata query filters for Chroma
            # Example: {"$and": [{"document_name": "..."}, {"type": "..."}]}
            filter_list = []
            for k, v in filters.items():
                if v is not None:
                    filter_list.append({k: v})
            
            if len(filter_list) > 1:
                chroma_filter = {"$and": filter_list}
            elif len(filter_list) == 1:
                chroma_filter = filter_list[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=chroma_filter if chroma_filter else None
        )
        
        # Format results
        formatted_results = []
        if results and results["ids"] and results["ids"][0]:
            n_items = len(results["ids"][0])
            for i in range(n_items):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    "score": 1.0 - results["distances"][0][i] if results["distances"] else 1.0 # Cosine score
                })
        return formatted_results

    def delete_document(self, document_name: str):
        """Deletes all chunks belonging to a specific document name."""
        self.collection.delete(
            where={"document_name": document_name}
        )
        print(f"[VectorStore] Deleted document '{document_name}' from {self.collection_name}.")

    def list_documents(self) -> list[str]:
        """Returns a list of unique document names indexed in this collection."""
        # Query all items to inspect their document metadata
        all_metadata = self.collection.get(include=["metadatas"])
        if not all_metadata or not all_metadata["metadatas"]:
            return []
        
        doc_names = set()
        for meta in all_metadata["metadatas"]:
            if "document_name" in meta:
                doc_names.add(meta["document_name"])
        return sorted(list(doc_names))

    def get_stats(self) -> dict:
        """Returns statistical counts of elements in the index."""
        all_data = self.collection.get(include=["metadatas"])
        total_chunks = len(all_data["ids"]) if all_data and all_data["ids"] else 0
        
        stats = {
            "total_chunks": total_chunks,
            "text_count": 0,
            "table_count": 0,
            "image_count": 0
        }
        
        if all_data and all_data["metadatas"]:
            for meta in all_data["metadatas"]:
                t = meta.get("type", "text")
                if t == "text":
                    stats["text_count"] += 1
                elif t == "table":
                    stats["table_count"] += 1
                elif t == "image":
                    stats["image_count"] += 1
                    
        return stats

    def reset_db(self):
        """Resets the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 50,
                "hnsw:M": 16
            }
        )
        print(f"[VectorStore] Reset complete for collection: {self.collection_name}")
