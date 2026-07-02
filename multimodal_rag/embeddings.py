import torch
from transformers import AutoTokenizer, AutoModel
from multimodal_rag.config import LOCAL_EMBED_MODEL

class LocalEmbedder:
    """Fallback local embedding generator using PyTorch & Transformers."""
    def __init__(self, model_name=LOCAL_EMBED_MODEL):
        print(f"[Embedding] Initializing local embedding model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.dimension = 384
        print(f"[Embedding] Local model loaded on device: {self.device}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        # Batch tokenization
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Mean Pooling - Take attention mask into account for correct averaging
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        return embeddings.cpu().numpy().tolist()

class EmbeddingPipeline:
    """Unified Embedding Manager using local Hugging Face Sentence Transformers."""
    def __init__(self):
        self.embedder = LocalEmbedder()
        self.dimension = self.embedder.dimension
        self.is_local = True

    def reset(self):
        """No-op reset for local offline embedding pipeline."""
        pass

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embedder.embed([text])
        return embeddings[0] if embeddings else []
