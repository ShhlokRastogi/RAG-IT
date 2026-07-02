import torch
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from multimodal_rag.config import get_openai_api_key, OPENAI_EMBED_MODEL, LOCAL_EMBED_MODEL

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

class OpenAIEmbedder:
    """Remote embedding generator using OpenAI's API."""
    def __init__(self, api_key: str, model_name=OPENAI_EMBED_MODEL):
        print(f"[Embedding] Initializing OpenAI embeddings with model: {model_name}...")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.dimension = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        # Clean inputs to avoid empty strings/None causing API failures
        cleaned_texts = [t.replace("\n", " ").strip() for t in texts]
        cleaned_texts = [t if t else " " for t in cleaned_texts]
        
        # Call OpenAI Embeddings API (batch size is handled by API automatically)
        response = self.client.embeddings.create(
            input=cleaned_texts,
            model=self.model_name
        )
        # Record tokens
        if hasattr(response, 'usage') and response.usage:
            from multimodal_rag.config import TokenTracker
            TokenTracker.add_embedding_tokens(response.usage.prompt_tokens)
            
        # Sort by index to maintain original order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

class EmbeddingPipeline:
    """Unified Embedding Manager that dynamically switches between OpenAI and Local fallback."""
    def __init__(self):
        self.api_key = get_openai_api_key()
        self.embedder = None
        self.dimension = None
        self.reset()

    def reset(self):
        """Recheck API key and reinitialize the embedder."""
        from multimodal_rag.config import get_use_local_embeddings
        self.api_key = get_openai_api_key()
        
        if get_use_local_embeddings():
            self._init_local()
        elif self.api_key:
            try:
                self.embedder = OpenAIEmbedder(api_key=self.api_key)
                self.dimension = self.embedder.dimension
                self.is_local = False
            except Exception as e:
                print(f"[Embedding] Failed to initialize OpenAI Embeddings: {e}. Falling back to local.")
                self._init_local()
        else:
            self._init_local()

    def _init_local(self):
        self.embedder = LocalEmbedder()
        self.dimension = self.embedder.dimension
        self.is_local = True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embedder.embed([text])
        return embeddings[0] if embeddings else []
