import os
import json
from pathlib import Path

# Base workspace directory
WORKSPACE_DIR = Path("c:/D/multimodal rag/rag-it")

# Data directories
DATA_DIR = WORKSPACE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "documents"
MEDIA_DIR = DATA_DIR / "cache" / "media"
DB_DIR = DATA_DIR / "vector_db"
KEYWORD_DIR = DATA_DIR / "cache" / "keyword"

# Ensure directories exist
for directory in [UPLOAD_DIR, MEDIA_DIR, DB_DIR, KEYWORD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Settings persistence path
SETTINGS_FILE = WORKSPACE_DIR / "settings.json"

# Models Configuration
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o"
OPENAI_VLM_MODEL = "gpt-4o"

LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_settings():
    """Load settings from the persistent JSON file."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_settings(settings: dict):
    """Save settings to the persistent JSON file."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving settings: {e}")

def get_openai_api_key():
    """Get the OpenAI API key from settings file, .env file, or environment variables."""
    # 1. Check settings file
    settings = get_settings()
    if settings.get("openai_api_key"):
        return settings["openai_api_key"]
    
    # 2. Check environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
        
    # 3. Try to read from a local .env file
    env_file = WORKSPACE_DIR / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("OPENAI_API_KEY="):
                        key = line.strip().split("=", 1)[1]
                        # Remove quotes if present
                        if key.startswith(('"', "'")) and key.endswith(('"', "'")):
                            key = key[1:-1]
                        return key
        except Exception:
            pass
    return None

def save_openai_api_key(key: str):
    """Save the OpenAI API key to the settings file."""
    settings = get_settings()
    settings["openai_api_key"] = key
    save_settings(settings)

def get_use_local_embeddings() -> bool:
    """Check if the system should use local offline embeddings. Defaults to True."""
    settings = get_settings()
    return settings.get("use_local_embeddings", True)

class TokenTracker:
    """Tracks cumulative OpenAI token usage during RAG execution loops."""
    prompt_tokens = 0
    completion_tokens = 0
    embedding_tokens = 0

    @classmethod
    def reset(cls):
        cls.prompt_tokens = 0
        cls.completion_tokens = 0
        cls.embedding_tokens = 0

    @classmethod
    def add_chat_tokens(cls, prompt: int, completion: int):
        cls.prompt_tokens += prompt
        cls.completion_tokens += completion

    @classmethod
    def add_embedding_tokens(cls, tokens: int):
        cls.embedding_tokens += tokens
