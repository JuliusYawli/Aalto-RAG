"""
RAG Application Configuration
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG application"""
    
    # LLM Provider Configuration
    USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
    USE_HUGGINGFACE = os.getenv("USE_HUGGINGFACE", "true").lower() == "true"
    
    # Ollama Configuration (Free, Local)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    
    # Hugging Face Configuration (Free, Cloud)
    HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "HuggingFaceH4/zephyr-7b-beta")
    HUGGINGFACE_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # Optional, for rate limits
    
    # OpenAI Configuration (Fallback if USE_OLLAMA=false and USE_HUGGINGFACE=false)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
    # Vector Store Configuration
    VECTORSTORE_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "vectorstore")
    )
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval Configuration
    TOP_K_RESULTS = 4
    
    # Document Directory
    DOCUMENTS_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "documents")
    )
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if cls.USE_OLLAMA:
            # Ollama doesn't need API keys
            return True
        elif cls.USE_HUGGINGFACE:
            # Hugging Face works without API token (but may have rate limits)
            return True
        else:
            # OpenAI needs API key
            if not cls.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY not found. Please set it in .env file or environment, "
                    "or set USE_HUGGINGFACE=true to use free Hugging Face models."
                )
        return True
