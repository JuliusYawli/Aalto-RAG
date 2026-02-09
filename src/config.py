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
    USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"
    
    # Ollama Configuration (Free, Local)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    
    # OpenAI Configuration (Fallback if USE_OLLAMA=false)
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
        else:
            # OpenAI needs API key
            if not cls.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY not found. Please set it in .env file or environment, "
                    "or set USE_OLLAMA=true to use free local models."
                )
        return True
