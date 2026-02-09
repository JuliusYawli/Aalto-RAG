"""
Vector Store Module
Handles document embedding and vector storage using FAISS
"""
import os
import pickle
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from src.config import Config


class VectorStore:
    """Manage vector store for document embeddings"""
    
    def __init__(self, persist_directory: str, embedding_model: str = None):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist vector store
            embedding_model: Embedding model to use (optional, uses Config default)
        """
        self.persist_directory = persist_directory
        
        # Choose embeddings based on configuration
        if Config.USE_OLLAMA:
            print(f"Using Ollama embeddings: {Config.OLLAMA_EMBEDDING_MODEL}")
            self.embeddings = OllamaEmbeddings(
                model=Config.OLLAMA_EMBEDDING_MODEL,
                base_url=Config.OLLAMA_BASE_URL
            )
        else:
            model = embedding_model or Config.EMBEDDING_MODEL
            print(f"Using OpenAI embeddings: {model}")
            self.embeddings = OpenAIEmbeddings(model=model)
        
        self.vectorstore: Optional[FAISS] = None
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Create a new vector store from documents
        
        Args:
            documents: List of documents to embed
            
        Returns:
            FAISS vector store
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        print(f"Creating vector store with {len(documents)} document chunks...")
        
        # Create the vector store
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        # Save to disk
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vectorstore.save_local(self.persist_directory)
        
        print(f"Vector store created and persisted to {self.persist_directory}")
        return self.vectorstore
    
    def load_vectorstore(self) -> FAISS:
        """
        Load existing vector store from disk
        
        Returns:
            FAISS vector store
        """
        if not os.path.exists(self.persist_directory):
            raise ValueError(
                f"Vector store not found at {self.persist_directory}. "
                "Please create it first."
            )
        
        print(f"Loading vector store from {self.persist_directory}...")
        self.vectorstore = FAISS.load_local(
            self.persist_directory,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        return self.vectorstore
    
    def get_vectorstore(self) -> FAISS:
        """
        Get the vector store (load if not already loaded)
        
        Returns:
            FAISS vector store
        """
        if self.vectorstore is None:
            self.vectorstore = self.load_vectorstore()
        return self.vectorstore
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the vector store
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Load or create it first.")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def delete_vectorstore(self):
        """Delete the vector store from disk"""
        import shutil
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"Vector store deleted from {self.persist_directory}")
        self.vectorstore = None
