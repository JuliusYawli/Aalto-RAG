"""
Vector Store Module
Handles document embedding and vector storage using ChromaDB
"""
import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


class VectorStore:
    """Manage vector store for document embeddings"""
    
    def __init__(self, persist_directory: str, embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist vector store
            embedding_model: OpenAI embedding model to use
        """
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore: Optional[Chroma] = None
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector store from documents
        
        Args:
            documents: List of documents to embed
            
        Returns:
            ChromaDB vector store
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        print(f"Creating vector store with {len(documents)} document chunks...")
        
        # Create the vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"Vector store created and persisted to {self.persist_directory}")
        return self.vectorstore
    
    def load_vectorstore(self) -> Chroma:
        """
        Load existing vector store from disk
        
        Returns:
            ChromaDB vector store
        """
        if not os.path.exists(self.persist_directory):
            raise ValueError(
                f"Vector store not found at {self.persist_directory}. "
                "Please create it first."
            )
        
        print(f"Loading vector store from {self.persist_directory}...")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        return self.vectorstore
    
    def get_vectorstore(self) -> Chroma:
        """
        Get the vector store (load if not already loaded)
        
        Returns:
            ChromaDB vector store
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
