"""
Document Loader Module
Handles loading and processing of various document formats
"""
import os
from typing import List
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DocumentLoader:
    """Load and process documents from various formats"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document loader
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document based on file extension
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext == '.docx':
                loader = Docx2txtLoader(file_path)
            elif file_ext == '.txt':
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        
        except Exception as e:
            print(f"Error loading document {file_path}: {str(e)}")
            return []
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of all document chunks from directory
        """
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return []
        
        all_documents = []
        
        # Load text files
        try:
            text_loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            all_documents.extend(text_loader.load())
        except Exception as e:
            print(f"Error loading text files: {str(e)}")
        
        # Load PDF files
        for filename in os.listdir(directory_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                all_documents.extend(self.load_document(file_path))
        
        # Load DOCX files
        for filename in os.listdir(directory_path):
            if filename.endswith('.docx'):
                file_path = os.path.join(directory_path, filename)
                all_documents.extend(self.load_document(file_path))
        
        # Split documents into chunks
        if all_documents:
            return self.text_splitter.split_documents(all_documents)
        
        return []
