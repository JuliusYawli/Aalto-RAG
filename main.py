"""
Main RAG Application
Command-line interface for the RAG system
"""
import os
import sys
import argparse
from src.config import Config
from src.document_loader import DocumentLoader
from src.vectorstore import VectorStore
from src.rag_chain import RAGChain


class RAGApplication:
    """Main RAG Application"""
    
    def __init__(self):
        """Initialize RAG application"""
        Config.validate()
        
        self.document_loader = DocumentLoader(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        self.vector_store = VectorStore(
            persist_directory=Config.VECTORSTORE_PATH,
            embedding_model=Config.EMBEDDING_MODEL
        )
        
        self.rag_chain = None
    
    def index_documents(self, directory_path: str = None):
        """
        Index documents from directory
        
        Args:
            directory_path: Path to documents directory (default: Config.DOCUMENTS_PATH)
        """
        if directory_path is None:
            directory_path = Config.DOCUMENTS_PATH
        
        print(f"\n📚 Loading documents from: {directory_path}")
        documents = self.document_loader.load_directory(directory_path)
        
        if not documents:
            print("❌ No documents found or loaded. Please add documents to the directory.")
            return False
        
        print(f"✅ Loaded {len(documents)} document chunks")
        
        # Delete existing vector store if it exists
        if os.path.exists(Config.VECTORSTORE_PATH):
            print("🗑️  Deleting existing vector store...")
            self.vector_store.delete_vectorstore()
        
        # Create new vector store
        print("\n🔧 Creating vector store...")
        self.vector_store.create_vectorstore(documents)
        print("✅ Vector store created successfully!")
        
        return True
    
    def load_chain(self):
        """Load or create the RAG chain"""
        if not os.path.exists(Config.VECTORSTORE_PATH):
            print("❌ Vector store not found. Please index documents first using --index")
            return False
        
        vectorstore = self.vector_store.load_vectorstore()
        self.rag_chain = RAGChain(
            vectorstore=vectorstore,
            llm_model=Config.LLM_MODEL,
            top_k=Config.TOP_K_RESULTS
        )
        
        return True
    
    def ask_question(self, question: str):
        """
        Ask a question and get an answer
        
        Args:
            question: Question to ask
        """
        if self.rag_chain is None:
            if not self.load_chain():
                return
        
        print(f"\n❓ Question: {question}")
        print("\n🔍 Searching for relevant information...")
        
        response = self.rag_chain.ask(question)
        
        print(f"\n💡 Answer:\n{response['result']}")
        
        if response.get('source_documents'):
            print(f"\n📄 Sources ({len(response['source_documents'])} documents):")
            for i, doc in enumerate(response['source_documents'], 1):
                source = doc.metadata.get('source', 'Unknown')
                print(f"\n  [{i}] {source}")
                print(f"      {doc.page_content[:200]}...")
    
    def interactive_mode(self):
        """Run in interactive Q&A mode"""
        if not self.load_chain():
            return
        
        print("\n🤖 RAG Application - Interactive Mode")
        print("=" * 50)
        print("Ask questions based on your indexed documents.")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                self.ask_question(question)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RAG Application for Domain-Specific Question Answering"
    )
    
    parser.add_argument(
        '--index',
        action='store_true',
        help='Index documents from the documents directory'
    )
    
    parser.add_argument(
        '--documents-path',
        type=str,
        help='Path to documents directory (default: ./documents)'
    )
    
    parser.add_argument(
        '--question',
        type=str,
        help='Ask a single question'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive Q&A mode'
    )
    
    args = parser.parse_args()
    
    try:
        app = RAGApplication()
        
        # Index documents if requested
        if args.index:
            success = app.index_documents(args.documents_path)
            if not success:
                sys.exit(1)
            
            # If only indexing was requested, exit
            if not args.question and not args.interactive:
                return
        
        # Ask single question if provided
        if args.question:
            app.ask_question(args.question)
        
        # Run interactive mode if requested or no other action
        elif args.interactive or (not args.index and not args.question):
            app.interactive_mode()
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
