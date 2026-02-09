"""
Streamlit Web Interface for RAG Application
Run with: streamlit run app.py
"""
import streamlit as st
from src.config import Config
from src.document_loader import DocumentLoader
from src.vectorstore import VectorStore
from src.rag_chain import RAGChain
import os

# Page configuration
st.set_page_config(
    page_title="Aalto RAG System",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_rag():
    """Initialize the RAG system"""
    try:
        Config.validate()
        
        if not os.path.exists(Config.VECTORSTORE_PATH):
            st.error("⚠️ Vector store not found. Please index documents first using the sidebar.")
            return None
        
        vector_store = VectorStore(persist_directory=Config.VECTORSTORE_PATH)
        vectorstore = vector_store.load_vectorstore()
        
        rag_chain = RAGChain(
            vectorstore=vectorstore,
            top_k=Config.TOP_K_RESULTS
        )
        
        return rag_chain
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

def index_documents():
    """Index documents from the documents directory"""
    try:
        with st.spinner("📚 Loading documents..."):
            document_loader = DocumentLoader(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            documents = document_loader.load_directory(Config.DOCUMENTS_PATH)
            
            if not documents:
                st.error("No documents found in the documents directory.")
                return False
            
            st.success(f"✅ Loaded {len(documents)} document chunks")
        
        with st.spinner("🔧 Creating vector store..."):
            vector_store = VectorStore(persist_directory=Config.VECTORSTORE_PATH)
            
            # Delete existing if it exists
            if os.path.exists(Config.VECTORSTORE_PATH):
                vector_store.delete_vectorstore()
            
            vector_store.create_vectorstore(documents)
            st.success("✅ Vector store created successfully!")
            
        return True
    except Exception as e:
        st.error(f"Error indexing documents: {str(e)}")
        return False

# Header
st.title("🤖 Aalto RAG System")
st.markdown("Ask questions based on your indexed documents using **free Ollama models**!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Show current configuration
    if Config.USE_OLLAMA:
        st.success("🆓 Using Ollama (Free)")
        st.info(f"**Model:** {Config.OLLAMA_MODEL}")
        st.info(f"**Embeddings:** {Config.OLLAMA_EMBEDDING_MODEL}")
    else:
        st.warning("💳 Using OpenAI (Paid)")
        st.info(f"**Model:** {Config.LLM_MODEL}")
    
    st.divider()
    
    # Document indexing
    st.header("📄 Document Management")
    
    if os.path.exists(Config.VECTORSTORE_PATH):
        st.success("✅ Documents indexed")
    else:
        st.warning("⚠️ No documents indexed")
    
    if st.button("🔄 Re-index Documents", use_container_width=True):
        if index_documents():
            st.session_state.rag_chain = None  # Reset to reload
            st.rerun()
    
    st.divider()
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    # Information
    st.header("ℹ️ About")
    st.markdown("""
    This RAG system uses:
    - **Ollama** for free local LLM
    - **FAISS** for vector storage
    - **LangChain** for orchestration
    
    Add your documents to the `documents/` folder and click "Re-index Documents" to update.
    """)

# Initialize RAG if not already done
if st.session_state.rag_chain is None:
    st.session_state.rag_chain = initialize_rag()

# Main chat interface
if st.session_state.rag_chain is not None:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source}")
    
    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching and generating answer..."):
                try:
                    response = st.session_state.rag_chain.ask(question)
                    answer = response["result"]
                    sources = [doc.metadata.get("source", "Unknown") for doc in response["source_documents"]]
                    
                    st.markdown(answer)
                    
                    # Add assistant message to chat
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Show sources in expander
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:** {source}")
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })

else:
    # Show welcome message if RAG not initialized
    st.info("👈 Please index documents using the sidebar to get started!")
    
    st.markdown("""
    ### Getting Started
    
    1. **Add documents** to the `documents/` folder
    2. **Click "Re-index Documents"** in the sidebar
    3. **Start asking questions!**
    
    ### Example Questions
    - What are the best practices for machine learning?
    - How do transformers work in NLP?
    - What is RAG and how does it work?
    """)
