"""
RAG Chain Module
Implements the Retrieval-Augmented Generation chain
"""
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama, HuggingFaceHub
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import Config


class RAGChain:
    """RAG chain for question answering"""
    
    def __init__(
        self, 
        vectorstore, 
        llm_model: str = None,
        temperature: float = 0.0,
        top_k: int = 4
    ):
        """
        Initialize RAG chain
        
        Args:
            vectorstore: Vector store for document retrieval
            llm_model: Model to use (optional, uses Config default)
            temperature: Temperature for LLM (0.0 = deterministic)
            top_k: Number of documents to retrieve
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        
        # Choose LLM based on configuration (Priority: Groq > Ollama > HuggingFace > OpenAI)
        if Config.USE_GROQ and Config.GROQ_API_KEY:
            model = llm_model or Config.GROQ_MODEL
            print(f"Using Groq LLM: {model}")
            self.llm = ChatGroq(
                groq_api_key=Config.GROQ_API_KEY,
                model_name=model,
                temperature=temperature
            )
        elif Config.USE_OLLAMA:
            model = llm_model or Config.OLLAMA_MODEL
            print(f"Using Ollama LLM: {model}")
            self.llm = Ollama(
                model=model,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=temperature
            )
        elif Config.USE_HUGGINGFACE:
            model = llm_model or Config.HUGGINGFACE_MODEL
            print(f"Using Hugging Face LLM: {model}")
            self.llm = HuggingFaceHub(
                repo_id=model,
                model_kwargs={
                    "temperature": temperature,
                    "max_new_tokens": 512,
                },
                huggingfacehub_api_token=Config.HUGGINGFACE_API_TOKEN
            )
        else:
            model = llm_model or Config.LLM_MODEL
            print(f"Using OpenAI LLM: {model}")
            self.llm = ChatOpenAI(model=model, temperature=temperature)
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        
        # Create custom prompt template
        template = """You are a helpful assistant that answers questions based on the provided context. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """
        
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # Create RAG chain using LCEL (LangChain Expression Language)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.qa_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask(self, question: str) -> Dict:
        """
        Ask a question and get an answer with source documents
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary with 'result' and 'source_documents'
        """
        # Get source documents
        source_docs = self.retriever.invoke(question)
        
        # Get answer
        answer = self.qa_chain.invoke(question)
        
        return {
            "result": answer,
            "source_documents": source_docs
        }
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents for a query without generating answer
        
        Args:
            query: Query string
            
        Returns:
            List of relevant documents
        """
        return self.vectorstore.similarity_search(query, k=self.top_k)
