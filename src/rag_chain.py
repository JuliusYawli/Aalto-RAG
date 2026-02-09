"""
RAG Chain Module
Implements the Retrieval-Augmented Generation chain
"""
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document


class RAGChain:
    """RAG chain for question answering"""
    
    def __init__(
        self, 
        vectorstore, 
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        top_k: int = 4
    ):
        """
        Initialize RAG chain
        
        Args:
            vectorstore: Vector store for document retrieval
            llm_model: OpenAI model to use
            temperature: Temperature for LLM (0.0 = deterministic)
            top_k: Number of documents to retrieve
        """
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.top_k = top_k
        
        # Create custom prompt template
        self.prompt_template = """You are a helpful assistant that answers questions based on the provided context. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """
        
        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": self.top_k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.PROMPT}
        )
    
    def ask(self, question: str) -> Dict:
        """
        Ask a question and get an answer with source documents
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary with 'result' and 'source_documents'
        """
        response = self.qa_chain.invoke({"query": question})
        return response
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents for a query without generating answer
        
        Args:
            query: Query string
            
        Returns:
            List of relevant documents
        """
        return self.vectorstore.similarity_search(query, k=self.top_k)
