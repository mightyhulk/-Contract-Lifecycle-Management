#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLM RAG Pipeline and Chatbot

This module implements a RAG pipeline and chatbot interface for the CLM system.
"""

import os
import logging
import streamlit as st
from typing import List, Dict, Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.document_processor import DocumentProcessor
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/chatbot.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class CLMChatbot:
    """Chatbot for querying contract information using RAG."""
    
    def __init__(self):
        """Initialize the CLM chatbot."""
        self.document_processor = DocumentProcessor()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
        
        # Load document index
        if not self.document_processor.load_index():
            logger.warning("No document index found. Please run document_processor.py first.")
    
    def setup_rag_pipeline(self):
        """Set up the RAG pipeline."""
        # Template for generating a response based on context
        template = """You are a helpful assistant for a Contract Lifecycle Management system.
Use the following contract document excerpts to answer the question.
If you don't know the answer or can't find relevant information in the context, say so - don't make up information.
Always cite your sources by mentioning the document name and page number where appropriate.

Context:
{context}

Question:
{question}

Answer:
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join([
                f"Document: {doc.metadata.get('source_file', 'Unknown')}" + 
                (f", Page: {doc.metadata.get('page_number', 'N/A')}" if doc.metadata.get('file_type') == 'pdf' else "") + 
                f"\n{doc.page_content}"
                for doc in docs
            ])
        
        # Set up RAG pipeline
        rag_chain = (
            {"context": lambda x: format_docs(self.document_processor.similarity_search(x["question"])), 
             "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def find_similar_documents(self, document_name: str) -> List[Dict[str, Any]]:
        """Find documents similar to the specified document."""
        similar_docs = self.document_processor.find_similar_documents(document_name)
        return similar_docs
    
def run_chatbot():
    """Run the Streamlit chatbot interface."""
    st.set_page_config(page_title="CLM Assistant", page_icon="📄")
    
    st.title("Contract Lifecycle Management Assistant")
    
    # Initialize the chatbot
    chatbot = CLMChatbot()
    rag_chain = chatbot.setup_rag_pipeline()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Document similarity search
    with st.sidebar:
        st.header("Document Similarity")
        st.write("Find similar documents:")
        
        # Get list of documents
        contracts_dir = "data/contracts"
        document_list = []
        if os.path.exists(contracts_dir):
            document_list = [f for f in os.listdir(contracts_dir) if os.path.isfile(os.path.join(contracts_dir, f))]
        
        selected_doc = st.selectbox("Select a document", options=document_list)
        
        if st.button("Find Similar Documents"):
            if selected_doc:
                with st.spinner("Finding similar documents..."):
                    similar_docs = chatbot.find_similar_documents(selected_doc)
                    if similar_docs:
                        st.write("Similar documents:")
                        for doc_name, similarity in similar_docs:
                            st.write(f"- {doc_name} (Similarity: {similarity:.2f})")
                    else:
                        st.write("No similar documents found.")
            else:
                st.error("Please select a document first.")
    
    # Chat input
    if prompt := st.chat_input("Ask about contract information..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = rag_chain.invoke({"question": prompt})
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)

if __name__ == "__main__":
    run_chatbot()