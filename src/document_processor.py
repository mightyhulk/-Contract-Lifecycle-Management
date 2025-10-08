#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Document Processor

This module handles the ingestion, processing, and indexing of contract documents.
"""

import os
import re
import logging
import datetime
from typing import List, Dict, Any, Tuple, Optional
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader,
    UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/document_processor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process and index documents for the CLM system."""
       
    def __init__(self, contracts_dir="data/contracts", faiss_index_path="data/faiss_index"):
        """Initialize the document processor."""
        self.contracts_dir = contracts_dir
        self.faiss_index_path = faiss_index_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # Choose model via env var, default to gemini-embedding-001 but normalize for API
        raw_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")

        if "/" not in raw_model:
            model_for_api = f"models/{raw_model}"
        else:
            model_for_api = raw_model

        logger.info(f"Using Google embedding model: raw='{raw_model}' normalized='{model_for_api}'")

        # Initialize embeddings with a normalization + fallback attempt
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=model_for_api,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                task_type="RETRIEVAL_QUERY",
                title="Contract Analysis",
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize GoogleGenerativeAIEmbeddings with "
                f"model='{model_for_api}': {e}. Trying raw model name '{raw_model}' as fallback."
            )
            try:
                # Try the raw model string as a fallback (some SDK versions expect the model id)
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model=raw_model,
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    task_type="RETRIEVAL_QUERY",
                    title="Contract Analysis",
                )
            except Exception as e2:
                # If it still fails, log and re-raise so failure is obvious at startup
                logger.error(
                    "Failed to initialize GoogleGenerativeAIEmbeddings with both "
                    f"'{model_for_api}' and '{raw_model}'. Please check your model name, "
                    "your GOOGLE_API_KEY, and your langchain-google-genai / google-genai versions.",
                    exc_info=True,
                )
                raise

        self.vector_db = None

        # Regular expressions for extracting dates and entities
        self.date_pattern = r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b)"
        self.email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        self.phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
        self.address_pattern = r"\d+\s+[A-Za-z]+\s+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Court|Ct|Way|Parkway|Pkwy|Place|Pl),\s+[A-Za-z\s]+,\s+[A-Z]{2}\s+\d{5}"


    def load_documents(self) -> List[Document]:
        """Load all documents from the contracts directory."""
        documents = []
        
        if not os.path.exists(self.contracts_dir):
            logger.error(f"Contracts directory does not exist: {self.contracts_dir}")
            return documents
        
        for filename in os.listdir(self.contracts_dir):
            file_path = os.path.join(self.contracts_dir, filename)
            
            if not os.path.isfile(file_path):
                continue
                
            try:
                if filename.lower().endswith('.pdf'):
                    logger.info(f"Loading PDF: {filename}")
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    # Add source information
                    for doc in docs:
                        doc.metadata['source_file'] = filename
                        doc.metadata['file_type'] = 'pdf'
                        doc.metadata['page_number'] = doc.metadata.get('page', 0) + 1
                        # Extract dates if present
                        self._extract_metadata(doc)
                    documents.extend(docs)
                    
                elif filename.lower().endswith('.docx'):
                    logger.info(f"Loading DOCX: {filename}")
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata['source_file'] = filename
                        doc.metadata['file_type'] = 'docx'
                        # Extract dates if present
                        self._extract_metadata(doc)
                    documents.extend(docs)
                    
                elif filename.lower().endswith('.txt'):
                    logger.info(f"Loading TXT: {filename}")
                    loader = TextLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata['source_file'] = filename
                        doc.metadata['file_type'] = 'txt'
                        # Extract dates if present
                        self._extract_metadata(doc)
                    documents.extend(docs)
                    
                else:
                    logger.info(f"Loading using UnstructuredFileLoader: {filename}")
                    try:
                        loader = UnstructuredFileLoader(file_path)
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata['source_file'] = filename
                            doc.metadata['file_type'] = 'unstructured'
                            # Extract dates if present
                            self._extract_metadata(doc)
                        documents.extend(docs)
                    except Exception as e:
                        logger.error(f"Error loading file with UnstructuredFileLoader: {filename}, error: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error loading document {filename}: {str(e)}")
        
        logger.info(f"Loaded {len(documents)} document chunks from {len(set(doc.metadata['source_file'] for doc in documents))} files")
        return documents
    
    def _extract_metadata(self, doc: Document) -> None:
        """Extract metadata from document content like dates, contacts, etc."""
        content = doc.page_content
        
        # Extract dates
        dates = re.findall(self.date_pattern, content)
        if dates:
            # Try to parse dates to a standard format
            doc.metadata['extracted_dates'] = dates
        
        # Extract emails
        emails = re.findall(self.email_pattern, content)
        if emails:
            doc.metadata['emails'] = emails
        
        # Extract phone numbers
        phones = re.findall(self.phone_pattern, content)
        if phones:
            doc.metadata['phones'] = phones
        
        # Extract addresses
        addresses = re.findall(self.address_pattern, content)
        if addresses:
            doc.metadata['addresses'] = addresses
        
        # Look for specific contract information
        if "EFFECTIVE DATE:" in content or "Creation Date:" in content:
            doc.metadata['has_creation_date'] = True
            
        if "EXPIRATION DATE:" in content or "Expiration Date:" in content:
            doc.metadata['has_expiration_date'] = True
            
        if "RENEWAL DATE:" in content or "Renewal Date:" in content:
            doc.metadata['has_renewal_date'] = True
    
    def process_and_index(self) -> None:
        """Process documents and create a vector index."""
        # Load all documents
        documents = self.load_documents()
        
        if not documents:
            logger.warning("No documents were loaded, cannot create index")
            return
        
        # Split documents into chunks
        logger.info("Splitting documents into chunks")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} document sections")
        
        # Create FAISS vector store
        logger.info("Creating FAISS vector index")
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        
        # Save the index
        os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
        self.vector_db.save_local(self.faiss_index_path)
        logger.info(f"FAISS index saved to {self.faiss_index_path}")
    

    def load_index(self) -> bool:
        """Load the existing FAISS index."""
        try:
            if os.path.exists(self.faiss_index_path):
                logger.info(f"Loading FAISS index from {self.faiss_index_path}")
                self.vector_db = FAISS.load_local(
                    self.faiss_index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True  
                )
                logger.info("FAISS index loaded successfully.")
                return True
            else:
                logger.warning(f"No existing FAISS index found at {self.faiss_index_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            return False

    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for documents similar to the query."""
        if not self.vector_db:
            if not self.load_index():
                logger.error("No vector database available for search")
                return []
        
        try:
            results = self.vector_db.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def find_similar_documents(self, document_name: str, k: int = 3) -> List[Tuple[str, float]]:
        """Find documents similar to the specified document."""
        if not self.vector_db:
            if not self.load_index():
                logger.error("No vector database available for similarity comparison")
                return []
        
        try:
            # First, load and embed the target document
            target_document = None
            for filename in os.listdir(self.contracts_dir):
                if filename == document_name:
                    file_path = os.path.join(self.contracts_dir, filename)
                    if filename.lower().endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                    elif filename.lower().endswith('.docx'):
                        loader = Docx2txtLoader(file_path)
                    elif filename.lower().endswith('.txt'):
                        loader = TextLoader(file_path)
                    else:
                        loader = UnstructuredFileLoader(file_path)
                    
                    target_document = loader.load()[0]
                    break
            
            if not target_document:
                logger.error(f"Document not found: {document_name}")
                return []
            
            # Get document embedding
            doc_embedding = self.embeddings.embed_query(target_document.page_content)
            
            # Search similar documents
            similar_docs = self.vector_db.similarity_search_by_vector(doc_embedding, k=k+1)  # +1 because the document might find itself
            
            # Filter out the target document itself and format results
            similar_doc_names = []
            for doc in similar_docs:
                similar_filename = doc.metadata.get('source_file')
                if similar_filename and similar_filename != document_name:
                    # Calculate similarity score (placeholder - in a real system you'd calculate this from embeddings)
                    similarity = 0.85 + 0.1 * (1 - (similar_docs.index(doc) / len(similar_docs)))  # Just a demo score
                    similar_doc_names.append((similar_filename, similarity))
            
            return similar_doc_names
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            return []
    
    def extract_contract_entities(self) -> List[Dict[str, Any]]:
        """Extract contract entities like parties, dates, etc. from all documents."""
        documents = self.load_documents()
        contract_entities = []
        
        for doc in documents:
            try:
                # Extract contract name/ID
                contract_id = None
                if "Contract ID:" in doc.page_content:
                    match = re.search(r"Contract ID: (CNT-\d+)", doc.page_content)
                    if match:
                        contract_id = match.group(1)
                
                # Extract parties
                parties = []
                party_pattern = r"(.*?) \(hereinafter referred to as \"(.*?)\"\)"
                party_matches = re.findall(party_pattern, doc.page_content)
                for match in party_matches:
                    parties.append({"name": match[0].strip(), "reference": match[1]})
                
                # Extract dates
                dates = {}
                if "EFFECTIVE DATE:" in doc.page_content:
                    match = re.search(r"EFFECTIVE DATE: (.*?)$", doc.page_content, re.MULTILINE)
                    if match:
                        dates["effective"] = match.group(1).strip()
                
                if "EXPIRATION DATE:" in doc.page_content:
                    match = re.search(r"EXPIRATION DATE: (.*?)$", doc.page_content, re.MULTILINE)
                    if match:
                        dates["expiration"] = match.group(1).strip()
                
                if "RENEWAL DATE:" in doc.page_content:
                    match = re.search(r"RENEWAL DATE: (.*?)$", doc.page_content, re.MULTILINE)
                    if match:
                        dates["renewal"] = match.group(1).strip()
                
                # Extract addresses
                addresses = {}
                for party in parties:
                    address_pattern = f"{re.escape(party['name'])}.*?Address: (.*?)$"
                    match = re.search(address_pattern, doc.page_content, re.MULTILINE)
                    if match:
                        addresses[party["name"]] = match.group(1).strip()
                
                # Skip if we don't have enough information
                if not contract_id and not parties and not dates:
                    continue
                
                contract_entities.append({
                    "source_file": doc.metadata["source_file"],
                    "contract_id": contract_id,
                    "parties": parties,
                    "dates": dates,
                    "addresses": addresses
                })
                
            except Exception as e:
                logger.error(f"Error extracting entities from {doc.metadata.get('source_file')}: {str(e)}")
        
        return contract_entities
    
    def find_conflicts(self) -> List[Dict[str, Any]]:
        """Find conflicts in contract information."""
        entities = self.extract_contract_entities()
        conflicts = []
        
        # Group entities by contract_id or contract name pattern
        grouped_entities = {}
        for entity in entities:
            key = entity.get("contract_id", "")
            if not key:
                # Try to extract contract name from filename
                filename = entity.get("source_file", "")
                if "_Contract" in filename:
                    key = filename.split("_Contract")[0]
            
            if key:
                if key not in grouped_entities:
                    grouped_entities[key] = []
                grouped_entities[key].append(entity)
        
        # Check for conflicts within each group
        for key, group in grouped_entities.items():
            if len(group) <= 1:
                continue
                
            # Check for address conflicts
            address_conflicts = {}
            for entity in group:
                for party_name, address in entity.get("addresses", {}).items():
                    if party_name not in address_conflicts:
                        address_conflicts[party_name] = {}
                    
                    if address not in address_conflicts[party_name]:
                        address_conflicts[party_name][address] = []
                    
                    address_conflicts[party_name][address].append(entity["source_file"])
            
            # Report conflicts where a party has multiple addresses
            for party_name, addresses in address_conflicts.items():
                if len(addresses) > 1:
                    conflicts.append({
                        "type": "address",
                        "party": party_name,
                        "conflict_details": [
                            {"value": addr, "files": files}
                            for addr, files in addresses.items()
                        ]
                    })
            
            # Check for date conflicts (expiration, renewal)
            date_conflicts = {}
            for entity in group:
                for date_type, date_value in entity.get("dates", {}).items():
                    if date_type not in date_conflicts:
                        date_conflicts[date_type] = {}
                    
                    if date_value not in date_conflicts[date_type]:
                        date_conflicts[date_type][date_value] = []
                    
                    date_conflicts[date_type][date_value].append(entity["source_file"])
            
            # Report conflicts in dates
            for date_type, dates in date_conflicts.items():
                if len(dates) > 1:
                    conflicts.append({
                        "type": "date",
                        "date_type": date_type,
                        "conflict_details": [
                            {"value": date, "files": files}
                            for date, files in dates.items()
                        ]
                    })
        
        return conflicts
    
    def find_expiring_contracts(self, days_threshold: int = 30) -> List[Dict[str, Any]]:
        """Find contracts that will expire within the specified days."""
        entities = self.extract_contract_entities()
        expiring_contracts = []
        
        current_date = datetime.datetime.now()
        
        for entity in entities:
            expiration_date_str = entity.get("dates", {}).get("expiration")
            if not expiration_date_str:
                continue
            
            try:
                # Try to parse the date string
                expiration_date = None
                date_formats = [
                    "%B %d, %Y",  # January 01, 2023
                    "%m/%d/%Y",   # 01/01/2023
                    "%m-%d-%Y",   # 01-01-2023
                    "%d %B %Y",   # 01 January 2023
                ]
                
                for date_format in date_formats:
                    try:
                        expiration_date = datetime.datetime.strptime(expiration_date_str, date_format)
                        break
                    except ValueError:
                        continue
                
                if not expiration_date:
                    continue
                
                days_remaining = (expiration_date - current_date).days
                
                if 0 < days_remaining <= days_threshold:
                    expiring_contracts.append({
                        "source_file": entity["source_file"],
                        "expiration_date": expiration_date_str,
                        "days_remaining": days_remaining,
                        "contract_id": entity.get("contract_id", "Unknown"),
                        "parties": [p["name"] for p in entity.get("parties", [])]
                    })
                    
            except Exception as e:
                logger.error(f"Error parsing expiration date {expiration_date_str}: {str(e)}")
        
        return expiring_contracts

if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_and_index()