# Contract Lifecycle Management (CLM) Automation System

This project implements an intelligent platform for Contract Lifecycle Management (CLM) that can ingest, process, analyze, and provide insights about contract documents.

## Project Overview

The CLM Automation system offers the following features:

1. **Document Ingestion & Indexing**:
   - Automatically loads documents from a designated folder
   - Uses FAISS for efficient vector storage and similarity search
   - Implements chunking for better context retrieval

2. **RAG Pipeline**:
   - Built with LangChain to retrieve relevant document chunks
   - Uses Google's Gemini models for embeddings and LLM

3. **Daily Report Generation**:
   - AI agent that identifies approaching contract expirations
   - Detects conflicts in contract information
   - Generates and sends email reports

4. **Chatbot Interface**:
   - Simple Streamlit interface for querying contract information
   - Provides AI answers with source citations

5. **Document Similarity**:
   - Finds semantically similar documents for version comparison

## Technology Stack

- **LLM**: Google's Gemini 2.5 Flash model
- **Embeddings**: Google's Gemini Embedding model
- **Vector Database**: FAISS
- **Framework**: LangChain, LangGraph
- **Interface**: Streamlit
- **Document Processing**: PyPDF2, python-docx, pytesseract (for OCR)

### Why FAISS?

FAISS (Facebook AI Similarity Search) was chosen for the vector database because:
- It's optimized for similarity search with dense vectors
- Provides excellent performance for the scale of this application
- Offers a straightforward API that integrates well with LangChain
- Can be easily deployed locally without requiring a separate service

## Project Structure

```
clm2/
├── data/
│   └── contracts/       # Contract documents
├── logs/                # Log files
├── src/
│   ├── data_generator.py      # Generates synthetic contract data
│   ├── document_processor.py  # Processes and indexes documents
│   ├── agent.py              # AI agent for daily reports
│   └── chatbot.py            # Streamlit chatbot interface
├── main.py              # Main entry point
└── requirements.txt     # Python dependencies
```

## Setup and Usage

### Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up Google AI API access (required for Gemini models)
create a .env file and setup the gemini api key


### Running the System

The system can be run using the main.py script with various options:

```bash
python main.py
```

### Synthetic Dataset

The system generates a diverse set of synthetic contract documents:
- PDFs (standard and scanned)
- Word documents
- Text files
- Unstructured text (meeting notes, emails)

These documents include:
- Different contract versions
- Deliberately conflicting information
- Various date types (creation, expiration, renewal)
- Optional metadata

## Features in Detail

### Document Processing

Documents are processed through these steps:
1. Loading from the contracts folder
2. Text extraction (including OCR for scanned PDFs)
3. Metadata extraction and augmentation
4. Chunking for context preservation
5. Embedding and indexing in FAISS

### RAG Implementation

The RAG (Retrieval-Augmented Generation) pipeline:
1. Embeds the user query
2. Retrieves relevant document chunks from FAISS
3. Formats the context with document citations
4. Generates an AI response using the Gemini model

### Daily Reports

The AI agent:
1. Identifies contracts expiring in the next 30 days
2. Detects conflicts in contract information
3. Uses LangGraph for structured workflow
4. Generates an executive summary
5. Formats and sends an email report

### Document Similarity

The system finds similar documents by:
1. Embedding the target document
2. Performing similarity search in the vector space
3. Ranking and returning similar documents with scores