#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLM System Main Entry Point

This script serves as the main entry point for the CLM system.
It provides a Streamlit UI with buttons for generating data, 
processing files, and report generation.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import subprocess
import streamlit as st
import base64


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/contracts", exist_ok=True)
    
    logger.info("Environment set up successfully")

def generate_data():
    """Generate synthetic contract data."""
    logger.info("Generating synthetic contract data...")
    
    try:
        from src.data_generator import generate_dataset
        num_documents = generate_dataset()
        
        # Get the list of generated files
        contracts_dir = "data/contracts"
        generated_files = [f for f in os.listdir(contracts_dir) if os.path.isfile(os.path.join(contracts_dir, f))]
        generated_files.sort()  # Sort files alphabetically
        
        logger.info(f"Generated {num_documents} synthetic documents")
        return True, f"Successfully generated {num_documents} synthetic documents", generated_files
    except Exception as e:
        error_msg = f"Error generating data: {str(e)}"
        logger.error(error_msg)
        return False, error_msg, []

def process_documents():
    """Process and index the documents."""
    logger.info("Processing and indexing documents...")
    
    try:
        from src.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        processor.process_and_index()
        
        # Count the number of processed files
        contracts_dir = "data/contracts"
        num_files = len([f for f in os.listdir(contracts_dir) if os.path.isfile(os.path.join(contracts_dir, f))])
        
        success_msg = f"Successfully processed and indexed {num_files} documents"
        logger.info(success_msg)
        return True, success_msg
    except Exception as e:
        error_msg = f"Error processing documents: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def run_daily_report():
    """Run the daily report agent."""
    logger.info("Running daily report agent...")
    
    try:
        from src.agent import CLMAgent
        agent = CLMAgent(recipient_email="rahulraj349434@gmail.com")
        
        # Updated to capture the report text
        report_text = agent._generate_report(
            agent.document_processor.find_expiring_contracts(days_threshold=30),
            agent.document_processor.find_conflicts()
        )
        
        # Save the report to a file
        report_file_path = f"logs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file_path, "w") as f:
            f.write(report_text)
        
        # Also send the email
        agent._send_email_report(report_text)
        
        success_msg = f"Daily report generated successfully and saved to {report_file_path}"
        logger.info(success_msg)
        return True, success_msg, report_text, report_file_path
    except Exception as e:
        error_msg = f"Error generating daily report: {str(e)}"
        logger.error(error_msg)
        return False, error_msg, None, None

def get_download_link(file_path, link_text):
    """Generate a download link for a file."""
    with open(file_path, "r") as f:
        file_content = f.read()
    
    b64 = base64.b64encode(file_content.encode()).decode()
    filename = os.path.basename(file_path)
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'

def run_streamlit_ui():
    """Run the integrated Streamlit UI."""
    logger.info("Starting integrated Streamlit UI...")
    
    # Make sure the src directory is in the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        
    # Import CLMChatbot here to avoid circular imports
    from src.chatbot import CLMChatbot
    
    st.set_page_config(page_title="CLM System", page_icon="📄", layout="wide")
    
    st.title("Contract Lifecycle Management System")
    
    # Sidebar for system operations
    with st.sidebar:
        st.header("System Operations")
        
        # Generate Data button
        if st.button("Generate Synthetic Data"):
            with st.spinner("Generating synthetic data..."):
                success, message, generated_files = generate_data()
                if success:
                    st.success(message)
                    
                    # Display the generated files in an expander
                    with st.expander("Generated Files", expanded=True):
                        if generated_files:
                            st.write("The following files were generated:")
                            
                            # Group files by contract type for better organization
                            contract_types = {}
                            for file in generated_files:
                                # Extract the contract type (first part of the filename before '_')
                                contract_type = file.split('_')[0] if '_' in file else "Other"
                                if contract_type not in contract_types:
                                    contract_types[contract_type] = []
                                contract_types[contract_type].append(file)
                            
                            # Display files organized by contract type
                            for contract_type, files in contract_types.items():
                                st.write(f"**{contract_type}**:")
                                for file in files:
                                    st.write(f"- {file}")
                        else:
                            st.write("No files were generated.")
                else:
                    st.error(message)
        
        # Process Documents button
        if st.button("Process & Index Documents"):
            with st.spinner("Processing documents..."):
                success, message = process_documents()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Generate Report button
        if st.button("Generate Daily Report"):
            with st.spinner("Generating daily report..."):
                success, message, report_text, report_path = run_daily_report()
                if success:
                    st.success(message)
                    
                    # Create download link for the report
                    if report_path and os.path.exists(report_path):
                        download_link = get_download_link(report_path, "Download Report")
                        st.markdown(download_link, unsafe_allow_html=True)
                        
                        # Display report preview
                        with st.expander("Report Preview", expanded=True):
                            st.markdown(report_text)
                else:
                    st.error(message)
        
        # Document similarity section
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
                try:
                    # Import the document processor directly to avoid issues with the chatbot
                    from src.document_processor import DocumentProcessor
                    processor = DocumentProcessor()
                    
                    with st.spinner("Finding similar documents..."):
                        similar_docs = processor.find_similar_documents(selected_doc)
                        if similar_docs:
                            st.write("Similar documents:")
                            for doc_name, similarity in similar_docs:
                                st.write(f"- {doc_name} (Similarity: {similarity:.2f})")
                        else:
                            st.write("No similar documents found.")
                except Exception as e:
                    st.error(f"Error finding similar documents: {str(e)}")
                    logger.error(f"Error in document similarity: {str(e)}")
            else:
                st.error("Please select a document first.")
    
    # Main chat interface
    try:
        from src.chatbot import CLMChatbot
        
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
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        st.info("Please make sure to process documents first.")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='CLM System')
    parser.add_argument('--generate-data', action='store_true', help='Generate synthetic data')
    parser.add_argument('--process', action='store_true', help='Process and index documents')
    parser.add_argument('--report', action='store_true', help='Run daily report')
    parser.add_argument('--no-ui', action='store_true', help='Run without UI')
    parser.add_argument('--all', action='store_true', help='Run all components')
    
    args = parser.parse_args()
    
    setup_environment()
    
    # Command line operations mode
    if args.no_ui or args.generate_data or args.process or args.report or args.all:
        # If no specific argument or --all, run everything
        if args.all:
            generate_data()
            process_documents()
            run_daily_report()
        else:
            if args.generate_data:
                generate_data()
            
            if args.process:
                process_documents()
            
            if args.report:
                run_daily_report()
        
        logger.info("CLM system operations completed")
    # Default UI mode
    else:
        try:
            # If this script is being run directly (not with streamlit run)
            # Re-run it using streamlit
            if not os.environ.get('STREAMLIT_RUNNING'):
                logger.info("Starting Streamlit server...")
                os.environ['STREAMLIT_RUNNING'] = '1'
                subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
            else:
                # If streamlit is already running this script, run the UI
                run_streamlit_ui()
        except Exception as e:
            logger.error(f"Error running Streamlit UI: {str(e)}")
            return False

if __name__ == "__main__":
    main()