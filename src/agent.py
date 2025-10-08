#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI Agent for CLM Daily Reports

This module implements an AI agent that generates daily reports for contract lifecycle management.
"""

import os
import smtplib
import logging
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from src.document_processor import DocumentProcessor
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict, Any

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/agent.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class CLMAgent:
    """AI Agent for generating daily contract reports."""
    
    def __init__(self, recipient_email="rahulraj349434@gmail.com"):
        """Initialize the CLM agent."""
        self.recipient_email = recipient_email
        self.document_processor = DocumentProcessor()
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
    
    def run_daily_report(self) -> None:
        """Generate and send the daily contract report."""
        logger.info("Starting daily report generation")
        
        # Collect data for the report
        expiring_contracts = self.document_processor.find_expiring_contracts(days_threshold=30)
        conflicts = self.document_processor.find_conflicts()
        
        # Generate the report using a state graph
        report = self._generate_report(expiring_contracts, conflicts)
        
        # Send the report via email
        self._send_email_report(report)
        
        logger.info("Daily report completed")
    


    def _generate_report(self, expiring_contracts: List[Dict[str, Any]], conflicts: List[Dict[str, Any]]) -> str:
        """Generate a concise daily report using LangGraph."""
        
        @dataclass
        class AgentState:
            messages: List = field(default_factory=list)
            expiring_contracts: List[Dict[str, Any]] = field(default_factory=list)
            conflicts: List[Dict[str, Any]] = field(default_factory=list)
            report_text: str = ""

        def get_content(resp):
            """Extract text from AI response."""
            return getattr(resp, "content", None) or (resp.get("content") if isinstance(resp, dict) else str(resp))
        
        def analyze_expiring(state):
            if not state.expiring_contracts:
                summary = "No contracts are expiring in the next 30 days."
            else:
                info = "\n".join([f"- {c['source_file']}: Expires on {c['expiration_date']} ({c['days_remaining']} days)"
                                for c in state.expiring_contracts])
                resp = self.llm.invoke(state.messages + [HumanMessage(content=f"Summarize expiring contracts:\n{info}")])
                summary = get_content(resp)
            state.messages.append(AIMessage(content=summary))
            state.report_text += f"\n## EXPIRING CONTRACTS\n{summary}\n"
            return state

        def analyze_conflicts(state):
            if not state.conflicts:
                summary = "No conflicts detected in the contract documents."
            else:
                info = []
                for c in state.conflicts:
                    label = f"{c['party']}" if c["type"] == "address" else f"{c['date_type'].capitalize()} date"
                    for d in c["conflict_details"]:
                        info.append(f"- {label}: {d['value']} in files: {', '.join(d['files'])}")
                resp = self.llm.invoke(state.messages + [HumanMessage(content=f"Explain these conflicts:\n{chr(10).join(info)}")])
                summary = get_content(resp)
            state.messages.append(AIMessage(content=summary))
            state.report_text += f"\n## CONFLICTS DETECTED\n{summary}\n"
            return state

        def generate_summary(state):
            msg = "Provide an executive summary of key findings and recommended actions."
            resp = self.llm.invoke(state.messages + [HumanMessage(content=msg)])
            summary = get_content(resp)
            state.report_text = f"# CONTRACT LIFECYCLE MANAGEMENT DAILY REPORT\n\n" \
                                f"**Date:** {datetime.datetime.now():%B %d, %Y}\n\n" \
                                f"## EXECUTIVE SUMMARY\n{summary}\n" + state.report_text
            return state

        workflow = StateGraph(AgentState)
        workflow.add_node("expiring", analyze_expiring)
        workflow.add_node("conflicts", analyze_conflicts)
        workflow.add_node("summary", generate_summary)
        workflow.add_edge("expiring", "conflicts")
        workflow.add_edge("conflicts", "summary")
        workflow.add_edge("summary", END)
        workflow.set_entry_point("expiring")
        app = workflow.compile()

        result = app.invoke({
            "messages": [HumanMessage(content="Generate a daily CLM report.")],
            "expiring_contracts": expiring_contracts,
            "conflicts": conflicts
        })

        # Unified extraction logic
        if hasattr(result, "report_text"): return result.report_text
        if isinstance(result, dict):
            for k in ("report_text", "report", "output"):
                if k in result: return result[k]
            state = result.get("state") or {}
            if isinstance(state, dict): return state.get("report_text", "")
            if hasattr(state, "report_text"): return state.report_text
        return str(result)


    
    def _send_email_report(self, report_text: str) -> None:
        """Send the report via email."""
        try:
            # For demonstration, we'll just log the email rather than sending it
            logger.info(f"Would send email to: {self.recipient_email}")
            logger.info(f"Email content:\n{report_text}")
            
            # In a real implementation, you'd use something like:
            """
            sender_email = "clm-system@example.com"
            
            msg = MIMEMultipart()
            msg['Subject'] = f"CLM Daily Report - {datetime.datetime.now().strftime('%B %d, %Y')}"
            msg['From'] = sender_email
            msg['To'] = self.recipient_email
            
            # Convert markdown to HTML (simplified)
            html_content = f"<pre>{report_text}</pre>"  # In real system, use a proper markdown to HTML converter
            
            msg.attach(MIMEText(html_content, 'html'))
            msg.attach(MIMEText(report_text, 'plain'))
            
            with smtplib.SMTP('smtp.example.com') as server:
                server.login(sender_email, 'password')
                server.send_message(msg)
            """
        except Exception as e:
            logger.error(f"Error sending email report: {str(e)}")

if __name__ == "__main__":
    agent = CLMAgent(recipient_email="rahulraj349434@gmail.com")
    agent.run_daily_report()