# app.py
import streamlit as st
import json
import time
import pandas as pd
import numpy as np
import os
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional

# Try to import Groq early
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq library not available. Please install with: pip install groq")

# Import your modules
from embedding_model import EmbeddingModel
from data_loader import KGDataLoader
from connection import Neo4jConnection
from config import Config
from extraction import EntityExtractor
from similarity import SimilarityCalculator
from path_finder import PathFinder
from cot import ChainOfThoughtGenerator
from pipeline import BiomedicalPipeline
from analysis import PipelineAnalyzer
from demo import BiomedicalPipelineDemo

# Configuration
GROQ_API_KEY = "add_your_api"
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Page configuration
st.set_page_config(
    page_title="Fertility Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Theme
st.markdown("""
<style>
:root {
    --tanit-teal: #1AC3AE;
    --tanit-teal-light: #6AE8D8;
    --tanit-teal-lighter: rgba(26, 195, 174, 0.2);
    --tanit-purple: #6D529F;
    --tanit-purple-light: #9D7FD3;
    --tanit-indigo: #3F4080;
    --bg-main: #0A0D12;
    --bg-card: rgba(22, 27, 34, 0.92);
    --border: rgba(46, 52, 64, 0.6);
    --text-main: #F0F4F8;
    --text-secondary: #B0BEC5;
}

body {
    background: linear-gradient(135deg, #0A0D12 0%, #0E1117 100%);
    color: var(--text-main);
    font-family: 'Inter', sans-serif;
}

.main-header {
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(45deg, #1AC3AE 0%, #00B5B8 25%, #6D529F 75%, #3F4080 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
}

/* System Status Styles */
.system-status-container {
    background: rgba(22, 27, 34, 0.7);
    border-radius: 18px;
    padding: 1.2rem;
    margin-bottom: 1.2rem;
    border: 1px solid var(--border);
    backdrop-filter: blur(20px) saturate(180%);
}

.status-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.8rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
}

.status-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-main);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.status-icon {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    position: relative;
}

.status-icon::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

.status-indicator {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    padding: 0.4rem 0.8rem !important;
    border-radius: 20px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
}

.status-online {
    background: rgba(26, 195, 174, 0.15) !important;
    color: var(--tanit-teal-light) !important;
    border: 1px solid rgba(26, 195, 174, 0.3) !important;
}

.status-warning {
    background: rgba(26, 195, 174, 0.15) !important;
    color: var(--tanit-teal-light) !important;
    border: 1px solid rgba(26, 195, 174, 0.3) !important;
}

.status-error {
    background: rgba(26, 195, 174, 0.1) !important;
    color: rgba(106, 232, 216, 0.6) !important;
    border: 1px solid rgba(26, 195, 174, 0.2) !important;
}

.status-details {
    margin-top: 0.8rem;
    padding-top: 0.8rem;
    border-top: 1px solid rgba(26, 195, 174, 0.3);
}

.status-detail-item {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-bottom: 0.5rem !important;
    font-size: 0.9rem !important;
}

.status-detail-label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

.status-detail-value {
    color: var(--text-main) !important;
    font-weight: 600 !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.3rem !important;
}

.status-value-teal {
    color: var(--tanit-teal-light) !important;
    font-weight: 600 !important;
    padding: 0.2rem 0.5rem !important;
    background: rgba(26, 195, 174, 0.1) !important;
    border-radius: 6px !important;
    border: 1px solid rgba(26, 195, 174, 0.2) !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    font-size: 0.9rem !important;
    white-space: nowrap !important;
}

.performance-indicator {
    display: inline-block !important;
    padding: 0.2rem 0.6rem !important;
    border-radius: 12px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    margin-left: 0.5rem !important;
    white-space: nowrap !important;
}

.performance-excellent {
    background: rgba(26, 195, 174, 0.2) !important;
    color: var(--tanit-teal-light) !important;
    border: 1px solid rgba(26, 195, 174, 0.3) !important;
}

.performance-good {
    background: rgba(26, 195, 174, 0.15) !important;
    color: rgba(106, 232, 216, 0.9) !important;
    border: 1px solid rgba(26, 195, 174, 0.25) !important;
}

.performance-fair {
    background: rgba(26, 195, 174, 0.1) !important;
    color: rgba(106, 232, 216, 0.7) !important;
    border: 1px solid rgba(26, 195, 174, 0.2) !important;
}

/* Chat message styling */
.chat-message {
    display: block;
    margin: 16px 0;
    padding: 16px;
    border-radius: 12px;
    width: 100%;
    box-sizing: border-box;
}

.user-message {
    margin-left: auto;
    background: rgba(33, 150, 243, 0.15);
    border: 1px solid rgba(33, 150, 243, 0.3);
    border-radius: 12px 12px 4px 12px;
    max-width: 80%;
    text-align: left;
}

.bot-message {
    margin-right: auto;
    background: rgba(26, 195, 174, 0.15);
    border: 1px solid rgba(26, 195, 174, 0.3);
    border-radius: 12px 12px 12px 4px;
    max-width: 80%;
    text-align: left;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, var(--tanit-teal) 0%, var(--tanit-purple) 50%, var(--tanit-indigo) 100%) !important;
    color: white !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    padding: 0.85rem 2.2rem !important;
    border: none !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 12px 30px rgba(26, 195, 174, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 20px 40px rgba(26, 195, 174, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# Helper function to generate LLM answer from question + chain-of-thought
def generate_llm_answer(question: str, chain_of_thought: Dict) -> str:
    """Generate a new LLM answer based on question and chain-of-thought."""
    try:
        if not st.session_state.llm_configured:
            return "LLM not configured for answer generation."
        
        client = st.session_state.llm_client
        model = st.session_state.llm_model
        
        # Extract CoT reasoning
        reasoning = ""
        if isinstance(chain_of_thought, dict):
            if "reasoning_steps" in chain_of_thought:
                reasoning = "\n".join([f"{i+1}. {step}" for i, step in enumerate(chain_of_thought["reasoning_steps"])])
            elif "chain_of_thought" in chain_of_thought:
                if isinstance(chain_of_thought["chain_of_thought"], dict):
                    cot = chain_of_thought["chain_of_thought"]
                    if "reasoning_steps" in cot:
                        steps = cot["reasoning_steps"]
                        reasoning = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
        
        # Prepare prompt
        prompt = f"""You are a fertility medicine specialist. Based on the following question and biomedical reasoning, provide a clear, clinical answer .

QUESTION FROM PATIENT/CLINICIAN:
{question}

BIOMEDICAL ANALYSIS AND REASONING:
{reasoning}

INSTRUCTIONS FOR YOUR ANSWER:
1. Directly address the question in the first sentence
2. Summarize the key biomedical findings
3. Provide specific clinical recommendations if applicable
4. Mention any fertility-specific implications
5. Use empathetic, patient-friendly language
6. Highlight evidence-based approaches
7. Keep answer comprehensive but concise (3-5 paragraphs)

YOUR CLINICAL ANSWER:
"""
        
        # Generate response
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": """You are a fertility specialist with expertise in reproductive medicine. 
                    You provide evidence-based, compassionate clinical advice. 
                    You explain complex medical concepts in understandable terms.
                    You always consider fertility implications in your answers.
                    """
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400,
            top_p=0.9
        )
        
        answer = response.choices[0].message.content
        st.session_state.debug_logs.append(f"Generated LLM answer with {len(answer)} characters")
        
        return answer
        
    except Exception as e:
        st.session_state.debug_logs.append(f"Error in generate_llm_answer: {str(e)}")
        return f"⚠️ I encountered an error generating the detailed answer. Here's the analysis summary instead:\n\n"

# Helper function for detailed responses
def _build_detailed_response(question: str, result: Dict) -> str:
    """Build a detailed chatbot response from pipeline results with new LLM generation."""
    response = ""
        
    # Get CoT data
    cot_data = result.get("chain_of_thought", {}).get("chain_of_thought", {})
    
    # Generate new LLM answer based on question and CoT
    if cot_data and st.session_state.llm_configured:
        try:
            # Generate LLM answer
            llm_answer = generate_llm_answer(question, cot_data)
            
            # Display LLM answer prominently
            response += f"{llm_answer}\n\n"
            
            # Add separator
            response += "---\n\n"
                        
        except Exception as e:
            st.session_state.debug_logs.append(f"Error generating LLM answer: {e}")
            # Fall back to original CoT answer
            if cot_data.get('final_answer'):
                response += f" Clinical Insight\n\n"
                response += f"{cot_data['final_answer']}\n\n"
    else:
        # Use original CoT if available
        if cot_data and cot_data.get('final_answer'):
            response += f"  Clinical Insight\n\n"
            response += f"{cot_data['final_answer']}\n\n"
    
    # Quick summary
    summary = result.get("summary", {})
    response += f"**Quick Summary:**\n"
    response += f"- Entities extracted: {summary.get('entities_processed', 0)}\n"
    response += f"- Entities mapped to KG: {summary.get('entities_successfully_mapped', 0)}\n"
    response += f"- Biomedical pathways found: {summary.get('paths_found', 0)}\n\n"
    
    # Show CoT reasoning if available
    if cot_data and cot_data.get('reasoning_steps'):
        response += f"Reasoning Process\n\n"
        for i, step in enumerate(cot_data['reasoning_steps'], 1):
            response += f"{i}. {step}\n"
        response += "\n"
    
    # Confidence score
    confidence = cot_data.get('confidence_score', 0) if cot_data else 0
    if confidence > 0:
        response += f"**Confidence:** {confidence:.2f}/1.0\n\n"
    
    # Key entities
    entity_extraction = result.get("entity_extraction", {})
    extracted_entities = entity_extraction.get("extracted_entities", [])
    if extracted_entities:
        response += f"Key Biomedical Entities\n\n"
        
        # Group by type
        diseases = [e for e in extracted_entities if e.get('type', '').lower() == 'disease']
        drugs = [e for e in extracted_entities if e.get('type', '').lower() == 'drug']
        
        if diseases:
            response += f"**🦠 Diseases identified:**\n"
            for disease in diseases[:3]:
                response += f"- {disease['name']} (confidence: {disease.get('confidence', 0):.2f})\n"
            if len(diseases) > 3:
                response += f"- ... and {len(diseases) - 3} more\n"
            response += "\n"
        
        if drugs:
            response += f"**💊 Drugs identified:**\n"
            for drug in drugs[:3]:
                response += f"- {drug['name']} (confidence: {drug.get('confidence', 0):.2f})\n"
            if len(drugs) > 3:
                response += f"- ... and {len(drugs) - 3} more\n"
            response += "\n"
    
    # Pathway information
    path_discovery = result.get("path_discovery", {})
    selected_path = path_discovery.get("selected_path")
    if selected_path:
        response += f"Key Biomedical Pathway\n\n"
        
        path_desc = selected_path.get('detailed_description', selected_path.get('path_string', ''))
        if len(path_desc) > 150:
            path_desc = path_desc[:150] + "..."
        
        response += f"{path_desc}\n\n"
        
        confidence = path_discovery.get("selected_path_confidence", 0)
        if confidence >= 0.8:
            conf_text = "High clinical relevance"
        elif confidence >= 0.6:
            conf_text = "Moderate clinical relevance"
        else:
            conf_text = "Potential clinical relevance"
        
        response += f"**Pathway relevance:** {conf_text} (score: {confidence:.2f})\n\n"
    
    # Fertility context
    clinical_context = result.get("clinical_context", {})
    fertility_info = clinical_context.get("fertility_relevance", {})
    if fertility_info.get("is_fertility_question", False):
        response += f"## 🎯 Fertility Context\n\n"
        response += f"This analysis is specifically relevant to fertility and reproductive health.\n\n"
    
    return response

# Helper function for displaying entity mapping
def _display_entity_mapping(mapping_result: Dict) -> None:
    """Display entity mapping results in a structured way."""
    if not mapping_result:
        st.info("No mapping results available.")
        return
    
    # Check if we have the correct structure
    if "mapped_node" not in mapping_result:
        st.warning("Mapping result structure is invalid.")
        st.json(mapping_result)
        return
    
    mapped_node = mapping_result.get("mapped_node")
    query_entity = mapping_result.get("query_entity", {})
    
    if not mapped_node:
        st.error(f"No KG node found for: {query_entity.get('name', 'Unknown')}")
        if mapping_result.get("reason"):
            st.caption(f"Reason: {mapping_result.get('reason')}")
        return
    
    # Display mapping info
    col1, col2, col3 = st.columns([3, 2, 2])
    
    with col1:
        st.markdown(f"**Query Entity:** {query_entity.get('name', 'Unknown')}")
        st.markdown(f"**Type:** {query_entity.get('type', 'Unknown')}")
    
    with col2:
        st.markdown(f"**Mapped to KG:** {mapped_node.get('node_name', 'Unknown')}")
        st.markdown(f"**KG Type:** {mapped_node.get('entity_type', 'Unknown')}")
    
    with col3:
        # Get mapping stage and apply styling
        mapping_stage = mapping_result.get("mapping_stage", "unknown")
        stage_classes = {
            "property_exact": ("stage-exact", "🎯 Exact Match"),
            "property_enhanced": ("stage-enhanced", "⚡ Enhanced"),
            "property_llm": ("stage-llm", "🤖 LLM Assisted"),
            "property_fallback": ("stage-fallback", "⚠️ Fallback"),
            "no_candidates": ("stage-fallback", "❌ No Candidates"),
            "error": ("stage-fallback", "❌ Error")
        }
        
        badge_class, stage_text = stage_classes.get(mapping_stage, ("stage-fallback", mapping_stage))
        st.markdown(f"""
        <span class='mapping-stage-badge {badge_class}'>
            {stage_text}
        </span>
        """, unsafe_allow_html=True)
        
        # Enhanced score
        enhanced_score = mapped_node.get("enhanced_score", 0)
        st.metric("Enhanced Score", f"{enhanced_score:.3f}")
    
    # Progress bar for enhanced score
    enhanced_score = mapped_node.get("enhanced_score", 0)
    st.progress(float(enhanced_score))
    
    # Score details
    with st.expander("📊 Score Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Similarity Score:** {mapped_node.get('similarity_score', 0):.4f}")
            st.markdown(f"**Enhanced Score:** {mapped_node.get('enhanced_score', 0):.4f}")
            st.markdown(f"**Match Type:** {mapped_node.get('match_type', 'unknown')}")
        
        with col2:
            st.markdown(f"**Property Match:** {mapped_node.get('property_match_type', 'N/A')}")
            if "llm_property_matches" in mapped_node:
                matches = mapped_node['llm_property_matches']
                if isinstance(matches, list):
                    st.markdown(f"**LLM Properties:** {', '.join(matches)}")
                else:
                    st.markdown(f"**LLM Properties:** {matches}")
    
    # Property matches
    if "property_info" in mapping_result:
        property_info = mapping_result["property_info"]
        with st.expander("🔍 Property Matching Details"):
            st.json(property_info)
    
    # LLM reasoning if available
    if "llm_reason" in mapped_node and mapped_node["llm_reason"]:
        with st.expander("🤖 LLM Reasoning"):
            st.write(mapped_node["llm_reason"])

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables."""
    # System components
    if 'neo4j_conn' not in st.session_state:
        st.session_state.neo4j_conn = Neo4jConnection()
    
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = KGDataLoader()
    
    if 'entity_extractor' not in st.session_state:
        st.session_state.entity_extractor = EntityExtractor()
    
    if 'similarity_calc' not in st.session_state:
        st.session_state.similarity_calc = SimilarityCalculator()
    
    if 'path_finder' not in st.session_state:
        st.session_state.path_finder = PathFinder(st.session_state.neo4j_conn)
    
    if 'cot_generator' not in st.session_state:
        st.session_state.cot_generator = ChainOfThoughtGenerator()
    
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = BiomedicalPipeline(
            entity_extractor=st.session_state.entity_extractor,
            similarity_calculator=st.session_state.similarity_calc,
            path_finder=st.session_state.path_finder,
            cot_generator=st.session_state.cot_generator
        )
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = PipelineAnalyzer()
    
    if 'demo' not in st.session_state:
        st.session_state.demo = BiomedicalPipelineDemo(
            pipeline=st.session_state.pipeline,
            cot_generator=st.session_state.cot_generator,
            analyzer=st.session_state.analyzer
        )
    
    # Chat state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # System state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'embeddings_computed' not in st.session_state:
        st.session_state.embeddings_computed = False
    
    # Debug state
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []
    
    # AUTO-CONFIGURE GROQ LLM
    if 'llm_configured' not in st.session_state:
        try:
            if GROQ_AVAILABLE:
                client = Groq(api_key=GROQ_API_KEY)
                st.session_state.llm_configured = True
                st.session_state.llm_model = LLM_MODEL
                st.session_state.llm_client = client
                st.session_state.llm_status = "online"
                st.session_state.debug_logs.append(f"✅ Groq client initialized for clinical reasoning with model: {LLM_MODEL}")
            else:
                st.session_state.llm_configured = False
                st.session_state.llm_model = None
                st.session_state.llm_client = None
                st.session_state.llm_status = "error"
                st.session_state.debug_logs.append("⚠️ Groq library not available. LLM features will be limited.")
        except Exception as e:
            st.session_state.debug_logs.append(f"❌ Error initializing Groq LLM: {e}")
            st.session_state.llm_configured = False
            st.session_state.llm_model = None
            st.session_state.llm_client = None
            st.session_state.llm_status = "error"
    
    # Current processing state
    if 'current_processing' not in st.session_state:
        st.session_state.current_processing = None
    
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None

# Initialize the app
initialize_session_state()

def add_debug_log(message: str):
    """Add a debug log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.debug_logs.append(f"[{timestamp}] {message}")

def display_debug_info():
    """Display debug information if debug mode is enabled."""
    if st.session_state.debug_mode and st.session_state.debug_logs:
        with st.expander(" Debug Logs", expanded=False):
            for log in st.session_state.debug_logs[-20:]:  # Show last 20 logs
                st.text(log)

# Sidebar
with st.sidebar:
    # Centered logo section
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
    """, unsafe_allow_html=True)
    
    tanit_logo_path = r"C:\Users\hedik\Desktop\KG_tanit\test\image\tanit.png"
    
    try:
        st.image(tanit_logo_path, width=200)
    except Exception as e:
        st.markdown("""
        <h1 style="color: #FF6B6B; margin: 0;">👶</h1>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <p style="color: #666; font-size: 0.9rem; margin: 0;">Fertility Chatbot</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    
    # ================= SYSTEM STATUS INDICATORS =================
    
    # Check system status
    neo4j_status = "offline"
    data_status = "offline"
    embedding_status = "offline"
    llm_status = "offline"
    
    try:
        if st.session_state.neo4j_conn.test_connection():
            neo4j_status = "online"
        else:
            neo4j_status = "error"
    except Exception as e:
        neo4j_status = "error"
    
    # Data status
    if st.session_state.data_loaded:
        nodes = st.session_state.data_loader.get_nodes()
        if nodes and len(nodes) > 0:
            data_status = "online"
        else:
            data_status = "warning"
    else:
        data_status = "offline"
    
    # Embeddings status
    if st.session_state.embeddings_computed:
        embedding_status = "online"
    else:
        embedding_status = data_status  # Follow data status
    
    # LLM status
    if st.session_state.llm_configured:
        llm_status = "online"
    else:
        llm_status = "error"
    
    # Determine overall status
    overall_status = "online"
    if "error" in [neo4j_status, data_status, embedding_status, llm_status]:
        overall_status = "error"
    elif "offline" in [neo4j_status, data_status, embedding_status, llm_status]:
        overall_status = "offline"
    elif "warning" in [neo4j_status, data_status, embedding_status, llm_status]:
        overall_status = "warning"
    
    # Determine performance
    if overall_status == "online":
        performance = "excellent"
    elif overall_status == "warning":
        performance = "good"
    else:
        performance = "fair"
    
    # Map statuses to emojis
    status_emoji_map = {
        "online": "✅",
        "warning": "⚠️",
        "offline": "❌",
        "error": "🚫"
    }
    
    # Get node count
    node_count = len(st.session_state.data_loader.get_nodes()) if st.session_state.data_loaded else 0
    data_text = f"{node_count} nodes" if data_status in ["online", "warning"] else "No data"
    
    # Display system status using direct HTML (FIXED VERSION)
    st.markdown(f"""
    <div class="system-status-container">
        <div class="status-header">
            <div class="status-title">
                <div class="status-icon"></div>
                System Status
            </div>
            <div class="status-indicator status-{overall_status}">
                {overall_status.upper()}
                <span class="performance-indicator performance-{performance}">
                    {performance.upper()}
                </span>
            </div>
        </div>
        <div class="status-details">
            <div class="status-detail-item">
                <span class="status-detail-label">Neo4j:</span>
                <span class="status-detail-value">
                    <span class="status-value-teal">
                        {status_emoji_map.get(neo4j_status, "❓")} {'Connected' if neo4j_status == 'online' else 'Disconnected'}
                    </span>
                </span>
            </div>
            <div class="status-detail-item">
                <span class="status-detail-label">KG Data:</span>
                <span class="status-detail-value">
                    <span class="status-value-teal">
                        {status_emoji_map.get(data_status, "📊")} {data_text}
                    </span>
                </span>
            </div>
            <div class="status-detail-item">
                <span class="status-detail-label">Embeddings:</span>
                <span class="status-detail-value">
                    <span class="status-value-teal">
                        {status_emoji_map.get(embedding_status, "❌")} {'Ready' if embedding_status == 'online' else 'Not computed'}
                    </span>
                </span>
            </div>
            <div class="status-detail-item">
                <span class="status-detail-label">LLM:</span>
                <span class="status-detail-value">
                    <span class="status-value-teal">
                        {status_emoji_map.get(llm_status, "❌")} {'Configured' if llm_status == 'online' else 'Error'}
                    </span>
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display LLM status separately
    model_name = st.session_state.get('llm_model', 'Groq').split('/')[-1]
    st.markdown(f"""
    <div class="system-status-container">
        <div class="status-header">
            <div class="status-title">
                <div class="status-icon"></div>
                LLM Engine
            </div>
            <div class="status-indicator status-{llm_status}">
                {llm_status.upper()}
            </div>
        </div>
        <div class="status-details">
            <div class="status-detail-item">
                <span class="status-detail-label">Model:</span>
                <span class="status-detail-value">
                    <span class="status-value-teal">{model_name}</span>
                </span>
            </div>
            <div class="status-detail-item">
                <span class="status-detail-label">Status:</span>
                <span class="status-detail-value">
                    <span class="status-value-teal">
                        {status_emoji_map.get(llm_status, "❓")} {'Ready' if llm_status == 'online' else 'Error'}
                    </span>
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ================= QUICK ACTIONS =================
    st.divider()
    st.markdown("### Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh"):
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear Chat", use_container_width=True, key="clear_chat"):
            st.session_state.messages = []
            st.session_state.current_results = None
            st.rerun()
    
    if st.button(" Load KG Data", use_container_width=True, type="primary", key="load_kg_data"):
        with st.spinner("Loading knowledge graph data..."):
            try:
                success, messages = st.session_state.data_loader.load_data(compute_embeddings=True)
                if success:
                    st.session_state.data_loaded = True
                    st.session_state.embeddings_computed = True
                    
                    # Configure pipeline with loaded data
                    nodes = st.session_state.data_loader.get_nodes()
                    embeddings = st.session_state.data_loader.kg_embeddings
                    
                    add_debug_log(f"Loaded {len(nodes)} nodes with embeddings shape: {embeddings.shape if embeddings is not None else 'None'}")
                    
                    st.session_state.pipeline.set_kg_data(nodes, embeddings)
                    
                    st.success("✅ Data loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load data")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                add_debug_log(f"Error loading KG data: {traceback.format_exc()}")
    
    # Debug toggle
    st.divider()
    st.markdown("### 🔧 Debug Options")
    
    debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode, key="debug_toggle")
    if debug_mode != st.session_state.debug_mode:
        st.session_state.debug_mode = debug_mode
        st.rerun()
    
    if st.button("🪲 Clear Debug Logs", use_container_width=True, key="clear_debug"):
        st.session_state.debug_logs = []
        st.rerun()
    
    st.divider()
    
    # ================= ABOUT SECTION =================
    st.markdown("""
    ### ℹ️ About
    This chatbot uses:
    - **Neo4j KG** with fertility data
    - **Biomedical Entity Recognition**
    - **Enhanced Similarity Matching**
    - **Path Discovery** between entities
    - **Chain-of-Thought Reasoning**
    
    *Specialized for fertility and reproductive medicine*
    """)

# Main Chat Interface
st.markdown("""
<div class="header-container">
    <h1 class="main-header">EPTanit</h1>
</div>
""", unsafe_allow_html=True)

# Display debug info if enabled
display_debug_info()

# Chat container
chat_container = st.container()

with chat_container:
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Fertility Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        elif message["role"] == "system":
            st.markdown(f"""
            <div class="chat-message system-message">
                <strong>⚙️ System:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)

# Input area
st.divider()

# Quick question buttons
st.markdown("### 💭 Quick Questions")
col1, col2, col3 = st.columns(3)

quick_questions = [
    "Is Clomiphene citrate effective for PCOS?",
    "How does Metformin help with infertility?",
    "What is endometriosis treatment?"
]

with col1:
    if st.button(quick_questions[0], use_container_width=True, key="q1"):
        st.session_state.user_input = quick_questions[0]

with col2:
    if st.button(quick_questions[1], use_container_width=True, key="q2"):
        st.session_state.user_input = quick_questions[1]

with col3:
    if st.button(quick_questions[2], use_container_width=True, key="q3"):
        st.session_state.user_input = quick_questions[2]

# User input
user_input = st.text_area(
    "💬 Ask about fertility treatments, diseases, or relationships:",
    height=100,
    placeholder="Example: What is the relationship between Polycystic ovary syndrome and Clomiphene citrate?",
    key="user_input"
)

col1, col2 = st.columns([4, 1])
with col1:
    analysis_level = "Full Pipeline"

with col2:
    send_button = st.button("Send", use_container_width=True, type="primary", key="send_button")

# Process user input
if send_button and user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Check if system is ready
    if not st.session_state.data_loaded:
        st.session_state.messages.append({
            "role": "system",
            "content": "❌ Please load knowledge graph data first using the 'Load KG Data' button in the sidebar."
        })
        st.rerun()
    
    # Process the question
    with st.spinner("🔍 Analyzing your question..."):
        try:
            add_debug_log(f"Starting analysis for question: {user_input}")
            add_debug_log(f"Analysis level: {analysis_level}")
            
            # Check if pipeline has KG data
            if not hasattr(st.session_state.pipeline, 'kg_nodes') or st.session_state.pipeline.kg_nodes is None:
                st.session_state.messages.append({
                    "role": "system",
                    "content": "❌ Pipeline not initialized with KG data. Please reload KG data."
                })
                st.rerun()
            
            # Run appropriate analysis based on level
            if analysis_level == "Basic":
                # Basic entity extraction and matching
                add_debug_log("Running basic entity extraction...")
                result = st.session_state.pipeline._extract_entities(user_input)
                
                response = f"I found {len(result)} biomedical entities in your question:\n\n"
                for entity in result:
                    entity_type = entity.get('type', 'Unknown')
                    badge_class = "disease-badge" if entity_type.lower() == "disease" else "drug-badge"
                    response += f"<span class='entity-badge {badge_class}'>{entity['name']} ({entity_type})</span> "
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            elif analysis_level == "Enhanced":
                # Enhanced matching with simple answer
                add_debug_log("Running enhanced pipeline with CoT...")
                try:
                    result = st.session_state.pipeline.extract_and_map_entities_with_cot(user_input)
                    add_debug_log(f"Pipeline completed. Result keys: {list(result.keys()) if result else 'No result'}")
                    
                    st.session_state.current_results = result
                    
                    # Extract key information
                    summary = result.get("summary", {})
                    cot = result.get("chain_of_thought", {}).get("chain_of_thought", {})
                    
                    response = f"## 🔬 Analysis Complete\n\n"
                    
                    # Entity summary
                    response += f"**📊 Entities Found:** {summary.get('entities_processed', 0)} extracted, {summary.get('entities_successfully_mapped', 0)} mapped to knowledge graph\n\n"
                    
                    # Paths found
                    response += f"**🛤️ Pathways Discovered:** {summary.get('paths_found', 0)}\n\n"
                    
                    # Generate LLM answer if available
                    if cot and st.session_state.llm_configured:
                        try:
                            llm_answer = generate_llm_answer(user_input, cot)
                            response += f"## 🎯 Clinical Answer\n\n{llm_answer}\n\n"
                        except Exception as e:
                            add_debug_log(f"Error generating LLM answer: {e}")
                            if cot.get('final_answer'):
                                response += f"## 🎯 Clinical Insight\n\n{cot['final_answer']}\n\n"
                    elif cot and cot.get('final_answer'):
                        response += f"## 🎯 Clinical Insight\n\n{cot['final_answer']}\n\n"
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                except Exception as pipeline_error:
                    add_debug_log(f"Pipeline error: {str(pipeline_error)}")
                    add_debug_log(f"Traceback: {traceback.format_exc()}")
                    
                    # Try simplified version
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"⚠️ Enhanced analysis failed. Trying basic analysis..."
                    })
                    
                    # Fall back to basic extraction
                    result = st.session_state.pipeline._extract_entities(user_input)
                    
                    response = f"I found {len(result)} biomedical entities:\n\n"
                    for entity in result:
                        entity_type = entity.get('type', 'Unknown')
                        badge_class = "disease-badge" if entity_type.lower() == "disease" else "drug-badge"
                        response += f"<span class='entity-badge {badge_class}'>{entity['name']} ({entity_type})</span> "
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                
            else:  # Full Pipeline
                # Complete pipeline with detailed results
                add_debug_log("Running full pipeline analysis...")
                try:
                    result = st.session_state.pipeline.extract_and_map_entities_with_cot(user_input)
                    st.session_state.current_results = result
                    
                    # Build comprehensive response with NEW LLM generation
                    response = _build_detailed_response(user_input, result)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                except Exception as full_pipeline_error:
                    add_debug_log(f"Full pipeline error: {str(full_pipeline_error)}")
                    add_debug_log(f"Traceback: {traceback.format_exc()}")
                    
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"❌ Full pipeline analysis failed. Error: {str(full_pipeline_error)}"
                    })
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": user_input,
                "analysis_level": analysis_level,
                "results": result if 'result' in locals() else None
            })
            
        except Exception as e:
            error_msg = f"❌ Error processing question: {str(e)}"
            add_debug_log(error_msg)
            add_debug_log(f"Full traceback: {traceback.format_exc()}")
            
            st.session_state.messages.append({
                "role": "system",
                "content": error_msg
            })
    
    st.rerun()

# Demo Section
st.divider()
st.markdown('<h2 class="sub-header">⚙️ System Information</h2>', unsafe_allow_html=True)

if st.session_state.data_loaded:
    nodes = st.session_state.data_loader.get_nodes()
    if nodes:
        # Statistics
        disease_count = sum(1 for node in nodes if node.get('entity_type') == 'disease')
        drug_count = sum(1 for node in nodes if node.get('entity_type') == 'drug')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Nodes", len(nodes))
        with col2:
            st.metric("Disease Nodes", disease_count)
        with col3:
            st.metric("Drug Nodes", drug_count)
        with col4:
            if st.session_state.data_loader.kg_embeddings is not None:
                try:
                    dim = st.session_state.data_loader.kg_embeddings.shape[1]
                    st.metric("Embedding Dim", dim)
                except:
                    st.metric("Embeddings", "Error")
            else:
                st.metric("Embeddings", "Not computed")
        
        # Sample nodes
        with st.expander("🔍 View Sample KG Nodes"):
            sample_nodes = nodes[:5]
            for node in sample_nodes:
                with st.expander(f"{node['node_name']} ({node['entity_type']})"):
                    st.write(f"**Labels:** {', '.join(node.get('labels', []))}")
                    st.write(f"**Properties:** {len(node.get('properties', {}))}")
                    
                    # Show key properties
                    key_props = ["node_name", "description", "indication", "mondo_definitions"]
                    for prop in key_props:
                        if prop in node.get('properties', {}):
                            value = node['properties'][prop]
                            if len(str(value)) > 100:
                                value = str(value)[:100] + "..."
                            st.write(f"**{prop}:** {value}")
else:
    st.info("Load knowledge graph data to see system information.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
    <i>Fertility Knowledge Graph Chatbot </i><br>
    <i>Powered by Neo4j</i><br>
    <i>For educational and research purposes in fertility medicine</i>
</div>
""", unsafe_allow_html=True)