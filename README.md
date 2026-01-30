# 🧬 Medical AI with Knowledge Graphs: Fertility KG-RAG System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.14+-green.svg)](https://neo4j.com)
[![Groq](https://img.shields.io/badge/LLM-Groq_LLaMA--4-purple.svg)](https://groq.com)

> **Advanced Medical AI System** combining Knowledge Graphs with Retrieval-Augmented Generation for clinical fertility reasoning.

## 🎯 Overview

**Nurtura** is a specialized medical AI system that implements **Knowledge Graph–augmented Retrieval-Augmented Generation (KG-RAG)** for fertility and reproductive medicine. The system combines biomedical knowledge graphs with large language models to provide evidence-based clinical insights.

### ✨ Key Innovations

- **🔗 Hybrid KG-RAG Architecture**: Combines structured biomedical knowledge with LLM reasoning
- **🏥 Domain-Specialized**: Focused on fertility and reproductive medicine
- **🧠 Multi-Stage Reasoning**: Implements chain-of-thought with clinical validation
- **📊 Property-Aware Entity Mapping**: Advanced semantic matching with biomedical properties

## 🏗️ Architecture

```mermaid
graph TB
    A[User Query] --> B[Biomedical Entity Recognition]
    B --> C[KG Entity Mapping]
    C --> D[Graph Path Discovery]
    D --> E[Context Retrieval]
    E --> F[LLM Chain-of-Thought]
    F --> G[Clinical Answer Generation]
    G --> H[Evidence Attribution]
    
    subgraph "Knowledge Graph Layer"
        I[Neo4j Database]
        J[Disease Nodes]
        K[Drug Nodes]
        L[Relationship Edges]
    end
    
    subgraph "LLM Layer"
        M[Groq LLaMA-4]
        N[Clinical Reasoning]
        O[Answer Synthesis]
    end
    
    C --> I
    D --> I
    E --> I
    F --> M
