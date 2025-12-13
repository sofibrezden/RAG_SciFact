# 🔍 RAG system

A comprehensive Retrieval-Augmented Generation (RAG) system with multiple retrieval strategies, reranking capabilities, and an interactive Streamlit interface.

## 🗒️ Dataset

This project uses the **`beir/scifact`** dataset from the **BEIR benchmark**, which is designed for scientific fact verification. 

It contains **biomedical research paper abstracts** that serve as evidence for supporting or refuting scientific claims. 
The dataset is commonly used to evaluate information retrieval and evidence-based question answering systems. 

In this project, **SciFact** is used purely as a document corpus for retrieval-augmented generation (RAG), without relying on its original claim–evidence annotations. Its high-quality, domain-specific content makes it well suited for experimenting with lexical, semantic, and hybrid retrieval methods.

## 🌟 Features

- **Multiple Retrieval Methods**: BM25, Semantic Search, and Hybrid approaches
- **Advanced Reranking**: Cross-encoder reranking for improved relevance
- **LLM Integration**: OpenAI GPT integration for answer generation
- **Conversation History**: Support for follow-up questions and context
- **Interactive UI**: Clean Streamlit interface with configurable parameters
- **Semantic Chunking**: Intelligent document chunking using semantic boundaries

## 🏗️ Architecture

### Core Components

#### 1. Retrieval System (`src/rag/retrievers/`)
- **BM25Retriever**: Traditional keyword-based retrieval using BM25 algorithm
- **SemanticRetriever**: Dense vector retrieval using sentence transformers
- **HybridRetriever**: Combines BM25 and semantic search for optimal results
- **BaseRetriever**: Abstract base class defining the retriever interface

#### 2. Document Processing (`src/rag/`)
- **Chunking**: Semantic chunking using LangChain's SemanticChunker
- **Reranking**: Cross-encoder reranking for relevance scoring

#### 3. LLM Integration (`src/llm/`)
- **BaseLLM**: Abstract interface for LLM providers
- **OpenAILLM**: OpenAI API integration with conversation support
- **Prompt Management**: Configurable system and user prompts

#### 4. Web Interface (`app.py`)
- Streamlit-based interactive interface
- Real-time configuration of retrieval parameters
- Conversation history management
- Document source display

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- OpenAI API key (for LLM features)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd rag
```

2. **Install dependencies**:
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## 🎯 Usage

### Basic Search

1. **Configure Settings**: Use the sidebar to:
   - Enter your OpenAI API key
   - Select retrieval method (BM25, Semantic, or Hybrid)
   - Enable/disable reranking
   - Enable/disable LLM answer generation
   - Adjust top-k results and document count

2. **Ask Questions**: Enter your question in the search box and click "🔍 Search"

3. **View Results**: 
   - Generated answers (if LLM is enabled)
   - Retrieved document chunks with source information
   - Conversation history for follow-up questions

### Retrieval Methods

#### BM25 Retriever
- **Best for**: Keyword-based queries, exact term matching
- **Algorithm**: BM25 (Best Matching 25) with Okapi normalization
- **Use case**: "Find documents about machine learning algorithms"

#### Semantic Retriever
- **Best for**: Conceptual queries, semantic similarity
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Use case**: "What are the benefits of neural networks?"

#### Hybrid Retriever
- **Best for**: Balanced approach combining keyword and semantic search
- **Strategy**: Combines results from both BM25 and semantic retrievers
- **Use case**: Most general queries for comprehensive coverage

### Advanced Features

#### Reranking
- Uses cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Reorders retrieved documents by relevance to the query
- Improves precision of top results

#### Conversation Support
- Maintains conversation history for follow-up questions
- Context-aware responses using previous exchanges
- Clear conversation history option

## 🔧 Configuration

### Model Configuration

The system uses several pre-trained models:

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**: OpenAI GPT-4.1 (configurable)

### Chunking Strategy

- **Method**: Semantic chunking using sentence transformers
- **Threshold**: 95th percentile for breakpoint detection
- **Benefits**: Maintains semantic coherence within chunks

## 📁 Project Structure

```
rag/
├── app.py                          # Main Streamlit application
├── pyproject.toml                  # Project configuration and dependencies
├── src/
│   ├── llm/                        # LLM integration
│   │   ├── base.py                 # Abstract LLM interface
│   │   └── openai.py               # OpenAI implementation
│   ├── rag/                        # RAG system components
│   │   ├── chunking.py             # Document chunking logic
│   │   ├── reranker.py             # Cross-encoder reranking
│   │   ├── prompts/                # LLM prompts
│   │   │   ├── system_prompt.txt   # System instructions
│   │   │   └── user_prompt.txt     # User query template
│   │   └── retrievers/             # Retrieval implementations
│   │       ├── base.py             # Abstract retriever interface
│   │       ├── bm25.py             # BM25 retriever
│   │       ├── semantic.py         # Semantic retriever
│   │       └── hybrid.py           # Hybrid retriever
│   └── utils.py                    # Utility functions
└── README.md                       # This file
```

### Example Questions

This system is designed for **knowledge-seeking question answering** over a scientific document corpus.
It works best with **general, factual questions** that ask for definitions, mechanisms, or biological relevance,
rather than questions about a specific paper or study.

**Examples of suitable questions:**
- What is gene expression?
- Why is DNA methylation important?
- How does methylation relate to disease?
- Why is bisulfite sequencing used?
- What are peripheral blood mononuclear cells?

These questions are intentionally **cross-document** and **topic-oriented**, allowing the system to retrieve
and synthesize evidence from multiple sources. The goal is to provide concise, evidence-grounded answers
based on retrieved scientific documents.


## 🌐 Live Demo

You can try the app here:  
👉 [Demo](link)

![Demo GIF](demo/demo.gif)