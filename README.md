# Ollama VectorChat 🧠💬
An intelligent, document-aware chatbot that blends **FAISS vector search**, **token-based chunking**, and **multi-model LLMs (LLaMA, Mistral, Gemma)** via **Ollama** — all wrapped in a clean, responsive **Streamlit UI**.

## ✨ Features
- **Multi-Model Support**: Choose between LLaMA, Mistral, and Gemma.  
- **Document Ingestion**: Upload PDFs or text files for contextual Q&A.  
- **Token-Based Chunking**: Efficient retrieval using semantic embedding.  
- **FAISS Vector Database**: Fast similarity search for top-k context.  
- **Safe Prompt Mode**: Bypass context on errors or privacy-sensitive queries.  
- **Dynamic Themes**: Switch between light/dark UI with custom chat avatars.  
- **Auto Ollama Handling**: Checks and auto-starts Ollama if not running.  
- **Persistent Chat History**: Review prior Q&A with retrieved contexts.

---

## 🚀 Tech Stack
- **Frontend/UI**: Streamlit (Custom theming and components)  
- **Vector DB**: FAISS (Facebook AI Similarity Search)  
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)  
- **LLM Backend**: Ollama (LLaMA, Mistral, Gemma)  
- **Tokenization**: Hugging Face Transformers (BERT-based)  
- **Document Parsing**: PyMuPDF (PDF extraction)

---

## ⚡ Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/<your-username>/ollama-vectorchat.git
cd ollama-vectorchat

🎯 Usage
Upload PDF/Text documents

Select your preferred LLM model (auto-detect installed models)

Ask contextual questions — top-k results retrieved via FAISS

Toggle Safe Prompt Mode if document context isn’t needed

📌 Why this Project?
This project demonstrates real-world RAG (Retrieval Augmented Generation) architecture with:

Offline-first LLM inference via Ollama

High-performance vector similarity search

Clean UX with theming and custom avatars


