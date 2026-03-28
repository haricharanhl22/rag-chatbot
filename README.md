# RAG Document Q&A Chatbot

> Ask questions over any PDF using local LLMs — runs 100% free, no API costs

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask questions about it. Built with LangChain, FAISS vector database, and Ollama for completely local, privacy-first AI inference.

## How It Works
```
PDF → Chunks → Embeddings → FAISS Index (offline, once)
                                    ↓
Question → Embed → Similarity Search → Top 4 Chunks → LLM → Answer
```

## Features

- **100% local** — runs on your machine, no data sent to external APIs
- **Any PDF** — lecture notes, research papers, documentation
- **Source pages** — shows which pages were used to answer
- **Fast retrieval** — FAISS vector search across thousands of chunks
- **Web UI** — clean Streamlit interface

## Tech Stack

- **LangChain** — RAG pipeline and LCEL chain
- **FAISS** — vector database for semantic search
- **Ollama** — local LLM inference (llama3.2)
- **nomic-embed-text** — local embedding model
- **Streamlit** — web interface
- **Python** — core language

## Setup
```bash
git clone https://github.com/haricharanhl22/rag-chatbot
cd rag-chatbot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Install Ollama from [ollama.com](https://ollama.com) then:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Usage
```bash
# Index your PDF (run once)
python ingest.py your_document.pdf

# Launch the chatbot
streamlit run app.py
```

## Author

**Hari Charan Hosakote Lokesh**
- GitHub: [@haricharanhl22](https://github.com/haricharanhl22)
- LinkedIn: [haricharanhl22](https://linkedin.com/in/haricharanhl22)
