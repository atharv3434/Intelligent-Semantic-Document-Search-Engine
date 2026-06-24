# AI-Powered Semantic Search Engine 🧠🔍

An enterprise-ready NLP microservice designed to perform context-aware, semantic document retrieval using deep learning sentences transformer embeddings and dense vector indexing (`FAISS`).

## 🔮 How it works
Traditional search tools rely on raw keyword matches (e.g., matching "WFH" to "WFH"). This system maps both queries and text data into a high-dimensional vector space. It understands context, allowing a query like *"Can I work from my house?"* to successfully retrieve document pages talking about *"Remote work guidelines"* without sharing an exact word match.

## ✨ Technical Highlights
- **Transformer Vectorization**: Leverages the `all-MiniLM-L6-v2` bi-encoder network to derive sentence-level semantics.
- **Sub-millisecond Vector Querying**: Employs `Facebook AI Similarity Search (FAISS)` to perform ultra-fast L2 distance matrix searches.
- **Containerized API**: Encapsulated via Docker and served through an asynchronous `FastAPI` endpoint.

## 🏃 Quick Execution Setup

1. **Build and Run via Docker Engine:**
   ```bash
   docker build -t semantic-search .
   docker run -p 8000:8000 semantic-search