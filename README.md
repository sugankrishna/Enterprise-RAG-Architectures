# Enterprise RAG Architectures 🚀

This repository contains two parallel Retrieval-Augmented Generation (RAG) architectures designed to explore the trade-offs between strict data sovereignty (Offline) and massive cloud scalability (Online).

## 🔒 Project 1: 100% Air-Gapped Local RAG
Engineered for strict data privacy without internet reliance.

* **The Challenge:** Running an LLM, a vector database, and a Streamlit UI simultaneously without overflowing a 4GB VRAM GPU (NVIDIA RTX 3050).
* **The Architecture:** * Implemented **FastMCP (Model Context Protocol)** to decouple the Streamlit frontend from the FAISS indexing backend.
  * Enforced a strict 800-character vector chunking limit to prevent memory collisions.
  * Deployed Google's **Gemma 2 (2B)** via Ollama for instant, grounded inference directly from VRAM.

## ☁️ Project 2: Cloud-Native Enterprise RAG
Built for scalability, handling massive document contexts with sub-second latency.

* **The Architecture:** * Integrated **Gemini 2.5 Flash** and **Pinecone Vector DB** via LangChain.
  * **Security First:** Designed around enterprise constraints, utilizing Data Loss Prevention (DLP) concepts for PII sanitization and VPC Service Controls for secure cloud perimeters.

---
**Author**: Sugan Krishna G
