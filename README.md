<div align="center">

# 🚀 AirGap-RAG (AMD Match 3 Edition)
### *Production-Ready, Offline & Fail-Closed Agentic RAG Framework*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm--6.2-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://www.amd.com/en/products/software/rocm.html)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-FF6F61?style=for-the-badge&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Docker](https://img.shields.io/badge/Container-Docker--Offline-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Built for the **AMD AI League: Match 3 Challenge**. Designed to run 100% offline with strict latency limits, multi-format file support, and zero hallucinations.*

</div>

---

## 💡 Key Highlights

* 🔒 **Air-Gapped & Fully Offline (`--network none`):** Zero external API dependencies or cloud calls during runtime.
* 🎯 **Cite-or-Refuse Policy:** Strictly avoids hallucinations. If no direct evidence is found in the corpus, it gracefully returns empty answers and citations (`{"answer":"","citations":[]}`).
* ⚡ **High Efficiency Hybrid Search:** Combines **BM25 lexical ranking** with **sparse feature hashing** for ultra-fast indexing (<10 min) and sub-second query retrieval[cite: 5].
* 📄 **Multi-Format Document Parsing:** Ingests PDF, DOCX, CSV, XLSX, Text Logs, and Images (via OCR) with robust error-handling that automatically skips corrupted or encrypted files[cite: 5].
* 🔄 **LangGraph Workflow Integration:** Uses a state-based architecture (Planner ➔ Retriever ➔ Extractive Generator ➔ Guardrail) with an automatic sequential fallback[cite: 5].

---

## 🛠️ Tech Stack & Technologies

| Layer | Technology Used |
| :--- | :--- |
| **Language & Execution** | Python 3.10+, PyTorch (ROCm-compatible)[cite: 5] |
| **Orchestration** | LangGraph, LangChain Core[cite: 5] |
| **Search & Indexing** | Deterministic Sparse Feature Hashing + BM25 Lexical Ranking (`rank-bm25`)[cite: 5] |
| **Parsers & Ingestion** | PyMuPDF (`pypdf`), `python-docx`, `openpyxl`, PyTesseract (OCR)[cite: 5] |
| **Contract Validation** | Pydantic & Standard JSON Formatter[cite: 4, 5] |
| **Containerization** | Docker (`rocm/pytorch:rocm6.2_ubuntu22.04`)[cite: 5, 7] |

---

## 🗂️ Project Architecture & Directory Structure

```text
.
├── src/
│   ├── parser.py       # Multi-format tolerant corpus walker & reader[cite: 5]
│   ├── chunker.py      # Paragraph-aware chunking with relative path preservation[cite: 5]
│   ├── indexer.py      # Dual-engine indexing (Sparse Hashing + BM25)[cite: 5]
│   ├── retriever.py    # Hybrid retrieval engine[cite: 5]
│   ├── guardrail.py    # Fail-closed citation and lexical support validator[cite: 5]
│   └── agent.py        # LangGraph state workflow with fallback execution[cite: 5]
├── app.py              # Main CLI entry point handling --index and --query[cite: 5]
├── Dockerfile          # Offline ROCm-enabled container setup[cite: 5]
├── requirements.txt    # Production dependencies[cite: 5]
└── README.md           # Documentation