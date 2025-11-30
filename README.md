# CardioRAG: Cardiovascular Pharmacotherapy RAG Assistant  
![Python](https://img.shields.io/badge/python-3.9%2B-blue)  
![License: MIT](https://img.shields.io/badge/license-MIT-green)  
![Status: Production-Ready](https://img.shields.io/badge/status-production--ready-success)

**Ask clinical questions. Get evidence-based answers from cardiovascular pharmacotherapy textbook.**

---

## Overview

**CardioRAG** is a **Retrieval-Augmented Generation (RAG)** system that turns **PDF textbook on cardiovascular pharmacotherapy** into an **interactive AI assistant** for:

- **Diagnosis support**  
- **Treatment recommendations**  
- **Drug selection & dosing**  
- **Guideline lookup with citations**

> No hallucinations. Only answers backed by **source PDF**.

---

## Features

| Feature | Description |
|-------|-----------|
| **PDF Ingestion** | Full-text extraction from large medical PDFs |
| **Smart Chunking** | Recursive + Semantic chunking with overlap |
| **Vector Search** | ChromaDB + Sentence Transformers |
| **Multi-LLM Support** | OpenAI, Groq, Google Gemini |
| **Citation Tracking** | Every answer cites `chunk_id` from source |
| **Interactive CLI & Web UI** | Ask via terminal or Streamlit |
| **Production Ready** | `.env` config, persistent DB, error handling |

---

## Project Structure

```bash
cardio-rag/
│
├── src/
│   ├── vectordb.py          # Vector DB + advanced chunking
│   ├── app.py               # RAG assistant logic
│   └── ingest_pdf.py        # PDF → vector DB pipeline
│
├── ui/
│   ├── ui.py                # Streamlit web interface
│
├── data/
│   └── cardiovascular_pharmacotherapy.pdf   # ← Your PDF
│
├── chroma_db/               # Persistent vector store
├── .env                     # API keys
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourname/cardio-rag.git
cd cardio-rag
pip install -r requirements.txt
```

### 2. Add Your PDF

Place your PDF in the `data/` folder:

```bash
data/cardiovascular_pharmacotherapy.pdf
```

### 3. Set API Keys (`.env`)

```env
OPENAI_API_KEY=sk-...
# OR
GROQ_API_KEY=gq-...
# OR
GOOGLE_API_KEY=...
```

> **Recommended**: Use `gpt-4o` or `llama3-70b` for best clinical accuracy.

---

### 4. Ingest the PDF (One-Time)

```bash
python src/ingest_pdf.py
```

> Creates `chroma_db/` with embedded chunks.

---

### 5. Launch Assistant

#### Option A: Terminal (CLI)

```bash
python src/app.py
```

```
Q: Patient with STEMI, BP 110/70. First-line antithrombotic?
A: [Chunk 145] Aspirin 162–325 mg + P2Y12 inhibitor (prasugrel/ticagrelor preferred)...
```

#### Option B: Web UI (Streamlit)

```bash
streamlit run ui/ui.py
```

→ Open [http://localhost:8501](http://localhost:8501)

---

## Example Queries

```text
- "Best beta-blocker in heart failure with reduced EF?"
- "Contraindications to spironolactone in CKD?"
- "Loading dose of clopidogrel in PCI?"
- "Management of hypertensive urgency?"
```

All answers include **source chunk citation**.

---

## Advanced Chunking (in `vectordb.py`)

| Strategy | Use Case |
|--------|---------|
| `simple` | Fast prototyping |
| `recursive` | **Default** – best balance |
| `semantic` | Long-form narrative, preserves meaning |

Switch in `add_documents()`:

```python
chunks = self.chunk_text(content, strategy="semantic")
```

---

## Configuration

| Variable | Default | Description |
|--------|--------|-----------|
| `CHROMA_COLLECTION_NAME` | `cv_pharmacotherapy` | Vector DB collection |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model |
| `OPENAI_MODEL` | `gpt-4o` | Override LLM |

---

## For Developers

### Re-ingest with new PDF

```bash
rm -rf chroma_db/
python src/ingest_pdf.py
```

### Add More Sources

Update `ingest_pdf.py` to loop over multiple PDFs:

```python
for pdf in Path("data").glob("*.pdf"):
    docs.append({"content": extract(pdf), "metadata": {"source": pdf.name}})
```

---

## License

[MIT License](LICENSE) – Free to use, modify, and distribute.

---

## Disclaimer

> **Not for clinical decision-making without physician oversight.**  
> This tool supports education and research. Always verify with primary sources.

---

## Made in Ethiopia  
**Built with love for better cardiovascular care.**

---
*Powered by RAG, ChromaDB, and your trusted textbook.*