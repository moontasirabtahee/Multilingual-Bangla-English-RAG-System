# Multilingual Bangla-English RAG System

![GitHub last commit](https://img.shields.io/github/last-commit/moontasirabtahee/Multilingual-Bangla-English-RAG-System)

A production-ready Retrieval-Augmented Generation (RAG) system designed for processing scanned Bengali PDFs and answering questions in both Bangla and English. The project features a Jupyter Notebook for rapid experimentation and a modular, industry-standard pipeline in the `rag_multilingual` package — optimized with the **Qwen3-32b** model for deployment. The system supports end-to-end OCR extraction, translation, chunking, multilingual embedding, vector search, and context-aware question answering.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Pipeline Steps](#pipeline-steps)
- [Pipeline Flow (Production)](#pipeline-flow-production)
- [Experiments and Results](#experiments-and-results)
- [Sample Queries and Outputs](#sample-queries-and-outputs)
- [Setup Guide](#setup-guide)
- [Tools and Dependencies](#tools-and-dependencies)
- [API Documentation](#api-documentation)
- [Evaluation Metrics](#evaluation-metrics)
- [Technical Deep Dive](#technical-deep-dive)
- [Limitations](#limitations)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

This project tackles a real-world challenge in Bengali NLP: extracting knowledge from scanned Bengali documents and making it queryable through natural language — in both Bangla and English. The system processes scanned PDFs, extracts text via OCR, translates it, builds a multilingual knowledge base with FAISS vector search, and performs hybrid RAG-based question answering powered by state-of-the-art LLMs.

The repository includes two implementations:
1. **Experimentation Notebook** — A Jupyter Notebook for prototyping, benchmarking multiple LLMs, and visualizing embeddings.
2. **Production Pipeline** (`rag_multilingual`) — A modular, scalable architecture optimized with Qwen3-32b for deployment via a Flask REST API.

## Key Features
- **OCR Extraction** — Extracts text from scanned PDFs using Tesseract OCR with Bengali and English language support.
- **Translation** — Translates Bengali text to English using Deep Translator's GoogleTranslator for cross-lingual retrieval.
- **Chunking & Embedding** — Splits text into semantically coherent chunks and generates multilingual embeddings using Sentence Transformers.
- **Vector Search** — Uses FAISS for efficient semantic retrieval with optional keyword filtering.
- **RAG with LLMs** — Integrates Groq API for fast inference, with Qwen3-32b as the primary production model.
- **Visualization** — Visualizes chunk embeddings using t-SNE to explore semantic structure and clustering.
- **Modular Design** — Organized into independent, testable modules for scalability and maintainability.
- **Evaluation** — Built-in groundedness and relevance scoring for answer quality assessment.

## Architecture

### Experimentation Notebook
A Jupyter Notebook for rapid prototyping and model comparison, designed for Google Colab. It includes:
- Environment setup and dependency installation
- OCR extraction and Bengali-to-English translation
- Knowledge base construction with FAISS indexing
- Benchmarking four LLMs (Llama-3-70b, Qwen3-32b, Kimi-K2-Instruct, Deepseek-R1-Distill-Llama-70b)
- t-SNE visualization of chunk embeddings

### Production Pipeline (`rag_multilingual`)

| **Module** | **Responsibility** |
|---|---|
| `api/` | Flask REST API server for inference and health checks |
| `config/` | Centralized settings, paths, and constants |
| `data/` | Stores PDF, OCR output, translations, merged text, vectorized chunks, and FAISS index |
| `ingestion/` | Document ingestion: `ocr.py` (Bangla OCR), `translation.py` (Bangla→English), `preprocessor.py` (text cleaning/merging) |
| `knowledge_base/` | Knowledge base: `chunker.py` (text splitting), `vectorizer.py` (embedding), `vectordb.py` (FAISS index management) |
| `memory/` | Maintains short-term chat context for conversational continuity |
| `retrieval/` | Hybrid semantic and keyword search over the vector database |
| `generation/` | LLM answer generation with context preparation and Groq API integration |
| `evaluation/` | Computes groundedness and relevance scores for generated answers |
| `scripts/` | Utility scripts: `prepare_kb.py` (knowledge base preparation), `test_api.py` (API validation) |
| `main.py` | Entrypoint to launch the API server |

## Pipeline Steps

| **Step** | **Description** | **Tools Used** | **Output** |
|---|---|---|---|
| **1. Environment Setup** | Installs required packages | `poppler-utils`, `tesseract-ocr`, `tesseract-ocr-ben`, `pytesseract`, `deep_translator`, `langchain`, `faiss-cpu`, `sentence-transformers`, `groq`, `flask` | Installed dependencies |
| **2. PDF OCR Extraction** | Extracts text from a scanned Bengali PDF. Normalizes whitespace and line breaks | `pdf2image`, `pytesseract`, `re` | `extracted_text.txt` |
| **3. Translation** | Translates Bengali text to English in chunks (max 500 chars) | `deep_translator.GoogleTranslator` | `translated_text.txt` |
| **4. Merge Texts** | Combines Bengali and English text into a unified document | Python file I/O | `merged_text.txt` |
| **5. Chunking** | Splits merged text into overlapping chunks (350 chars, 50-char overlap) | `langchain.text_splitter.RecursiveCharacterTextSplitter` | List of text chunks |
| **6. Embedding** | Generates multilingual embeddings using `paraphrase-multilingual-mpnet-base-v2` | `sentence-transformers` | Array of chunk embeddings |
| **7. Visualization** | Visualizes first 100 chunk embeddings using t-SNE | `matplotlib`, `sklearn.manifold.TSNE` | t-SNE scatter plot |
| **8. Vector Index** | Builds a FAISS index for semantic search | `faiss` | FAISS index with embedded chunks |
| **9. Hybrid Search** | Retrieves top-k chunks with semantic and optional keyword matching | `sentence-transformers`, `faiss` | Relevant chunks with scores |
| **10. RAG with LLMs** | Queries LLMs (Qwen3-32b in production) with retrieved context | `groq` | Answers in Bangla |

## Pipeline Flow (Production)
1. **Ingestion** — OCR → Translation → Preprocessing
2. **Knowledge Base** — Chunking → Embedding → FAISS Indexing
3. **Serving & Retrieval** — User Query → Hybrid Retrieval → Context Assembly
4. **Generation** — LLM Prompting (Qwen3-32b) → Answer Generation
5. **Memory** — Maintains recent chat context for multi-turn conversations
6. **Evaluation** — Measures answer groundedness and relevance

## Experiments and Results

The notebook benchmarks four LLMs on Bengali comprehension questions from the text "অপরিচিতা" (Aparichita). The questions and expected answers are:

| **Question** | **Expected Answer** |
|---|---|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? | শস্তুনাথ বাবু |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামাকে |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | ১৫ বছর |

### Model Comparison

| **Rank** | **Model** | **Performance Summary** | **Sample Answers** |
|---|---|---|---|
| **1** | Qwen3-32b | Most accurate — correctly identified all three answers | সুপুরুষ: শস্তুনাথ বাবু ✓ · ভাগ্য দেবতা: মামাকে ✓ · Age: ১৫ বছর ✓ |
| **2** | Llama-3-70b-versatile | Correctly identified 2/3 — gave a descriptive answer for সুপুরুষ | সুপুরুষ: চুল পাকা, দাঁড়ি পাকা, সুঠাম পুরুষ ✗ · ভাগ্য দেবতা: মামাকে ✓ · Age: ১৫ বছর ✓ |
| **3** | Kimi-K2-Instruct | Correctly answered 1/3 — misidentified সুপুরুষ and ভাগ্য দেবতা | সুপুরুষ: শস্তুনাথ সেন ✗ · ভাগ্য দেবতা: কল্যাণী ✗ · Age: ১৫ বছর ✓ |
| **4** | Deepseek-R1-Distill-Llama-70b | Correctly answered 1/3 — incomplete answers for two questions | সুপুরুষ: শস্তুনাথ সেন ✗ · ভাগ্য দেবতা: incomplete ✗ · Age: ১৫ বছর ✓ |

> **Qwen3-32b** was selected for the production pipeline due to its superior accuracy across all test questions.

## Sample Queries and Outputs

| **Query (Bangla)** | **Query (English)** | **Output (Qwen3-32b)** | **Correct?** |
|---|---|---|---|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? | Who is called a nobleman in Anupam's language? | শস্তুনাথ বাবু | ✓ |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | Who is referred to as Anupam's fortune deity? | মামাকে | ✓ |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | What was Kalyani's actual age at the time of marriage? | ১৫ বছর | ✓ |

## Setup Guide

### Prerequisites
- Python 3.8+
- Tesseract OCR with Bengali language pack
- A [Groq API key](https://groq.com) for LLM inference

### Quick Start (Notebook)
1. **Clone the repository:**
   ```bash
   git clone https://github.com/moontasirabtahee/Multilingual-Bangla-English-RAG-System.git
   cd Multilingual-Bangla-English-RAG-System
   ```
2. **Run in Google Colab:**
   - Upload the notebook to Google Colab.
   - Upload your scanned Bengali PDF to `/content/`.
   - Set your Groq API key in the notebook (replace the placeholder).
3. **Install dependencies** (run the notebook's installation cells):
   ```bash
   !apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-ben
   !pip install -q pdf2image pytesseract deep_translator langchain-community langchain faiss-cpu sentence-transformers groq matplotlib
   ```
4. **Execute:** Run cells sequentially to process the PDF and query LLMs.

### Production Deployment (`rag_multilingual`)
1. **Clone the repository** (as above).
2. **Install dependencies:**
   ```bash
   pip install -r rag_multilingual/requirements.txt
   ```
3. **Configure settings** in `rag_multilingual/config/`.
4. **Prepare the knowledge base:**
   - Place your scanned Bengali PDF in `rag_multilingual/data/`.
   - Run:
     ```bash
     python rag_multilingual/scripts/prepare_kb.py
     ```
5. **Start the API server:**
   ```bash
   python rag_multilingual/main.py
   ```
6. **Test the API:**
   ```bash
   python rag_multilingual/scripts/test_api.py
   ```

## Tools and Dependencies

| **Tool / Library** | **Purpose** |
|---|---|
| `poppler-utils` | PDF to image conversion |
| `tesseract-ocr`, `tesseract-ocr-ben` | OCR for Bengali and English |
| `pdf2image` | Convert PDF pages to images |
| `pytesseract` | Python interface for Tesseract |
| `deep_translator` | Bengali to English translation |
| `langchain`, `langchain-community` | Text splitting and RAG pipeline |
| `faiss-cpu` | Vector search index |
| `sentence-transformers` | Multilingual embeddings (`paraphrase-multilingual-mpnet-base-v2`) |
| `groq` | LLM API integration (Qwen3-32b in production) |
| `matplotlib`, `sklearn` | t-SNE visualization |
| `flask` | REST API server for production pipeline |

## API Documentation

The production pipeline exposes a Flask REST API with the following endpoints:

| **Endpoint** | **Method** | **Description** | **Parameters** |
|---|---|---|---|
| `/health` | GET | Returns API server status | None |
| `/query` | POST | Submits a question and returns an answer with context | `query` (str, required), `keyword` (str, optional) |

## Evaluation Metrics

The `evaluation/` module computes two key metrics for generated answers:
- **Groundedness** — Measures how well the answer is supported by the retrieved context chunks.
- **Relevance** — Measures how directly the answer addresses the user's query.

## Technical Deep Dive

1. **Why Tesseract OCR?**
   - Tesseract is robust for multilingual OCR, including Bengali, and `pdf2image` handles scanned PDFs effectively. Both are open-source and widely supported.
   - **Challenge:** Low-quality scans caused character misrecognition (e.g., similar Bengali glyphs). Whitespace and line breaks were normalized using regex.

2. **Chunking Strategy**
   - Character-based chunking with `RecursiveCharacterTextSplitter` (350 chars, 50-char overlap).
   - Ensures consistent chunk sizes for dense Bengali text, with overlap maintaining semantic continuity for queries spanning chunk boundaries.

3. **Why `paraphrase-multilingual-mpnet-base-v2`?**
   - Supports 50+ languages including Bengali, optimized for semantic similarity tasks.
   - Uses a multilingual MPNet backbone fine-tuned on paraphrase tasks, encoding text into dense vectors that cluster semantically similar content.

4. **Similarity Search**
   - FAISS `IndexFlatL2` for L2 distance-based similarity — efficient for dense vectors and aligned with the embedding model's training objective.
   - Flat index suits small-to-medium datasets, balancing speed and accuracy.

5. **Hybrid Retrieval**
   - Combines FAISS semantic retrieval with keyword-based filtering, prioritizing exact matches when keywords are provided.
   - Multilingual embeddings ensure cross-lingual alignment between Bengali queries and English content.

6. **Handling Vague Queries**
   - Vague queries may retrieve less relevant chunks due to low semantic similarity.
   - Mitigated through prompt engineering, increased chunk overlap, and optional keyword filtering.

## Limitations
- **OCR Accuracy** — Tesseract struggles with low-quality scans or complex page layouts.
- **Translation Quality** — GoogleTranslator may misinterpret nuanced or literary Bengali text.
- **LLM Accuracy** — Models may misinterpret context in rare cases, though Qwen3-32b was highly accurate in testing.
- **Compute Resources** — FAISS indexing and embedding generation may be slow on free-tier cloud environments.

## License
This project is open source. See the repository for license details.

## Acknowledgments
Built with [LangChain](https://langchain.com), [Sentence Transformers](https://www.sbert.net/), [FAISS](https://github.com/facebookresearch/faiss), [Groq](https://groq.com), and [Flask](https://flask.palletsprojects.com/).
