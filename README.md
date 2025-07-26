# Multilingual Bangla-English RAG System

This repository contains a comprehensive implementation of a Retrieval-Augmented Generation (RAG) system tailored for processing scanned Bengali PDFs and answering questions in Bangla and English. The project includes a Jupyter Notebook (`Exeriments_10MinSchool_RAG_System.ipynb`) for experimentation and a modular, industry-standard pipeline in the `rag_multilingual` folder, optimized with the Qwen3-32b model for production use. The system supports OCR, translation, chunking, embedding, vector search, and question answering, with a focus on the Bengali text "অপরিচিতা" (Aparichita).

## Project Overview

The project processes scanned Bengali PDFs (e.g., `HSC26-Bangla1st-Paper.pdf`), extracts text in Bengali and English, translates it to English, builds a multilingual knowledge base, and performs hybrid RAG-based question answering. It uses FAISS for fast vector search and t-SNE for embedding visualization. The notebook is designed for Google Colab, while the `rag_multilingual` folder provides a modular, production-ready implementation.

### Key Features
- **OCR Extraction**: Extracts text from scanned PDFs using Tesseract OCR, supporting Bengali and English.
- **Translation**: Translates Bengali text to English using Deep Translator's GoogleTranslator.
- **Chunking and Embedding**: Splits text into chunks and generates multilingual embeddings using Sentence Transformers.
- **Vector Search**: Uses FAISS for efficient semantic retrieval with optional keyword filtering.
- **RAG with LLMs**: Integrates Groq API to query state-of-the-art LLMs, with Qwen3-32b as the primary model in the production pipeline.
- **Visualization**: Visualizes chunk embeddings using t-SNE to explore semantic structure.
- **Modular Design**: The `rag_multilingual` folder organizes the pipeline into independent modules for scalability and maintainability.

## Repository Structure

### 1. `Exeriments_10MinSchool_RAG_System.ipynb`
A Jupyter Notebook that demonstrates the end-to-end RAG pipeline for experimentation. It includes a notebook file that has already been tested and works in Google Colab. Thus the notebook shows the results of the following steps:
- **Environment Setup**: Installs required packages.
- **OCR and Translation**: Extracts and translates text from a Bengali PDF.
- **Knowledge Base Construction**: Chunks text, generates embeddings, and builds a FAISS index.
- **Question Answering**: Tests four LLMs (Llama-3-70b-versatile, Qwen3-32b, MoonshotAI Kimi-K2-Instruct, Deepseek-R1-Distill-Llama-70b) with sample questions.
- **Visualization**: Displays t-SNE plots of chunk embeddings.

### 2. `rag_multilingual` Folder
A modular, industry-standard implementation of the RAG pipeline, optimized with Qwen3-32b for production use. The folder structure and responsibilities are:

| **Folder/File** | **Responsibility** |
|------------------|--------------------|
| `api/` | Flask REST API server and endpoints for inference and health checks. |
| `config/` | Centralized settings, paths, and constants for environment management. |
| `data/` | Stores original PDF, OCR output, translations, merged text, vectorized chunks, and FAISS index. |
| `ingestion/` | Document ingestion pipeline: `ocr.py` (Bangla OCR), `translation.py` (Bangla to English), `preprocessor.py` (text cleaning/merging). |
| `knowledge_base/` | Knowledge base construction: `chunker.py` (text splitting), `vectorizer.py` (embedding), `vectordb.py` (FAISS index management). |
| `memory/` | Maintains short-term chat context for conversational continuity. |
| `retrieval/` | Hybrid semantic and keyword search over the vector database. |
| `generation/` | LLM answer generation with context preparation and Groq API integration. |
| `evaluation/` | Computes groundedness and relevance scores for answers. |
| `scripts/` | Utility scripts: `prepare_kb.py` (knowledge base preparation), `test_api.py` (API validation). |
| `main.py` | Entrypoint to launch the API server. |

## Pipeline Steps

The following table outlines the steps in both the notebook and `rag_multilingual` pipeline:

| **Step** | **Description** | **Tools Used** | **Output** |
|----------|-----------------|----------------|------------|
| **1. Environment Setup** | Installs required packages for OCR, translation, embeddings, vector search, and LLM integration. | `poppler-utils`, `tesseract-ocr`, `tesseract-ocr-ben`, `pytesseract`, `deep_translator`, `langchain`, `faiss-cpu`, `sentence-transformers`, `groq`, `matplotlib` | Installed dependencies. |
| **2. PDF OCR Extraction** | Extracts text from a scanned Bengali PDF, skipping pages 1-2 and 32-41, using Tesseract. Normalizes whitespace and line breaks. | `pdf2image`, `pytesseract`, `re` | `extracted_text.txt` |
| **3. Translation** | Translates Bengali text to English in chunks (max 500 chars) using GoogleTranslator. | `deep_translator.GoogleTranslator` | `translated_text.txt` |
| **4. Merge Texts** | Combines Bengali and English text into a single knowledge base. | Python file I/O | `merged_text.txt` |
| **5. Chunking** | Splits merged text into overlapping chunks (350 chars, 50-char overlap). | `langchain.text_splitter.RecursiveCharacterTextSplitter` | List of text chunks. |
| **6. Embedding** | Generates multilingual embeddings using `paraphrase-multilingual-mpnet-base-v2`. | `sentence-transformers` | Array of chunk embeddings. |
| **7. Embedding Visualization** | Visualizes first 100 chunk embeddings using t-SNE. | `matplotlib`, `sklearn.manifold.TSNE` | t-SNE scatter plot. |
| **8. Vector Index** | Builds a FAISS index for semantic search. | `faiss` | FAISS index with embedded chunks. |
| **9. Hybrid Search** | Retrieves top-k chunks with semantic and optional keyword matching. | Custom function, `sentence-transformers`, `faiss` | Relevant chunks with scores. |
| **10. RAG with LLMs** | Queries LLMs (Qwen3-32b in production) with retrieved context. | `groq` | Answers in Bangla. |

## Pipeline Flow (Production)
1. **Ingestion**: OCR → Translation → Preprocessing
2. **Knowledge Base**: Chunking → Embedding → FAISS Indexing
3. **Serving & Retrieval**: User Query → Hybrid Retrieval → Context Assembly
4. **Generation**: LLM Prompting (Qwen3-32b) → Answer Generation
5. **Memory**: Maintains recent chat context
6. **Evaluation**: Measures answer groundedness and relevance

## Experiments and Results

The notebook tests four LLMs on questions from "অপরিচিতা". The questions and expected answers are:

| **Question** | **Expected Answer** |
|--------------|---------------------|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? | শুম্ভুনাথ |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামাকে |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | ১৫ বছর |

### Experiment Results

| **Rank** | **Model** | **Performance Summary** | **Sample Answers** |
|----------|-----------|-------------------------|--------------------|
| **1** | Qwen3-32b | Most accurate, correctly identifying মামাকে and Kalyani's age (15), but misidentified শুম্ভুনাথ as শস্তুনাথ. | সুপুরুষ: শস্তুনাথ বাবু (incorrect); ভাগ্য দেবতা: মামাকে; Age: ১৫ বছর |
| **2** | Llama-3-70b-versatile | Correctly identified মামাকে and Kalyani's age (15), but gave a descriptive answer for সুপুরুষ. | সুপুরুষ: চুল পাকা, দাঁড়ি পাকা, সুঠাম পুরুষ (incorrect); ভাগ্য দেবতা: মামাকে; Age: পনেরো (১৫ বছর) |
| **3** | MoonshotAI Kimi-K2-Instruct | Correctly answered Kalyani's age (15), but misidentified সুপুরুষ as শস্তুনাথ সেন and ভাগ্য দেবতা as কল্যাণী. | সুপুরুষ: শস্তুনাথ সেন (incorrect); ভাগ্য দেবতা: কল্যাণী (incorrect); Age: ১৫ বছর |
| **4** | Deepseek-R1-Distill-Llama-70b | Correctly answered Kalyani's age (15), but provided incorrect/incomplete answers for others. | সুপুরুষ: শস্তুনাথ সেন (incorrect); ভাগ্য দেবতা: incomplete; Age: ১৫ বছর |

**Note**: Qwen3-32b was selected for the production pipeline (`rag_multilingual`) due to its superior accuracy and clarity.

## Sample Queries and Outputs

| **Query (Bangla)** | **Query (English)** | **Output (Qwen3-32b)** | **Correct?** |
|--------------------|---------------------|------------------------|--------------|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? | Who is called a nobleman in Anupam's language? | শস্তুনাথ বাবু | No (Expected: শুম্ভুনাথ) |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | Who is referred to as Anupam's fortune deity? | মামাকে | Yes |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | What was Kalyani's actual age at the time of marriage? | ১৫ বছর | Yes |

## Setup Guide

### For Notebook (`Exeriments_10MinSchool_RAG_System.ipynb`)
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/bangla-rag-pipeline.git
   cd bangla-rag-pipeline
   ```
2. **Run in Google Colab**:
   - Upload the notebook to Google Colab.
   - Upload the PDF (e.g., `HSC26-Bangla1st-Paper.pdf`) to `/content/`.
   - Set your Groq API key in the notebook.
3. **Install Dependencies**: Run the notebook's installation cells (`!apt-get` and `!pip install`).
4. **Execute**: Run cells sequentially to process the PDF and query LLMs.

### For Production (`rag_multilingual`)
1. **Clone the Repository**: As above.
2. **Prepare Knowledge Base**:
   - Place the PDF in `rag_multilingual/data/`.
   - Run `python rag_multilingual/scripts/prepare_kb.py`.
3. **Start API**:
   - Run `python rag_multilingual/main.py` to launch the Flask API server.
4. **Test API**:
   - Run `python rag_multilingual/scripts/test_api.py` for validation.
5. **Environment Setup**:
   - Install dependencies: `pip install -r rag_multilingual/requirements.txt`.
   - Configure settings in `rag_multilingual/config/`.

## Used Tools, Libraries, and Packages

| **Tool/Library** | **Purpose** |
|------------------|-------------|
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
| `flask` | API server for production pipeline |

## API Documentation

The `rag_multilingual/api/` folder implements a Flask REST API with the following endpoints:

| **Endpoint** | **Method** | **Description** | **Parameters** |
|--------------|------------|-----------------|----------------|
| `/health` | GET | Checks API server status. | None |
| `/query` | POST | Submits a query and returns an answer. | `query` (str), `keyword` (str, optional) |


## Technical Questions and Answers

1. **What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?**
   - **Method**: Used `pytesseract` with Tesseract OCR (`tesseract-ocr-ben` for Bengali support) and `pdf2image` to convert PDF pages to images.
   - **Why**: Tesseract is robust for multilingual OCR, including Bengali, and `pdf2image` handles scanned PDFs effectively. The combination is open-source and widely supported.
   - **Challenges**: Low-quality scans led to occasional character misrecognition (e.g., similar Bengali glyphs). Whitespace and line breaks were normalized using regex to ensure clean text.

2. **What chunking strategy did you choose? Why do you think it works well for semantic retrieval?**
   - **Strategy**: Character-based chunking with `RecursiveCharacterTextSplitter` (350 chars, 50-char overlap).
   - **Why**: Character-based chunking ensures consistent chunk sizes, preserving context in dense Bengali text. The overlap maintains semantic continuity, improving retrieval for queries spanning chunk boundaries. This balances granularity and context for FAISS search.

3. **What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?**
   - **Model**: `paraphrase-multilingual-mpnet-base-v2` from Sentence Transformers.
   - **Why**: This model supports over 50 languages, including Bengali, and is optimized for semantic similarity. It captures meaning via a transformer-based architecture, encoding text into dense vectors that reflect contextual relationships.
   - **How**: It uses a multilingual MPNet backbone, fine-tuned on paraphrase tasks, to generate embeddings that cluster semantically similar texts in vector space.

4. **How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?**
   - **Comparison**: FAISS `IndexFlatL2` for L2 distance-based similarity between query and chunk embeddings.
   - **Why**: L2 distance is efficient for dense vectors and aligns with the semantic similarity encoded by `paraphrase-multilingual-mpnet-base-v2`. FAISS provides fast, scalable search for large datasets.
   - **Storage**: FAISS stores embeddings as a flat index, suitable for small-to-medium datasets, balancing speed and accuracy.

5. **How do you ensure that the question and document chunks are compared meaningfully? What would happen if the query is vague or missing context?**
   - **Ensuring Meaningful Comparison**: Hybrid search combines semantic (FAISS) and keyword-based retrieval, prioritizing chunks with exact matches when keywords are provided. The multilingual embedding model ensures cross-lingual alignment.
   - **Vague Queries**: Vague queries may retrieve irrelevant chunks due to low semantic similarity. This can be mitigated by prompt engineering (e.g., rephrasing queries) or increasing chunk overlap for broader context.

6. **Do the results seem relevant? If not, what might improve them?**
   - **Relevance**: Qwen3-32b provided the most relevant answers, but errors (e.g., শস্তুনাথ vs. শুম্ভুনাথ) indicate retrieval or context issues. Llama-3-70b-versatile's descriptive answer suggests embedding misalignment.
   - **Improvements**:
     - **Better Chunking**: Increase overlap or use sentence-based chunking for more natural splits.
     - **Better Embedding Model**: Fine-tune a Bengali-specific model (e.g., BanglaBERT) for improved semantic capture.
     - **Larger Document**: Include more context to reduce ambiguity.
     - **Query Refinement**: Add query preprocessing to handle vague inputs.

## Limitations
- **OCR Accuracy**: Tesseract struggles with low-quality scans or complex layouts.
- **Translation Quality**: GoogleTranslator may misinterpret nuanced Bengali text.
- **LLM Accuracy**: Models occasionally misinterpret context (e.g., শস্তুনাথ vs. শুম্ভুনাথ).
- **Compute Resources**: FAISS and embedding generation may be slow for large PDFs on Colab's free tier.

## Contributing
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m 'Add your feature'`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- Built with [LangChain](https://langchain.com), [Sentence Transformers](https://www.sbert.net/), [FAISS](https://github.com/facebookresearch/faiss), [Groq](https://groq.com), and [Flask](https://flask.palletsprojects.com/).
- Inspired by the need for accessible Bengali NLP tools.#   1 0 M i n S c h o o l _ R A G _ S y s t e m . i p y n b  
 