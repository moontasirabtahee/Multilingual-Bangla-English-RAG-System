# -*- coding: utf-8 -*-
"""
Script to prepare the knowledge base: OCR, Translation, Chunking, Embedding, Indexing.
Run this only once unless you update the PDF.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dotenv import load_dotenv
from ..config import settings
from ..ingestion.ocr import extract_text_from_pdf
from ..ingestion.translation import translate_text
from ..ingestion.preprocessor import merge_bengali_english
from ..knowledge_base.chunker import chunk_text
from ..knowledge_base.vectorizer import get_model, embed_chunks
from ..knowledge_base.vectordb import build_faiss_index, save_chunks

def prepare_knowledge_base():
    # Ensure data directory exists
    os.makedirs(settings.DATA_DIR, exist_ok=True)

    # Validate GROQ_API_KEY
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set")

    # OCR
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(settings.PDF_PATH)
    with open(settings.EXTRACTED_TEXT_PATH, 'w', encoding='utf-8') as f:
        f.write(extracted_text)

    # Translate
    print("Translating text...")
    translated_text = translate_text(extracted_text)
    with open(settings.TRANSLATED_TEXT_PATH, 'w', encoding='utf-8') as f:
        f.write(translated_text)

    # Merge
    print("Merging Bengali and English text...")
    merged_text = merge_bengali_english(extracted_text, translated_text)
    with open(settings.MERGED_TEXT_PATH, 'w', encoding='utf-8') as f:
        f.write(merged_text)

    # Chunking
    print("Chunking text...")
    chunks = chunk_text(merged_text)
    save_chunks(chunks)

    # Embedding
    print("Generating embeddings...")
    model = get_model()
    chunk_embeddings = embed_chunks(model, chunks)

    # FAISS index
    print("Building FAISS index...")
    build_faiss_index(chunk_embeddings)
    print("Knowledge base prepared and indexed successfully.")

if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file
    prepare_knowledge_base()
