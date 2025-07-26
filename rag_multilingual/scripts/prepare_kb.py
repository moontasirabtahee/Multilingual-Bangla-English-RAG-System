# -*- coding: utf-8 -*-

"""
Script to prepare the knowledge base: OCR, Translation, Chunking, Embedding, Indexing.
Run this only once unless you update the PDF.
"""

from ..config import settings
from ..ingestion.ocr import extract_text_from_pdf
from ..ingestion.translation import translate_text
from ..ingestion.preprocessor import merge_bengali_english
from ..knowledge_base.chunker import chunk_text
from ..knowledge_base.vectorizer import get_model, embed_chunks
from ..knowledge_base.vectordb import build_faiss_index, save_chunks

def prepare_knowledge_base():
    # OCR
    extracted_text = extract_text_from_pdf(settings.PDF_PATH)
    with open(settings.EXTRACTED_TEXT_PATH, 'w', encoding='utf-8') as f:
        f.write(extracted_text)

    # Translate
    translated_text = translate_text(extracted_text)
    with open(settings.TRANSLATED_TEXT_PATH, 'w', encoding='utf-8') as f:
        f.write(translated_text)

    # Merge
    merged_text = merge_bengali_english(extracted_text, translated_text)
    with open(settings.MERGED_TEXT_PATH, 'w', encoding='utf-8') as f:
        f.write(merged_text)

    # Chunking
    chunks = chunk_text(merged_text)
    save_chunks(chunks)

    # Embedding
    model = get_model()
    chunk_embeddings = embed_chunks(model, chunks)

    # FAISS index
    build_faiss_index(chunk_embeddings)
    print("Knowledge base prepared and indexed.")

if __name__ == "__main__":
    prepare_knowledge_base()