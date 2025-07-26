import os

DATA_DIR = os.environ.get("RAG_DATA_DIR", "data")
PDF_PATH = os.path.join(DATA_DIR, "HSC26-Bangla1st-Paper.pdf")
EXTRACTED_TEXT_PATH = os.path.join(DATA_DIR, "extracted_text.txt")
TRANSLATED_TEXT_PATH = os.path.join(DATA_DIR, "translated_text.txt")
MERGED_TEXT_PATH = os.path.join(DATA_DIR, "merged_text.txt")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.npy")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

SHORT_TERM_MEMORY_SIZE = 5