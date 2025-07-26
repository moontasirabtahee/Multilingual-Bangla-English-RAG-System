import numpy as np
import faiss
from ..config import settings


def build_faiss_index(chunk_embeddings):
    embedding_dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    embeddings_np = np.array(chunk_embeddings).astype('float32')
    index.add(embeddings_np)
    faiss.write_index(index, settings.EMBEDDINGS_PATH)
    return index


def load_faiss_index():
    return faiss.read_index(settings.EMBEDDINGS_PATH)


def save_chunks(chunks):
    np.save(settings.CHUNKS_PATH, np.array(chunks))


def load_chunks():
    return np.load(settings.CHUNKS_PATH, allow_pickle=True)
# -*- coding: utf-8 -*-

