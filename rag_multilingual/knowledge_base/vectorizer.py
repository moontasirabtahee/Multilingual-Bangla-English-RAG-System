# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer
import config.settings as settings

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.MODEL_NAME)
    return _model

def embed_chunks(model, chunks):
    return model.encode(chunks, show_progress_bar=True)