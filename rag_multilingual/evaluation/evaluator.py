from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_groundedness(answer, retrieved_chunks, model):
    ans_emb = model.encode([answer])
    chunk_embs = model.encode([doc for doc, _ in retrieved_chunks])
    sims = cosine_similarity(ans_emb, chunk_embs)[0]
    return float(np.max(sims)), float(np.mean(sims))

def rag_evaluate(query, answer, retrieved, model):
    groundedness, mean_sim = compute_groundedness(answer, retrieved, model)
    best_chunk, best_score = max(retrieved, key=lambda x: x[1])
    relevance = 1.0 / (1.0 + best_score)
    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": [doc for doc, _ in retrieved],
        "groundedness": groundedness,
        "relevance": relevance
    }