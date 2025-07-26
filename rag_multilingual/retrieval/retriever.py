def hybrid_search(query, index, chunks, model, k=3, keyword=None):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    semantic_results = [(chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]
    if keyword:
        keyword_results = [res for res in semantic_results if keyword.lower() in res[0].lower()]
        hybrid_results = keyword_results + [res for res in semantic_results if res not in keyword_results]
        return hybrid_results[:k]
    return semantic_results