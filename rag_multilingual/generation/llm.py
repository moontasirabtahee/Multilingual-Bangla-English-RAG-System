from groq import Groq
from config.settings import GROQ_API_KEY
from retrieval.retriever import hybrid_search

def generate_answer_qwen3(prompt: str, temperature: float = 1, max_tokens: int = 1024, top_p: float = 1):
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=top_p,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content

def process_query_qwen3(query, index, chunks, model, keyword=None, history_context=""):
    retrieved = hybrid_search(query, index, chunks, model, k=3, keyword=keyword)
    context_lines = [f"Document: {doc} [Score: {score:.4f}]" for doc, score in retrieved]
    context_str = "\n".join(context_lines)
    prompt = (
        f"{history_context}\n\n"
        f"Using the following knowledge base excerpts:\n{context_str}\n\n"
        f"Answer this question in a single line in Bangla: {query}"
    )
    answer = generate_answer_qwen3(prompt)
    return answer, retrieved