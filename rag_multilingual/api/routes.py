from flask import Blueprint, request ,jsonify
from ..config import settings
from ..knowledge_base.vectorizer import get_model
from ..knowledge_base.vectordb import load_faiss_index, load_chunks
from ..generation.llm import process_query_qwen3
from ..memory.memory import short_term_memory
from ..evaluation.evaluator import rag_evaluate

rag_blueprint = Blueprint("rag", __name__)

@rag_blueprint.route('/rag/query', methods=['POST'])
def rag_query():
    data = request.json
    user_query = data.get('query', '')
    user_id = data.get('user_id', 'anonymous')
    keyword = data.get('keyword', None)

    # Load KB (persistent in prod, lazy load here for demo)
    chunks = load_chunks()
    index = load_faiss_index()
    model = get_model()

    # Short-term memory context
    history_context = short_term_memory.get_context()

    answer, retrieved = process_query_qwen3(user_query, index, chunks, model, keyword, history_context)
    eval_scores = rag_evaluate(user_query, answer, retrieved, model)

    short_term_memory.add(user_id, user_query, answer)
    return jsonify({"answer": answer, "evaluation": eval_scores})

@rag_blueprint.route('/rag/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})