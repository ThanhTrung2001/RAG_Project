"""
A3 - Tầng RAG: dùng lại search_core.search() của A1 rồi nhờ LLM viết câu trả lời có trích dẫn.

Luồng: retrieve() -> build_context() -> build_prompt() -> call_llm().
retrieve() gọi search() của A1; tuỳ chọn rerank bằng cross-encoder MiniLM (A1) hoặc DeBERTa distill (A2).
Có 2 pipeline: answer() (Naive RAG) và answer_agentic() (tự đánh giá, viết lại query, search lại),
kèm Memory hội thoại lưu trong RAM theo session_id.
Yêu cầu: Ollama chạy ở localhost:11434 và đã `ollama pull llama3.1`.
"""
import requests

from search_core import search

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.1"   # đổi nếu đã pull model khác trong Ollama

RERANK_POOL = 30   # số ứng viên đưa vào rerank; cùng pool với step_ablation_rerank.py


# ---------------------------------------------------------------------------
# Memory: lịch sử hội thoại theo session, lưu trong RAM (mất khi restart)
# ---------------------------------------------------------------------------
_SESSION_STORE = {}   # {session_id: [{"query": ..., "answer": ...}, ...]}
MAX_HISTORY_TURNS = 3   # giữ vài lượt gần nhất để prompt không quá dài


def get_history(session_id: str):
    """Lịch sử của session, rỗng nếu chưa có."""
    return _SESSION_STORE.get(session_id, [])


def append_history(session_id: str, query: str, answer_text: str):
    """Thêm 1 lượt hỏi-đáp, bỏ lượt cũ nhất nếu vượt MAX_HISTORY_TURNS."""
    history = _SESSION_STORE.setdefault(session_id, [])
    history.append({"query": query, "answer": answer_text})
    if len(history) > MAX_HISTORY_TURNS:
        del history[0]


def clear_history(session_id: str):
    _SESSION_STORE.pop(session_id, None)


def _format_history(history):
    """Lịch sử dạng text để chèn vào prompt (mỗi câu trả lời cắt 150 ký tự)."""
    if not history:
        return ""
    lines = ["Lịch sử hội thoại trước đó (để hiểu ngữ cảnh câu hỏi hiện tại):"]
    for turn in history:
        lines.append(f'- Khách hỏi: "{turn["query"]}" -> Đã trả lời: "{turn["answer"][:150]}..."')
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Context và prompt
# ---------------------------------------------------------------------------

def build_context(results):
    """Mỗi kết quả search 1 dòng, đánh số [1], [2]... để LLM trích dẫn."""
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r['caption']} (score={r['score']})")
    return "\n".join(lines)


def build_prompt(query, context, history=None):
    """Prompt ràng buộc: chỉ dùng context, trích dẫn [n], nói rõ nếu không tìm thấy."""
    history_block = _format_history(history) if history else ""

    return f"""Bạn là trợ lý tư vấn sản phẩm. CHỈ được dùng thông tin trong danh sách sản phẩm dưới đây để trả lời. KHÔNG được bịa thêm chi tiết (giá, tính năng...) không có trong danh sách. Nếu danh sách không có sản phẩm phù hợp, hãy nói rõ là không tìm thấy thay vì đoán.
{history_block}
Danh sách sản phẩm tìm được:
{context}

Câu hỏi của người dùng: "{query}"

Trả lời ngắn gọn bằng tiếng Việt, TRÍCH DẪN số thứ tự trong ngoặc vuông (ví dụ [1]) mỗi khi nhắc tới 1 sản phẩm cụ thể. Nếu câu hỏi hiện tại tham chiếu tới lượt hỏi trước (ví dụ "còn màu khác không", "cái đó giá bao nhiêu"), dùng lịch sử hội thoại ở trên để hiểu đúng ý khách đang hỏi về sản phẩm nào."""


def call_llm(prompt: str) -> str:
    """Gọi Ollama /api/generate (không streaming). Muốn đổi sang API khác chỉ cần sửa hàm này."""
    response = requests.post(OLLAMA_URL, json={
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"]


# ---------------------------------------------------------------------------
# Retrieval: search() của A1, tuỳ chọn rerank trước khi đưa vào context
# ---------------------------------------------------------------------------

def retrieve(query: str, k: int, components=None, reranker: str = None):
    """
    reranker=None: search() mặc định của A1 (BM25 + Dense + RRF), lấy k kết quả.
    reranker="cross_encoder" | "deberta": lấy RERANK_POOL ứng viên, xếp lại, giữ k kết quả đầu.
    """
    if reranker is None:
        return search(query, query_type="text", k=k, components=components)
    return search(query, query_type="text", k=k, components=components,
                  top_n=RERANK_POOL, reranker=reranker, rerank_pool=RERANK_POOL)


# ---------------------------------------------------------------------------
# Naive RAG (baseline)
# ---------------------------------------------------------------------------

def answer(query: str, k: int = 5, components=None, session_id: str = None,
           reranker: str = None):
    """
    Search 1 lần -> gọi LLM 1 lần (0 lần nếu search rỗng).
    session_id: bật Memory (đọc và ghi lịch sử); None = mỗi câu độc lập.
    reranker:   None, "cross_encoder" hoặc "deberta" (xem retrieve()).
    Trả về {"answer", "sources", "reranker"}; sources là kết quả search để hiển thị kèm.
    """
    history = get_history(session_id) if session_id else []

    results = retrieve(query, k=k, components=components, reranker=reranker)

    if not results:
        # Không gọi LLM với context rỗng để tránh bịa.
        fallback = "Không tìm thấy sản phẩm nào phù hợp với câu hỏi này."
        if session_id:
            append_history(session_id, query, fallback)
        return {"answer": fallback, "sources": [], "reranker": reranker}

    context = build_context(results)
    prompt = build_prompt(query, context, history=history)
    llm_answer = call_llm(prompt)

    if session_id:
        append_history(session_id, query, llm_answer)

    return {"answer": llm_answer, "sources": results, "reranker": reranker}


# ---------------------------------------------------------------------------
# Agentic RAG: LLM tự đánh giá context, nếu chưa đủ thì viết lại query và search lại
# ---------------------------------------------------------------------------

def judge_sufficiency(query: str, context: str) -> str:
    """LLM trả "CÓ" hoặc "KHÔNG: <lý do>"."""
    prompt = f"""Câu hỏi: "{query}"
Danh sách sản phẩm tìm được:
{context}

Danh sách trên có đủ thông tin để trả lời câu hỏi không?
Trả lời CHỈ theo đúng 1 trong 2 dạng:
"CÓ"
hoặc
"KHÔNG: <lý do ngắn gọn>" """
    return call_llm(prompt).strip()


def rewrite_query(original_query: str, reason: str) -> str:
    """LLM viết lại query cụ thể hơn dựa trên lý do còn thiếu."""
    prompt = f"""Người dùng hỏi: "{original_query}"
Kết quả tìm kiếm hiện tại KHÔNG đủ vì: {reason}
Hãy viết lại thành 1 câu truy vấn NGẮN GỌN, dùng từ khoá cụ thể hơn để tìm kiếm tốt hơn.
CHỈ trả về câu truy vấn mới, không giải thích gì thêm."""
    return call_llm(prompt).strip()


def answer_agentic(query: str, k: int = 5, components=None, max_iters: int = 2,
                   session_id: str = None, reranker: str = None):
    """
    Lặp tối đa max_iters lần: search -> judge -> (chưa đủ) rewrite. Sau đó 1 lượt trả lời.
    Với max_iters=2: 2 lượt gọi LLM nếu đủ ngay, 4 lượt nếu phải search lại.
    reranker áp dụng cho mọi lượt search (xem retrieve()).
    Trả về thêm "final_query_used" (query cuối dùng để search).
    """
    history = get_history(session_id) if session_id else []

    current_query = query
    results, context = [], ""

    for attempt in range(max_iters):
        results = retrieve(current_query, k=k, components=components, reranker=reranker)
        context = build_context(results) if results else "(không có kết quả)"

        verdict = judge_sufficiency(query, context)   # luôn so với query gốc
        if verdict.upper().startswith("CÓ") or attempt == max_iters - 1:
            break

        reason = verdict.split(":", 1)[-1].strip() if ":" in verdict else verdict
        current_query = rewrite_query(current_query, reason)

    # Câu trả lời cuối dựa trên query gốc của người dùng, không phải query đã viết lại.
    prompt = build_prompt(query, context, history=history)
    llm_answer = call_llm(prompt)

    if session_id:
        append_history(session_id, query, llm_answer)

    return {
        "answer": llm_answer,
        "sources": results,
        "final_query_used": current_query,
        "reranker": reranker,
    }
