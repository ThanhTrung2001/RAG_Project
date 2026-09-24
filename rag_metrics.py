"""
Metric đánh giá RAG tự viết, theo ý tưởng RAGAS (đơn giản hoá, không dùng thư viện ragas).

- context_precision_at_k: công thức thuần, dạng Average Precision trong top-k.
- faithfulness: LLM làm giám khảo, tỷ lệ claim trong câu trả lời được context hỗ trợ.
- answer_relevancy: cosine(embedding câu hỏi, embedding câu trả lời), model all-MiniLM-L6-v2.
Context recall dùng lại recall_at_k của metrics.py.
"""
import re

from metrics import recall_at_k as context_recall_at_k


# ---------------------------------------------------------------------------
# Context Precision@k
# ---------------------------------------------------------------------------

def context_precision_at_k(retrieved_ids, relevant_ids, k):
    """
    Tổng precision@i tại các hạng i đúng, chia cho min(số id đúng, k).
    Đúng nhưng xếp hạng thấp thì điểm thấp. None nếu không có ground truth.
    """
    relevant_ids = set(relevant_ids)
    if not relevant_ids:
        return None

    hits = 0
    score_sum = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            hits += 1
            score_sum += hits / i

    denom = min(len(relevant_ids), k)
    return score_sum / denom if denom > 0 else None


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

_CLAIM_PATTERN = re.compile(r"CLAIM:\s*(.+?)\s*\|\s*(SUPPORTED|NOT_SUPPORTED)", re.IGNORECASE)


def _build_faithfulness_prompt(answer_text: str, context_text: str) -> str:
    return f"""Bạn là người chấm điểm "grounding" cho 1 hệ thống RAG. Nhiệm vụ: tách CÂU TRẢ LỜI dưới đây thành từng CLAIM (nhận định/thông tin cụ thể, độc lập với nhau), rồi với MỖI claim, xác định claim đó có được NGỮ CẢNH hỗ trợ hay không.

Ngữ cảnh (nguồn thông tin DUY NHẤT được phép dùng):
{context_text}

Câu trả lời cần chấm:
{answer_text}

Với MỖI claim tìm được, in ĐÚNG 1 dòng theo định dạng sau (không thêm gì khác):
CLAIM: <nội dung claim> | SUPPORTED
hoặc
CLAIM: <nội dung claim> | NOT_SUPPORTED

SUPPORTED = thông tin này CÓ trong ngữ cảnh. NOT_SUPPORTED = claim này KHÔNG có trong ngữ cảnh (kể cả khi nghe có vẻ đúng ngoài đời) -- bám sát NGUYÊN VĂN ngữ cảnh, không dùng kiến thức ngoài."""


def faithfulness(answer_text: str, context_text: str, llm_call_fn) -> float:
    """
    Số claim SUPPORTED / tổng số claim (1 lượt gọi llm_call_fn).
    llm_call_fn: hàm (prompt) -> str, truyền vào để không phụ thuộc cứng rag_core.
    None nếu answer/context rỗng hoặc không parse được claim nào.
    """
    if not answer_text.strip() or not context_text.strip():
        return None

    prompt = _build_faithfulness_prompt(answer_text, context_text)
    raw_output = llm_call_fn(prompt)

    matches = _CLAIM_PATTERN.findall(raw_output)
    if not matches:
        return None

    supported = sum(1 for _, verdict in matches if verdict.upper() == "SUPPORTED")
    return supported / len(matches)


# ---------------------------------------------------------------------------
# Answer Relevancy (đơn giản hoá: RAGAS gốc sinh câu hỏi giả định từ câu trả lời)
# ---------------------------------------------------------------------------

_embedder = None   # lazy load khi cần


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


def answer_relevancy(answer_text: str, question_text: str) -> float:
    """Cosine giữa câu hỏi và câu trả lời. None nếu câu trả lời rỗng."""
    if not answer_text.strip():
        return None

    model = _get_embedder()
    embs = model.encode([question_text, answer_text], normalize_embeddings=True)
    # Vector đã chuẩn hoá nên dot product = cosine.
    return float(embs[0] @ embs[1])
