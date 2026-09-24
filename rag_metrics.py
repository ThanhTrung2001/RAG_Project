"""
================================================================================
METRIC ĐÁNH GIÁ RAG TỰ VIẾT — Context Precision, Faithfulness, Answer Relevancy
================================================================================

TẠI SAO Recall@k/nDCG@k (metrics.py, A1) KHÔNG ĐỦ CHO A3?
    Recall/nDCG chỉ đo TẦNG RETRIEVAL (tìm đúng sản phẩm không). RAG có THÊM
    1 tầng nữa: TẦNG GENERATION (LLM viết câu trả lời từ context tìm được).
    2 tầng này có thể sai ĐỘC LẬP với nhau:
        - Retrieval đúng (tìm đúng sản phẩm) NHƯNG Generation vẫn có thể
          BỊA (hallucinate) chi tiết không có trong context -- Recall không
          bắt được lỗi này, cần Faithfulness.
        - Retrieval đúng, Generation không bịa, NHƯNG trả lời LẠC ĐỀ (không
          thực sự trả lời câu hỏi người dùng) -- cần Answer Relevancy.
    Đây chính là khung "tách retrieval vs generation" đã nêu ở BAO_CAO_A3.md
    mục 10c -- 3 metric dưới đây biến khung đó từ ĐỊNH TÍNH thành ĐO ĐƯỢC.

TẠI SAO TỰ VIẾT, KHÔNG DÙNG THƯ VIỆN `ragas`?
    ragas mặc định kỳ vọng OpenAI API cho phần LLM-giám khảo -- cấu hình lại
    để dùng Ollama local phức tạp hơn, và là "hộp đen" khó giải thích chi
    tiết bên trong khi bảo vệ đồ án (cùng lý do A1 tự viết Recall/nDCG thay
    vì dùng sklearn). 3 hàm dưới đây cài lại đúng Ý TƯỞNG của 3 metric này
    theo cách đơn giản hoá CÓ CHỦ ĐÍCH, giải thích rõ trade-off ở từng hàm.

3 METRIC:
    1. context_precision_at_k() -- KHÔNG cần LLM, thuần công thức (giống
       Average Precision@k) -- đo retrieval có xếp hạng đúng ĐÚNG sản phẩm
       lên TRÊN các sản phẩm sai không (khác Recall: không quan tâm thứ hạng).
    2. faithfulness() -- CẦN LLM (dùng Ollama làm "giám khảo"): tách câu trả
       lời thành từng claim, hỏi LLM claim đó có được context hỗ trợ không.
    3. answer_relevancy() -- CẦN embedding model: đo câu trả lời có THỰC SỰ
       nói về đúng câu hỏi không, bằng cosine similarity(embedding(câu hỏi),
       embedding(câu trả lời)) -- ĐƠN GIẢN HOÁ so với RAGAS gốc (RAGAS sinh
       lại nhiều "câu hỏi giả định" từ câu trả lời rồi so với câu hỏi gốc,
       tốn thêm nhiều lượt gọi LLM) -- đánh đổi: nhanh hơn, ít lượt gọi LLM
       hơn, nhưng kém tinh vi hơn bản gốc. Đủ dùng làm proxy tự động cho A3.
"""
import re

from metrics import recall_at_k as context_recall_at_k   # context recall = recall_at_k
                                                            # đã viết ở A1, dùng lại nguyên vẹn
                                                            # (không viết lại lần 2)


# ---------------------------------------------------------------------------
# 1. CONTEXT PRECISION@K -- thuần công thức, không cần LLM
# ---------------------------------------------------------------------------

def context_precision_at_k(retrieved_ids, relevant_ids, k):
    """
    Đo retrieval có xếp các sản phẩm ĐÚNG lên TRÊN các sản phẩm SAI không --
    khác Recall (chỉ đếm CÓ tìm được hay không, không quan tâm thứ hạng).

    Công thức (Average Precision giới hạn trong top-k):
        precision@i = (số sản phẩm đúng trong i kết quả đầu) / i
        Context Precision@k = tổng(precision@i, với MỌI i mà kết quả tại
                               hạng i là ĐÚNG) / min(số sản phẩm đúng, k)

    VÍ DỤ TÍNH TAY: relevant_ids={3}, retrieved=[1,2,3,4,5] (k=5)
        Sản phẩm đúng (id=3) nằm ở hạng 3 -> precision@3 = 1/3
        Context Precision@5 = (1/3) / 1 = 0.333
        (nếu id=3 nằm ở hạng 1 thay vì hạng 3: precision@1=1/1=1.0 -> điểm
        cao hơn hẳn -- đúng ý nghĩa "phạt nếu đúng nhưng xếp hạng thấp",
        giống tinh thần nDCG nhưng công thức khác, không dùng log2)
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
# 2. FAITHFULNESS -- cần LLM làm "giám khảo" (dùng lại call_llm() của rag_core)
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
    Trả về tỷ lệ claim ĐƯỢC NGỮ CẢNH HỖ TRỢ / tổng số claim -- 1.0 nghĩa là
    KHÔNG hallucination (mọi thông tin trong câu trả lời đều bắt nguồn từ
    context), 0.0 nghĩa là bịa hoàn toàn.

    llm_call_fn: hàm (prompt: str) -> str -- truyền vào thay vì import cứng
                 rag_core.call_llm(), để file này KHÔNG phụ thuộc trực tiếp
                 cách gọi LLM cụ thể (dễ test độc lập, giống thiết kế
                 evaluate_all(search_fn=...) ở metrics.py).

    Trả về None nếu không parse được claim nào (câu trả lời quá ngắn, hoặc
    LLM trả sai định dạng -- KHÔNG tính là 0.0, vì 0.0 nghĩa là "có claim
    nhưng sai hết", khác với "không có claim nào để chấm").
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
# 3. ANSWER RELEVANCY -- cần embedding model (đơn giản hoá so với RAGAS gốc,
#    xem giải thích trade-off ở docstring đầu file)
# ---------------------------------------------------------------------------

_embedder = None   # lazy load, giống _reranker trong search_core.py -- chỉ tải
                     # model khi THỰC SỰ cần, không phải mọi lần import file này


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


def answer_relevancy(answer_text: str, question_text: str) -> float:
    """
    Cosine similarity giữa embedding(câu hỏi gốc) và embedding(câu trả lời)
    -- điểm càng cao, câu trả lời càng "cùng chủ đề" với câu hỏi. Không bắt
    được lỗi tinh vi (câu trả lời đúng chủ đề nhưng sai chi tiết) -- đó là
    việc của Faithfulness, không phải Answer Relevancy.

    Trả về None nếu câu trả lời rỗng (trường hợp fallback "không tìm thấy
    sản phẩm" của rag_core.answer() -- không có gì để so sánh).
    """
    if not answer_text.strip():
        return None

    model = _get_embedder()
    embs = model.encode([question_text, answer_text], normalize_embeddings=True)
    # normalize_embeddings=True -- vector đã chuẩn hoá độ dài 1, nên dot
    # product == cosine similarity, giống lý do chuẩn hoá CLIP embedding ở A1.
    return float(embs[0] @ embs[1])
