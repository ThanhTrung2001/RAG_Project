"""
================================================================================
A3 — TẦNG RAG: MỞ RỘNG search() CỦA A1 THÀNH TRỢ LÝ TRẢ LỜI TỰ NHIÊN
================================================================================

QUAN HỆ VỚI A1 — TẠI SAO KHÔNG VIẾT LẠI SEARCH?
    File này KHÔNG chứa bất kỳ logic retrieval nào (không có BM25, không có
    FAISS). Nó CHỈ import và gọi lại search_core.search() nguyên vẹn -- đúng
    như đã note ở phần "Vì sao dùng registry pattern": code A1 viết ra để
    dùng lại, không phải viết lại. RAG = Search (đã có) + 2 bước MỚI:

        search() (A1, không đổi) -> build_context() -> build_prompt() -> call_llm()

TẠI SAO KHÔNG DÙNG CHUNG COMPONENT_REGISTRY VỚI BM25/DENSE?
    COMPONENT_REGISTRY trong search_core.py dành cho các chiến lược chạy
    SONG SONG rồi fusion (BM25 và Dense đều chạy, rồi hợp nhất kết quả).
    "Sinh câu trả lời" là bước chạy TUẦN TỰ, SAU KHI đã có kết quả search --
    không fusion với gì cả. Vì bản chất khác nhau, đây là 1 pipeline riêng,
    không đăng ký chung registry (đúng câu trả lời trong Phụ lục Q&A báo cáo A1).

TẠI SAO PHẢI "BUILD CONTEXT" CÓ ĐÁNH SỐ [1][2][3]?
    Đây là kỹ thuật GROUNDING quan trọng nhất để giảm hallucination: LLM chỉ
    được phép dùng thông tin có ĐÁNH SỐ trong context, và phải TRÍCH DẪN số
    đó khi trả lời. Nếu không làm vậy, LLM có thể "bịa" tên sản phẩm, giá
    tiền không có thật -- đây chính là rủi ro đạo đức AI (G3.2) cần nêu
    trong báo cáo A3.

TẠI SAO GỌI LLM QUA OLLAMA (LOCAL), KHÔNG GỌI API TRẢ PHÍ?
    Ollama chạy model LLM ngay trên máy, miễn phí, không cần API key --
    phù hợp để demo trong lớp mà không lo phát sinh chi phí hay lộ key.
    Nếu muốn đổi sang OpenAI/Anthropic/Gemini API, CHỈ CẦN SỬA HÀM
    call_llm() -- mọi hàm khác trong file này không cần đổi gì (đây chính
    là lý do tách call_llm() thành 1 hàm riêng, không gọi API rải rác
    khắp nơi).

CÀI ĐẶT TRƯỚC KHI DÙNG:
    1. Cài Ollama: https://ollama.com
    2. Tải 1 model:  ollama pull llama3.1        (hoặc model nhẹ hơn: qwen2.5:3b)
    3. Ollama tự chạy server ở http://localhost:11434 sau khi cài
"""
import requests

from search_core import search

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.1"   # đổi tên model khác nếu đã pull model khác trong Ollama


# ---------------------------------------------------------------------------
# MEMORY -- lưu lịch sử hội thoại theo session, để hệ thống "nhớ" ngữ cảnh
# câu hỏi trước (đúng yêu cầu "A3 đưa lên chatbot" -- chatbot thật sự phải
# hiểu hội thoại NHIỀU LƯỢT, không phải chỉ trả lời độc lập từng câu).
#
# TẠI SAO DÙNG DICT TRONG RAM, KHÔNG DÙNG DATABASE?
#     Ở quy mô demo (1 vài phiên chat cùng lúc), dict Python trong RAM là
#     đủ và đơn giản nhất để CHỨNG MINH có thành phần Memory trong kiến
#     trúc (Search -> RAG -> Agent controller -> Memory theo sơ đồ môn
#     học). Nhược điểm: mất hết lịch sử khi restart server, không dùng
#     được nếu chạy nhiều server song song (không chia sẻ RAM) -- chấp
#     nhận được cho phạm vi 1 đồ án môn học, KHÔNG phù hợp cho production
#     thật (khi đó cần Redis/database để lưu bền và chia sẻ giữa nhiều máy).
# ---------------------------------------------------------------------------
_SESSION_STORE = {}   # {session_id: [{"query": ..., "answer": ...}, ...]}
MAX_HISTORY_TURNS = 3   # chỉ giữ 3 lượt gần nhất -- lịch sử dài vô hạn sẽ làm
                          # prompt quá dài, tốn chi phí gọi LLM và dễ loãng ngữ cảnh


def get_history(session_id: str):
    """Lấy lịch sử hội thoại của 1 session -- rỗng nếu session mới/không tồn tại."""
    return _SESSION_STORE.get(session_id, [])


def append_history(session_id: str, query: str, answer_text: str):
    """Ghi thêm 1 lượt hỏi-đáp vào lịch sử, tự cắt bớt nếu vượt MAX_HISTORY_TURNS."""
    history = _SESSION_STORE.setdefault(session_id, [])
    history.append({"query": query, "answer": answer_text})
    if len(history) > MAX_HISTORY_TURNS:
        del history[0]   # xoá lượt CŨ NHẤT, giữ lại các lượt gần đây hơn


def clear_history(session_id: str):
    """Xoá lịch sử 1 session -- dùng khi người dùng bấm 'bắt đầu hội thoại mới'."""
    _SESSION_STORE.pop(session_id, None)


def _format_history(history):
    """Biến lịch sử thành đoạn text ngắn gọn để chèn vào prompt."""
    if not history:
        return ""
    lines = ["Lịch sử hội thoại trước đó (để hiểu ngữ cảnh câu hỏi hiện tại):"]
    for turn in history:
        lines.append(f'- Khách hỏi: "{turn["query"]}" -> Đã trả lời: "{turn["answer"][:150]}..."')
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# BƯỚC A3-1: Build context có đánh số, để LLM trích dẫn được nguồn
# ---------------------------------------------------------------------------

def build_context(results):
    """
    results: list dict trả về từ search_core.search() -- mỗi phần tử có
             id, image_path, caption, score.
    Trả về: 1 đoạn text, mỗi sản phẩm 1 dòng, đánh số [1], [2], ... để LLM
             trích dẫn lại đúng số này khi trả lời (không trích dẫn bằng id
             thô, vì [1][2][3] dễ đọc và tự nhiên hơn trong câu trả lời).
    """
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r['caption']} (score={r['score']})")
    return "\n".join(lines)


def build_prompt(query, context, history=None):
    """
    Prompt CỐ Ý ràng buộc chặt: chỉ được dùng context, phải trích dẫn số,
    phải nói rõ nếu không tìm thấy -- đây là 3 câu quan trọng nhất để
    giảm hallucination, không phải chi tiết trang trí.

    history: list lịch sử hội thoại (từ get_history()) -- None hoặc [] thì
             hành vi giống hệt bản gốc (không đổi gì cho code cũ gọi hàm
             này mà không truyền history).
    """
    history_block = _format_history(history) if history else ""

    return f"""Bạn là trợ lý tư vấn sản phẩm. CHỈ được dùng thông tin trong danh sách sản phẩm dưới đây để trả lời. KHÔNG được bịa thêm chi tiết (giá, tính năng...) không có trong danh sách. Nếu danh sách không có sản phẩm phù hợp, hãy nói rõ là không tìm thấy thay vì đoán.
{history_block}
Danh sách sản phẩm tìm được:
{context}

Câu hỏi của người dùng: "{query}"

Trả lời ngắn gọn bằng tiếng Việt, TRÍCH DẪN số thứ tự trong ngoặc vuông (ví dụ [1]) mỗi khi nhắc tới 1 sản phẩm cụ thể. Nếu câu hỏi hiện tại tham chiếu tới lượt hỏi trước (ví dụ "còn màu khác không", "cái đó giá bao nhiêu"), dùng lịch sử hội thoại ở trên để hiểu đúng ý khách đang hỏi về sản phẩm nào."""


# ---------------------------------------------------------------------------
# BƯỚC A3-2: Gọi LLM -- điểm DUY NHẤT chạm vào model sinh text
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    """
    Gọi model qua Ollama REST API. stream=False -- nhận nguyên câu trả lời
    1 lần thay vì streaming từng token (đơn giản hơn cho A3 baseline;
    streaming là 1 hướng nâng cấp UX có thể làm thêm sau).
    """
    response = requests.post(OLLAMA_URL, json={
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    })
    response.raise_for_status()   # raise lỗi rõ ràng nếu Ollama chưa chạy/model chưa pull
    return response.json()["response"]


# ---------------------------------------------------------------------------
# BƯỚC A3-3: Naive RAG -- pipeline đơn giản nhất, dùng làm BASELINE (5đ)
# ---------------------------------------------------------------------------

def answer(query: str, k: int = 5, components=None, session_id: str = None):
    """
    Naive RAG: search 1 lần -> build context -> build prompt -> gọi LLM 1 lần.
    Đây là pipeline TỐI THIỂU cần có để đạt phần baseline của rubric A3.

    session_id: nếu có, đọc lịch sử hội thoại của session này để đưa vào
                prompt (Memory), và LƯU LẠI lượt hỏi-đáp này vào lịch sử sau
                khi có câu trả lời. None -> hành vi y hệt bản gốc, không có
                Memory (mỗi câu hỏi độc lập) -- giữ tương thích ngược cho
                nơi nào gọi answer() mà không cần hội thoại nhiều lượt.

    Trả về dict có "sources" (kết quả search THẬT) kèm theo "answer" (text
    LLM sinh ra) -- để frontend hiển thị được ẢNH/NGUỒN THẬT bên cạnh câu
    trả lời, cho phép người dùng tự kiểm chứng LLM có trích dẫn đúng không.
    """
    history = get_history(session_id) if session_id else []

    results = search(query, query_type="text", k=k, components=components)

    if not results:
        # Không có gì để đưa vào context -- trả lời thẳng, KHÔNG gọi LLM
        # (gọi LLM với context rỗng dễ khiến nó tự bịa câu trả lời).
        fallback = "Không tìm thấy sản phẩm nào phù hợp với câu hỏi này."
        if session_id:
            append_history(session_id, query, fallback)
        return {"answer": fallback, "sources": []}

    context = build_context(results)
    prompt = build_prompt(query, context, history=history)
    llm_answer = call_llm(prompt)

    if session_id:
        append_history(session_id, query, llm_answer)

    return {"answer": llm_answer, "sources": results}


# ---------------------------------------------------------------------------
# BƯỚC A3-4 (NÂNG CẤP — tuỳ chọn): Agentic RAG -- tự đánh giá và search lại
# ---------------------------------------------------------------------------
# Khác Naive RAG ở chỗ: sau khi search, LLM TỰ ĐÁNH GIÁ kết quả có đủ để trả
# lời không. Nếu không đủ, LLM tự viết lại câu query rồi search LẦN NỮA --
# lặp tối đa max_iters lần. Đây là dạng đơn giản của "Agentic RAG" (B5-6).

def judge_sufficiency(query: str, context: str) -> str:
    """Hỏi LLM: context hiện tại có đủ để trả lời câu hỏi gốc không?"""
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
    """Nhờ LLM viết lại câu query cụ thể/dễ search hơn, dựa trên lý do vừa thiếu."""
    prompt = f"""Người dùng hỏi: "{original_query}"
Kết quả tìm kiếm hiện tại KHÔNG đủ vì: {reason}
Hãy viết lại thành 1 câu truy vấn NGẮN GỌN, dùng từ khoá cụ thể hơn để tìm kiếm tốt hơn.
CHỈ trả về câu truy vấn mới, không giải thích gì thêm."""
    return call_llm(prompt).strip()


def answer_agentic(query: str, k: int = 5, components=None, max_iters: int = 2, session_id: str = None):
    """
    Vòng lặp: search -> tự đánh giá -> (nếu chưa đủ) viết lại query -> search lại.
    max_iters giới hạn số lần lặp -- BẮT BUỘC phải có giới hạn, nếu không
    LLM có thể lặp vô hạn nếu nó liên tục đánh giá "chưa đủ".

    session_id: giống hệt vai trò trong answer() -- đọc/ghi lịch sử hội thoại
                nếu có, None thì không dùng Memory (tương thích ngược).
    """
    history = get_history(session_id) if session_id else []

    current_query = query
    results, context = [], ""

    for attempt in range(max_iters):
        results = search(current_query, query_type="text", k=k, components=components)
        context = build_context(results) if results else "(không có kết quả)"

        verdict = judge_sufficiency(query, context)   # luôn đánh giá so với query GỐC,
                                                          # không phải query đã viết lại
        if verdict.upper().startswith("CÓ") or attempt == max_iters - 1:
            break

        reason = verdict.split(":", 1)[-1].strip() if ":" in verdict else verdict
        current_query = rewrite_query(current_query, reason)

    # Câu trả lời CUỐI CÙNG luôn build từ query GỐC của người dùng (không phải
    # bản đã viết lại) -- người dùng hỏi gì thì nhận câu trả lời đúng cho câu đó,
    # dù bên trong hệ thống đã tự đổi query để search cho tốt hơn.
    prompt = build_prompt(query, context, history=history)
    llm_answer = call_llm(prompt)

    if session_id:
        append_history(session_id, query, llm_answer)

    return {
        "answer": llm_answer,
        "sources": results,
        "final_query_used": current_query,   # để debug/demo: xem hệ thống đã tự
                                               # viết lại query thành gì
    }
