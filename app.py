"""
API FastAPI cho tìm kiếm đa phương thức (A1) và chatbot RAG (A3), kèm 2 trang web riêng.

Chạy: uvicorn app:app --reload --port 8000
  - http://localhost:8000/search : giao diện tìm kiếm A1
  - http://localhost:8000/chat   : giao diện chatbot A3
Cần có dữ liệu và index từ step1-3; /api/v1/ask cần Ollama (xem rag_core.py).
"""
import time
import io

import requests
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from search_core import search, available_components, available_rerankers, get_product
from rag_core import answer as rag_answer, answer_agentic as rag_answer_agentic, clear_history
from rag_core import LLM_MODEL

app = FastAPI(title="A1 Multimodal Search + A3 RAG API")

# data/images/000001.jpg -> /images/000001.jpg cho thẻ <img> ở frontend.
app.mount("/images", StaticFiles(directory="data/images"), name="images")
# frontend/style.css -> /static/style.css, dùng chung cho 2 trang.
app.mount("/static", StaticFiles(directory="frontend"), name="static")


def _parse_components(components: str | None):
    """"bm25,dense" -> ["bm25", "dense"]; None hoặc rỗng -> None (search() bật hết)."""
    if not components:
        return None
    return [c.strip() for c in components.split(",") if c.strip()]


# ---------------------------------------------------------------------------
# Trang web
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return RedirectResponse("/search")


@app.get("/search")
def search_page():
    return FileResponse("frontend/search.html")


@app.get("/chat")
def chat_page():
    return FileResponse("frontend/chat.html")


# ---------------------------------------------------------------------------
# A1: tìm kiếm
# ---------------------------------------------------------------------------

@app.get("/api/v1/components")
def list_components():
    """Danh sách component đã đăng ký, frontend dùng để sinh checkbox."""
    return {"components": available_components()}


@app.get("/api/v1/search")
def search_text(
    q: str = Query(..., description="Câu truy vấn text"),
    k: int = 10,
    components: str | None = Query(None, description="vd: 'bm25,dense' -- rỗng = bật hết"),
    rerank: bool = Query(False, description="True = rerank bằng cross-encoder MiniLM sau RRF"),
):
    """Tìm bằng text, vd GET /api/v1/search?q=folding+pocket+knife&k=10&components=bm25,dense&rerank=true"""
    start = time.time()
    results = search(q, query_type="text", k=k, components=_parse_components(components),
                     use_rerank=rerank)
    latency_ms = round((time.time() - start) * 1000, 1)

    return {
        "query": q,
        "components_used": _parse_components(components) or available_components(),
        "rerank": rerank,
        "latency_ms": latency_ms,
        "results": results
    }


@app.get("/api/v1/products/{product_id}")
def product_detail(product_id: int):
    """Chi tiết 1 sản phẩm (tên, brand, category, mô tả) để hiển thị khi bấm vào sản phẩm."""
    product = get_product(product_id)
    if product is None:
        raise HTTPException(404, f"Không có sản phẩm id={product_id}")
    return product


@app.post("/api/v1/search_by_image")
async def search_by_image(
    file: UploadFile = File(...),
    k: int = 10,
    components: str | None = Query(None),
):
    """Tìm bằng ảnh upload (multipart, field "file"). Không có rerank vì cross-encoder cần text."""
    start = time.time()
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = search(image, query_type="image", k=k, components=_parse_components(components))
    latency_ms = round((time.time() - start) * 1000, 1)

    return {
        "query": f"[image: {file.filename}]",
        "components_used": _parse_components(components) or available_components(),
        "rerank": False,
        "latency_ms": latency_ms,
        "results": results
    }


# ---------------------------------------------------------------------------
# A3: chatbot RAG
# ---------------------------------------------------------------------------

@app.get("/api/v1/rerankers")
def list_rerankers():
    """Reranker dùng được cho /api/v1/ask ("deberta" chỉ có khi đã tải checkpoint)."""
    return {"rerankers": available_rerankers()}


@app.get("/api/v1/ask")
def ask(
    q: str = Query(..., description="Câu hỏi tự nhiên, vd: 'giày leo núi chống nước'"),
    k: int = 5,
    agentic: bool = Query(False, description="True = tự đánh giá và search lại nếu thiếu"),
    session_id: str = Query(None, description="ID phiên chat -- có thì hệ thống nhớ ngữ cảnh câu trước"),
    reranker: str = Query("none", description="none | cross_encoder | deberta"),
):
    """
    Search + sinh câu trả lời. agentic=False: naive RAG; True: tự đánh giá, search tối đa 2 lượt.
    session_id: giữ lịch sử hội thoại theo phiên; bỏ trống thì mỗi câu hỏi độc lập.
    reranker: xếp lại ứng viên trước khi đưa vào context (xem rag_core.retrieve()).
    """
    if reranker == "none":
        reranker = None
    elif reranker not in available_rerankers():
        raise HTTPException(400, f"reranker không hợp lệ hoặc chưa có model: {reranker}")

    try:
        if agentic:
            return rag_answer_agentic(q, k=k, session_id=session_id, reranker=reranker)
        return rag_answer(q, k=k, session_id=session_id, reranker=reranker)
    except requests.HTTPError as e:
        # Ollama có chạy nhưng trả lỗi (thường là chưa pull đúng model).
        raise HTTPException(503, f"{e}. Xem model đã có bằng `ollama list`; "
                                 "đổi model bằng biến môi trường OLLAMA_MODEL.")
    except requests.RequestException as e:
        raise HTTPException(503, f"Không kết nối được Ollama ({type(e).__name__}) khi gọi model "
                                 f"{LLM_MODEL}. Kiểm tra Ollama đã chạy chưa.")


@app.post("/api/v1/reset_session")
def reset_session(session_id: str = Query(...)):
    """Xoá lịch sử hội thoại của một session."""
    clear_history(session_id)
    return {"status": "ok", "session_id": session_id}
