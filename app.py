"""
API FastAPI cho tìm kiếm đa phương thức (A1) và hỏi đáp RAG (A3), kèm giao diện web.

Chạy: uvicorn app:app --reload --port 8000, rồi mở http://localhost:8000
Cần có dữ liệu và index từ step1-3; /api/v1/ask cần thêm rag_core.
"""
import time
import io

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from search_core import search, available_components
from rag_core import answer as rag_answer, answer_agentic as rag_answer_agentic, clear_history

app = FastAPI(title="A1 Multimodal Search API")

# data/images/000001.jpg -> /images/000001.jpg cho thẻ <img> ở frontend.
app.mount("/images", StaticFiles(directory="data/images"), name="images")


def _parse_components(components: str | None):
    """"bm25,dense" -> ["bm25", "dense"]; None hoặc rỗng -> None (search() bật hết)."""
    if not components:
        return None
    return [c.strip() for c in components.split(",") if c.strip()]


@app.get("/api/v1/components")
def list_components():
    """Danh sách component đã đăng ký, frontend dùng để sinh checkbox."""
    return {"components": available_components()}


@app.get("/api/v1/search")
def search_text(
    q: str = Query(..., description="Câu truy vấn text"),
    k: int = 10,
    components: str | None = Query(None, description="vd: 'bm25,dense' -- rỗng = bật hết"),
):
    """Tìm bằng text, vd GET /api/v1/search?q=folding+pocket+knife&k=10&components=bm25,dense"""
    start = time.time()
    results = search(q, query_type="text", k=k, components=_parse_components(components))
    latency_ms = round((time.time() - start) * 1000, 1)

    return {
        "query": q,
        "components_used": _parse_components(components) or available_components(),
        "latency_ms": latency_ms,
        "results": results
    }


@app.post("/api/v1/search_by_image")
async def search_by_image(
    file: UploadFile = File(...),
    k: int = 10,
    components: str | None = Query(None),
):
    """Tìm bằng ảnh upload (multipart, field "file")."""
    start = time.time()
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = search(image, query_type="image", k=k, components=_parse_components(components))
    latency_ms = round((time.time() - start) * 1000, 1)

    return {
        "query": f"[image: {file.filename}]",
        "components_used": _parse_components(components) or available_components(),
        "latency_ms": latency_ms,
        "results": results
    }


@app.get("/")
def root():
    return FileResponse("frontend/index.html")


# RAG (A3), tách riêng khỏi /api/v1/search.
@app.get("/api/v1/ask")
def ask(
    q: str = Query(..., description="Câu hỏi tự nhiên, vd: 'giày leo núi chống nước'"),
    k: int = 5,
    agentic: bool = Query(False, description="True = tự đánh giá và search lại nếu thiếu"),
    session_id: str = Query(None, description="ID phiên chat -- có thì hệ thống nhớ ngữ cảnh câu trước"),
):
    """
    Search + sinh câu trả lời. agentic=False: naive RAG; True: tự đánh giá, search tối đa 2 lượt.
    session_id: giữ lịch sử hội thoại theo phiên; bỏ trống thì mỗi câu hỏi độc lập.
    """
    if agentic:
        result = rag_answer_agentic(q, k=k, session_id=session_id)
    else:
        result = rag_answer(q, k=k, session_id=session_id)
    return result


@app.post("/api/v1/reset_session")
def reset_session(session_id: str = Query(...)):
    """Xoá lịch sử hội thoại của một session."""
    clear_history(session_id)
    return {"status": "ok", "session_id": session_id}
