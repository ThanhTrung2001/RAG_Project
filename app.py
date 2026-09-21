"""
================================================================================
BƯỚC 7 — API SERVICE (FastAPI)
================================================================================

TẠI SAO TÁCH RIÊNG API, KHÔNG ĐỂ FRONTEND GỌI THẲNG search()?
    Đây là nguyên tắc TÁCH LỚP (separation of concerns): frontend (chạy
    trong trình duyệt người dùng) không bao giờ được phép chạm trực tiếp
    vào model CLIP hay 2 file index -- nó CHỈ gửi HTTP request và nhận JSON.

    Lý do thực tế: model CLIP nặng (cần load vào RAM/VRAM), 2 file index
    cũng cần đọc từ đĩa -- những thứ này CHỈ nên tồn tại ở PHÍA SERVER (nơi
    file app.py này chạy), không thể "gửi" chúng sang trình duyệt người dùng.
    API chính là "cánh cửa" duy nhất kết nối 2 phía.

    Lợi ích phụ: đổi TOÀN BỘ thuật toán bên trong search_core.py (thêm
    rerank, đổi model CLIP...) thì FRONTEND KHÔNG CẦN SỬA GÌ -- miễn là API
    vẫn nhận đúng tham số và trả đúng format JSON như cũ.

TẠI SAO CÓ ENDPOINT /api/v1/components RIÊNG?
    Đây là "cầu nối" giữa registry pattern (bên search_core.py) và UI
    checkbox (bên frontend/index.html). Thay vì hardcode "bm25", "dense"
    trong code HTML, frontend GỌI endpoint này để hỏi "hiện có những
    component nào" -- thêm 1 component mới trong search_core.py, checkbox
    mới TỰ XUẤT HIỆN trên UI mà không cần sửa 1 dòng HTML nào.

CHẠY: uvicorn app:app --reload --port 8000
      rồi mở trình duyệt: http://localhost:8000
YÊU CẦU TRƯỚC: đã chạy xong step1, step2, step3.
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

# Mount thư mục ảnh thành 1 route tĩnh: file data/images/000001.jpg sẽ
# truy cập được qua URL /images/000001.jpg -- frontend dùng URL này để
# hiển thị <img src="...">, không cần API riêng để "tải" từng ảnh.
app.mount("/images", StaticFiles(directory="data/images"), name="images")


def _parse_components(components: str | None):
    """
    Chuyển chuỗi query string "bm25,dense" thành list ["bm25", "dense"].
    None hoặc chuỗi rỗng -> trả về None, để search() tự hiểu là "bật hết".
    """
    if not components:
        return None
    return [c.strip() for c in components.split(",") if c.strip()]


@app.get("/api/v1/components")
def list_components():
    """
    Frontend gọi endpoint này NGAY KHI TRANG WEB TẢI LÊN để biết cần vẽ
    bao nhiêu checkbox, tên gì -- xem chi tiết trong docstring đầu file.
    """
    return {"components": available_components()}


@app.get("/api/v1/search")
def search_text(
    q: str = Query(..., description="Câu truy vấn text"),
    k: int = 10,
    components: str | None = Query(None, description="vd: 'bm25,dense' -- rỗng = bật hết"),
):
    """
    Endpoint tìm kiếm bằng TEXT. Ví dụ gọi:
        GET /api/v1/search?q=folding+pocket+knife&k=10&components=bm25,dense
    """
    start = time.time()   # đo thời gian xử lý -- hữu ích khi phân tích latency
                            # trong báo cáo, và để debug nếu search bị chậm bất thường
    results = search(q, query_type="text", k=k, components=_parse_components(components))
    latency_ms = round((time.time() - start) * 1000, 1)

    return {
        "query": q,
        # Nếu người dùng không chỉ định components, trả về danh sách ĐẦY ĐỦ
        # (không phải None) -- để frontend luôn biết chính xác cấu hình nào
        # vừa chạy, dù có truyền tham số hay không.
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
    """
    Endpoint tìm kiếm bằng ẢNH. Dùng POST (không phải GET) vì phải gửi file
    nhị phân trong body request -- GET không phù hợp để gửi dữ liệu lớn.
    """
    start = time.time()
    image_bytes = await file.read()             # đọc toàn bộ file ảnh gửi lên thành bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")   # bytes -> PIL.Image
                                                                    # (định dạng search_core.py cần)
    results = search(image, query_type="image", k=k, components=_parse_components(components))
    latency_ms = round((time.time() - start) * 1000, 1)

    return {
        "query": f"[image: {file.filename}]",   # không có "câu chữ" thật, ghi tên file
                                                    # để người dùng biết đã search bằng ảnh nào
        "components_used": _parse_components(components) or available_components(),
        "latency_ms": latency_ms,
        "results": results
    }


@app.get("/")
def root():
    """Trả thẳng file HTML khi mở http://localhost:8000 -- không cần server tĩnh riêng."""
    return FileResponse("frontend/index.html")


# ---------------------------------------------------------------------------
# A3 — Endpoint RAG. TÁCH RIÊNG khỏi /api/v1/search (A1) -- không sửa gì
# endpoint cũ, vì frontend/script A1 (step8_run_ablation.py) vẫn cần
# /api/v1/search hoạt động y nguyên để ablation A1 tiếp tục đúng.
# ---------------------------------------------------------------------------

@app.get("/api/v1/ask")
def ask(
    q: str = Query(..., description="Câu hỏi tự nhiên, vd: 'giày leo núi chống nước'"),
    k: int = 5,
    agentic: bool = Query(False, description="True = tự đánh giá và search lại nếu thiếu"),
    session_id: str = Query(None, description="ID phiên chat -- có thì hệ thống nhớ ngữ cảnh câu trước"),
):
    """
    Endpoint RAG: search (A1) + sinh câu trả lời tự nhiên (A3).
    agentic=false (mặc định) -> Naive RAG, 1 lần search, đủ cho baseline A3.
    agentic=true             -> tự đánh giá + search lại tối đa 2 lần.
    session_id: TUỲ CHỌN -- nếu client gửi kèm (giữ NGUYÊN 1 giá trị cho cả
                phiên chat), hệ thống nhớ được các câu hỏi trước trong cùng
                session để hiểu ngữ cảnh (vd "còn màu khác không" sau khi đã
                hỏi về 1 sản phẩm cụ thể). Không gửi -> mỗi câu hỏi độc lập,
                y hệt hành vi trước khi có Memory (tương thích ngược).
    """
    if agentic:
        result = rag_answer_agentic(q, k=k, session_id=session_id)
    else:
        result = rag_answer(q, k=k, session_id=session_id)
    return result


@app.post("/api/v1/reset_session")
def reset_session(session_id: str = Query(...)):
    """Xoá lịch sử hội thoại của 1 session -- dùng khi người dùng bấm 'Cuộc trò chuyện mới'."""
    clear_history(session_id)
    return {"status": "ok", "session_id": session_id}
