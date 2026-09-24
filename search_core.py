"""
================================================================================
BƯỚC 4 — LÕI TRUY HỒI: BM25 + DENSE (FAISS) + RRF FUSION
================================================================================

THIẾT KẾ TỔNG QUAN — "REGISTRY PATTERN" LÀ GÌ VÀ TẠI SAO DÙNG?
    Thay vì viết cứng 2 tham số boolean (use_bm25, use_dense) rồi if/else
    trong hàm search(), mỗi kỹ thuật retrieval được viết thành 1 HÀM ĐỘC LẬP
    và TỰ ĐĂNG KÝ vào 1 "sổ đăng ký" chung (COMPONENT_REGISTRY) thông qua
    decorator @component("tên").

    Lợi ích cụ thể:
      1. search() không cần biết BM25/Dense hoạt động ra sao bên trong --
         chỉ cần biết "gọi những tên nào có trong danh sách được yêu cầu".
      2. Thêm 1 kỹ thuật mới (ví dụ rerank bằng cross-encoder) chỉ cần viết
         thêm 1 hàm + gắn decorator -- KHÔNG cần sửa search(), KHÔNG cần sửa
         app.py, KHÔNG cần sửa frontend (frontend tự hỏi qua API xem có gì).
      3. Ablation study (bước 8) và API sản phẩm thật (bước 7) DÙNG CHUNG
         đúng 1 cơ chế bật/tắt này -- không viết logic 2 lần ở 2 nơi khác nhau.

CƠ CHẾ RRF FUSION (Reciprocal Rank Fusion) — TẠI SAO KHÔNG CỘNG ĐIỂM THẲNG?
    BM25 trả điểm dạng 0-15+ (tuỳ độ dài văn bản), cosine similarity (dense)
    trả điểm 0-1 -- 2 THANG ĐO HOÀN TOÀN KHÁC NHAU, cộng thẳng vào nhau là
    phép so sánh vô nghĩa (giống cộng "độ C" với "phần trăm").

    RRF né hoàn toàn vấn đề này bằng cách CHỈ DÙNG THỨ HẠNG (rank), không
    dùng điểm số gốc:
        score(doc) = sum( 1 / (k + rank_trong_moi_danh_sach) )
    Một tài liệu đứng hạng 1 ở CẢ 2 hệ thống sẽ có điểm RRF cao nhất --
    dù điểm số gốc của 2 hệ thống chênh lệch bao nhiêu đi nữa.

CHẠY: file này KHÔNG chạy trực tiếp -- được import bởi app.py và
      step8_run_ablation.py.
YÊU CẦU TRƯỚC: đã chạy xong step1, step2, step3 (cần có data/dense.index,
      data/bm25.pkl, data/catalog.jsonl).
"""
import json
import pickle

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CATALOG_PATH = "data/catalog.jsonl"
DENSE_INDEX_PATH = "data/dense.index"
BM25_INDEX_PATH = "data/bm25.pkl"
MODEL_NAME = "openai/clip-vit-base-patch32"

# ---------------------------------------------------------------------------
# KHỞI TẠO 1 LẦN DUY NHẤT KHI MODULE ĐƯỢC IMPORT (không phải mỗi lần search)
# ---------------------------------------------------------------------------
# Load model CLIP và 2 index vào RAM ngay khi file này được import lần đầu --
# việc này hơi chậm (vài giây) nhưng chỉ xảy ra 1 LẦN khi app.py khởi động,
# không lặp lại ở mỗi request search -- đó là lý do các dòng dưới đây nằm
# NGOÀI mọi hàm, ở cấp module (biến bắt đầu bằng "_" quy ước là "riêng tư",
# không nên import trực tiếp từ file khác).

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = CLIPModel.from_pretrained(MODEL_NAME).to(_device).eval()
# use_fast=False: tránh lỗi "fast image processor" đổi mặc định ở bản transformers
# mới, từng gây lỗi get_image_features() trả sai kiểu dữ liệu (đã gặp thật trên Kaggle)
_processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=False)

_dense_index = faiss.read_index(DENSE_INDEX_PATH)   # đọc lại FAISS index đã build ở bước 3

with open(BM25_INDEX_PATH, "rb") as f:
    _bm25_data = pickle.load(f)
_bm25 = _bm25_data["bm25"]          # object BM25Okapi đã build sẵn ở bước 3
_bm25_ids = _bm25_data["ids"]       # danh sách id, để biết vị trí thứ i trong BM25
                                      # tương ứng sản phẩm nào (BM25 tự nó chỉ biết index 0,1,2...)

# Đọc catalog vào 1 dict {id: row} để tra cứu NHANH (O(1)) khi cần lấy
# lại image_path/caption từ 1 id kết quả -- thay vì phải quét cả file mỗi lần.
_catalog = {}
with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        _catalog[row["id"]] = row


# ---------------------------------------------------------------------------
# HÀM ENCODE — biến query (text hoặc ảnh) thành vector cùng không gian với
# vector đã lưu trong FAISS (đã tính sẵn ở bước 2)
# ---------------------------------------------------------------------------

def _encode_text(text: str) -> np.ndarray:
    """Encode 1 câu text thành vector 512 chiều, dùng CLIP text encoder."""
    with torch.no_grad():   # không cần tính gradient khi chỉ suy luận (inference)
        inputs = _processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(_device)
        feat = _model.get_text_features(**inputs)
        # Chuẩn hoá giống hệt cách đã làm ở bước 2 -- BẮT BUỘC phải nhất quán,
        # nếu không cosine similarity sẽ tính sai.
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")


def _encode_image(image: Image.Image) -> np.ndarray:
    """Encode 1 ảnh (PIL.Image) thành vector 512 chiều, dùng CLIP image encoder."""
    with torch.no_grad():
        inputs = _processor(images=[image], return_tensors="pt").to(_device)
        feat = _model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")


# ---------------------------------------------------------------------------
# REGISTRY — nơi mỗi kỹ thuật retrieval "tự giới thiệu" bản thân
# ---------------------------------------------------------------------------

# Dict toàn cục: key = tên component (str), value = hàm xử lý.
# Được điền vào bởi decorator component() ngay bên dưới -- KHÔNG cần code nào
# khác chủ động thêm phần tử vào dict này.
COMPONENT_REGISTRY = {}


def component(name: str):
    """
    Decorator: gắn @component("tên") lên 1 hàm để hàm đó TỰ ĐỘNG được thêm
    vào COMPONENT_REGISTRY ngay khi Python đọc tới định nghĩa hàm (tức là
    ngay khi file này được import, trước cả khi search() được gọi lần nào).

    Cách decorator hoạt động: component("bm25") trả về hàm "wrapper".
    Python sau đó áp wrapper lên hàm được decorate (_run_bm25), và vì
    wrapper() return chính func đó KHÔNG SỬA GÌ, nên _run_bm25 vẫn hoạt
    động y hệt bình thường khi được gọi -- decorator ở đây chỉ có tác dụng
    PHỤ (side effect) là ghi vào dict, không đổi hành vi hàm gốc.
    """
    def wrapper(func):
        COMPONENT_REGISTRY[name] = func
        return func
    return wrapper


@component("bm25")
def _run_bm25(query, query_type: str, top_n: int):
    """
    Chiến lược 1: BM25 (sparse, dựa trên tần suất từ khoá).
    Trả về: list các tuple (id, score), đã sắp theo score giảm dần.
    """
    if query_type != "text":
        # BM25 cần TỪ để so khớp -- ảnh không có "từ" nào cả.
        # Trả rỗng thay vì raise lỗi, để _fuse_all() tự bỏ qua danh sách rỗng
        # này một cách êm đẹp (search bằng ảnh vẫn chạy được, chỉ mất nhánh BM25).
        return []

    tokens = query.lower().split()   # tokenize query giống hệt cách tokenize
                                       # dữ liệu ở bước 3 (BẮT BUỘC nhất quán)
    scores = _bm25.get_scores(tokens)   # trả về mảng điểm cho TOÀN BỘ sản phẩm
                                          # trong index (không chỉ top-k)
    # argsort trả về chỉ số sắp XĂNG dần -- [::-1] đảo ngược thành giảm dần,
    # rồi cắt lấy top_n chỉ số điểm cao nhất.
    ranked = np.argsort(scores)[::-1][:top_n]
    return [(_bm25_ids[i], float(scores[i])) for i in ranked]


@component("dense")
def _run_dense(query, query_type: str, top_n: int):
    """
    Chiến lược 2: Dense (dựa trên CLIP embedding + FAISS).
    Trả về: list các tuple (id, score) -- score là cosine similarity.
    """
    # Encode khác nhau tuỳ query_type, nhưng cả 2 cho ra vector CÙNG không
    # gian -- đây chính là "phép màu" của CLIP đã giải thích ở bước 2.
    query_vec = _encode_text(query) if query_type == "text" else _encode_image(query)

    # .search(query_vec, top_n) trả về 2 mảng: khoảng cách/điểm số, và index.
    # FAISS hỗ trợ search NHIỀU query cùng lúc (batch), nên kết quả có thêm
    # 1 chiều -- ở đây ta chỉ search 1 query nên lấy phần tử [0].
    scores, ids = _dense_index.search(query_vec, top_n)
    return list(zip(ids[0].tolist(), scores[0].tolist()))


def rrf_fusion(list_a, list_b, k: int = 60):
    """
    Hợp nhất 2 danh sách (id, score) thành 1 danh sách duy nhất bằng RRF.
    k=60: giá trị Cormack et al. (2009) tìm được qua 1 thí nghiệm pilot nhỏ
    và giữ nguyên cho các thí nghiệm sau -- CHÍNH TÁC GIẢ ghi rõ "the choice
    was not critical" (bảng kết quả gốc: MAP gần như không đổi từ k=20 tới
    k=100, dao động 0.2134-0.2147). Dùng lại giá trị này không phải vì nó
    "tối ưu tuyệt đối" mà vì paper gốc đã chỉ ra RRF không nhạy cảm với
    việc chọn k -- không cần tự tinh chỉnh thêm.

    Công thức: mỗi tài liệu được cộng dồn điểm 1/(k + rank + 1) cho MỖI
    danh sách nó xuất hiện -- xuất hiện càng cao (rank nhỏ) và càng nhiều
    danh sách thì điểm càng cao.
    """
    fused = {}   # dict: id -> điểm RRF cộng dồn

    for rank, (doc_id, _) in enumerate(list_a):
        # .get(doc_id, 0.0): nếu id chưa từng xuất hiện, coi như đang có 0.0 điểm
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, (doc_id, _) in enumerate(list_b):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    # Sắp xếp theo điểm giảm dần -- trả về list (id, score) giống định dạng đầu vào
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _fuse_all(results_list, rrf_k: int = 60):
    """
    Hợp nhất N danh sách (không giới hạn chỉ 2) -- để nếu sau này thêm
    component thứ 3 (ví dụ color-histogram similarity), code KHÔNG cần sửa.
    Cách làm: fusion lần lượt từng cặp một (fold), giống hàm reduce().

    rrf_k: truyền thẳng xuống rrf_fusion() -- xem giải thích ở đó. Tham số
           hoá ra ngoài (thay vì hardcode) để step10_sweep_rrf.py sweep được
           nhiều giá trị k khác nhau qua search(), không cần sửa code ở đây.
    """
    non_empty = [r for r in results_list if r]   # bỏ qua danh sách rỗng
                                                    # (ví dụ BM25 trả [] khi query là ảnh)
    if len(non_empty) == 0:
        return []
    if len(non_empty) == 1:
        return non_empty[0]   # chỉ 1 component bật -- không cần fusion, trả thẳng
    fused = non_empty[0]
    for other in non_empty[1:]:
        fused = rrf_fusion(fused, other, k=rrf_k)
    return fused


def search(query, query_type: str, k: int = 10, components=None, top_n: int = 50,
           use_rerank: bool = False, rrf_k: int = 60):
    """
    Hàm CHÍNH -- được gọi từ app.py (API thật) và step8_run_ablation.py (ablation).

    query:       str (nếu query_type="text") hoặc PIL.Image (nếu query_type="image")
    query_type:  "text" hoặc "image"
    k:           số kết quả CUỐI CÙNG muốn trả về (top-k hiển thị cho người dùng)
    components:  list tên component muốn BẬT, ví dụ ["bm25", "dense"].
                 None -> mặc định bật TẤT CẢ component đã đăng ký trong registry.
                 Đây chính là input mà UI checkbox / ablation script gửi vào.
    top_n:       số ứng viên lấy từ MỖI nhánh TRƯỚC KHI fusion (rộng hơn k để
                 fusion có đủ "nguyên liệu" chọn lọc, không bị cắt cụt quá sớm)
    use_rerank:  True -> sau khi fusion, chấm lại điểm bằng cross-encoder
                 (xem hàm rerank() bên dưới) trước khi cắt xuống k. False
                 (mặc định) -> giữ nguyên hành vi cũ, không đổi gì cho A1
                 gốc -- đây là lý do use_rerank có default False, để không
                 phá code cũ đang chạy khi thêm tính năng mới.
    rrf_k:       tham số k của công thức RRF (xem rrf_fusion()). Mặc định 60
                 giữ nguyên hành vi gốc -- chỉ đổi khi cố ý sweep (xem
                 step10_sweep_rrf.py) để đo độ nhạy của fusion với k.
    """
    if components is None:
        components = list(COMPONENT_REGISTRY.keys())   # bật hết nếu không chỉ định

    # Gọi lần lượt từng component được yêu cầu, bỏ qua tên không hợp lệ
    # (không raise lỗi -- để UI có gửi nhầm tên cũng không sập cả hệ thống).
    results_by_component = []
    for name in components:
        fn = COMPONENT_REGISTRY.get(name)
        if fn is None:
            continue
        results_by_component.append(fn(query, query_type, top_n))

    fused = _fuse_all(results_by_component, rrf_k=rrf_k)

    # QUAN TRỌNG: nếu use_rerank=True, KHÔNG cắt xuống k ngay -- lấy nguyên
    # top_n để rerank có đủ ứng viên chọn lọc lại (xem docstring hàm rerank()
    # phía dưới, giải thích tại sao cắt sớm sẽ làm mất ý nghĩa của rerank).
    pool = fused if use_rerank else fused[:k]

    # Cắt xuống đúng k kết quả cuối cùng, rồi "làm giàu" từ id -> thông tin
    # đầy đủ (ảnh, caption) để trả về cho client -- client không cần biết
    # gì về id nội bộ, chỉ cần nhận thẳng thông tin hiển thị được.
    results = []
    for doc_id, score in pool:
        row = _catalog[doc_id]
        results.append({
            "id": doc_id,
            "image_path": row["image_path"],
            "caption": row["caption"],
            "score": round(float(score), 4)   # làm tròn cho gọn khi hiển thị/log
        })

    if use_rerank and query_type == "text":
        # Rerank CHỈ áp dụng được cho query dạng text -- cross-encoder cần
        # cặp (text, text) để so sánh. Query bằng ảnh không có "câu chữ" để
        # ghép cặp, nên bỏ qua rerank trong trường hợp đó (giữ nguyên kết
        # quả từ BM25/Dense/RRF, không báo lỗi -- degrade êm ái).
        results = rerank(query, results)

    return results[:k]


def available_components():
    """
    Trả về danh sách tên component đã đăng ký -- dùng bởi app.py để expose
    qua API (/api/v1/components), để frontend TỰ SINH checkbox thay vì
    hardcode tên trong HTML.
    """
    return list(COMPONENT_REGISTRY.keys())


# ---------------------------------------------------------------------------
# RERANK — các hàm dưới đây nhận danh sách ứng viên đã có (sau fusion) và
# chỉ sắp xếp lại, không tự tìm ứng viên mới. Vì chạy tuần tự sau fusion
# nên chúng không đăng ký vào COMPONENT_REGISTRY như BM25/Dense.
#
# Mọi hàm rerank nhận và trả về list dict {id, image_path, caption, score}.
# ---------------------------------------------------------------------------

# Cross-encoder pointwise (Nogueira & Cho, 2019, arXiv:1901.04085).
# Paper gốc dùng BERT-Large; ở đây dùng MiniLM đã distill để chạy được trên CPU.

_reranker = None   # lazy load: chỉ tải model rerank khi THỰC SỰ được gọi tới,
                     # vì rerank là tính năng tuỳ chọn -- không phải ai cũng bật


def _get_reranker():
    """Tải cross-encoder 1 lần duy nhất, dùng lại cho các lần rerank sau."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query: str, candidates: list):
    """
    candidates: list dict đã có {id, image_path, caption, score} -- LẤY TỪ
                DANH SÁCH RỘNG (vd top-50 sau fusion), KHÔNG PHẢI top-10 đã
                cắt gọn -- rerank cần đủ "nguyên liệu" để chọn lọc lại, nếu
                đưa vào danh sách đã cắt hẹp thì rerank chỉ sắp xếp lại thứ
                tự trong 10 cái sẵn có, mất hết ý nghĩa "tìm lại ứng viên tốt
                mà fusion xếp hạng thấp bị bỏ lỡ".

    Cách hoạt động: cross-encoder nhận ĐỒNG THỜI cặp (query, caption) --
    khác hẳn BM25/Dense (mỗi cái encode RIÊNG query và document rồi mới so
    sánh gián tiếp qua similarity). Nhận đồng thời cho phép model "chú ý"
    (attention) qua lại giữa 2 chuỗi -- CHÍNH XÁC HƠN nhưng CHẬM HƠN nhiều
    (không thể tính sẵn trước như FAISS -- phải chạy model cho MỖI CẶP một).
    Đây là lý do rerank chỉ áp dụng cho danh sách đã được BM25/Dense thu hẹp
    trước (vài chục ứng viên), không áp dụng cho toàn bộ catalog.
    """
    model = _get_reranker()
    pairs = [[query, c["caption"]] for c in candidates]
    scores = model.predict(pairs)   # điểm càng cao càng liên quan

    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 4)

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


# ---------------------------------------------------------------------------
# RankGPT — Sun et al. (EMNLP 2023), "Is ChatGPT Good at Search? Investigating
# Large Language Models as Re-Ranking Agents". Xem docs/BAO_CAO_A2.md.
#
# Cài theo rank_gpt.py của github.com/sunnweiwei/RankGPT:
#   - Prompt chat nhiều lượt ở Appendix A.5: mỗi ứng viên là một lượt user,
#     lượt cuối yêu cầu LLM trả về thứ tự dạng [2] > [1] > ...
#   - temperature = 0, mỗi passage cắt còn tối đa 300 từ.
#   - Đọc output: lấy mọi số, bỏ số trùng và số ngoài phạm vi, ứng viên bị
#     thiếu được nối vào cuối theo thứ tự cũ (footnote 9 của paper).
#   - Sliding window từ cuối danh sách lên đầu (mục 3.2), window = 20,
#     step = 10 như mục 6.1.
#
# Khác bản gốc: LLM chạy local qua Ollama (/api/chat) thay vì OpenAI API.
# ---------------------------------------------------------------------------

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
RANKGPT_WINDOW_SIZE = 20
RANKGPT_STEP = 10
RANKGPT_MAX_WORDS = 300


def _rankgpt_messages(query: str, captions: list):
    """Prompt permutation generation dạng chat, giữ nguyên câu chữ của Appendix A.5."""
    num = len(captions)
    messages = [
        {"role": "system",
         "content": "You are RankGPT, an intelligent assistant that can rank passages "
                    "based on their relevancy to the query."},
        {"role": "user",
         "content": f"I will provide you with {num} passages, each indicated by number "
                    f"identifier []. Rank them based on their relevance to query: {query}."},
        {"role": "assistant", "content": "Okay, please provide the passages."},
    ]
    for i, caption in enumerate(captions, start=1):
        passage = " ".join(caption.split()[:RANKGPT_MAX_WORDS])
        messages.append({"role": "user", "content": f"[{i}] {passage}"})
        messages.append({"role": "assistant", "content": f"Received passage [{i}]"})
    messages.append({
        "role": "user",
        "content": f"Search Query: {query}.\nRank the {num} passages above based on their "
                   "relevance to the search query. The passages should be listed in descending "
                   "order using identifiers, and the most relevant passages should be listed "
                   "first, and the output format should be [] > [], e.g., [1] > [2]. Only "
                   "response the ranking results, do not say any word or explain.",
    })
    return messages


def _ollama_chat(messages: list, llm_model: str) -> str:
    import requests

    response = requests.post(OLLAMA_CHAT_URL, json={
        "model": llm_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},
    })
    response.raise_for_status()
    return response.json()["message"]["content"]


def _parse_permutation(raw_output: str, num: int):
    """
    Chuyển output của LLM thành thứ tự đủ `num` ứng viên (index 0-based).

    Trả về (order, stats). stats đếm lỗi theo cách phân loại của Table 10:
        repetition   -- số lần một số thứ tự bị lặp lại
        missing      -- số ứng viên không có trong output (được nối vào cuối)
        out_of_range -- số xuất hiện trong output nhưng không phải số thứ tự hợp lệ
        rejection    -- 1 nếu output không có số thứ tự hợp lệ nào (LLM từ chối
                        hoặc trả lời lạc đề); khi đó không tính missing
    """
    import re

    stats = {"repetition": 0, "missing": 0, "out_of_range": 0, "rejection": 0}
    order, seen = [], set()
    for token in re.findall(r"\d+", raw_output):
        idx = int(token) - 1
        if not 0 <= idx < num:
            stats["out_of_range"] += 1
        elif idx in seen:
            stats["repetition"] += 1
        else:
            order.append(idx)
            seen.add(idx)

    missing = [i for i in range(num) if i not in seen]
    if order:
        stats["missing"] = len(missing)
    else:
        stats["rejection"] = 1
    return order + missing, stats


def rankgpt_rerank_with_stats(query: str, candidates: list, llm_model: str = "llama3.1",
                              window_size: int = RANKGPT_WINDOW_SIZE, step: int = RANKGPT_STEP):
    """
    Rerank bằng sliding window từ cuối lên đầu. Ví dụ 30 ứng viên, window 20,
    step 10: xếp lại [10:30] trước, rồi [0:20] -- ứng viên tốt ở cuối được
    đẩy dần lên đầu.

    Trả về (candidates đã xếp lại, stats cộng dồn qua mọi cửa sổ).
    """
    ranked = list(candidates)
    totals = {"windows": 0, "repetition": 0, "missing": 0, "out_of_range": 0, "rejection": 0}

    end = len(ranked)
    start = max(0, end - window_size)
    while end > 0:
        window = ranked[start:end]
        raw_output = _ollama_chat(_rankgpt_messages(query, [c["caption"] for c in window]), llm_model)
        order, stats = _parse_permutation(raw_output, len(window))
        ranked[start:end] = [window[i] for i in order]

        totals["windows"] += 1
        for key, value in stats.items():
            totals[key] += value

        if start == 0:
            break
        end -= step
        start = max(0, start - step)

    return ranked, totals


def rankgpt_rerank(query: str, candidates: list, llm_model: str = "llama3.1"):
    """Như rankgpt_rerank_with_stats() nhưng chỉ trả về danh sách đã xếp lại."""
    ranked, _ = rankgpt_rerank_with_stats(query, candidates, llm_model=llm_model)
    return ranked


# ---------------------------------------------------------------------------
# Reranker distill từ ChatGPT — mục 4 và 7 của paper RankGPT.
#
# Cross-encoder DeBERTa được huấn luyện (RankNet loss) để bắt chước thứ tự
# do ChatGPT sinh ra cho 10K query MS MARCO. Checkpoint "deberta-10k-rank_net"
# tải thủ công ở mục "Download data and model" của repo RankGPT, giải nén
# vào DISTILLED_RERANKER_PATH (thư mục chứa config.json).
#
# Cách nạp và chấm điểm theo specialization.py: AutoModelForSequenceClassification
# với 1 nhãn, điểm = logit, ghép (query, passage) với max_length = 500.
# ---------------------------------------------------------------------------

DISTILLED_RERANKER_PATH = "models/deberta-10k-rank_net"
# save_pretrained() trong specialization.py chỉ lưu model; nếu thư mục không có
# tokenizer thì dùng tokenizer của model gốc trong lệnh huấn luyện ở README.
DISTILLED_TOKENIZER_FALLBACK = "microsoft/deberta-v3-base"

_distilled_reranker = None


def _get_distilled_reranker():
    global _distilled_reranker
    if _distilled_reranker is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(DISTILLED_RERANKER_PATH)
        except (OSError, ValueError):
            tokenizer = AutoTokenizer.from_pretrained(DISTILLED_TOKENIZER_FALLBACK)
        model = AutoModelForSequenceClassification.from_pretrained(DISTILLED_RERANKER_PATH)
        _distilled_reranker = (tokenizer, model.to(_device).eval())
    return _distilled_reranker


def distilled_rerank(query: str, candidates: list):
    tokenizer, model = _get_distilled_reranker()
    inputs = tokenizer([query] * len(candidates), [c["caption"] for c in candidates],
                       padding=True, truncation=True, max_length=500,
                       return_tensors="pt").to(_device)
    with torch.no_grad():
        scores = model(**inputs).logits[:, 0].tolist()

    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 4)

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
