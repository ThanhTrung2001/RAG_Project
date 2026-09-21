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


def _fuse_all(results_list):
    """
    Hợp nhất N danh sách (không giới hạn chỉ 2) -- để nếu sau này thêm
    component thứ 3 (ví dụ color-histogram similarity), code KHÔNG cần sửa.
    Cách làm: fusion lần lượt từng cặp một (fold), giống hàm reduce().
    """
    non_empty = [r for r in results_list if r]   # bỏ qua danh sách rỗng
                                                    # (ví dụ BM25 trả [] khi query là ảnh)
    if len(non_empty) == 0:
        return []
    if len(non_empty) == 1:
        return non_empty[0]   # chỉ 1 component bật -- không cần fusion, trả thẳng
    fused = non_empty[0]
    for other in non_empty[1:]:
        fused = rrf_fusion(fused, other)
    return fused


def search(query, query_type: str, k: int = 10, components=None, top_n: int = 50, use_rerank: bool = False):
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

    fused = _fuse_all(results_by_component)

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
# RERANK (cross-encoder, pointwise) -- dựa trên Nogueira & Cho (2019),
# "Passage Re-ranking with BERT", arXiv:1901.04085 -- paper KHÔNG phải paper
# chọn cho A2 (A2 chọn RankGPT, xem hàm rankgpt_rerank() bên dưới và
# BAO_CAO_A2.md) nhưng vẫn là nền tảng lý thuyết đúng cho kỹ thuật
# cross-encoder pointwise này -- trích dẫn ở đây cho đầy đủ nguồn gốc kỹ
# thuật, không cần đọc thêm báo cáo riêng cho phần này.
#
# TẠI SAO KHÔNG ĐĂNG KÝ CHUNG COMPONENT_REGISTRY VỚI BM25/DENSE?
#     BM25 và Dense là 2 nhánh CHẠY SONG SONG rồi fusion (mỗi nhánh tự tìm
#     ứng viên riêng). Rerank thì khác hẳn về bản chất: nó chạy TUẦN TỰ,
#     SAU KHI đã có danh sách ứng viên từ fusion -- không tự tìm gì mới, chỉ
#     CHẤM LẠI ĐIỂM cho danh sách đã có. Trộn 2 loại logic khác nhau vào
#     chung 1 registry sẽ làm sai ý nghĩa của "component chạy song song".
#
# TẠI SAO PAPER GỐC DÙNG BERT ĐẦY ĐỦ NHƯNG Ở ĐÂY DÙNG MiniLM (NHỎ HƠN)?
#     Paper gốc (2019) dùng BERT-Large để đạt SOTA tuyệt đối, nhưng đó là
#     bài toán research chấp nhận chi phí tính toán lớn. Ở đây dùng bản
#     MiniLM đã distill lại (nhỏ hơn nhiều, dùng được trên CPU trong thời
#     gian hợp lý) -- đánh đổi 1 phần độ chính xác lấy tốc độ, PHÙ HỢP với
#     yêu cầu thực tế của A1 (chạy demo trực tiếp, không có GPU server riêng).
# ---------------------------------------------------------------------------

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
# RANKGPT RERANK -- dựa trên Sun et al. (2023) "Is ChatGPT Good at Search?
# Investigating Large Language Models as Re-Ranking Agents" (EMNLP 2023,
# tên kỹ thuật phổ biến: RankGPT). Xem BAO_CAO_A2.md để hiểu đầy đủ lý
# thuyết + kết quả gốc.
#
# KHÁC GÌ VỚI rerank() (CROSS-ENCODER, Nogueira & Cho) Ở TRÊN?
#     rerank() là "pointwise": chấm ĐIỂM RIÊNG cho từng cặp (query, document)
#     rồi sắp xếp theo điểm -- mỗi document được đánh giá ĐỘC LẬP, không biết
#     gì về các document khác trong danh sách.
#
#     rankgpt_rerank() là "listwise": đưa NGUYÊN CẢ DANH SÁCH ứng viên vào
#     1 prompt DUY NHẤT, yêu cầu LLM trả về thứ tự hoán vị (permutation)
#     trực tiếp -- LLM "nhìn thấy" toàn bộ danh sách CÙNG LÚC, có thể so
#     sánh CHÉO giữa các ứng viên với nhau (vd "sản phẩm A phù hợp hơn B vì
#     B tuy cùng loại nhưng đắt hơn nhiều"), điều mà cross-encoder (chỉ so
#     mỗi document với query, không so document với nhau) không làm được.
#
# TẠI SAO THAM SỐ HOÁ llm_model?
#     Đây chính là điểm mấu chốt để làm ablation như ví dụ giảng viên đưa ra
#     ("RankGPT dở vì dùng model mã nguồn mở kém hiệu quả"): paper gốc dùng
#     GPT-4/ChatGPT (mạnh, biết làm theo hướng dẫn tốt). Nếu đổi sang model
#     nhỏ/yếu hơn (vd qwen2.5:3b so với llama3.1), model YẾU dễ trả về sai
#     định dạng, bỏ sót số, hoặc lặp số -- gây SAI LỆCH kết quả rerank. Có
#     tham số này để CHẠY THỰC NGHIỆM so sánh, không chỉ khẳng định suông.
#
# TẠI SAO CHỈ 1 CỬA SỔ, KHÔNG "SLIDING WINDOW" ĐẦY ĐỦ NHƯ PAPER GỐC?
#     Paper gốc dùng sliding window để rerank danh sách DÀI HƠN giới hạn
#     ngữ cảnh (context window) của LLM -- trượt cửa sổ ~20 tài liệu qua
#     toàn bộ candidate pool nhiều vòng. Ở quy mô A1 (pool top-50, mỗi
#     caption ngắn), 1 cửa sổ duy nhất vẫn vừa context của các model dùng
#     qua Ollama -- đơn giản hoá CÓ CHỦ ĐÍCH cho phạm vi đồ án, không phải
#     thiếu sót không biết kỹ thuật gốc.
# ---------------------------------------------------------------------------

def rankgpt_rerank(query: str, candidates: list, llm_model: str = "llama3.1"):
    """
    candidates: list dict {id, image_path, caption, score} -- lấy từ pool
                rộng (vd top-50 sau RRF), giống hệt input của rerank() cross-encoder.
    llm_model:  tên model Ollama dùng để rerank -- đổi giá trị này để chạy
                thực nghiệm so sánh model mạnh/yếu (đúng tinh thần paper).
    Trả về: candidates đã sắp xếp lại theo đánh giá của LLM.
    """
    import re
    import requests

    numbered = "\n".join(f"[{i + 1}] {c['caption']}" for i, c in enumerate(candidates))

    # Prompt "permutation generation" -- yêu cầu LLM trả về THỨ TỰ, không
    # phải điểm số riêng lẻ từng cái (đó sẽ là pointwise, không phải RankGPT).
    prompt = f"""Sắp xếp lại danh sách sản phẩm dưới đây theo mức độ liên quan GIẢM DẦN với câu truy vấn.

Câu truy vấn: "{query}"

Danh sách sản phẩm:
{numbered}

CHỈ trả về đúng 1 dòng liệt kê số thứ tự theo mức độ liên quan giảm dần, cách nhau bằng dấu ">", ví dụ: [3] > [1] > [2]. KHÔNG giải thích gì thêm, KHÔNG bỏ sót số nào."""

    response = requests.post("http://localhost:11434/api/generate", json={
        "model": llm_model, "prompt": prompt, "stream": False
    })
    response.raise_for_status()
    raw_output = response.json()["response"]

    # Parse "[3] > [1] > [2]" thành list index 0-based [2, 0, 1]
    order = [int(x) - 1 for x in re.findall(r"\[(\d+)\]", raw_output)]

    # PHÒNG THỦ QUAN TRỌNG: model YẾU (đúng vấn đề paper/giảng viên nêu) hay
    # trả thiếu số, thừa số, lặp số, hoặc số ngoài phạm vi -- nếu không xử
    # lý, sẽ MẤT ứng viên hoặc lỗi chương trình. Code phải đảm bảo LUÔN trả
    # đủ danh sách gốc, kể cả khi LLM trả lời sai định dạng.
    seen = set()
    reordered = []
    for idx in order:
        if 0 <= idx < len(candidates) and idx not in seen:
            reordered.append(candidates[idx])
            seen.add(idx)
    for i, c in enumerate(candidates):
        if i not in seen:
            reordered.append(c)   # bù các ứng viên LLM bỏ sót, giữ nguyên thứ tự gốc

    return reordered
