"""
Lõi truy hồi: BM25 + Dense (CLIP + FAISS), gộp bằng RRF, rerank tuỳ chọn.

Query text hoặc ảnh -> list dict {id, image_path, caption, score}.
Không chạy trực tiếp; được import bởi app.py, rag_core.py và các script step*.
Cần có sẵn data/catalog.jsonl, data/dense.index, data/bm25.pkl (step1-3).
"""
import json
import os
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

# Nạp model và index một lần lúc import, dùng chung cho mọi lần search.
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = CLIPModel.from_pretrained(MODEL_NAME).to(_device).eval()
# use_fast=False: fast image processor ở transformers mới làm get_image_features() trả sai kiểu.
_processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=False)

_dense_index = faiss.read_index(DENSE_INDEX_PATH)

with open(BM25_INDEX_PATH, "rb") as f:
    _bm25_data = pickle.load(f)
_bm25 = _bm25_data["bm25"]
_bm25_ids = _bm25_data["ids"]       # vị trí i trong BM25 -> id sản phẩm

_catalog = {}                        # {id: row} để tra image_path/caption theo id
with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        _catalog[row["id"]] = row


# CLIP (Radford et al., 2021): text và ảnh được encode vào cùng một không gian.
def _encode_text(text: str) -> np.ndarray:
    """Text -> vector CLIP 512 chiều, đã chuẩn hoá L2."""
    with torch.no_grad():
        inputs = _processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(_device)
        feat = _model.get_text_features(**inputs)
        # Chuẩn hoá như lúc build index để inner product = cosine.
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")


def _encode_image(image: Image.Image) -> np.ndarray:
    """PIL.Image -> vector CLIP 512 chiều, đã chuẩn hoá L2."""
    with torch.no_grad():
        inputs = _processor(images=[image], return_tensors="pt").to(_device)
        feat = _model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")


# Registry: tên component -> hàm retrieval, dùng chung cho API và ablation.
COMPONENT_REGISTRY = {}


def component(name: str):
    """Decorator đăng ký hàm retrieval vào COMPONENT_REGISTRY dưới tên `name`."""
    def wrapper(func):
        COMPONENT_REGISTRY[name] = func
        return func
    return wrapper


@component("bm25")
def _run_bm25(query, query_type: str, top_n: int):
    """BM25 trên search_text. Trả về top_n (id, score) giảm dần; [] nếu query là ảnh."""
    if query_type != "text":
        return []

    tokens = query.lower().split()   # phải cùng cách tách từ với step3
    scores = _bm25.get_scores(tokens)
    ranked = np.argsort(scores)[::-1][:top_n]
    return [(_bm25_ids[i], float(scores[i])) for i in ranked]


@component("dense")
def _run_dense(query, query_type: str, top_n: int):
    """CLIP + FAISS. Trả về top_n (id, cosine) giảm dần; query là text hoặc ảnh."""
    query_vec = _encode_text(query) if query_type == "text" else _encode_image(query)

    # Vị trí vector trong FAISS chính là id sản phẩm (catalog id = 0..N-1).
    scores, ids = _dense_index.search(query_vec, top_n)
    return list(zip(ids[0].tolist(), scores[0].tolist()))


def rrf_fusion(list_a, list_b, k: int = 60):
    """
    Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum 1/(k + rank),
    rank tính từ 1. Chỉ dùng thứ hạng nên không cần chuẩn hoá điểm BM25 và cosine.
    """
    fused = {}

    for rank, (doc_id, _) in enumerate(list_a):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, (doc_id, _) in enumerate(list_b):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _fuse_all(results_list, rrf_k: int = 60):
    """Gộp N danh sách bằng RRF theo từng cặp, bỏ qua danh sách rỗng."""
    non_empty = [r for r in results_list if r]
    if len(non_empty) == 0:
        return []
    if len(non_empty) == 1:
        return non_empty[0]
    fused = non_empty[0]
    for other in non_empty[1:]:
        fused = rrf_fusion(fused, other, k=rrf_k)
    return fused


def search(query, query_type: str, k: int = 10, components=None, top_n: int = 50,
           use_rerank: bool = False, rrf_k: int = 60, reranker: str = None,
           rerank_pool: int = None):
    """
    query / query_type: str với "text", PIL.Image với "image".
    k:           số kết quả trả về.
    components:  tên component cần bật, vd ["bm25", "dense"]; None = bật hết, tên lạ bị bỏ qua.
    top_n:       số ứng viên lấy từ mỗi nhánh trước khi gộp.
    use_rerank:  True tương đương reranker="cross_encoder".
    rrf_k:       hằng số k của RRF.
    reranker:    None, "cross_encoder" hoặc "deberta" (xem RERANKERS); chỉ áp dụng cho text.
    rerank_pool: số ứng viên đầu sau RRF đưa vào rerank; None = toàn bộ pool.
    """
    if use_rerank and reranker is None:
        reranker = "cross_encoder"
    if reranker is not None and reranker not in RERANKERS:
        raise ValueError(f"reranker không hợp lệ: {reranker}. Chọn một trong {list(RERANKERS)}")
    if components is None:
        components = list(COMPONENT_REGISTRY.keys())

    results_by_component = []
    for name in components:
        fn = COMPONENT_REGISTRY.get(name)
        if fn is None:
            continue
        results_by_component.append(fn(query, query_type, top_n))

    fused = _fuse_all(results_by_component, rrf_k=rrf_k)

    # Cross-encoder cần cặp (text, text) nên query ảnh bỏ qua rerank.
    do_rerank = reranker is not None and query_type == "text"

    # Khi rerank, giữ cả pool (hoặc rerank_pool) để rerank có thể kéo ứng viên hạng thấp lên top-k.
    if do_rerank:
        pool = fused[:rerank_pool] if rerank_pool else fused
    else:
        pool = fused[:k]

    results = []
    for doc_id, score in pool:
        row = _catalog[doc_id]
        results.append({
            "id": doc_id,
            "image_path": row["image_path"],
            "caption": row["caption"],
            "score": round(float(score), 4)
        })

    if do_rerank:
        results = RERANKERS[reranker](query, results)

    return results[:k]


def available_components():
    """Tên các component đã đăng ký (frontend dùng để sinh checkbox)."""
    return list(COMPONENT_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Rerank: sắp xếp lại danh sách ứng viên sau fusion, không tìm ứng viên mới.
# Mọi hàm rerank nhận và trả về list dict {id, image_path, caption, score}.
# ---------------------------------------------------------------------------

# Cross-encoder pointwise (Nogueira & Cho, 2019, arXiv:1901.04085).
# Paper gốc dùng BERT-Large; ở đây dùng MiniLM đã distill để chạy được trên CPU.

_reranker = None   # lazy load: chỉ tải khi có lệnh rerank đầu tiên


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query: str, candidates: list):
    """Chấm cặp (query, caption) bằng cross-encoder, thêm rerank_score và sắp giảm dần."""
    model = _get_reranker()
    pairs = [[query, c["caption"]] for c in candidates]
    scores = model.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 4)

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


# ---------------------------------------------------------------------------
# Reranker distill từ ChatGPT — Sun et al. (EMNLP 2023), "Is ChatGPT Good at
# Search?", mục 4 và 7 (permutation distillation). Xem docs/BAO_CAO_A2.md.
#
# Cross-encoder DeBERTa-v3-base được huấn luyện (RankNet loss) để bắt chước thứ tự
# do ChatGPT sinh ra cho 10K query MS MARCO. Checkpoint "deberta-10k-rank_net"
# tải thủ công ở mục "Download data and model" của github.com/sunnweiwei/RankGPT,
# giải nén vào DISTILLED_RERANKER_PATH (thư mục chứa config.json).
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


# Reranker chọn được qua search(reranker=...): cross-encoder MiniLM (A1) hoặc DeBERTa distill (A2).
RERANKERS = {
    "cross_encoder": rerank,
    "deberta": distilled_rerank,
}


def available_rerankers():
    """Tên các reranker dùng được; "deberta" chỉ có khi đã tải checkpoint vào DISTILLED_RERANKER_PATH."""
    return [name for name in RERANKERS
            if name != "deberta" or os.path.isdir(DISTILLED_RERANKER_PATH)]
