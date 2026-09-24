"""
Lõi truy hồi: BM25 + Dense (CLIP + FAISS), gộp bằng RRF, rerank tuỳ chọn.

Query text hoặc ảnh -> list dict {id, image_path, caption, score}.
Không chạy trực tiếp; được import bởi app.py, rag_core.py và các script step*.
Cần có sẵn data/catalog.jsonl, data/dense.index, data/bm25.pkl (step1-3).
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
           use_rerank: bool = False, rrf_k: int = 60):
    """
    query / query_type: str với "text", PIL.Image với "image".
    k:          số kết quả trả về.
    components: tên component cần bật, vd ["bm25", "dense"]; None = bật hết, tên lạ bị bỏ qua.
    top_n:      số ứng viên lấy từ mỗi nhánh trước khi gộp.
    use_rerank: rerank toàn bộ pool sau RRF bằng cross-encoder rồi mới cắt k (chỉ với text).
    rrf_k:      hằng số k của RRF.
    """
    if components is None:
        components = list(COMPONENT_REGISTRY.keys())

    results_by_component = []
    for name in components:
        fn = COMPONENT_REGISTRY.get(name)
        if fn is None:
            continue
        results_by_component.append(fn(query, query_type, top_n))

    fused = _fuse_all(results_by_component, rrf_k=rrf_k)

    # Khi rerank, giữ cả pool để rerank có thể kéo ứng viên hạng thấp lên top-k.
    pool = fused if use_rerank else fused[:k]

    results = []
    for doc_id, score in pool:
        row = _catalog[doc_id]
        results.append({
            "id": doc_id,
            "image_path": row["image_path"],
            "caption": row["caption"],
            "score": round(float(score), 4)
        })

    if use_rerank and query_type == "text":
        # Cross-encoder cần cặp (text, text) nên query ảnh bỏ qua rerank.
        results = rerank(query, results)

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
