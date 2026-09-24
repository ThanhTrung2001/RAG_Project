"""
================================================================================
ABLATION — SO SÁNH TRƯỜNG DỮ LIỆU ĐƯA VÀO INDEX (mục 5.4 template A1)
================================================================================

CÂU HỎI THỰC NGHIỆM: BM25 index trên "chỉ title" (caption ngắn) có tệ hơn
"title + description" (search_text, đang dùng ở step3_build_index.py) không?
Và thêm nhánh Dense (ảnh) vào (hybrid) cải thiện thêm bao nhiêu nữa?

TẠI SAO KHÔNG SỬA step3_build_index.py ĐỂ CHẠY 3 LẦN?
    step3 ghi ĐÈ data/bm25.pkl mỗi lần chạy -- nếu chạy 3 lần với 3 trường
    khác nhau sẽ mất bản BM25 "title+description" đang dùng cho app.py/A3.
    Script này build 2 chỉ mục BM25 TẠM TRONG RAM (không ghi ra đĩa, không
    đụng tới data/bm25.pkl) để so sánh, giữ nguyên toàn bộ hệ thống thật.

3 CẤU HÌNH:
    1. "title only"              -- BM25 trên riêng caption (tên sản phẩm)
    2. "title + description"     -- BM25 trên search_text (đang dùng thật ở A1)
    3. "title + description + ảnh" -- (2) fusion RRF với Dense (CLIP ảnh),
                                        CHÍNH LÀ cấu hình "hybrid" của step8

CHẠY: python step9_ablation_index_fields.py
YÊU CẦU TRƯỚC: đã có data/catalog.jsonl, data/eval_set.jsonl, data/dense.index
      (không cần data/bm25.pkl vì script tự build BM25 riêng trong RAM).
"""
import json

import numpy as np
from rank_bm25 import BM25Okapi

from search_core import _run_dense, rrf_fusion, _catalog
from metrics import evaluate_all

EVAL_SET_PATH = "data/eval_set.jsonl"
CATALOG_PATH = "data/catalog.jsonl"
K = 10
TOP_N = 50   # số ứng viên lấy từ mỗi nhánh trước khi fusion -- khớp mặc định của search()


def load_eval_set():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_catalog_rows():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def simple_tokenize(text: str):
    """Giống hệt step3_build_index.py -- BẮT BUỘC tokenize nhất quán để so
    sánh công bằng, không phải do field khác mà do cách tách từ khác."""
    return text.lower().split()


def build_bm25(rows, field_getter):
    """
    field_getter: hàm (row) -> str, chọn field nào đưa vào BM25 (caption
    hoặc search_text). Trả về (bm25, ids) giống cấu trúc data/bm25.pkl.
    """
    tokenized = [simple_tokenize(field_getter(r)) for r in rows]
    bm25 = BM25Okapi(tokenized)
    ids = [r["id"] for r in rows]
    return bm25, ids


def bm25_search(bm25, ids, query: str, top_n: int):
    """Y hệt logic _run_bm25() trong search_core.py, viết lại độc lập ở đây
    vì search_core._run_bm25 gắn cứng với self._bm25 (bản title+description
    đã build sẵn ở step3) -- không nhận bm25 tuỳ ý làm tham số."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = np.argsort(scores)[::-1][:top_n]
    return [(ids[i], float(scores[i])) for i in ranked]


def make_search_fn(bm25, ids, fuse_with_dense: bool):
    """
    Trả về 1 hàm search_fn(query, query_type, k, ...) tương thích chữ ký
    với search_core.search() -- để dùng lại NGUYÊN VẸN metrics.evaluate_all()
    (không viết lại vòng lặp tính Recall/nDCG/MRR/Latency lần thứ 2).
    """

    def search_fn(query, query_type, k=K, **_ignored):
        bm25_results = bm25_search(bm25, ids, query, TOP_N)

        if fuse_with_dense:
            dense_results = _run_dense(query, query_type, TOP_N)
            fused = rrf_fusion(bm25_results, dense_results) if dense_results else bm25_results
        else:
            fused = bm25_results

        fused = fused[:k]
        results = []
        for doc_id, score in fused:
            row = _catalog[doc_id]
            results.append({
                "id": doc_id, "image_path": row["image_path"],
                "caption": row["caption"], "score": round(float(score), 4),
            })
        return results

    return search_fn


def main():
    eval_set = load_eval_set()
    rows = load_catalog_rows()
    catalog = {r["id"]: r for r in rows}

    bm25_title, ids = build_bm25(rows, lambda r: r["caption"])
    bm25_full, _ = build_bm25(rows, lambda r: r.get("search_text", r["caption"]))

    configs = [
        ("title only",                 make_search_fn(bm25_title, ids, fuse_with_dense=False)),
        ("title + description",        make_search_fn(bm25_full, ids, fuse_with_dense=False)),
        ("title + description + ảnh",  make_search_fn(bm25_full, ids, fuse_with_dense=True)),
    ]

    print(f"Ablation theo trường dữ liệu index -- {len(eval_set)} câu query, k={K}\n")
    header = f"{'Cấu hình':<28} {'Recall@'+str(K):<10} {'nDCG@'+str(K):<10} {'MRR':<8}"
    print(header)
    print("-" * len(header))

    for name, fn in configs:
        metrics = evaluate_all(search_fn=fn, eval_set=eval_set, k=K, catalog=catalog)

        def fmt(key):
            v = metrics[key]
            return f"{v:.3f}" if v is not None else "N/A"

        print(f"{name:<28} {fmt('recall@k'):<10} {fmt('ndcg@k'):<10} {fmt('mrr'):<8}")

    print("\n-> So 'title only' với 'title + description': đo giá trị của mô tả dài")
    print("   (description) so với chỉ tên sản phẩm khi khớp từ khoá BM25.")
    print("-> So 'title + description' với '... + ảnh': đo đóng góp của nhánh Dense/CLIP")
    print("   khi thêm vào BM25 đã dùng field đầy đủ nhất -- đây chính là hàng")
    print("   'hybrid' trong step8_run_ablation.py, dùng lại để đối chiếu 3 dòng cùng bảng.")


if __name__ == "__main__":
    main()
