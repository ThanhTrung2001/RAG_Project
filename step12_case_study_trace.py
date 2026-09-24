"""
================================================================================
CASE STUDY — TRUY VẾT 1 CÂU QUERY QUA TỪNG TẦNG (mục 6.1-6.4 template A1)
================================================================================

VẤN ĐỀ: step8_run_ablation.py chỉ cho biết KẾT QUẢ CUỐI CÙNG (sau fusion,
sau rerank) có đúng hay không -- KHÔNG cho biết BM25 xếp id đúng ở hạng
bao nhiêu TRƯỚC KHI vào RRF, Dense xếp hạng bao nhiêu, RRF gộp lại thành
hạng bao nhiêu, và rerank có ĐẨY LÊN hay ĐẨY XUỐNG so với hạng RRF.
Không có công cụ này thì không viết được case study "tầng nào cứu/tầng
nào làm hỏng" cho mục 6 của báo cáo.

CÁCH DÙNG: sửa CASES bên dưới -- mỗi case là 1 (query, query_type, target_id).
Mặc định điền sẵn vài ví dụ đã tìm thấy ở Bước 6 (error analysis notebook):
    - id=888 "rubber bag..." : bm25 đúng một mình nhưng hybrid làm rớt hạng
      (case RRF "làm loãng" kết quả BM25 đúng)
    - id=241 "rug for stairs...": bm25 một mình SAI, dense/hybrid cứu được
      (case dense bổ khuyết BM25)
    - id=1681 "device that helps...": SAI ở mọi tầng (case khó nhất)
Thêm/sửa case khác tuỳ ý -- lấy target_id + query trực tiếp từ data/eval_set.jsonl.

CHẠY: python step12_case_study_trace.py
"""
import json

from search_core import _run_bm25, _run_dense, rrf_fusion, rerank, _catalog

TOP_N = 50   # khớp mặc định của search() -- pool trước khi fusion/rerank

CASES = [
    {"query": "rubber bag you fill with hot water to stay warm", "query_type": "text", "target_id": 888},
    {"query": "rug for stairs so people don't slip", "query_type": "text", "target_id": 241},
    {"query": "device that helps someone relearn to walk after an injury", "query_type": "text", "target_id": 1681},
    {"query": "round basket for serving bread", "query_type": "text", "target_id": 1396},
]


def rank_of(pairs, target_id):
    """pairs: list (id, score) đã sắp theo hạng -- trả về (hạng 1-based, score)
    hoặc (None, None) nếu target_id không nằm trong danh sách (rớt khỏi pool)."""
    for rank, (doc_id, score) in enumerate(pairs, start=1):
        if doc_id == target_id:
            return rank, score
    return None, None


def to_result_dicts(pairs):
    """(id, score) -> dict {id, image_path, caption, score} -- định dạng
    rerank() cần (xem docstring rerank() trong search_core.py)."""
    results = []
    for doc_id, score in pairs:
        row = _catalog[doc_id]
        results.append({
            "id": doc_id, "image_path": row["image_path"],
            "caption": row["caption"], "score": round(float(score), 4),
        })
    return results


def trace(query, query_type, target_id):
    bm25_pairs = _run_bm25(query, query_type, TOP_N)
    dense_pairs = _run_dense(query, query_type, TOP_N)

    if bm25_pairs and dense_pairs:
        fused_pairs = rrf_fusion(bm25_pairs, dense_pairs)
    else:
        fused_pairs = bm25_pairs or dense_pairs

    reranked = rerank(query, to_result_dicts(fused_pairs[:TOP_N])) if fused_pairs else []
    reranked_pairs = [(r["id"], r["rerank_score"]) for r in reranked]

    bm25_rank, _ = rank_of(bm25_pairs, target_id)
    dense_rank, _ = rank_of(dense_pairs, target_id)
    rrf_rank, _ = rank_of(fused_pairs, target_id)
    rerank_rank, _ = rank_of(reranked_pairs, target_id)

    row_row = _catalog.get(target_id, {})
    print(f"Query: \"{query}\"")
    print(f"Target id={target_id} | caption: {row_row.get('caption')}")
    print(f"  Hạng ở BM25   (top-{TOP_N}): {bm25_rank if bm25_rank else 'KHÔNG có trong pool'}")
    print(f"  Hạng ở Dense  (top-{TOP_N}): {dense_rank if dense_rank else 'KHÔNG có trong pool'}")
    print(f"  Hạng sau RRF  (top-{TOP_N}): {rrf_rank if rrf_rank else 'KHÔNG có trong pool'}")
    print(f"  Hạng sau Rerank(top-{TOP_N}): {rerank_rank if rerank_rank else 'KHÔNG có trong pool'}")

    # Diễn giải TỰ ĐỘNG bước nào "cứu"/"làm hỏng" -- so 2 tầng liền kề.
    def describe_change(before, after, from_name, to_name):
        if before is None and after is None:
            return f"  -> {from_name} và {to_name}: cả 2 đều KHÔNG tìm thấy id đúng."
        if before is None and after is not None:
            return f"  -> {to_name} TÌM LẠI ĐƯỢC id đúng mà {from_name} bỏ lỡ hoàn toàn (hạng {after})."
        if before is not None and after is None:
            return f"  -> {to_name} LÀM RỚT id đúng ra khỏi top-{TOP_N} (đang ở hạng {before} tại {from_name})."
        if after < before:
            return f"  -> {to_name} CẢI THIỆN hạng: {before} -> {after}."
        if after > before:
            return f"  -> {to_name} LÀM TỆ HƠN hạng: {before} -> {after}."
        return f"  -> {to_name} giữ nguyên hạng ({after})."

    print(describe_change(bm25_rank, rrf_rank, "BM25 một mình", "RRF (fusion với Dense)"))
    print(describe_change(dense_rank, rrf_rank, "Dense một mình", "RRF (fusion với BM25)"))
    print(describe_change(rrf_rank, rerank_rank, "RRF", "Rerank (cross-encoder)"))
    print()


def main():
    print(f"Trace {len(CASES)} case study qua từng tầng: BM25 -> Dense -> RRF -> Rerank\n")
    for case in CASES:
        trace(case["query"], case["query_type"], case["target_id"])


if __name__ == "__main__":
    main()
