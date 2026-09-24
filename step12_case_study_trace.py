"""
Truy vết hạng của một id đúng qua từng tầng: BM25 -> Dense -> RRF -> Rerank (cross-encoder).

Sửa CASES bên dưới (query, query_type, target_id lấy từ data/eval_set.jsonl).
Chạy: python step12_case_study_trace.py (cần index từ step1-3).
"""
import json

from search_core import _run_bm25, _run_dense, rrf_fusion, rerank, _catalog

TOP_N = 50   # bằng top_n mặc định của search()

CASES = [
    {"query": "rubber bag you fill with hot water to stay warm", "query_type": "text", "target_id": 888},
    {"query": "rug for stairs so people don't slip", "query_type": "text", "target_id": 241},
    {"query": "device that helps someone relearn to walk after an injury", "query_type": "text", "target_id": 1681},
    {"query": "round basket for serving bread", "query_type": "text", "target_id": 1396},
]


def rank_of(pairs, target_id):
    """Hạng (1-based) và score của target_id trong list (id, score); (None, None) nếu không có."""
    for rank, (doc_id, score) in enumerate(pairs, start=1):
        if doc_id == target_id:
            return rank, score
    return None, None


def to_result_dicts(pairs):
    """(id, score) -> dict {id, image_path, caption, score} cho rerank()."""
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

    def describe_change(before, after, from_name, to_name):
        """So hạng giữa hai tầng liền kề (None = không có trong danh sách)."""
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
