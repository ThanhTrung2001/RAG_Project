"""
Metric đánh giá retrieval tự cài: Recall@k, nDCG@k (nhị phân và đa mức 0/1/2), MRR.

evaluate_all() chạy một hàm search trên eval set -> dict metric trung bình + latency.
Không chạy trực tiếp; được import bởi các script step*.
"""
import math


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Tỉ lệ id đúng nằm trong top-k. None nếu query không có ground truth."""
    if not relevant_ids:
        # None để evaluate_all() bỏ qua câu này thay vì tính là 0.
        return None

    top_k = set(retrieved_ids[:k])
    hit = len(top_k & set(relevant_ids))
    return hit / len(relevant_ids)


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """nDCG@k với relevance nhị phân (đúng=1, sai=0)."""
    relevant_ids = set(relevant_ids)

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)   # i từ 0 nên log2(i + 2) = log2(rank + 1)

    # IDCG: mọi id đúng xếp ở đầu, tối đa k id.
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        # relevant_ids rỗng hoặc k=0
        return None
    return dcg / idcg


def mrr(retrieved_ids, relevant_ids):
    """Reciprocal rank của id đúng đầu tiên (1/hạng); 0.0 nếu không có, None nếu không có ground truth."""
    if not relevant_ids:
        return None
    relevant_ids = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k_graded(retrieved_ids, relevant_ids, k, catalog):
    """
    nDCG@k đa mức: 2 = đúng id, 1 = cùng category với một id đúng, 0 = khác.
    catalog: dict {id: row}, dùng để tra category lúc chấm.
    """
    relevant_ids = set(relevant_ids)
    relevant_categories = {
        catalog[rid]["category"] for rid in relevant_ids if rid in catalog
    }

    def relevance(doc_id):
        if doc_id in relevant_ids:
            return 2.0
        row = catalog.get(doc_id)
        if row and row["category"] in relevant_categories:
            return 1.0
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        dcg += relevance(doc_id) / math.log2(i + 2)

    # IDCG: các id đúng (rel=2) đứng đầu, phần còn lại của top-k giả định là rel=1.
    n_perfect = min(len(relevant_ids), k)
    idcg = sum(2.0 / math.log2(i + 2) for i in range(n_perfect))
    idcg += sum(1.0 / math.log2(i + 2) for i in range(n_perfect, k))

    if idcg == 0:
        return None
    return dcg / idcg


def evaluate_all(search_fn, eval_set, k=10, catalog=None, **search_kwargs):
    """
    Chạy search_fn trên từng câu của eval_set, trả về trung bình Recall@k, nDCG@k, MRR,
    nDCG@k đa mức (chỉ khi có catalog) và latency (ms: avg, p50, p95).
    eval_set: list dict {"query", "query_type", "relevant_ids"}.
    search_kwargs: truyền thẳng vào search_fn, vd components=["bm25"].
    """
    import time

    recalls, ndcgs, mrrs, ndcgs_graded = [], [], [], []
    latencies = []   # ms

    for item in eval_set:
        t0 = time.perf_counter()
        results = search_fn(item["query"], item["query_type"], k=k, **search_kwargs)
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved_ids = [r["id"] for r in results]

        r = recall_at_k(retrieved_ids, item["relevant_ids"], k)
        n = ndcg_at_k(retrieved_ids, item["relevant_ids"], k)
        m = mrr(retrieved_ids, item["relevant_ids"])

        if r is not None:
            recalls.append(r)
        if n is not None:
            ndcgs.append(n)
        if m is not None:
            mrrs.append(m)
        if catalog is not None:
            ng = ndcg_at_k_graded(retrieved_ids, item["relevant_ids"], k, catalog)
            if ng is not None:
                ndcgs_graded.append(ng)

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else None
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else None

    return {
        "recall@k": sum(recalls) / len(recalls) if recalls else None,
        "ndcg@k": sum(ndcgs) / len(ndcgs) if ndcgs else None,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else None,
        "ndcg@k_graded": sum(ndcgs_graded) / len(ndcgs_graded) if ndcgs_graded else None,
        "latency_ms_avg": sum(latencies) / len(latencies) if latencies else None,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "n_queries": len(eval_set)
    }
