"""
Ablation reranker trên catalog A1, cùng một pool ứng viên hybrid (BM25 + Dense + RRF):
    1. Không rerank
    2. Cross-encoder MiniLM (ms-marco-MiniLM-L-6-v2, Nogueira & Cho 2019) -- reranker của A1
    3. DeBERTa distill từ ChatGPT (deberta-10k-rank_net) -- paper A2 (Sun et al., EMNLP 2023),
       mục 4 và 7; bỏ qua nếu chưa có models/deberta-10k-rank_net

Metric: nDCG@1/5/10 (như paper), Recall@10, thời gian rerank trung bình mỗi câu.
Dữ liệu: data/eval_set.jsonl (250 câu, mỗi câu 1 sản phẩm đúng).

CHẠY: python step_ablation_rerank.py
Kết quả in ra màn hình và lưu vào data/rerank_ablation_results.json.
"""
import json
import os
import time

from metrics import ndcg_at_k, recall_at_k
from search_core import DISTILLED_RERANKER_PATH, distilled_rerank, rerank, search

EVAL_SET_PATH = "data/eval_set.jsonl"
RESULTS_PATH = "data/rerank_ablation_results.json"

POOL_SIZE = 30            # số ứng viên hybrid đưa vào rerank; cùng giá trị với rag_core.RERANK_POOL
NDCG_CUTOFFS = (1, 5, 10)
RECALL_K = 10


def load_eval_set():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def build_configs():
    """Mỗi cấu hình là (tên, hàm). Hàm nhận (query, candidates), trả candidates đã xếp lại."""
    configs = [
        ("Không rerank", lambda q, c: c),
        ("Cross-encoder MiniLM", rerank),
    ]
    if os.path.isdir(DISTILLED_RERANKER_PATH):
        configs.append(("DeBERTa distill từ ChatGPT", distilled_rerank))
    else:
        print(f"Bỏ qua DeBERTa distill: không thấy thư mục {DISTILLED_RERANKER_PATH}\n")
    return configs


def run_config(rerank_fn, eval_set):
    ndcgs = {k: [] for k in NDCG_CUTOFFS}
    recalls, latencies = [], []

    for item in eval_set:
        query, relevant_ids = item["query"], item["relevant_ids"]
        candidates = search(query, query_type="text", k=POOL_SIZE,
                            components=["bm25", "dense"], top_n=POOL_SIZE)

        start = time.perf_counter()
        ranked = rerank_fn(query, candidates) if candidates else candidates
        latencies.append(time.perf_counter() - start)

        ranked_ids = [c["id"] for c in ranked]
        for k in NDCG_CUTOFFS:
            ndcgs[k].append(ndcg_at_k(ranked_ids, relevant_ids, k))
        recalls.append(recall_at_k(ranked_ids, relevant_ids, RECALL_K))

    result = {f"ndcg@{k}": mean(ndcgs[k]) for k in NDCG_CUTOFFS}
    result[f"recall@{RECALL_K}"] = mean(recalls)
    result["rerank_latency_sec_avg"] = mean(latencies)
    return result


def fmt(value):
    return "N/A" if value is None else f"{value:.3f}"


def print_table(results):
    metric_keys = [f"ndcg@{k}" for k in NDCG_CUTOFFS] + [f"recall@{RECALL_K}", "rerank_latency_sec_avg"]
    headers = [f"nDCG@{k}" for k in NDCG_CUTOFFS] + [f"Recall@{RECALL_K}", "Rerank(s)"]

    print(f"\n{'Cấu hình':<30}" + "".join(f"{h:>11}" for h in headers))
    print("-" * (30 + 11 * len(headers)))
    for r in results:
        print(f"{r['name']:<30}" + "".join(f"{fmt(r[key]):>11}" for key in metric_keys))


def main():
    eval_set = load_eval_set()
    print(f"Ablation reranker: {len(eval_set)} câu hỏi, pool = {POOL_SIZE} ứng viên hybrid\n")

    results = []
    for name, fn in build_configs():
        print(f"Đang chạy: {name}...")
        results.append({"name": name, **run_config(fn, eval_set)})

    print_table(results)

    output = {
        "eval_set": EVAL_SET_PATH,
        "num_queries": len(eval_set),
        "pool_size": POOL_SIZE,
        "results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả vào {RESULTS_PATH}")


if __name__ == "__main__":
    main()
