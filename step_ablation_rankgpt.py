"""
================================================================================
ABLATION RERANK CHO A2 — RankGPT (Sun et al., EMNLP 2023) TRÊN CATALOG A1
================================================================================

Tái hiện trên catalog sản phẩm của A1 hai thí nghiệm của paper:
    - Mục 6.6 / Table 6: LLM mã nguồn mở làm reranker bằng permutation generation.
    - Mục 7 / Table 7:   model nhỏ distill từ ChatGPT thay cho LLM.

Các cấu hình, cùng một pool ứng viên từ hybrid BM25 + Dense của A1:
    1. Không rerank
    2. Cross-encoder MiniLM (ms-marco-MiniLM-L-6-v2)
    3. DeBERTa distill từ ChatGPT (deberta-10k-rank_net) -- bỏ qua nếu chưa tải
    4. RankGPT với từng model Ollama trong RANKGPT_MODELS

Metric: nDCG@1/5/10 như paper, thêm Recall@10 và thời gian rerank trung bình.
Với RankGPT còn đếm lỗi permutation theo cách phân loại của Table 10.

Khác paper:
    - Paper rerank top-100 BM25 trên TREC-DL/BEIR, nhãn nhiều mức.
    - Ở đây rerank top-30 hybrid trên các câu hỏi text của data/eval_set.jsonl
      (250 câu), mỗi câu một sản phẩm đúng (nhãn nhị phân).

Chuẩn bị:
    ollama pull llama3.1
    ollama pull qwen2.5:0.5b
    (tuỳ chọn) tải deberta-10k-rank_net từ repo RankGPT, giải nén vào
    models/deberta-10k-rank_net

CHẠY: python step_ablation_rankgpt.py
Kết quả in ra màn hình và lưu vào data/rankgpt_ablation_results.json.
"""
import json
import os
import time

import requests

from metrics import ndcg_at_k, recall_at_k
from search_core import (DISTILLED_RERANKER_PATH, distilled_rerank,
                         rankgpt_rerank_with_stats, rerank, search)

EVAL_SET_PATH = "data/eval_set.jsonl"
RESULTS_PATH = "data/rankgpt_ablation_results.json"

POOL_SIZE = 30            # với window 20, step 10: mỗi câu hỏi 2 lượt gọi LLM
NDCG_CUTOFFS = (1, 5, 10)
RECALL_K = 10
RANKGPT_MODELS = ["llama3.1", "qwen2.5:0.5b"]
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
STAT_KEYS = ("windows", "repetition", "missing", "out_of_range", "rejection")


def load_eval_set():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def build_configs():
    """Mỗi cấu hình là (tên, hàm). Hàm nhận (query, candidates), trả (candidates, stats hoặc None)."""
    configs = [
        ("Không rerank", lambda q, c: (c, None)),
        ("Cross-encoder MiniLM", lambda q, c: (rerank(q, c), None)),
    ]

    if os.path.isdir(DISTILLED_RERANKER_PATH):
        configs.append(("DeBERTa distill từ ChatGPT", lambda q, c: (distilled_rerank(q, c), None)))
    else:
        print(f"Bỏ qua DeBERTa distill: không thấy thư mục {DISTILLED_RERANKER_PATH}\n")

    if RANKGPT_MODELS and not ollama_available():
        print(f"Bỏ qua RankGPT: không kết nối được Ollama tại {OLLAMA_TAGS_URL}\n")
        return configs

    for model in RANKGPT_MODELS:
        configs.append((f"RankGPT ({model})",
                        lambda q, c, m=model: rankgpt_rerank_with_stats(q, c, llm_model=m)))
    return configs


def ollama_available():
    try:
        requests.get(OLLAMA_TAGS_URL, timeout=3).raise_for_status()
        return True
    except requests.RequestException:
        return False


def run_config(rerank_fn, eval_set):
    ndcgs = {k: [] for k in NDCG_CUTOFFS}
    recalls, latencies = [], []
    stats_total = None

    for item in eval_set:
        query, relevant_ids = item["query"], item["relevant_ids"]
        candidates = search(query, query_type="text", k=POOL_SIZE,
                            components=["bm25", "dense"], top_n=POOL_SIZE)

        start = time.perf_counter()
        ranked, stats = rerank_fn(query, candidates) if candidates else (candidates, None)
        latencies.append(time.perf_counter() - start)

        ranked_ids = [c["id"] for c in ranked]
        for k in NDCG_CUTOFFS:
            ndcgs[k].append(ndcg_at_k(ranked_ids, relevant_ids, k))
        recalls.append(recall_at_k(ranked_ids, relevant_ids, RECALL_K))

        if stats is not None:
            stats_total = stats_total or dict.fromkeys(STAT_KEYS, 0)
            for key in STAT_KEYS:
                stats_total[key] += stats[key]

    result = {f"ndcg@{k}": mean(ndcgs[k]) for k in NDCG_CUTOFFS}
    result[f"recall@{RECALL_K}"] = mean(recalls)
    result["rerank_latency_sec_avg"] = mean(latencies)
    result["permutation_stats"] = stats_total
    return result


def fmt(value):
    return "N/A" if value is None else f"{value:.3f}"


def print_tables(results):
    metric_keys = [f"ndcg@{k}" for k in NDCG_CUTOFFS] + [f"recall@{RECALL_K}", "rerank_latency_sec_avg"]
    headers = [f"nDCG@{k}" for k in NDCG_CUTOFFS] + [f"Recall@{RECALL_K}", "Rerank(s)"]

    print(f"\n{'Cấu hình':<30}" + "".join(f"{h:>11}" for h in headers))
    print("-" * (30 + 11 * len(headers)))
    for r in results:
        print(f"{r['name']:<30}" + "".join(f"{fmt(r[key]):>11}" for key in metric_keys))

    rankgpt_rows = [r for r in results if r["permutation_stats"] is not None]
    if rankgpt_rows:
        print("\nLỗi permutation của RankGPT (cộng dồn mọi cửa sổ, cách phân loại theo Table 10):")
        print(f"{'Cấu hình':<30}" + "".join(f"{k:>14}" for k in STAT_KEYS))
        for r in rankgpt_rows:
            print(f"{r['name']:<30}" + "".join(f"{r['permutation_stats'][k]:>14}" for k in STAT_KEYS))


def main():
    eval_set = load_eval_set()
    print(f"Ablation rerank: {len(eval_set)} câu hỏi, pool = {POOL_SIZE} ứng viên hybrid\n")

    results = []
    for name, fn in build_configs():
        print(f"Đang chạy: {name}...")
        results.append({"name": name, **run_config(fn, eval_set)})

    print_tables(results)

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
