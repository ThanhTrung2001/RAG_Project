"""
Ablation theo component: bm25 / dense / hybrid / hybrid + rerank, cùng gọi search() như app.py.

Input: data/eval_set.jsonl, data/catalog.jsonl -> in bảng Recall@10, nDCG@10, nDCG đa mức, MRR, latency.
Chạy: python step8_run_ablation.py (cần index từ step1-3).
"""
import json

from search_core import search, available_components
from metrics import evaluate_all

EVAL_SET_PATH = "data/eval_set.jsonl"
K = 10

# Mỗi cấu hình chỉ khác hybrid ở đúng một thành phần.
CONFIGS = [
    {"name": "bm25 only",           "components": ["bm25"]},
    {"name": "dense only",          "components": ["dense"]},
    {"name": "hybrid (bm25+dense)", "components": ["bm25", "dense"]},
    {"name": "hybrid + rerank",     "components": ["bm25", "dense"], "use_rerank": True},
]


def load_eval_set():
    rows = []
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_catalog():
    """catalog.jsonl -> {id: row}, dùng cho nDCG đa mức."""
    rows = {}
    with open("data/catalog.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def main():
    print(f"Component đã đăng ký trong hệ thống: {available_components()}")

    eval_set = load_eval_set()
    catalog = load_catalog()
    print(f"Chạy ablation trên {len(eval_set)} câu query, k={K}\n")

    header = (f"{'Cấu hình':<24} {'Recall@'+str(K):<10} {'nDCG@'+str(K):<10} "
              f"{'nDCG_gr@'+str(K):<11} {'MRR':<8} {'Lat_avg(ms)':<12} {'Lat_p95(ms)':<12}")
    print(header)
    print("-" * len(header))

    for cfg in CONFIGS:
        metrics = evaluate_all(
            search_fn=search,
            eval_set=eval_set,
            k=K,
            catalog=catalog,
            components=cfg["components"],
            use_rerank=cfg.get("use_rerank", False),
        )

        def fmt(key, spec=".3f"):
            v = metrics[key]
            return format(v, spec) if v is not None else "N/A"

        print(f"{cfg['name']:<24} {fmt('recall@k'):<10} {fmt('ndcg@k'):<10} "
              f"{fmt('ndcg@k_graded'):<11} {fmt('mrr'):<8} "
              f"{fmt('latency_ms_avg', '.1f'):<12} {fmt('latency_ms_p95', '.1f'):<12}")

    print("\n-> Chênh lệch điểm số giữa 2 cấu hình = đóng góp của thành phần bị tắt.")
    print("-> nDCG_gr = nDCG đa mức (đúng id=2, cùng category=1, khác=0) -- xem metrics.ndcg_at_k_graded")
    print("-> MRR = hạng của kết quả đúng ĐẦU TIÊN, đảo ngược (1/hạng) -- xem metrics.mrr")
    print("-> Lat_avg/Lat_p95 = latency mỗi lần gọi search(), đo bằng time.perf_counter()")
    print("-> Copy bảng này vào báo cáo, kèm 2-3 ví dụ câu query sai cụ thể cho mỗi cấu hình")
    print("   (đọc lại từng câu trong eval_set.jsonl, xem search() trả về gì -- đây chính")
    print("   là bước 9 'error analysis', không code sẵn được vì cần bạn tự nhìn kết quả.)")


if __name__ == "__main__":
    main()
