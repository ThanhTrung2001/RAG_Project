"""
Bước 17 (A3): đánh giá tầng Retrieval của chatbot khi hỏi bằng ảnh, không cần LLM.

Dùng đúng hàm rag_core.retrieve(query="", image=..., k=5) mà /api/v1/ask_image gọi, trên 100 truy vấn
ảnh của data/eval_set_image.jsonl (Bước 13: crop / rotate / brightness / ...). Đo Context Recall@5 và
Context Precision@5 (cùng metric với step_ablation_rag.py) theo từng loại biến đổi.
Không có rerank: reranker chấm cặp (chữ, caption) nên không áp dụng cho câu hỏi chỉ có ảnh.

Chạy: python step17_eval_image_rag_retrieval.py
Output: in bảng + data/image_rag_retrieval_results.json
"""
import json
import time

from PIL import Image

from metrics import recall_at_k
from rag_core import retrieve
from rag_metrics import context_precision_at_k

EVAL_SET_IMAGE_PATH = "data/eval_set_image.jsonl"
RESULTS_PATH = "data/image_rag_retrieval_results.json"
K = 5   # cùng k với step_ablation_rag.py


def summarize(rows):
    def avg(key):
        values = [r[key] for r in rows if r[key] is not None]
        return sum(values) / len(values) if values else None
    return {
        "n": len(rows),
        "context_recall": avg("context_recall"),
        "context_precision": avg("context_precision"),
        "top1_hit": avg("top1_hit"),
        "latency_sec": avg("latency_sec"),
    }


def main():
    with open(EVAL_SET_IMAGE_PATH, "r", encoding="utf-8") as f:
        eval_set = [json.loads(line) for line in f]
    print(f"Retrieval của chatbot với câu hỏi bằng ảnh: {len(eval_set)} truy vấn, k={K}\n")

    rows = []
    for item in eval_set:
        image = Image.open(item["query_image"]).convert("RGB")
        start = time.time()
        results = retrieve("", k=K, image=image)
        latency = time.time() - start

        retrieved_ids = [r["id"] for r in results]
        relevant_ids = item["relevant_ids"]
        rows.append({
            "query_image": item["query_image"],
            "transform": item["transform"],
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            "context_recall": recall_at_k(retrieved_ids, relevant_ids, K),
            "context_precision": context_precision_at_k(retrieved_ids, relevant_ids, K),
            "top1_hit": 1.0 if retrieved_ids[:1] and retrieved_ids[0] in relevant_ids else 0.0,
            "latency_sec": latency,
        })

    by_transform = {}
    for r in rows:
        by_transform.setdefault(r["transform"], []).append(r)
    summary = {"all": summarize(rows),
               "by_transform": {t: summarize(rs) for t, rs in sorted(by_transform.items())}}

    def fmt(v):
        return f"{v:.3f}" if v is not None else "N/A"

    print(f"{'Biến đổi':<12} {'n':>4} {'CtxRecall@5':>12} {'CtxPrec@5':>10} {'Đúng ở [1]':>11} {'Thời gian (s)':>14}")
    print("-" * 68)
    for name, s in [*summary["by_transform"].items(), ("TẤT CẢ", summary["all"])]:
        print(f"{name:<12} {s['n']:>4} {fmt(s['context_recall']):>12} {fmt(s['context_precision']):>10} "
              f"{fmt(s['top1_hit']):>11} {fmt(s['latency_sec']):>14}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"k": K, "eval_set": EVAL_SET_IMAGE_PATH, "summary": summary, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {RESULTS_PATH}")


if __name__ == "__main__":
    main()
