"""
Bước 14: đánh giá truy vấn ảnh (Recall@K, nDCG@K, MRR) theo từng loại biến đổi.

Input: data/eval_set_image.jsonl + data/query_images/ (Bước 13); dùng search_core.search và metrics.py.
Chạy: python step14_eval_image_queries.py
"""
import json

from PIL import Image

from search_core import search
from metrics import recall_at_k, ndcg_at_k, mrr

EVAL_SET_IMAGE_PATH = "data/eval_set_image.jsonl"
K = 10
COMPONENTS = ["bm25", "dense"]   # bm25 tự trả [] với query ảnh nên chỉ dense có tác dụng


def load_eval_set_image():
    with open(EVAL_SET_IMAGE_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    eval_set = load_eval_set_image()
    print(f"Đánh giá {len(eval_set)} câu query ẢNH, k={K}, components={COMPONENTS}\n")

    per_transform = {}   # transform -> list {recall, ndcg, mrr}
    rows_detail = []

    for item in eval_set:
        # search() nhận PIL.Image cho query ảnh, không nhận đường dẫn.
        image = Image.open(item["query_image"]).convert("RGB")
        results = search(image, query_type="image", k=K, components=COMPONENTS)
        retrieved_ids = [r["id"] for r in results]

        r = recall_at_k(retrieved_ids, item["relevant_ids"], K)
        n = ndcg_at_k(retrieved_ids, item["relevant_ids"], K)
        m = mrr(retrieved_ids, item["relevant_ids"])

        transform = item["transform"]
        per_transform.setdefault(transform, []).append({"recall": r, "ndcg": n, "mrr": m})
        rows_detail.append({
            "transform": transform, "target_id": item["relevant_ids"][0],
            "source_caption": item["source_caption"], "recall": r,
            "top1_caption": results[0]["caption"] if results else None,
            "rank": retrieved_ids.index(item["relevant_ids"][0]) + 1
                    if item["relevant_ids"][0] in retrieved_ids else None,
        })

    print(f"{'Transform':<12} {'n':<4} {'Recall@'+str(K):<10} {'nDCG@'+str(K):<10} {'MRR':<8}")
    print("-" * 46)
    for transform, rows in per_transform.items():
        recalls = [r["recall"] for r in rows if r["recall"] is not None]
        ndcgs = [r["ndcg"] for r in rows if r["ndcg"] is not None]
        mrrs = [r["mrr"] for r in rows if r["mrr"] is not None]
        print(f"{transform:<12} {len(rows):<4} "
              f"{sum(recalls)/len(recalls):.3f}      {sum(ndcgs)/len(ndcgs):.3f}      "
              f"{sum(mrrs)/len(mrrs):.3f}")

    all_rows = [r for rows in per_transform.values() for r in rows]

    def avg(key):
        values = [r[key] for r in all_rows if r[key] is not None]
        return sum(values) / len(values)

    print("-" * 46)
    print(f"{'TỔNG':<12} {len(eval_set):<4} {avg('recall'):.3f}      {avg('ndcg'):.3f}      {avg('mrr'):.3f}")

    misses = [row for row in rows_detail if row["recall"] != 1.0]
    print(f"\nKhông tìm thấy sản phẩm đúng trong top-{K}: {len(misses)}/{len(rows_detail)}")
    for row in misses:
        print(f"  [MISS][{row['transform']:<10}] id={row['target_id']} "
              f"({row['source_caption'][:40]}) -> top1: {row['top1_caption']}")

    not_top1 = [row for row in rows_detail if row["rank"] not in (None, 1)]
    print(f"Tìm thấy nhưng không ở hạng 1: {len(not_top1)}/{len(rows_detail)}")
    for row in not_top1:
        print(f"  [hạng {row['rank']}][{row['transform']:<10}] id={row['target_id']} "
              f"({row['source_caption'][:40]}) -> top1: {row['top1_caption']}")


if __name__ == "__main__":
    main()
