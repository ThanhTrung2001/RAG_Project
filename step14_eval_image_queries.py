"""
================================================================================
BƯỚC 14 — ĐÁNH GIÁ TRUY VẤN ẢNH (mục 4.3 template A1: "Kết quả truy vấn ảnh")
================================================================================

KHÁC GÌ VỚI step8_run_ablation.py?
    step8 đánh giá query TEXT (đọc thẳng string từ eval_set.jsonl). Query ẢNH
    không thể lưu trong .jsonl dưới dạng dùng ngay được -- .jsonl chỉ lưu
    ĐƯỜNG DẪN ảnh (`query_image`), phải tự mở bằng PIL.Image.open() trước khi
    đưa vào search() (đúng kiểu dữ liệu _encode_image() trong search_core.py
    yêu cầu: PIL.Image, không phải string).

CHẠY: python step14_eval_image_queries.py
YÊU CẦU TRƯỚC: đã chạy step13_build_image_eval_set.py (có data/eval_set_image.jsonl
      + data/query_images/*.jpg).
"""
import json

from PIL import Image

from search_core import search
from metrics import recall_at_k, ndcg_at_k, mrr

EVAL_SET_IMAGE_PATH = "data/eval_set_image.jsonl"
K = 10
COMPONENTS = ["bm25", "dense"]   # giống hybrid mặc định của A1 -- bm25 tự trả []
                                    # cho query ảnh (xem _run_bm25 trong search_core.py),
                                    # không cần loại bỏ tay khỏi danh sách này


def load_eval_set_image():
    with open(EVAL_SET_IMAGE_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    eval_set = load_eval_set_image()
    print(f"Đánh giá {len(eval_set)} câu query ẢNH, k={K}, components={COMPONENTS}\n")

    per_transform = {}   # transform -> list các dict {recall, ndcg, mrr}
    rows_detail = []

    for item in eval_set:
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

    all_recalls = [r["recall"] for rows in per_transform.values() for r in rows if r["recall"] is not None]
    print("-" * 46)
    print(f"{'TỔNG':<12} {len(eval_set):<4} {sum(all_recalls)/len(all_recalls):.3f}")

    print("\nChi tiết từng câu (để đối chiếu bằng mắt, viết case study mục 6 nếu cần):")
    for row in rows_detail:
        status = "OK" if row["recall"] == 1.0 else "MISS"
        print(f"  [{status:<4}][{row['transform']:<10}] id={row['target_id']} "
              f"({row['source_caption'][:40]}) -> top1: {row['top1_caption']}")

    print("\n-> Copy bảng theo transform vào BAO_CAO_TONG_HOP.md mục 4.3 (Kết quả truy vấn ảnh).")
    print("-> Nếu 1 loại transform (vd rotate) có Recall thấp hẳn so với 2 loại còn lại,")
    print("   đó là bằng chứng cụ thể về ĐIỂM YẾU của CLIP với phép biến đổi đó -- nêu")
    print("   trong phần hạn chế/đề xuất cải tiến của báo cáo.")


if __name__ == "__main__":
    main()
