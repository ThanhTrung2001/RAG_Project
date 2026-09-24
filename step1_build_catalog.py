"""
Bước 1: tải dataset Shopify/product-catalogue và chuẩn hoá thành catalog.

Output: data/catalog.jsonl (id, image_path, caption, search_text, category, brand)
        + data/images/*.jpg.
Chạy: python step1_build_catalog.py
"""
import json
import os
from datasets import load_dataset

# Cấu hình dataset
OUT_DIR = "data"
IMAGES_DIR = os.path.join(OUT_DIR, "images")
CATALOG_PATH = os.path.join(OUT_DIR, "catalog.jsonl")

DATASET_NAME = "Shopify/product-catalogue"
SPLIT = "train"
MAX_ITEMS = 2000                              # giới hạn để chạy nhanh trên máy cá nhân


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print(f"Đang tải dataset {DATASET_NAME} (split={SPLIT}) ...")
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    # Xáo trộn trước khi cắt để không lệch về vài category; seed cố định để tái lập.
    ds = ds.shuffle(seed=42)

    catalog = []
    n = min(MAX_ITEMS, len(ds))
    skipped = 0

    for i in range(n):
        item = ds[i]

        title = (item.get("product_title") or "").strip()
        description = (item.get("product_description") or "").strip()
        image = item.get("product_image")
        category = (item.get("ground_truth_category") or "").strip()
        brand = (item.get("ground_truth_brand") or "").strip()

        # Bỏ record không có ảnh hoặc không có cả title lẫn description.
        if image is None or (not title and not description):
            skipped += 1
            continue

        img_path = os.path.join(IMAGES_DIR, f"{i:06d}.jpg")
        image.convert("RGB").save(img_path)   # một số ảnh gốc là CMYK/RGBA

        # caption (ngắn) cho CLIP; search_text (title + description) cho BM25.
        caption = title if title else description[:200]
        search_text = f"{title} {description}".strip()

        catalog.append({
            "id": i,                      # id = vị trí trong dataset sau shuffle
            "image_path": img_path,
            "caption": caption,
            "search_text": search_text,
            "category": category,
            "brand": brand,
        })

        if i % 300 == 0:
            print(f"  đã xử lý {i}/{n} (bỏ qua {skipped} record thiếu dữ liệu)")

    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        for row in catalog:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Xong: {len(catalog)} sản phẩm, bỏ qua {skipped}. Lưu tại {CATALOG_PATH}")

    # Thống kê category top-level để kiểm tra độ đa dạng của mẫu.
    category_counts = {}
    for row in catalog:
        cat_top_level = row["category"].split(">")[0].strip() if row["category"] else "(không có category)"
        category_counts[cat_top_level] = category_counts.get(cat_top_level, 0) + 1

    print(f"\nPhân bố category (top-level, {len(category_counts)} nhóm khác nhau):")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:15]:
        pct = count / len(catalog) * 100
        print(f"  {cat:<45} {count:>5} ({pct:.1f}%)")
    max_pct = max(category_counts.values()) / len(catalog) * 100
    if max_pct > 40:
        print(f"\nCẢNH BÁO: 1 category chiếm tới {max_pct:.1f}% dữ liệu -- có thể")
        print("chưa đủ đa dạng, cân nhắc tăng MAX_ITEMS hoặc kiểm tra lại dataset gốc.")


if __name__ == "__main__":
    main()
