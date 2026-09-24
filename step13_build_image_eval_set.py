"""
Bước 13: tạo bộ query ảnh bằng cách biến đổi ảnh sản phẩm có trong eval set.

Lấy N_SAMPLES câu đầu của data/eval_set.jsonl, mỗi ảnh áp 1 phép biến đổi (crop/rotate/brightness).
Output: data/query_images/*.jpg + data/eval_set_image.jsonl.
Chạy: python step13_build_image_eval_set.py (cần data/eval_set.jsonl, data/catalog.jsonl, data/images/).
"""
import json
import os

from PIL import Image, ImageEnhance

EVAL_SET_PATH = "data/eval_set.jsonl"
CATALOG_PATH = "data/catalog.jsonl"
OUT_DIR = "data/query_images"
OUT_EVAL_PATH = "data/eval_set_image.jsonl"

N_SAMPLES = 12   # chia đều 3 phép biến đổi, mỗi loại 4 ảnh
TRANSFORMS = ["crop", "rotate", "brightness"]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def apply_transform(image: Image.Image, transform: str) -> Image.Image:
    """crop: cắt 15% mỗi cạnh; rotate: xoay 15 độ; brightness: giảm sáng còn 55%."""
    if transform == "crop":
        w, h = image.size
        margin_w, margin_h = int(w * 0.15), int(h * 0.15)
        return image.crop((margin_w, margin_h, w - margin_w, h - margin_h))
    if transform == "rotate":
        # Nền trắng giống ảnh sản phẩm; expand=False giữ kích thước (mất góc).
        return image.rotate(15, fillcolor=(255, 255, 255), expand=False)
    if transform == "brightness":
        return ImageEnhance.Brightness(image).enhance(0.55)
    raise ValueError(f"Không nhận diện được transform: {transform}")


def main():
    eval_set = load_jsonl(EVAL_SET_PATH)
    catalog = {r["id"]: r for r in load_jsonl(CATALOG_PATH)}
    os.makedirs(OUT_DIR, exist_ok=True)

    samples = eval_set[:N_SAMPLES]
    rows = []

    for i, item in enumerate(samples):
        target_id = item["relevant_ids"][0]
        row = catalog[target_id]
        transform = TRANSFORMS[i % len(TRANSFORMS)]

        image = Image.open(row["image_path"]).convert("RGB")
        transformed = apply_transform(image, transform)

        out_path = os.path.join(OUT_DIR, f"{target_id:06d}_{transform}.jpg")
        transformed.save(out_path)

        rows.append({
            "query_image": out_path,
            "query_type": "image",
            "transform": transform,
            "relevant_ids": [target_id],
            "source_caption": row["caption"],
        })
        print(f"  [{transform:<10}] id={target_id} -> {out_path}")

    with open(OUT_EVAL_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nXong: {len(rows)} câu query ảnh -> {OUT_EVAL_PATH}")
    print("Chạy tiếp: python step14_eval_image_queries.py")


if __name__ == "__main__":
    main()
