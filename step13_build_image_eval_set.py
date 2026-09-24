"""
================================================================================
BƯỚC 13 — DỰNG BỘ TRUY VẤN ẢNH BẰNG BIẾN ĐỔI (mục 2.3/4.3 template A1)
================================================================================

VẤN ĐỀ: `data/eval_set.jsonl` (Bước 4/6) chỉ có query dạng TEXT. Template A1
yêu cầu thêm ground truth cho query dạng ẢNH -- nhưng KHÔNG THỂ tự "viết"
1 câu hỏi ảnh như viết text. Cách làm chuẩn (và cách DUY NHẤT không cần tự
chụp ảnh mới): lấy ảnh SẢN PHẨM THẬT đã có trong `data/images/` (từ các id
đã biết đúng ở `eval_set.jsonl`), áp DUY NHẤT 1 phép biến đổi (crop/xoay/
đổi sáng) để mô phỏng "ảnh người dùng chụp lại" -- rồi kỳ vọng hệ thống vẫn
tìm lại được ĐÚNG sản phẩm gốc dù ảnh đầu vào không giống 100% ảnh gốc.

TẠI SAO CHỈ ĐỔI ĐÚNG 1 PHÉP BIẾN ĐỔI MỖI ẢNH?
    Cùng tư duy ablation "mỗi lần chỉ đổi 1 biến" (xem step8) -- nếu vừa
    crop vừa xoay vừa đổi sáng trong 1 ảnh, không biết phép biến đổi nào
    làm CLIP nhận diện sai (nếu có). Tách riêng 3 loại giúp so sánh: CLIP
    "chịu" được crop tốt hơn hay đổi sáng tốt hơn?

3 PHÉP BIẾN ĐỔI:
    - crop:       cắt bớt 15% viền mỗi cạnh (mô phỏng chụp gần/lệch khung)
    - rotate:     xoay 15 độ, nền trắng (mô phỏng chụp nghiêng máy)
    - brightness: giảm sáng còn 55% (mô phỏng chụp thiếu sáng/trong nhà)

CHẠY: python step13_build_image_eval_set.py
OUTPUT: data/query_images/*.jpg (ảnh đã biến đổi) + data/eval_set_image.jsonl
YÊU CẦU TRƯỚC: đã có data/eval_set.jsonl (Bước 4/6) và data/images/ (Bước 1).
"""
import json
import os

from PIL import Image, ImageEnhance

EVAL_SET_PATH = "data/eval_set.jsonl"
CATALOG_PATH = "data/catalog.jsonl"
OUT_DIR = "data/query_images"
OUT_EVAL_PATH = "data/eval_set_image.jsonl"

N_SAMPLES = 12   # 12 sản phẩm x 1 phép biến đổi mỗi cái = 12 câu, chia đều 3 loại (4 mỗi loại)
TRANSFORMS = ["crop", "rotate", "brightness"]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def apply_transform(image: Image.Image, transform: str) -> Image.Image:
    if transform == "crop":
        w, h = image.size
        margin_w, margin_h = int(w * 0.15), int(h * 0.15)
        return image.crop((margin_w, margin_h, w - margin_w, h - margin_h))
    if transform == "rotate":
        # fillcolor trắng thay vì đen mặc định -- giống nền thật hơn khi ảnh
        # sản phẩm thường chụp trên nền sáng; expand=False giữ nguyên kích
        # thước gốc (chấp nhận mất góc, giống ảnh chụp nghiêng thật).
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
        transform = TRANSFORMS[i % len(TRANSFORMS)]   # xoay vòng 3 loại, chia đều

        image = Image.open(row["image_path"]).convert("RGB")
        transformed = apply_transform(image, transform)

        out_path = os.path.join(OUT_DIR, f"{target_id:06d}_{transform}.jpg")
        transformed.save(out_path)

        rows.append({
            "query_image": out_path,
            "query_type": "image",
            "transform": transform,
            "relevant_ids": [target_id],
            "source_caption": row["caption"],   # để đối chiếu khi đọc kết quả bằng mắt
        })
        print(f"  [{transform:<10}] id={target_id} -> {out_path}")

    with open(OUT_EVAL_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nXong: {len(rows)} câu query ảnh -> {OUT_EVAL_PATH}")
    print("Chạy tiếp: python step14_eval_image_queries.py")


if __name__ == "__main__":
    main()
