"""
================================================================================
BƯỚC 1 — TẢI DATASET VÀ CHUẨN HOÁ THÀNH "CATALOG"
================================================================================

TẠI SAO CẦN BƯỚC NÀY (không thể bỏ qua, đi thẳng vào bước 2)?
    Dataset tải từ Hugging Face (Shopify/product-catalogue) có cấu trúc riêng
    của nó: ảnh là object PIL.Image lồng trong dict, tên field là
    product_title/product_description (dài dòng, không nhất quán về việc có
    rỗng hay không). Nếu để mọi bước sau (embedding, index, search, API) đều
    tự đọc trực tiếp từ dataset gốc, thì đổi dataset khác sẽ phải sửa RẤT
    NHIỀU chỗ trong code.

    Giải pháp: tạo một "lớp trung gian" duy nhất — file catalog.jsonl với
    schema CỐ ĐỊNH (id, image_path, caption, search_text, category, brand).
    Từ bước 2 trở đi, KHÔNG có dòng code nào biết tới "Shopify" hay
    "Hugging Face" nữa — chúng chỉ đọc catalog.jsonl. Đổi dataset khác
    chỉ cần sửa DUY NHẤT file này.

CHẠY: python step1_build_catalog.py
OUTPUT: data/catalog.jsonl (metadata) + data/images/*.jpg (ảnh thật)
"""
import json
import os
from datasets import load_dataset

# ---- Cấu hình: đổi ở đây nếu dùng dataset khác ----
OUT_DIR = "data"
IMAGES_DIR = os.path.join(OUT_DIR, "images")
CATALOG_PATH = os.path.join(OUT_DIR, "catalog.jsonl")

DATASET_NAME = "Shopify/product-catalogue"   # tên dataset trên Hugging Face Hub
SPLIT = "train"                               # dataset này có 2 split: train (38.6k), test (9.66k)
MAX_ITEMS = 2000                              # giới hạn số dòng để chạy nhanh trên máy cá nhân
                                                # (không cần GPU mạnh, không cần chờ lâu ở bước 2)


def main():
    # os.makedirs(..., exist_ok=True): tạo thư mục nếu chưa có, không lỗi nếu đã tồn tại
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print(f"Đang tải dataset {DATASET_NAME} (split={SPLIT}) ...")
    # load_dataset(..., split="train") trả về THẲNG 1 Dataset (không phải DatasetDict),
    # nên dùng ds[i] truy cập trực tiếp được, không cần ds["train"][i].
    # Lần đầu chạy sẽ tải file về và cache tại ~/.cache/huggingface/datasets/ --
    # các lần sau đọc cache, không cần internet nữa.
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    # QUAN TRỌNG: xáo trộn TRƯỚC KHI cắt lấy MAX_ITEMS dòng đầu.
    # Nếu không xáo trộn, dữ liệu gốc có thể đã được nhóm sẵn theo category
    # (nhiều dataset trên HF sắp xếp theo thứ tự crawl/theo nguồn) -- lấy
    # tuần tự N dòng đầu dễ nghiêng hẳn về 1-2 category, vi phạm yêu cầu
    # "đảm bảo tính đa dạng của mẫu được chọn, không lấy toàn một chủng loại".
    # seed=42 cố định để KẾT QUẢ TÁI LẬP ĐƯỢC -- chạy lại vẫn ra đúng 2000
    # sản phẩm giống hệt, không đổi mỗi lần chạy.
    ds = ds.shuffle(seed=42)

    catalog = []
    n = min(MAX_ITEMS, len(ds))   # tránh lỗi nếu MAX_ITEMS > số dòng thật có trong dataset
    skipped = 0

    for i in range(n):
        item = ds[i]

        # .get(...) trả về None nếu field không tồn tại, thay vì raise KeyError --
        # dataset thật LUÔN có vài dòng dữ liệu thiếu, phải phòng thủ ngay từ đầu.
        title = (item.get("product_title") or "").strip()
        description = (item.get("product_description") or "").strip()
        image = item.get("product_image")
        category = (item.get("ground_truth_category") or "").strip()
        brand = (item.get("ground_truth_brand") or "").strip()

        # BỎ QUA record thiếu ảnh (không có gì để hiển thị/tìm) hoặc thiếu
        # CẢ title lẫn description (không có gì để encode thành embedding/BM25).
        # Lọc ngay tại đây để các bước sau không phải xử lý dữ liệu rác.
        if image is None or (not title and not description):
            skipped += 1
            continue

        # Lưu ảnh thật ra đĩa (thay vì giữ trong RAM) -- bước 2 sẽ đọc lại từ đây.
        # :06d -> đệm số 0 phía trước cho đủ 6 chữ số (000000.jpg, 000001.jpg, ...)
        # để file luôn sắp xếp đúng thứ tự khi liệt kê thư mục.
        img_path = os.path.join(IMAGES_DIR, f"{i:06d}.jpg")
        image.convert("RGB").save(img_path)   # convert("RGB"): một số ảnh gốc là CMYK/RGBA,
                                                # ép về RGB để CLIP xử lý được nhất quán

        # TẠI SAO TÁCH RIÊNG "caption" VÀ "search_text"?
        #   - caption (chỉ title, ngắn gọn) -> dùng cho CLIP text embedding ở bước 2,
        #     vì CLIP được huấn luyện chủ yếu trên caption ngắn kiểu "a photo of X".
        #   - search_text (title + description, đầy đủ) -> dùng cho BM25 ở bước 3,
        #     vì BM25 xếp hạng theo tần suất từ khoá -- càng nhiều chữ càng dễ khớp
        #     đúng khi người dùng gõ một từ chỉ xuất hiện trong mô tả, không có trong tên.
        caption = title if title else description[:200]
        search_text = f"{title} {description}".strip()

        catalog.append({
            "id": i,                      # dùng chính index trong dataset gốc làm id --
                                            # đơn giản, không cần sinh id riêng
            "image_path": img_path,
            "caption": caption,
            "search_text": search_text,
            "category": category,          # chưa dùng ở bước search, để dành cho
            "brand": brand,                # error analysis (bước 9) sau này
        })

        if i % 300 == 0:
            print(f"  đã xử lý {i}/{n} (bỏ qua {skipped} record thiếu dữ liệu)")

    # Ghi ra file .jsonl: MỖI DÒNG là 1 JSON object riêng biệt (không phải 1 mảng lớn).
    # Định dạng này cho phép đọc từng dòng một mà không cần load cả file vào RAM
    # (quan trọng khi dataset lớn hơn nhiều so với 2000 dòng ở đây).
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        for row in catalog:
            # ensure_ascii=False: giữ nguyên tiếng Việt/ký tự Unicode thay vì
            # escape thành \uXXXX -- dễ đọc bằng mắt khi mở file kiểm tra thủ công.
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Xong: {len(catalog)} sản phẩm, bỏ qua {skipped}. Lưu tại {CATALOG_PATH}")

    # Kiểm tra ĐA DẠNG category -- in ra để tự xác nhận không bị nghiêng
    # về 1-2 category (yêu cầu bắt buộc khi lấy subset từ dataset lớn).
    # Đây là bước kiểm tra ĐƠN GIẢN (đếm tần suất), không phải thuật toán
    # lấy mẫu phân tầng (stratified sampling) phức tạp -- đủ để phát hiện
    # nếu shuffle không đủ (ví dụ dataset gốc có category chiếm 90% số dòng).
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
