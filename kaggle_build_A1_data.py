"""
================================================================================
SCRIPT DÙNG TRÊN KAGGLE — Gộp cả step1+step2+step3, xuất 1 file .zip để tải về
================================================================================

CÁCH DÙNG:
    1. Tạo notebook mới trên Kaggle
    2. Settings (bên phải) -> Accelerator: chọn GPU (GPU T4 x2 là đủ)
    3. Settings -> Internet: bật ON (bắt buộc, mặc định Kaggle tắt internet)
    4. Upload file này lên (hoặc copy nguyên nội dung vào 1 cell)
    5. QUAN TRỌNG -- nếu đã từng chạy thử trước đó trong cùng notebook này,
       BẮT BUỘC bấm Restart Session trước (menu "..." góc phải trên) --
       chỉ cài lại thư viện bằng pip KHÔNG đủ, vì bản cũ đã nạp vào RAM.
    6. Chạy: !pip install -q datasets transformers==4.57.1 faiss-cpu rank_bm25
       (dùng dấu == với số cụ thể, KHÔNG dùng dấu < -- ký tự < trong dòng lệnh
       shell dễ bị hiểu nhầm thành "input redirection" nếu gõ thiếu ngoặc kép,
       khiến việc ghim phiên bản bị bỏ qua âm thầm mà không báo lỗi rõ ràng)
    7. Kiểm tra lại đã cài đúng bản chưa (bước hay bị bỏ qua nhưng nên làm):
           !python -c "import transformers; print(transformers.__version__)"
       Kết quả PHẢI là 4.57.1 -- nếu ra số khác (vd 5.x), quay lại bước 5.
    8. Chạy: !python kaggle_build_A1_data.py
    9. Sau khi chạy xong, vào tab "Output" bên phải Kaggle, tải file
       a1_data_output.zip về máy

SAU KHI TẢI VỀ MÁY:
    Giải nén a1_data_output.zip đè vào thư mục data/ của project A1
    (giữ đúng cấu trúc data/catalog.jsonl, data/images/, data/dense.index,
    data/bm25.pkl) -- rồi chạy `uvicorn app:app --reload` như bình thường,
    KHÔNG cần chạy lại step1/step2/step3 trên máy nữa.

LƯU Ý QUAN TRỌNG:
    Kaggle session có giới hạn thời gian (thường 9-12 tiếng cho GPU) và
    KHÔNG tự lưu file sau khi session đóng -- phải tải file .zip về TRƯỚC
    KHI đóng notebook, hoặc lưu thành Kaggle Dataset để giữ lại lâu dài.
"""
import json
import os
import pickle
import zipfile

import faiss
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from transformers import CLIPModel, CLIPProcessor

# ---------------- Cấu hình (giống hệt step1-3 gốc) ----------------
OUT_DIR = "data"
IMAGES_DIR = os.path.join(OUT_DIR, "images")
CATALOG_PATH = os.path.join(OUT_DIR, "catalog.jsonl")
IMAGE_EMB_PATH = os.path.join(OUT_DIR, "image_embeddings.npy")
DENSE_INDEX_PATH = os.path.join(OUT_DIR, "dense.index")
BM25_INDEX_PATH = os.path.join(OUT_DIR, "bm25.pkl")

DATASET_NAME = "Shopify/product-catalogue"
SPLIT = "train"
MAX_ITEMS = 2000
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32


def step1_build_catalog():
    print("=== BƯỚC 1: Tải dataset, build catalog ===")
    os.makedirs(IMAGES_DIR, exist_ok=True)
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    ds = ds.shuffle(seed=42)   # xáo trộn trước khi cắt -- đảm bảo đa dạng category,
                                 # xem giải thích chi tiết trong step1_build_catalog.py

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

        if image is None or (not title and not description):
            skipped += 1
            continue

        img_path = os.path.join(IMAGES_DIR, f"{i:06d}.jpg")
        image.convert("RGB").save(img_path)

        caption = title if title else description[:200]
        search_text = f"{title} {description}".strip()

        catalog.append({
            "id": i, "image_path": img_path, "caption": caption,
            "search_text": search_text, "category": category, "brand": brand,
        })
        if i % 300 == 0:
            print(f"  đã xử lý {i}/{n}")

    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        for row in catalog:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Xong bước 1: {len(catalog)} sản phẩm, bỏ qua {skipped}\n")
    return catalog


def step2_build_embeddings(catalog):
    print("=== BƯỚC 2: Sinh embedding CLIP (dùng GPU nếu có) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=False)

    image_embs = []
    for start in range(0, len(catalog), BATCH_SIZE):
        batch = catalog[start:start + BATCH_SIZE]
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]

        with torch.no_grad():
            img_inputs = processor(images=images, return_tensors="pt").to(device)
            img_feat = model.get_image_features(**img_inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        image_embs.append(img_feat.cpu().numpy())
        print(f"  đã encode {start + len(batch)}/{len(catalog)}")

    image_embs = np.concatenate(image_embs, axis=0).astype("float32")
    np.save(IMAGE_EMB_PATH, image_embs)
    print(f"Xong bước 2: lưu {image_embs.shape} vào {IMAGE_EMB_PATH}\n")
    return image_embs


def step3_build_index(catalog, image_embs):
    print("=== BƯỚC 3: Build FAISS + BM25 ===")
    index = faiss.IndexFlatIP(image_embs.shape[1])
    index.add(image_embs)
    faiss.write_index(index, DENSE_INDEX_PATH)

    tokenized = [r["search_text"].lower().split() for r in catalog]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": [r["id"] for r in catalog]}, f)
    print(f"Xong bước 3: {index.ntotal} vector, {len(catalog)} document BM25\n")


def zip_output():
    print("=== Đóng gói data/ thành a1_data_output.zip ===")
    zip_path = "a1_data_output.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUT_DIR):
            for file in files:
                filepath = os.path.join(root, file)
                zf.write(filepath, filepath)
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Xong: {zip_path} ({size_mb:.1f} MB) -- vào tab Output bên phải để tải về")


if __name__ == "__main__":
    catalog = step1_build_catalog()
    image_embs = step2_build_embeddings(catalog)
    step3_build_index(catalog, image_embs)
    zip_output()
    print("\n=== HOÀN TẤT -- tải a1_data_output.zip về máy, giải nén vào data/ ===")
