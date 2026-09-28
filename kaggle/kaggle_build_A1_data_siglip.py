"""
Chạy step1 + step2 + step3 trên Kaggle (GPU) DÙNG SigLIP thay CLIP B/32, đóng gói data/ thành a1_data_output_siglip.zip.

TẠI SAO ĐỔI SANG SigLIP?
    Ablation embedding model (BAO_CAO_TONG_HOP.md mục 6.5, kaggle_ablation_embedding_models.py)
    đo thật trên 250 câu: SigLIP Recall@10=0.920 vs CLIP B/32=0.656 (dense-only) -- chênh lệch
    đủ lớn để đáng chuyển production sang SigLIP thay vì chỉ dừng ở so sánh lý thuyết.

KHÁC GÌ SO VỚI kaggle_build_A1_data.py (bản CLIP)?
    - MODEL_NAME đổi sang "google/siglip-base-patch16-224", dùng AutoModel/AutoProcessor
      (không phải CLIPModel/CLIPProcessor) -- cùng interface get_image_features()/get_text_features().
    - image_embeddings.npy giờ có dim=768 (thay vì 512) -- KHÔNG tương thích ngược với
      dense.index cũ build bằng CLIP, phải build lại HOÀN TOÀN, không trộn lẫn được.
    - BM25 (bm25.pkl) KHÔNG đổi gì -- không phụ thuộc embedding model, build lại y hệt cho
      đồng bộ nhưng số liệu bên trong sẽ giống hệt bản CLIP.
    - Bước encode ảnh KHÔNG cần lo padding="max_length" (quirk đó chỉ áp dụng khi encode TEXT
      của QUERY lúc search -- xem search_core.py sau khi cập nhật). Encode ẢNH ở đây dùng
      processor(images=...) bình thường, không đụng tới tokenizer/padding.

SAU KHI TẢI a1_data_output_siglip.zip VỀ MÁY:
    1. Giải nén ĐÈ vào thư mục data/ của project (thay data/image_embeddings.npy,
       data/dense.index, data/bm25.pkl, data/catalog.jsonl -- catalog giữ nguyên id vì
       cùng seed=42, không mất eval_set.jsonl/rag stuff vì chúng nằm ngoài các file này).
    2. Sửa search_core.py: đổi MODEL_NAME + CLIPModel/CLIPProcessor -> AutoModel/AutoProcessor,
       và THÊM đúng padding="max_length", max_length=64 khi encode TEXT (xem
       kaggle_ablation_embedding_models.py::encode_query_text để copy lại logic đã kiểm chứng).
    3. Chạy lại step8_run_ablation.py, step9_ablation_index_fields.py, step10_sweep_rrf.py,
       step12_case_study_trace.py, step13/14 (ảnh) -- MỌI số liệu phụ thuộc nhánh dense sẽ
       đổi vì embedding gốc đổi hẳn, không dùng lại số liệu CLIP cũ được.

Kaggle: bật GPU và Internet, restart session nếu đã cài thư viện khác phiên bản, rồi chạy
    !pip install -q datasets "transformers==4.57.1" sentencepiece faiss-cpu rank_bm25
    !python kaggle_build_A1_data_siglip.py
Tải a1_data_output_siglip.zip ở tab Output.
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
from transformers import AutoModel, AutoProcessor

# Cấu hình giống step1-3, chỉ đổi MODEL_NAME + tên file output để không đè lên bản CLIP
OUT_DIR = "data"
IMAGES_DIR = os.path.join(OUT_DIR, "images")
CATALOG_PATH = os.path.join(OUT_DIR, "catalog.jsonl")
IMAGE_EMB_PATH = os.path.join(OUT_DIR, "image_embeddings.npy")
DENSE_INDEX_PATH = os.path.join(OUT_DIR, "dense.index")
BM25_INDEX_PATH = os.path.join(OUT_DIR, "bm25.pkl")
OUTPUT_ZIP = "a1_data_output_siglip.zip"

DATASET_NAME = "Shopify/product-catalogue"
SPLIT = "train"
MAX_ITEMS = 2000
MODEL_NAME = "google/siglip-base-patch16-224"
BATCH_SIZE = 32


def step1_build_catalog():
    print("=== BƯỚC 1: Tải dataset, build catalog (cùng seed=42 nên cùng id với bản CLIP) ===")
    os.makedirs(IMAGES_DIR, exist_ok=True)
    ds = load_dataset(DATASET_NAME, split=SPLIT)
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
    print(f"=== BƯỚC 2: Sinh embedding SigLIP ({MODEL_NAME}, dùng GPU nếu có) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

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
    print(f"Xong bước 2: lưu {image_embs.shape} vào {IMAGE_EMB_PATH} (dim={image_embs.shape[1]}, so với 512 của CLIP B/32)\n")
    return image_embs


def step3_build_index(catalog, image_embs):
    print("=== BƯỚC 3: Build FAISS (dim mới từ SigLIP) + BM25 (không đổi) ===")
    index = faiss.IndexFlatIP(image_embs.shape[1])
    index.add(image_embs)
    faiss.write_index(index, DENSE_INDEX_PATH)

    tokenized = [r["search_text"].lower().split() for r in catalog]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": [r["id"] for r in catalog]}, f)
    print(f"Xong bước 3: {index.ntotal} vector, {len(catalog)} document BM25\n")


def zip_output():
    print(f"=== Đóng gói data/ thành {OUTPUT_ZIP} ===")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUT_DIR):
            for file in files:
                filepath = os.path.join(root, file)
                zf.write(filepath, filepath)
    size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"Xong: {OUTPUT_ZIP} ({size_mb:.1f} MB) -- vào tab Output bên phải để tải về")


if __name__ == "__main__":
    catalog = step1_build_catalog()
    image_embs = step2_build_embeddings(catalog)
    step3_build_index(catalog, image_embs)
    zip_output()
    print("\n=== HOÀN TẤT -- tải a1_data_output_siglip.zip về máy, giải nén ĐÈ vào data/, ===")
    print("=== rồi cập nhật search_core.py sang SigLIP (xem docstring đầu file này) ===")
