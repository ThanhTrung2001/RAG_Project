"""
Bước 2: sinh embedding CLIP cho ảnh và caption của catalog.

Input: data/catalog.jsonl, data/images/ (Bước 1).
Output: data/image_embeddings.npy, data/text_embeddings.npy (float32, đã chuẩn hoá L2).
Chạy: python step2_build_embeddings.py
"""
import json
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CATALOG_PATH = "data/catalog.jsonl"
OUT_IMAGE_EMB = "data/image_embeddings.npy"
OUT_TEXT_EMB = "data/text_embeddings.npy"

MODEL_NAME = "openai/clip-vit-base-patch32"   # CLIP nhỏ, chạy được trên CPU
BATCH_SIZE = 32


def load_catalog():
    rows = []
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dùng device: {device}")

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    rows = load_catalog()
    print(f"Đã load {len(rows)} ảnh từ catalog")

    image_embs, text_embs = [], []

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]

        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        captions = [r["caption"] for r in batch]

        with torch.no_grad():
            img_inputs = processor(images=images, return_tensors="pt").to(device)
            img_feat = model.get_image_features(**img_inputs)
            # Chuẩn hoá L2 để inner product == cosine (dùng với faiss.IndexFlatIP).
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            txt_inputs = processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)
            txt_feat = model.get_text_features(**txt_inputs)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        image_embs.append(img_feat.cpu().numpy())
        text_embs.append(txt_feat.cpu().numpy())

        print(f"  đã encode {start + len(batch)}/{len(rows)}")

    # FAISS yêu cầu float32.
    image_embs = np.concatenate(image_embs, axis=0).astype("float32")
    text_embs = np.concatenate(text_embs, axis=0).astype("float32")

    np.save(OUT_IMAGE_EMB, image_embs)
    np.save(OUT_TEXT_EMB, text_embs)
    print(f"Xong. Lưu {image_embs.shape} vào {OUT_IMAGE_EMB}")
    print(f"Xong. Lưu {text_embs.shape} vào {OUT_TEXT_EMB}")


if __name__ == "__main__":
    main()
