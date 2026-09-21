"""
================================================================================
BƯỚC 2 — SINH EMBEDDING CHO ẢNH VÀ TEXT BẰNG CLIP
================================================================================

TẠI SAO CẦN CLIP?
    CLIP là model được huấn luyện để ẢNH và TEXT mô tả đúng ảnh đó nằm GẦN
    NHAU trong CÙNG MỘT không gian vector (gọi là "joint embedding space").
    Đây là điều đặc biệt: encoder ảnh và encoder text là 2 mạng neural HOÀN
    TOÀN KHÁC NHAU, nhưng output của chúng có thể so sánh trực tiếp bằng
    cosine similarity.

    Nhờ vậy, một hệ thống có thể tìm theo cả 2 chiều:
      - Người dùng gõ text -> encode thành vector -> so với vector ẢNH đã lưu
      - Người dùng đưa ảnh  -> encode thành vector -> so với vector ẢNH/TEXT đã lưu

    Nếu không dùng model kiểu CLIP, sẽ phải có 2 không gian vector RIÊNG BIỆT
    (1 cho ảnh, 1 cho text) và KHÔNG so được trực tiếp với nhau.

TẠI SAO CHẠY 1 LẦN RỒI CACHE RA FILE, KHÔNG TÍNH LẠI MỖI LẦN SEARCH?
    Encode 2000 ảnh qua CLIP tốn vài phút (tuỳ máy) -- nếu phải chạy lại mỗi
    lần người dùng gõ 1 câu search thì hệ thống sẽ CỰC KỲ chậm. Vì vậy:
      - Embedding của CATALOG (dữ liệu có sẵn) -> tính 1 lần ở bước này, lưu ra .npy
      - Embedding của QUERY (câu người dùng gõ) -> tính real-time lúc search
        (chỉ 1 câu/1 ảnh, rất nhanh, không cần cache)

CHẠY: python step2_build_embeddings.py
OUTPUT: data/image_embeddings.npy, data/text_embeddings.npy
"""
import json
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CATALOG_PATH = "data/catalog.jsonl"
OUT_IMAGE_EMB = "data/image_embeddings.npy"
OUT_TEXT_EMB = "data/text_embeddings.npy"

MODEL_NAME = "openai/clip-vit-base-patch32"   # bản CLIP nhỏ, đủ tốt cho A1, chạy nhanh cả trên CPU
BATCH_SIZE = 32   # xử lý 32 ảnh/lần thay vì từng ảnh một -- tận dụng GPU/CPU hiệu quả hơn,
                   # và tránh tràn bộ nhớ nếu xử lý cả nghìn ảnh cùng lúc


def load_catalog():
    """Đọc lại catalog.jsonl đã tạo ở bước 1 -- mỗi dòng là 1 dict."""
    rows = []
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    # Tự động dùng GPU nếu máy có (nhanh hơn nhiều), fallback về CPU nếu không có.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dùng device: {device}")

    # .eval(): tắt các layer chỉ dùng khi train (dropout, batchnorm update) --
    # BẮT BUỘC khi chỉ dùng model để suy luận (inference), không training.
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)   # xử lý resize ảnh, tokenize text
                                                              # đúng chuẩn model này cần

    rows = load_catalog()
    print(f"Đã load {len(rows)} ảnh từ catalog")

    image_embs, text_embs = [], []

    # Chạy theo từng batch thay vì từng dòng một
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]

        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        captions = [r["caption"] for r in batch]

        # torch.no_grad(): tắt việc tính gradient (chỉ cần khi training) --
        # giúp chạy nhanh hơn và tốn ít RAM/VRAM hơn đáng kể.
        with torch.no_grad():
            # --- Encode ảnh ---
            img_inputs = processor(images=images, return_tensors="pt").to(device)
            img_feat = model.get_image_features(**img_inputs)
            # CHUẨN HOÁ vector về độ dài 1 (unit vector). Sau bước này:
            #   cosine_similarity(a, b) == dot_product(a, b)
            # Lý do quan trọng: cho phép dùng faiss.IndexFlatIP (inner product) ở bước 3,
            # vốn nhanh hơn tính cosine similarity thủ công, và giúp công thức RRF
            # ở bước 4 nhất quán khi so điểm giữa các query khác nhau.
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            # --- Encode text (caption) ---
            # padding=True: các câu caption dài ngắn khác nhau trong 1 batch được
            #   đệm thêm token để có cùng độ dài -- CLIP cần input hình chữ nhật đều.
            # truncation=True: cắt bớt nếu caption dài hơn giới hạn token của model.
            txt_inputs = processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)
            txt_feat = model.get_text_features(**txt_inputs)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        # .cpu().numpy(): chuyển tensor PyTorch (có thể đang ở GPU) về numpy array
        # trên CPU -- vì bước 3 (FAISS) làm việc với numpy, không phải tensor PyTorch.
        image_embs.append(img_feat.cpu().numpy())
        text_embs.append(txt_feat.cpu().numpy())

        print(f"  đã encode {start + len(batch)}/{len(rows)}")

    # Gộp tất cả batch lại thành 1 ma trận lớn duy nhất, shape (N, 512).
    # .astype("float32"): FAISS yêu cầu float32, PyTorch mặc định có thể trả float64.
    image_embs = np.concatenate(image_embs, axis=0).astype("float32")
    text_embs = np.concatenate(text_embs, axis=0).astype("float32")

    np.save(OUT_IMAGE_EMB, image_embs)
    np.save(OUT_TEXT_EMB, text_embs)
    print(f"Xong. Lưu {image_embs.shape} vào {OUT_IMAGE_EMB}")
    print(f"Xong. Lưu {text_embs.shape} vào {OUT_TEXT_EMB}")


if __name__ == "__main__":
    main()
