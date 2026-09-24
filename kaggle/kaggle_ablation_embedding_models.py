"""
================================================================================
SCRIPT DÙNG TRÊN KAGGLE — Ablation embedding model THẬT: CLIP B/32 vs
CLIP L/14 vs SigLIP (mục 5.1 template A1)
================================================================================

TẠI SAO CHẠY TRÊN KAGGLE, KHÔNG CHẠY LOCAL?
    CLIP ViT-L/14 encode 2000 ảnh trên CPU mất ~35-40 phút CHỈ CHO 1 MODEL
    (đã đo thật: ~1.1 giây/ảnh). Kaggle có GPU T4 miễn phí -- cùng việc đó
    chỉ mất vài phút. SigLIP còn cần thêm thư viện `sentencepiece` (thường
    không có sẵn trong venv local đã cài cho A1) -- Kaggle cài `pip install`
    lại từ đầu mỗi phiên nên không xung đột với requirements.txt của A1.

TẠI SAO KHÔNG DÙNG data/catalog.jsonl CÓ SẴN (tải qua .zip)?
    Script này TỰ TẢI LẠI dataset gốc (giống hệt step1/kaggle_build_A1_data.py,
    CÙNG seed=42 -> CÙNG 2000 sản phẩm, CÙNG thứ tự id) -- để không phải
    upload cả data/images/ (200MB ảnh) làm input Kaggle. Chỉ cần upload
    ĐÚNG 1 file nhỏ: data/eval_set.jsonl (30 câu query + id đúng đã viết ở
    Bước 4/6 local) -- đây là dữ liệu KHÔNG THỂ tự sinh lại, phải mang theo.

CÁCH DÙNG:
    1. Tạo Kaggle Notebook mới, Settings -> Accelerator: GPU (T4 x2 đủ)
    2. Settings -> Internet: ON (bắt buộc)
    3. Add Input -> Upload file `data/eval_set.jsonl` từ máy làm 1 Dataset
       riêng (đặt tên bất kỳ, ví dụ "a1-eval-set") -> gắn vào notebook
    4. Cài thư viện (dấu == với số cụ thể, không dùng dấu <):
           !pip install -q datasets "transformers==4.57.1" sentencepiece faiss-cpu
    5. Copy nguyên file này vào 1 cell (hoặc upload làm Dataset thứ 2, ít
       tiện hơn upload trực tiếp) -- SỬA biến EVAL_SET_PATH bên dưới cho
       khớp đường dẫn Kaggle thật tìm thấy ở tab "Input" (thường dạng
       /kaggle/input/<tên-dataset>/eval_set.jsonl)
    6. Chạy: !python kaggle_ablation_embedding_models.py
    7. Copy bảng kết quả in ra (hoặc tải results.json ở tab Output) về máy,
       dán vào BAO_CAO_TONG_HOP.md mục 5.1 (thay bảng nhận định lý thuyết
       bằng số liệu thật)

LƯU Ý:
    - Mỗi model encode xong sẽ GIẢI PHÓNG khỏi GPU trước khi tải model kế
      tiếp (torch.cuda.empty_cache()) -- tránh tràn VRAM khi chạy tuần tự
      3 model lớn trong cùng 1 session.
    - Đánh giá DENSE-ONLY (không BM25/RRF) cho cả 3 model -- vì mục đích là
      SO SÁNH CHẤT LƯỢNG EMBEDDING với nhau, không phải so hệ thống hybrid
      đầy đủ (BM25 không đổi giữa 3 lần chạy nên không cần lặp lại).
"""
import json
import math
import os
import time

import faiss
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset

# ---------------- Cấu hình ----------------
DATASET_NAME = "Shopify/product-catalogue"
SPLIT = "train"
MAX_ITEMS = 2000
BATCH_SIZE = 32
K = 10

# SỬA đường dẫn này cho khớp dataset bạn attach ở Kaggle (xem Bước 3 ở docstring trên).
# Nếu upload eval_set.jsonl vào CÙNG dataset đã dùng cho kaggle_build_A1_data.py
# (vd tên dataset "shopify-a1-research" như đã dùng ở lần chạy trước), đường dẫn
# sẽ là "/kaggle/input/<tên-dataset-của-bạn>/eval_set.jsonl".
EVAL_SET_PATH = "/kaggle/input/datasets/thanhtrungtran/shopify-a1-research/eval_set.jsonl"

MODELS = [
    {"name": "CLIP ViT-B/32 (baseline A1 gốc)", "hf_id": "openai/clip-vit-base-patch32", "family": "clip"},
    {"name": "CLIP ViT-L/14",                    "hf_id": "openai/clip-vit-large-patch14", "family": "clip"},
    {"name": "SigLIP base patch16-224",          "hf_id": "google/siglip-base-patch16-224", "family": "siglip"},
]


# ---------------- Metric tự viết (copy tối giản từ metrics.py -- xem file gốc
# để đọc giải thích lý thuyết đầy đủ, ở đây chỉ giữ lại phần TÍNH TOÁN) ----------------

def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return None
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(relevant_ids)


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    relevant_ids = set(relevant_ids)
    dcg = sum((1.0 if doc_id in relevant_ids else 0.0) / math.log2(i + 2)
              for i, doc_id in enumerate(retrieved_ids[:k]))
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else None


def mrr(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return None
    relevant_ids = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


# ---------------- Bước A: build lại catalog (giống hệt step1, CÙNG seed) ----------------

def build_catalog():
    print("=== Tải lại dataset gốc (cùng seed=42 -> cùng 2000 sản phẩm, cùng id với local) ===")
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    ds = ds.shuffle(seed=42)

    catalog = []
    n = min(MAX_ITEMS, len(ds))
    for i in range(n):
        item = ds[i]
        title = (item.get("product_title") or "").strip()
        description = (item.get("product_description") or "").strip()
        image = item.get("product_image")
        if image is None or (not title and not description):
            continue
        caption = title if title else description[:200]
        catalog.append({"id": i, "image": image.convert("RGB"), "caption": caption})
        if i % 300 == 0:
            print(f"  đã xử lý {i}/{n}")

    print(f"Xong: {len(catalog)} sản phẩm\n")
    return catalog


def load_eval_set():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------- Bước B: encode ảnh catalog + build FAISS cho 1 model ----------------

def load_model(hf_id, family, device):
    if family == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(hf_id).to(device).eval()
        processor = CLIPProcessor.from_pretrained(hf_id, use_fast=False)
    else:   # siglip -- cùng interface get_image_features/get_text_features như CLIP
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(hf_id).to(device).eval()
        processor = AutoProcessor.from_pretrained(hf_id)
    return model, processor


def encode_images(model, processor, catalog, device):
    embs = []
    for start in range(0, len(catalog), BATCH_SIZE):
        batch = catalog[start:start + BATCH_SIZE]
        images = [r["image"] for r in batch]
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt").to(device)
            feat = model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        embs.append(feat.cpu().numpy())
        if start % (BATCH_SIZE * 5) == 0:
            print(f"    encode ảnh {start + len(batch)}/{len(catalog)}")
    return np.concatenate(embs, axis=0).astype("float32")


def encode_query_text(model, processor, text, device, family="clip"):
    """
    family="siglip" PHẢI pad tới ĐỘ DÀI CỐ ĐỊNH (padding="max_length", 64 token)
    -- khác CLIP (padding=True, pad động theo câu ngắn nhất trong batch). Lý do:
    SigLIP KHÔNG dùng attention_mask, nó lấy biểu diễn tại VỊ TRÍ TOKEN CUỐI
    CÙNG của chuỗi đã pad. Lúc huấn luyện, SigLIP LUÔN pad tới 64 -- token
    "cuối cùng" luôn nằm ở vị trí 63. Nếu pad động (câu ngắn ra length 5),
    "token cuối" rơi vào vị trí 4 -- một vùng positional embedding CHƯA TỪNG
    được huấn luyện ở ngữ cảnh này, làm embedding gần như nhiễu ngẫu nhiên
    (đây chính là nguyên nhân SigLIP ra Recall/nDCG/MRR = 0.000 khi chạy lần
    đầu -- không phải model tệ, mà là dùng sai cách gọi processor).
    """
    with torch.no_grad():
        if family == "siglip":
            inputs = processor(text=[text], return_tensors="pt", padding="max_length",
                                truncation=True, max_length=64).to(device)
        else:
            inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
        feat = model.get_text_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")


# ---------------- Bước C: đánh giá dense-only cho 1 model ----------------

def evaluate_model(model_cfg, catalog, eval_set, device):
    print(f"\n### {model_cfg['name']} ({model_cfg['hf_id']}) ###")
    model, processor = load_model(model_cfg["hf_id"], model_cfg["family"], device)

    print("  Đang encode ảnh catalog...")
    image_embs = encode_images(model, processor, catalog, device)

    index = faiss.IndexFlatIP(image_embs.shape[1])
    index.add(image_embs)
    ids = [r["id"] for r in catalog]

    recalls, ndcgs, mrrs, latencies = [], [], [], []
    for item in eval_set:
        t0 = time.perf_counter()
        query_vec = encode_query_text(model, processor, item["query"], device, family=model_cfg["family"])
        scores, idxs = index.search(query_vec, K)
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved_ids = [ids[i] for i in idxs[0].tolist()]
        r = recall_at_k(retrieved_ids, item["relevant_ids"], K)
        n = ndcg_at_k(retrieved_ids, item["relevant_ids"], K)
        m = mrr(retrieved_ids, item["relevant_ids"])
        if r is not None:
            recalls.append(r)
        if n is not None:
            ndcgs.append(n)
        if m is not None:
            mrrs.append(m)

    embedding_dim = int(image_embs.shape[1])   # lưu lại TRƯỚC khi del, xem chú thích dưới

    # Giải phóng model khỏi GPU trước khi load model tiếp theo -- tránh tràn VRAM
    del model, processor, image_embs, index
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_cfg["name"],
        "recall@k": sum(recalls) / len(recalls) if recalls else None,
        "ndcg@k": sum(ndcgs) / len(ndcgs) if ndcgs else None,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else None,
        "latency_ms_avg": sum(latencies) / len(latencies) if latencies else None,
        "embedding_dim": embedding_dim,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("CẢNH BÁO: không thấy GPU -- kiểm tra lại Settings > Accelerator trên Kaggle,")
        print("script vẫn chạy được nhưng sẽ chậm gần như local (mất hàng chục phút/model).")

    catalog = build_catalog()
    eval_set = load_eval_set()
    print(f"Đã load {len(eval_set)} câu eval\n")

    results = []
    for model_cfg in MODELS:
        results.append(evaluate_model(model_cfg, catalog, eval_set, device))

    print("\n\n================ BẢNG KẾT QUẢ (copy vào BAO_CAO_TONG_HOP.md mục 5.1) ================")
    header = f"{'Model':<32} {'dim':<6} {'Recall@'+str(K):<10} {'nDCG@'+str(K):<10} {'MRR':<8} {'Latency_avg(ms)':<16}"
    print(header)
    print("-" * len(header))
    for r in results:
        def fmt(key, spec=".3f"):
            v = r[key]
            return format(v, spec) if v is not None else "N/A"
        print(f"{r['model']:<32} {r['embedding_dim']:<6} {fmt('recall@k'):<10} {fmt('ndcg@k'):<10} "
              f"{fmt('mrr'):<8} {fmt('latency_ms_avg', '.1f'):<16}")

    with open("embedding_model_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nĐã lưu embedding_model_ablation_results.json -- xem tab Output để tải về.")


if __name__ == "__main__":
    main()
