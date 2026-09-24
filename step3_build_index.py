"""
Bước 3: build index dense (FAISS, trên embedding ảnh) và sparse (BM25, trên search_text).

Input: data/catalog.jsonl (Bước 1), data/image_embeddings.npy (Bước 2).
Output: data/dense.index, data/bm25.pkl ({"bm25": BM25Okapi, "ids": [...]}).
Chạy: python step3_build_index.py
"""
import json
import pickle

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

CATALOG_PATH = "data/catalog.jsonl"
IMAGE_EMB_PATH = "data/image_embeddings.npy"
DENSE_INDEX_PATH = "data/dense.index"
BM25_INDEX_PATH = "data/bm25.pkl"


def simple_tokenize(text: str):
    """Tokenize tối giản: chữ thường + tách theo khoảng trắng (không bỏ dấu câu, không stemming)."""
    return text.lower().split()


def main():
    # Dense index (FAISS)
    image_embs = np.load(IMAGE_EMB_PATH)

    # Flat = brute-force chính xác; IP = inner product (== cosine vì vector đã chuẩn hoá ở Bước 2).
    dim = image_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(image_embs)

    faiss.write_index(index, DENSE_INDEX_PATH)
    print(f"Dense index: {index.ntotal} vectors, dim={dim} -> {DENSE_INDEX_PATH}")

    # Sparse index (BM25)
    rows = []
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # Dùng search_text (title + description) để có nhiều từ khoá hơn caption.
    tokenized_captions = [simple_tokenize(r.get("search_text", r["caption"])) for r in rows]
    bm25 = BM25Okapi(tokenized_captions)

    # BM25Okapi không có hàm save nên dùng pickle; "ids" map vị trí -> id sản phẩm.
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": [r["id"] for r in rows]}, f)
    print(f"BM25 index: {len(rows)} captions -> {BM25_INDEX_PATH}")


if __name__ == "__main__":
    main()
