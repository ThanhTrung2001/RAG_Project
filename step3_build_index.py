"""
================================================================================
BƯỚC 3 — BUILD 2 INDEX SONG SONG: FAISS (dense) + BM25 (sparse)
================================================================================

TẠI SAO CẦN INDEX, KHÔNG SO SÁNH VECTOR TRỰC TIẾP MỖI LẦN SEARCH?
    Với vài nghìn ảnh, so sánh tuần tự (brute-force) từng vector một VẪN đủ
    nhanh -- nhưng "index" ở đây (FAISS IndexFlatIP) thực chất CHÍNH LÀ cách
    làm brute-force này một cách tối ưu (dùng thư viện C++ tính toán ma trận
    nhanh hơn code Python thuần rất nhiều). Với dataset lớn hơn (hàng triệu
    ảnh), FAISS còn hỗ trợ các thuật toán index xấp xỉ (ANN) nhanh hơn nữa --
    nhưng ở quy mô A1, IndexFlatIP là đủ và cho kết quả CHÍNH XÁC TUYỆT ĐỐI
    (không xấp xỉ).

TẠI SAO CẦN CẢ BM25, KHÔNG DÙNG DUY NHẤT DENSE (CLIP EMBEDDING)?
    Dense embedding (CLIP) giỏi bắt được NGHĨA (semantic) -- "chó" và "cún"
    cho ra vector gần nhau dù chữ khác nhau hoàn toàn. Nhưng nó lại YẾU với
    việc khớp CHÍNH XÁC từ hiếm gặp: mã sản phẩm ("HP1044"), tên riêng
    thương hiệu ("SOG"), vì embedding có xu hướng "làm mờ" các chi tiết cụ
    thể để tối ưu cho similarity tổng quát.

    BM25 (thuật toán xếp hạng dựa trên tần suất từ, KHÔNG dùng embedding)
    làm ngược lại: rất mạnh khi khớp đúng từ/cụm từ, nhưng KHÔNG hiểu được
    đồng nghĩa hay diễn đạt khác nhau.

    Build CẢ 2 index song song để chúng bổ khuyết điểm yếu cho nhau -- việc
    hợp nhất 2 kết quả này (RRF fusion) nằm ở bước 4.

CHẠY: python step3_build_index.py
OUTPUT: data/dense.index (FAISS), data/bm25.pkl (BM25 + danh sách id)
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
    """
    Tách câu thành danh sách từ đơn giản nhất có thể: chữ thường + tách theo
    khoảng trắng. BM25 hoạt động trên "từ" (token), không phải chuỗi ký tự thô.
    Đây là bản tokenize NGÂY THƠ nhất -- không xử lý dấu câu, không stemming --
    nhưng đủ dùng cho A1 và dễ giải thích trong báo cáo. Cải tiến (nếu muốn
    nâng điểm engineering) có thể thêm loại bỏ stopword, xử lý số nhiều...
    """
    return text.lower().split()


def main():
    # ---------------- Dense index (FAISS) ----------------
    image_embs = np.load(IMAGE_EMB_PATH)   # đọc lại embedding đã tính ở bước 2

    # IndexFlatIP: "Flat" nghĩa là brute-force (so mọi vector, không xấp xỉ),
    # "IP" nghĩa là Inner Product -- vì embedding đã được chuẩn hoá độ dài 1
    # ở bước 2, nên inner product == cosine similarity, tính nhanh hơn.
    dim = image_embs.shape[1]   # 512 với model clip-vit-base-patch32
    index = faiss.IndexFlatIP(dim)
    index.add(image_embs)   # nạp toàn bộ vector vào index

    # write_index: lưu index ra file nhị phân -- lần sau chỉ cần đọc lại
    # bằng faiss.read_index(), không phải add() lại từ đầu.
    faiss.write_index(index, DENSE_INDEX_PATH)
    print(f"Dense index: {index.ntotal} vectors, dim={dim} -> {DENSE_INDEX_PATH}")

    # ---------------- Sparse index (BM25) ----------------
    rows = []
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # Dùng "search_text" (title + description, đầy đủ), KHÔNG dùng "caption"
    # (chỉ title, ngắn) -- vì BM25 cần càng nhiều từ khoá càng dễ khớp đúng.
    tokenized_captions = [simple_tokenize(r.get("search_text", r["caption"])) for r in rows]
    bm25 = BM25Okapi(tokenized_captions)   # BM25Okapi tự tính sẵn thống kê tần suất từ
                                             # trên toàn bộ corpus ngay khi khởi tạo

    # BM25Okapi không tự lưu ra file được (không có hàm save() sẵn) -- dùng
    # pickle để "đóng băng" toàn bộ object Python (bao gồm mọi thống kê đã tính)
    # ra đĩa. Lưu kèm "ids" để biết vị trí thứ i trong bm25 tương ứng sản phẩm nào.
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": [r["id"] for r in rows]}, f)
    print(f"BM25 index: {len(rows)} captions -> {BM25_INDEX_PATH}")


if __name__ == "__main__":
    main()
