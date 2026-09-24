# A1 — Hệ tìm kiếm đa phương thức (text + ảnh)

**Dataset:** Shopify/product-catalogue (Hugging Face) — 2000 sản phẩm đầu, ảnh + text + metadata.
**Kiến trúc:** 100% local, không cần server/tài khoản ngoài — FAISS (dense) + BM25 (sparse) + RRF fusion.
**Đọc trước khi làm:** `BAO_CAO_TONG_HOP.md` — lý thuyết + code + rubric đầy đủ, dùng để viết báo cáo nộp.

## Cài đặt
```bash
pip install -r requirements.txt --break-system-packages
```

## Chạy theo đúng thứ tự (bắt buộc — bước sau phụ thuộc bước trước)

```bash
# Bước 1: tải dataset Shopify/product-catalogue, chuẩn hoá thành catalog.jsonl
python step1_build_catalog.py

# Bước 2: sinh embedding CLIP cho ảnh + caption (bước tốn thời gian nhất)
python step2_build_embeddings.py

# Bước 3: build FAISS (dense) + BM25 (sparse)
python step3_build_index.py

# Bước 6 (làm TAY — không thể code sẵn): mở data/catalog.jsonl xem sản phẩm
# thật, tự viết 20-40 câu query + id đúng cho mỗi câu
cp data/eval_set_template.jsonl data/eval_set.jsonl
# mở file eval_set.jsonl vừa copy, viết thêm cho đủ số lượng

# Bước 8: chạy ablation, in bảng Recall@k / nDCG@k cho từng cấu hình
python step8_run_ablation.py

# Bước 7 + 10: chạy API + frontend demo
uvicorn app:app --reload --port 8000
# mở trình duyệt: http://localhost:8000
```

## Cấu trúc file — map với các bước trong BAO_CAO_TONG_HOP.md

| File | Bước | Vai trò |
|---|---|---|
| `step1_build_catalog.py` | 1 | Tải dataset, tạo `catalog.jsonl` + thư mục `images/` |
| `step2_build_embeddings.py` | 2 | Sinh CLIP embedding cho ảnh + caption, lưu ra `.npy` |
| `step3_build_index.py` | 3 | Build FAISS (`dense.index`) + BM25 (`bm25.pkl`) |
| `search_core.py` | 4 | Registry pattern: `@component("bm25")`, `@component("dense")`, `search()`, `rrf_fusion()` |
| `metrics.py` | 5 | `recall_at_k()`, `ndcg_at_k()`, `mrr()`, `ndcg_at_k_graded()` (relevance đa mức 0/1/2) tự viết tay (không dùng sklearn); `evaluate_all()` đo thêm latency avg/p95 |
| `data/eval_set_template.jsonl` | 6 | 8 câu mẫu thật từ dataset — cần viết thêm cho đủ 20-40 câu |
| `app.py` | 7 | FastAPI: `/api/v1/search`, `/api/v1/search_by_image`, `/api/v1/components` |
| `step8_run_ablation.py` | 8 | Chạy 4 cấu hình (bm25-only / dense-only / hybrid / hybrid+rerank), in bảng Recall@10, nDCG@10, nDCG đa mức, MRR, Latency |
| *(bạn tự làm, không code sẵn được)* | 9 | Error analysis — đọc kết quả ablation, gom nhóm lỗi theo nguyên nhân |
| `frontend/index.html` | 10 | UI tối giản, checkbox tự sinh từ `/api/v1/components` |
| `BAO_CAO_TONG_HOP.md` | 11 | Toàn bộ lý thuyết + code + rubric — rút gọn thành báo cáo ≤8 trang khi nộp |

### Ablation nâng cao (tuỳ chọn, đã có sẵn — dùng khi viết mục 5.x của báo cáo)

| File | Vai trò |
|---|---|
| `step9_ablation_index_fields.py` | So sánh BM25 index theo trường: title-only / title+description / +ảnh (build BM25 tạm trong RAM, không đụng `data/bm25.pkl` thật) |
| `step10_sweep_rrf.py` | Sweep `rrf_k ∈ {10,60,100}` × `top_n (N) ∈ {20,50,100}` — dùng tham số `rrf_k` mới thêm ở `search()` |
| `step12_case_study_trace.py` | Truy vết 1 câu qua từng tầng BM25 → Dense → RRF → Rerank, tự diễn giải tầng nào cứu/làm hỏng hạng |
| `step13_build_image_eval_set.py` | Dựng bộ truy vấn ẢNH bằng crop/rotate/brightness từ ảnh thật trong `data/images/` |
| `step14_eval_image_queries.py` | Đánh giá bộ truy vấn ảnh vừa dựng — Recall/nDCG/MRR theo từng loại transform |
| `kaggle/kaggle_ablation_embedding_models.py` | So CLIP B/32 vs CLIP L/14 vs SigLIP bằng số thật — chạy trên Kaggle (GPU), xem hướng dẫn trong docstring đầu file |

## Mọi file `.py` đều có comment đầy đủ theo cấu trúc

M��i file bắt đầu bằng 1 khối docstring giải thích:
- **Bước này giải quyết vấn đề gì** (tại sao cần, nếu bỏ qua thì sao)
- **Cách hoạt động** (thuật toán, công thức, ví dụ tính tay nếu có)
- **Cách chạy + output mong đợi**

Trong thân hàm, các dòng code có ý nghĩa không hiển nhiên (ví dụ `feat.norm()`,
`np.argsort()[::-1]`, decorator `@component`) đều có comment ngay bên cạnh
giải thích TẠI SAO viết như vậy — đọc trực tiếp trong file `.py` để hiểu sâu
hơn phần tóm tắt trong README này.

## Việc bạn cần tự làm thêm (không thể code sẵn vì phụ thuộc dataset/domain của bạn)

1. **Viết `data/eval_set.jsonl` thật** — 20-40 câu, xem ảnh thật trong
   `data/images/` để biết id nào đúng cho câu nào. Phần tốn công nhất
   nhưng bắt buộc (rubric chấm riêng tiêu chí này, 20% điểm).
2. **Error analysis (bước 9)** — sau khi chạy `step8_run_ablation.py`, lọc
   các câu có Recall thấp, mở ảnh lên xem, đối chiếu `category`/`brand`
   trong catalog, gom nhóm theo nguyên nhân. Không có code nào thay được
   việc bạn tự nhìn và phân tích (rubric 20%).
3. **(Tuỳ chọn, nâng điểm engineering) Thêm component rerank** — viết hàm
   dùng cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), gắn
   `@component("rerank")` trong `search_core.py`. Vì rerank cần chạy SAU
   khi đã có kết quả fusion (không phải 1 nhánh song song với BM25/Dense),
   cần thêm logic riêng cho bước hậu xử lý này — không đăng ký chung
   registry với 2 component hiện tại.
