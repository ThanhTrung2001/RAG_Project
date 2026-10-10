# KẾ HOẠCH TỔNG HỢP — A1, A2, A3, BÁO CÁO BPM
### Đọc file này đầu tiên — bản đồ toàn bộ project, việc gì đã xong, việc gì còn thiếu

---

## 0. Lưu ý mới từ giảng viên (đã áp dụng vào code + báo cáo)

| Note của giảng viên | Đã xử lý ra sao |
|---|---|
| Multimodal không bắt buộc, nhóm tự chọn phạm vi | Không cần đổi gì — A1 đã chọn multimodal (ảnh+text), ghi rõ trong báo cáo là lựa chọn, không phải ép buộc |
| Dataset nhỏ được khuyến khích, subset lớn phải đảm bảo đa dạng | **Đã sửa `step1_build_catalog.py`/`kaggle_build_A1_data.py`**: thêm `ds.shuffle(seed=42)` trước khi cắt lấy 2000 dòng, thêm in phân bố category để tự kiểm tra |
| Phải có ground truth (qrels) cho retrieval | `data/eval_set.jsonl` (250 câu, dùng CHUNG cho A1 và A3) — có `relevant_ids` (retrieval, A1) VÀ `answer_should_mention` (generation, A3) trên MỌI dòng, xem `step15_expand_eval_set.py` |
| Không cần train/fine-tune, dùng embedding có sẵn, phải giải thích lý do chọn model | **Đã thêm mục "Vì sao chọn CLIP ViT-B/32"** vào `BAO_CAO_TONG_HOP.md` — so sánh với ViT-L/14, SigLIP, sentence-transformers |
| Rerank (cross-encoder/VLM) phải đặt ở service layer, không đặt trong database | Đã đúng từ đầu — `rerank()` nằm trong `search_core.py` (service layer), không phải SQL |
| Tối thiểu 4 cấu hình ablation: BM25/Dense/Hybrid/Hybrid+rerank | ✅ Đã đủ 4/4 trong `step8_run_ablation.py` |
| Đối chiếu template `Bao_cao_A1_IS6303_template.docx`: cần thêm MRR + Latency (mục 4.1), sweep RRF k×N (5.2), so sánh reranker (5.3), ablation theo trường index (5.4), case study theo tầng (6.x) | ✅ Đã thêm `mrr()` + `evaluate_all()` đo latency vào `metrics.py`; thêm `rrf_k` xuyên suốt `search_core.py`; thêm 3 script mới `step9_ablation_index_fields.py`/`step10_sweep_rrf.py`/`step12_case_study_trace.py`; mục 5.3 dùng lại 2 dòng cuối bảng `step8` (không cần RankGPT trong A1). |
| Bộ truy vấn ẢNH bằng biến đổi thật (mục 2.3/4.3) | ✅ **Đã chạy xong** — `step13_build_image_eval_set.py` (crop/rotate/brightness từ 12 ảnh thật) + `step14_eval_image_queries.py` — Recall/nDCG/MRR = 1.000 cả 3 loại (CLIP rất bền với biến đổi mức vừa phải đã thử) |
| Ablation embedding model CLIP B/32 vs L/14 vs SigLIP bằng số thật (mục 5.1) | ✅ **Đã chạy xong trên Kaggle** — SigLIP thắng cả 3 metric (Recall 0.967 vs 0.767 baseline). Gặp + sửa 1 bug thật khi chạy: SigLIP cần `padding="max_length"` (khác CLIP), sai cách gọi ban đầu làm Recall = 0.000 tuyệt đối — chi tiết trong `BAO_CAO_TONG_HOP.md` mục 11.4, đáng đưa vào phần "bài học/hạn chế" của báo cáo |
| A3 phải chẩn đoán được tại sao tốt/xấu (kiểu "RankGPT dở vì model kém"), tách retrieval vs generation | **Đã thêm mục 10c** vào `BAO_CAO_A3.md` — khung chẩn đoán 2 tầng, có ví dụ cụ thể |
| A3 phải đưa lên dạng chatbot | **Đã sửa** `rag_core.py` (thêm Memory/session), `app.py` (endpoint nhận `session_id`), `frontend/index.html` (giao diện chat log đa lượt thay vì hộp hỏi-đáp đơn) |
| A3 phải "upgrade" hệ search A1 bằng LLM + ingestion, đánh giá bằng ĐỦ metric RAG (faithfulness, context precision...), ground truth tối thiểu 50 câu/người (nhóm 5 người → 250 câu, ÁP DỤNG TỪ A1, không phải file riêng cho A3) | ✅ Kiến trúc RAG đã đủ 3 tầng Retrieval→Augmentation(grounding)→Generation từ trước (xem mục 10 dưới). **Mới bổ sung lần này:** `rag_metrics.py` (context precision/faithfulness/answer relevancy tự viết); `step15_expand_eval_set.py` mở rộng THẲNG `data/eval_set.jsonl` (A1, 30→250 câu, tiếng Anh, có `answer_should_mention`) — **không tạo file `rag_eval_set.jsonl` riêng nữa** (bản trước đã xoá); sửa `step_ablation_rag.py` đọc đúng `eval_set.jsonl`. Đã CHẠY LẠI `step8/9/10_run_ablation*.py` trên 250 câu, cập nhật số liệu thật trong `BAO_CAO_TONG_HOP.md`. ⚠️ **step_ablation_rag.py (A3) chưa chạy** — cần Ollama trên máy thật, xem mục 4 BƯỚC |

---

## 1. Tổng quan 4 phần việc

| Phần | Môn | Đã xong | Còn thiếu |
|---|---|---|---|
| **A1** — Search | IS6303 (25%) | Code + báo cáo đầy đủ | Chạy thật lấy `eval_set.jsonl` + số liệu ablation thật |
| **A2** — Đọc paper | IS6303 (25%) | Báo cáo đầy đủ | Không còn gì |
| **A3** — RAG | IS6303 (đồ án cuối kỳ) | Code + báo cáo + **ablation vừa bổ sung** | Chạy thật lấy số liệu ablation RAG |
| **BPM** | IS6003 (ERP nâng cao) | Báo cáo + script mô phỏng | Không bắt buộc, có thể làm thêm nếu muốn số liệu thật |

---

## 2. Danh sách file — map với từng phần

### Code A1 (Search)
| File | Vai trò |
|---|---|
| `step1_build_catalog.py` | Tải dataset Shopify, tạo `catalog.jsonl` |
| `step2_build_embeddings.py` | Sinh CLIP embedding |
| `step3_build_index.py` | Build FAISS + BM25 |
| `search_core.py` | Registry pattern: BM25+Dense+RRF |
| `metrics.py` | Recall@k, nDCG@k, MRR, nDCG đa mức (0/1/2), latency avg/p95 — tự viết |
| `step8_run_ablation.py` | Ablation A1: BM25-only/Dense-only/Hybrid/Hybrid+rerank, đủ Recall/nDCG/nDCG đa mức/MRR/Latency |
| `step9_ablation_index_fields.py` | **[MỚI]** Ablation A1: title-only/title+desc/+ảnh (mục 5.4 template) |
| `step10_sweep_rrf.py` | **[MỚI]** Ablation A1: sweep rrf_k × top_n (mục 5.2 template) |
| `step12_case_study_trace.py` | **[MỚI]** Case study A1: trace 1 câu qua BM25/Dense/RRF/Rerank (mục 6 template) |
| `step13_build_image_eval_set.py` | **[MỚI]** Dựng `data/eval_set_image.jsonl` bằng crop/rotate/brightness từ ảnh thật (mục 2.3/4.3 template) |
| `step14_eval_image_queries.py` | **[MỚI]** Đánh giá query ảnh — ĐÃ CHẠY, Recall/nDCG/MRR = 1.000 cả 3 transform (mục 4.3 template) |
| `kaggle/kaggle_ablation_embedding_models.py` | **[MỚI]** So CLIP B/32 vs CLIP L/14 vs SigLIP bằng số thật, chạy trên Kaggle (GPU) — **chưa chạy**, xem hướng dẫn trong docstring đầu file (mục 5.1 template) |
| `app.py` | API — có cả endpoint A1 (`/api/v1/search`) và A3 (`/api/v1/ask`) |
| `frontend/index.html` | UI — cả search (A1) và hỏi-đáp (A3) |

### Code A3 (RAG)
| File | Vai trò |
|---|---|
| `rag_core.py` | Naive RAG + Agentic RAG |
| `rag_metrics.py` | **[MỚI]** Metric RAG tự viết: `context_precision_at_k()` (không cần LLM), `faithfulness()` (LLM giám khảo qua Ollama), `answer_relevancy()` (cosine similarity embedding, dùng `sentence-transformers/all-MiniLM-L6-v2`) |
| `step15_expand_eval_set.py` | **[MỚI]** Mở rộng THẲNG `data/eval_set.jsonl` (A1) từ 30 → 250 câu, TOÀN BỘ tiếng Anh: 30 câu A1 gốc (giữ nguyên) + 20 câu A3 gốc (dịch sang tiếng Anh) + 200 câu sinh bằng template — dùng CHUNG cho cả A1 và A3, không có file `rag_eval_set.jsonl` riêng nữa |
| `step_ablation_rag.py` | **[SỬA 2 LẦN]** (1) Bản đầu đọc NHẦM `eval_set.jsonl` lúc chưa có `answer_should_mention`; (2) bản sau tách riêng `rag_eval_set.jsonl` (tiếng Việt, SAI quy ước ngôn ngữ) — nay đã GỘP LẠI, đọc đúng `data/eval_set.jsonl` (250 câu, tiếng Anh, có `answer_should_mention`) + thêm đủ 6 metric (Context Recall/Precision, Faithfulness, Answer Relevancy, Mention hit rate, Citation rate) + Latency. **Chưa chạy** (cần Ollama, ước tính ~1000 lượt gọi LLM cho 250 câu x 2 cấu hình — có `SAMPLE_SIZE` để chạy thử nhỏ trước) |

### Code BPM
| File | Vai trò |
|---|---|
| `bpm_event_log_simulation.py` | Mô phỏng event log 3 kịch bản, công thức Loop |

### Báo cáo (4 file — mỗi file 1 mục đích, KHÔNG gộp chung)
| File | Nộp cho | Nội dung |
|---|---|---|
| `BAO_CAO_TONG_HOP.md` | IS6303 — A1 | Lý thuyết + code + rubric Search (16 phần) |
| `BAO_CAO_A2.md` | IS6303 — A2 | Tóm tắt paper RankGPT (Sun et al., EMNLP 2023) + liên hệ A1 |
| `BAO_CAO_A3.md` | IS6303 — Đồ án cuối kỳ | RAG mở rộng từ A1: Naive/Agentic, grounding, ví dụ Sales tư vấn |
| `BAO_CAO_BPM.md` | IS6003 — ERP nâng cao | Áp dụng A1/A3 vào tối ưu quy trình Sales, công thức Loop |

`HOC_LY_THUYET.md` và `README.md` là tài liệu hỗ trợ, không phải bài nộp.

---

## 3. Đối chiếu với yêu cầu chính thức của đồ án (đã gửi)

> 1. Hệ thống hoàn chỉnh — đủ thành phần
> 2. Cải tiến có kiểm chứng bằng ablation
> 3. Báo cáo, demo video, slide
> 4. Trách nhiệm cá nhân — trả lời câu hỏi

| Yêu cầu | Trạng thái SAU khi bổ sung lần này |
|---|---|
| 1. Hệ thống hoàn chỉnh | ✅ Search (A1) + RAG Naive/Agentic (A3) + API + Frontend chạy được end-to-end. ⚠️ Khối Recommendation/Memory (sơ đồ kiến trúc môn học) vẫn CHƯA build — xem mục 5 |
| 2. Cải tiến có kiểm chứng | ✅ A1 có `step8_run_ablation.py`. ✅ A3 giờ có `step_ablation_rag.py` (mới viết) — so Naive vs Agentic bằng Recall@k + tỷ lệ trích dẫn + latency |
| 3. Báo cáo/demo/slide | ✅ 4 báo cáo `.md` đầy đủ nội dung. ❌ Chưa convert sang Word/PDF để nộp chính thức. ❌ Chưa quay demo video. ❌ Chưa làm slide |
| 4. Trách nhiệm cá nhân | ✅ Mỗi báo cáo đều có Phụ lục Q&A — dùng ôn trước demo |

---

## 4. Việc CẦN LÀM NGAY — theo đúng thứ tự ưu tiên

```
BƯỚC 1 (bắt buộc, chưa làm)
  □ Chạy step1 → step2 → step3 (cần máy có internet tới Hugging Face)
  □ Viết eval_set.jsonl thật (20-40 câu, từ eval_set_template.jsonl)

BƯỚC 2 (bắt buộc, chưa làm)
  □ Cài Ollama + pull model (xem README.md)
  □ Chạy step8_run_ablation.py -- lấy số liệu A1 THẬT (thay số minh hoạ)
  □ Chạy step_ablation_rag.py -- lấy số liệu A3 THẬT (file mới, chưa từng chạy)
  □ Chạy bpm_event_log_simulation.py với catalog.jsonl THẬT (số hiện tại
    dùng 8 sản phẩm mẫu do mình tự tạo để minh hoạ)

BƯỚC 3 (bắt buộc, chưa làm)
  □ Làm error analysis (A1) -- đọc kết quả ablation, gom nhóm lỗi
  □ Cập nhật lại 4 báo cáo .md với số liệu THẬT thay vì số minh hoạ

BƯỚC 4 (bắt buộc, chưa làm)
  □ Convert 4 báo cáo .md thành Word/PDF theo đúng khuôn nộp bài
  □ Làm slide trình bày
  □ Quay demo video (nên quay cả 2: A1 search UI, và A3 hỏi-đáp RAG,
    kèm nhấn checkbox bật/tắt component để demo ablation trực tiếp)

BƯỚC 5 (nên làm nếu còn thời gian, KHÔNG bắt buộc)
  □ Build thêm 1 khối (Recommendation, gợi ý sản phẩm liên quan theo
    category) để "đủ thành phần" chắc chắn hơn theo sơ đồ kiến trúc môn học
  □ Cả nhóm cùng đọc Phụ lục Q&A trong 4 báo cáo, tự hỏi-đáp chéo nhau
    trước buổi bảo vệ
```

---

## 5. Về khối "Recommendation" còn thiếu (không bắt buộc nhưng nên biết)

Sơ đồ "kiến trúc tổng quát môn học" (Search → RAG → Recommendation) còn 1 khối chưa build. Nếu muốn hoàn thiện, hướng khả thi nhất (tái dùng dữ liệu có sẵn, không cần thêm gì mới):

- Dùng `category`/`brand` đã có sẵn trong `catalog.jsonl` (từ A1) — sau khi trả lời RAG, gợi ý thêm "Có thể bạn cũng quan tâm" dựa trên sản phẩm cùng category.
- Không bắt buộc theo đề bài gốc bạn gửi (4 yêu cầu ở mục 3 không nhắc "phải có đủ 3 khối Search/RAG/Recommendation") — chỉ nên làm nếu còn thời gian và muốn điểm cộng.

---

## 6. Thứ tự đọc file khi bắt tay vào làm

```
1. File này (KE_HOACH_TONG_HOP.md)          -- bản đồ tổng thể
2. README.md                                 -- cách cài đặt, chạy code
3. BAO_CAO_TONG_HOP.md                       -- hiểu A1 trước khi đọc A3
4. BAO_CAO_A2.md                             -- đọc song song, không phụ thuộc thứ tự
5. BAO_CAO_A3.md                             -- hiểu RAG mở rộng từ A1 ra sao
6. BAO_CAO_BPM.md                            -- hiểu ứng dụng thực tế + công thức Loop
7. Code (.py) theo thứ tự bước 1-8 trong README.md
```
