# BÁO CÁO A1 — Hệ tìm kiếm đa phương thức (text + ảnh)
### Môn: Thiết kế Hệ thống Thông tin Thông minh dựa trên Truy hồi (IS6303)

> **Nguồn số liệu:** mọi kết quả trong báo cáo được đo lại ngày 24–28/09/2026 trên code hiện tại, bằng các script ghi ở từng mục. Latency đo trên CPU của máy chạy thử nên sẽ dao động giữa các lần chạy. Kết quả so sánh embedding model (mục 6.5) được chạy trên Kaggle (GPU T4), trên bộ eval 250 câu hiện tại.
>
> **Cập nhật ngày 10/10/2026: ground truth nâng lên qrels đa mức 0–3** (mục 2.2). Đã đo lại mục 6.1, 6.2, 6.4 và phần đếm lỗi ở mục 8.1 theo qrels. Mục 6.1, 6.2 và 6.4 (250 câu) chạy trên Kaggle (GPU), nên latency ở các mục này không so sánh được với latency CPU trước đây. Mục 6.5 đo lại trên Kaggle cùng ngày; mục 6.3 chưa đo lại.
>
> **Căn cứ yêu cầu:** đề bài, rubric và các lưu ý của giảng viên chỉ có bản ghi lại trong `docs/KE_HOACH_TONG_HOP.md`; repo không có bản gốc.

Mục lục
1. Đề bài và rubric
2. Dữ liệu
3. Kiến trúc và luồng xử lý
4. Cài đặt các cơ chế
5. Phương pháp đánh giá
6. Kết quả thực nghiệm
7. Case study: truy vết từng tầng
8. Phân tích kết quả tốt và xấu (error analysis)
9. Hạn chế
10. Đối chiếu rubric
11. Cách chạy lại

---

## 1. Đề bài và rubric

Yêu cầu A1 (25% điểm môn) theo bản ghi trong repo:
1. Chọn dataset ảnh, text, metadata hoặc đa phương thức.
2. Xây hệ tìm kiếm ngữ nghĩa: input là ảnh hoặc text tự do, output là ảnh hoặc text liên quan.
3. Frontend đơn giản gọi hàm tìm kiếm qua API, hiển thị top-k.
4. Báo cáo ≤ 8 trang: giải thích dataset, phân tích kết quả tốt/xấu, ablation study.
5. Slide, demo, trả lời câu hỏi.

| Tiêu chí | Trọng số | Nội dung |
|---|---|---|
| Cài đặt đúng cơ chế | 30% | BM25, RRF, metric tự viết, tái lập được |
| Đánh giá đúng phương pháp | 20% | Tập đánh giá hợp lệ kèm qrels, metric đúng |
| Ablation có kiểm soát | 20% | Mỗi cấu hình khác đúng 1 thành phần, kết luận dựa trên số liệu |
| Error analysis | 20% | Taxonomy lỗi rõ ràng, đề xuất có căn cứ |
| Trả lời câu hỏi | 10% | Mọi thành viên hiểu hệ thống |

Các lưu ý của giảng viên (bản ghi trong repo):
- Multimodal không bắt buộc.
- Dataset nhỏ được khuyến khích; nếu lấy tập con thì phải đảm bảo đa dạng.
- Phải có qrels.
- Không cần fine-tune, nhưng phải giải thích vì sao chọn model.
- Rerank đặt ở service layer.
- Tối thiểu 4 cấu hình ablation: BM25 / Dense / Hybrid / Hybrid + rerank.

---

## 2. Dữ liệu

### 2.1 Catalog

- **Nguồn:** [`Shopify/product-catalogue`](https://huggingface.co/datasets/Shopify/product-catalogue) trên Hugging Face, giấy phép apache-2.0.
- **Quy mô gốc:** 48,3k sản phẩm (split train 38,6k, test 9,66k). Mỗi sản phẩm có ảnh, tiêu đề, mô tả, brand và category gán sẵn.
- **Tập dùng trong đồ án:** 2.000 sản phẩm lấy từ split train sau khi **xáo trộn với `seed=42`** (`step1_build_catalog.py`). Mục đích là tránh lệch về vài category nếu dataset gốc được sắp theo nguồn thu thập; seed cố định cho cùng 2.000 sản phẩm ở mọi lần chạy.
- **Lọc:** bỏ dòng không có ảnh, hoặc thiếu cả tiêu đề lẫn mô tả.

Mỗi sản phẩm được chuẩn hoá thành một dòng trong `data/catalog.jsonl`:

| Trường | Lấy từ | Dùng cho |
|---|---|---|
| `id` | chỉ số sau khi xáo trộn (0..1999) | khoá chung, trùng vị trí trong FAISS |
| `image_path` | ảnh lưu thành `data/images/{id:06d}.jpg` (RGB) | SigLIP, hiển thị |
| `caption` | `product_title` (hoặc 200 ký tự đầu của mô tả nếu thiếu title) | SigLIP text, cross-encoder |
| `search_text` | `title + " " + description` | BM25 |
| `category`, `brand` | `ground_truth_category`, `ground_truth_brand` | nDCG đa mức, error analysis |

Tách `caption` ngắn và `search_text` dài vì model dense (SigLIP) được huấn luyện trên caption ngắn, còn BM25 có lợi khi có nhiều từ khoá.

### 2.2 Tập đánh giá text: `data/eval_set.jsonl` (250 câu)

Mỗi dòng gồm `query`, `query_type="text"`, `qrels`, `relevant_ids`, `target_ids` và `answer_should_mention` (dùng cho A3). Thành phần theo `step15_expand_eval_set.py`:

| Nhóm | Số câu | Cách tạo |
|---|---|---|
| A1 gốc | 30 | Viết tay; trộn câu có từ khoá chính xác, câu diễn đạt lại và câu mơ hồ |
| A3 gốc | 20 | Viết tay (ban đầu tiếng Việt, viết lại bằng tiếng Anh cho khớp catalog) |
| Sinh từ mẫu | 200 | 10 mẫu câu, điền 8 từ đầu của caption và brand của sản phẩm chọn ngẫu nhiên (seed=42) |

**Hạn chế quan trọng:** nhóm 200 câu sinh từ mẫu chứa nguyên văn một phần caption, nên BM25 dễ khớp trúng. Vì vậy mục 6 báo cáo kết quả trên **cả 250 câu** và **riêng 50 câu viết tay** khi số liệu khác nhau đáng kể.

**Qrels (cập nhật 10/10/2026, `step16_build_qrels.py`):** ban đầu mỗi câu chỉ có 1 sản phẩm đúng. Nay ground truth được xây theo cách của TREC:
- **Pooling:** gộp top-10 của BM25, Dense và Hybrid, cộng sản phẩm đúng gốc. Trung bình 20,3 ứng viên/câu, tổng 5.064 cặp.
- **Gán nhãn 0–3** từ tên, brand, category và mô tả trong catalog: 3 = đúng hoàn toàn, 2 = liên quan tốt, 1 = một phần, 0 = không liên quan. Nhãn 2–3 được rà soát thủ công.
- `relevant_ids` = các sản phẩm nhãn ≥ 2. Có 240/250 câu vẫn chỉ có 1 sản phẩm như vậy, vì catalog trải rộng nhiều ngành hàng; 10 câu có 2–4 sản phẩm. Có 179/250 câu có sản phẩm nhãn 1.
- **Sửa 3 câu** mà truy vấn mâu thuẫn với chính sản phẩm đúng, ví dụ câu "slide stopper ... trombone" trong khi sản phẩm là chặn ống trượt kèn *trumpet*.

Quy trình chi tiết và thống kê nằm ở báo cáo A3 (`BAO_CAO_A3_IS6303.md` mục 3.2.1). Bản trước khi có qrels được lưu ở `data/eval_set_known_item_backup.jsonl`.

### 2.3 Tập đánh giá ảnh: `data/eval_set_image.jsonl` (100 câu)

- `step13_build_image_eval_set.py` lấy ảnh thật của **100 sản phẩm khác nhau**, theo thứ tự xuất hiện trong eval set text (bỏ sản phẩm trùng).
- Ground truth của mỗi câu là chính sản phẩm có ảnh bị biến đổi.
- Mỗi ảnh áp đúng **một** phép biến đổi, xoay vòng giữa 3 loại (34 crop, 33 rotate, 33 brightness):
- **crop:** cắt 15% viền.
- **rotate:** xoay 15°, nền trắng.
- **brightness:** giảm sáng còn 55%.

---

## 3. Kiến trúc và luồng xử lý

### 3.1 Offline: chuẩn bị dữ liệu (chạy một lần)

```
Bước 1  step1_build_catalog.py     HF dataset → shuffle(42) → 2000 sản phẩm → catalog.jsonl + images/
Bước 2  step2_build_embeddings.py  SigLIP encode ảnh (và caption), chuẩn hoá L2 → image_embeddings.npy (768 chiều)
Bước 3  step3_build_index.py       FAISS IndexFlatIP trên embedding ảnh → dense.index
                                   BM25Okapi trên search_text           → bm25.pkl
```

Ba bước này cũng chạy được trên Kaggle bằng `kaggle/kaggle_build_A1_data_siglip.py` (bản CLIP gốc: `kaggle_build_A1_data.py`, đã thay bằng bản SigLIP sau ablation mục 6.5). Thư mục `data/` hiện tại được tạo bằng cách đó. Dung lượng thực tế:

| File | Dung lượng |
|---|---|
| `data/images/` (2.000 ảnh) | 200 MB |
| `data/catalog.jsonl` | 2,3 MB |
| `data/image_embeddings.npy` | 4,0 MB |
| `data/dense.index` | 4,0 MB |
| `data/bm25.pkl` | 2,7 MB |

### 3.2 Online: một lượt tìm kiếm

```
Trình duyệt (/search)
  │  GET /api/v1/search?q=...&components=bm25,dense&rerank=true|false
  │  POST /api/v1/search_by_image (file ảnh)
  ▼
app.py → search_core.search()
  1. Gọi từng component đang bật (registry):
       bm25 : tách từ query → điểm BM25 trên toàn catalog → top_n=50   (query ảnh → bỏ qua)
       dense: SigLIP encode query (text hoặc ảnh) → FAISS tìm top_n=50 ảnh gần nhất
  2. RRF gộp các danh sách: score(d) = Σ 1/(60 + hạng)
  3. Nếu rerank=true và query là text: cross-encoder chấm lại toàn bộ pool sau RRF
  4. Cắt top-k (k=10), gắn thông tin ảnh/caption → JSON
  ▼
Trình duyệt hiển thị lưới kết quả, component đã dùng, trạng thái rerank, latency
```

Hai điểm thiết kế chính:
1. **Tách client và server:** trình duyệt chỉ gửi HTTP và nhận JSON; model và index chỉ nằm ở server.
2. **Registry pattern:** mỗi kỹ thuật tìm kiếm là một hàm, đăng ký bằng `@component("tên")`. Frontend hỏi `/api/v1/components` để tự sinh chip bật/tắt. Ablation script và API dùng chung `search()`, nên kết quả ablation phản ánh đúng hệ thống chạy thật.

---

## 4. Cài đặt các cơ chế

### 4.1 Dense: SigLIP + FAISS

- **Model:** `google/siglip-base-patch16-224` (đổi từ `openai/clip-vit-base-patch32` sau ablation mục 6.5 — xem lý do đổi bên dưới). Model đưa ảnh và text vào cùng không gian vector **768 chiều** (CLIP B/32 là 512), nên một index ảnh phục vụ được cả query text lẫn query ảnh.
- **Chuẩn hoá L2** mọi vector, nên tích vô hướng bằng cosine. Index là `faiss.IndexFlatIP`: so sánh chính xác với cả 2.000 vector, không xấp xỉ.
- **Lý do đổi từ CLIP B/32 sang SigLIP:** mục 6.5 đo thật trên 250 câu (dense-only) cho thấy SigLIP vượt CLIP B/32 rất rõ (Recall@10 0.920 so với 0.657) — chênh lệch đủ lớn để chuyển sang production thay vì chỉ dừng ở so sánh lý thuyết. Ảnh hưởng dây chuyền: hybrid (bm25+dense) từ 0.988 Recall (CLIP) lên **0.996** (SigLIP, ground truth cũ; 0.997 theo qrels, mục 6.1).
- **Khác CLIP ở cách encode TEXT:** SigLIP không dùng `attention_mask`, lấy biểu diễn ở token cuối của chuỗi đã pad CỐ ĐỊNH 64 token (khớp lúc pretrain) — dùng `padding=True` (kiểu CLIP) làm token cuối rơi vào vị trí sai, embedding gần như nhiễu ngẫu nhiên. Đã gặp đúng lỗi này khi ablation (Recall=0.000 tuyệt đối lần chạy đầu) trước khi sửa đúng `padding="max_length", max_length=64` (xem `search_core.py::_encode_text()`).
- **Không fine-tune**, theo lưu ý của giảng viên — chỉ đổi checkpoint có sẵn, không huấn luyện lại.

### 4.2 Sparse: BM25

- `rank_bm25.BM25Okapi` trên `search_text`, tách từ bằng `lower().split()`.
- Cách tách từ này không xử lý dấu câu, gạch nối, stopword hay stemming. Mục 8 có một lỗi thật do điểm này.

### 4.3 RRF (Cormack et al., 2009)

- Điểm BM25 (0 đến 15+) và cosine (0 đến 1) khác thang đo, nên không cộng thẳng. RRF chỉ dùng thứ hạng: `score(d) = Σ 1/(k + hạng)`, với k=60 và hạng tính từ 1.
- **Ví dụ tính tay (k=60):** BM25 trả [3, 1], Dense trả [3, 2]. Khi đó id 3 = 1/61 + 1/61 = 0.0328, còn id 1 = id 2 = 1/62 = 0.0161.

### 4.4 Rerank: cross-encoder

- `cross-encoder/ms-marco-MiniLM-L-6-v2` (Nogueira & Cho, 2019, bản MiniLM distill để chạy trên CPU).
- Model đọc đồng thời (query, caption) cho từng ứng viên, nên chính xác hơn nhưng chậm hơn nhiều so với tính sẵn vector.
- Rerank chạy trên **toàn bộ pool sau RRF** rồi mới cắt top-k, để có thể kéo ứng viên bị RRF xếp thấp lên đầu. Chỉ áp dụng cho query text.
- Bật/tắt từ giao diện `/search` (tham số `rerank`), mặc định tắt.

### 4.5 Metric (tự viết trong `metrics.py`)

| Metric | Công thức |
|---|---|
| Recall@k | \|top-k ∩ đúng\| / \|đúng\| |
| nDCG@k (nhị phân) | DCG = Σ rel_i / log2(i+1), i là hạng từ 1; IDCG là DCG lý tưởng |
| nDCG@k đa mức | rel = nhãn 0–3 trong `qrels`; IDCG = DCG của các nhãn trong qrels xếp giảm dần (trước 10/10: rel = 2 nếu đúng id, 1 nếu cùng category, 0 nếu khác) |
| MRR | 1 / hạng của kết quả đúng đầu tiên (0 nếu không có trong top-k) |
| Latency | `time.perf_counter()` quanh mỗi lần gọi `search()`; báo trung bình và p95 |

- **Ví dụ tính tay:** đúng = {3, 7}, trả về [3, 1, 7, 2, 9]. Khi đó Recall@5 = 1.0; DCG = 1 + 1/log2(4) = 1.5; IDCG = 1 + 1/log2(3) = 1.63; nDCG@5 ≈ 0.92.
- Ở 240/250 câu chỉ có 1 sản phẩm nhãn ≥ 2, nên nDCG@1 xấp xỉ tỷ lệ câu có sản phẩm đúng đứng hạng 1.
- nDCG đa mức (theo qrels) thưởng cả việc xếp sản phẩm liên quan một phần (nhãn 1) lên cao, thay vì chỉ quan tâm sản phẩm đúng. Bản cũ đoán mức 1 theo category và giả định cả top-k đều là sản phẩm cùng category, nên luôn cho điểm rất thấp (khoảng 0,35) và gần như không phân biệt được các cấu hình.

---

## 5. Phương pháp đánh giá

- Mọi cấu hình đều gọi cùng hàm `search()` với cùng eval set, chỉ đổi đúng một tham số.
- Chạy trên 250 câu text (mục 2.2) và 100 câu ảnh (mục 2.3), k=10.

| Script | Biến được thay đổi |
|---|---|
| `step8_run_ablation.py` | Component: bm25 / dense / hybrid / hybrid + rerank |
| `step9_ablation_index_fields.py` | Trường đưa vào BM25: title / title + description / + ảnh (BM25 tạm trong RAM, không ghi đè index thật) |
| `step10_sweep_rrf.py` | Tham số RRF: rrf_k × top_n |
| `step_ablation_rerank.py` | Loại reranker (không rerank / MiniLM / DeBERTa distill từ paper A2), trên cùng pool 30 ứng viên hybrid |
| `kaggle/kaggle_ablation_embedding_models.py` | Embedding model (dense-only) |
| `step14_eval_image_queries.py` | Loại biến đổi ảnh |
| `step12_case_study_trace.py` | Không phải ablation: truy vết hạng của 1 sản phẩm qua từng tầng |

---

## 6. Kết quả thực nghiệm

### 6.1 Ablation theo component (`step8_run_ablation.py`, 250 câu, k=10, qrels)

| Cấu hình | Recall@10 | nDCG@10 | nDCG đa mức@10 | MRR | Latency TB (ms) | Latency p95 (ms) |
|---|---|---|---|---|---|---|
| bm25 only | 0.972 | 0.953 | 0.861 | 0.950 | 5.4 | 7.4 |
| dense only | 0.917 | 0.847 | 0.798 | 0.828 | 10.9 | 11.5 |
| hybrid (bm25 + dense) | **0.997** | 0.958 | **0.901** | 0.946 | 14.4 | 16.6 |
| hybrid + rerank | 0.981 | **0.978** | 0.892 | **0.982** | 93.6 | 79.3 |

Chạy trên Kaggle (GPU) ngày 10/10/2026, dense = SigLIP. Latency đo trên GPU nên thấp hơn nhiều so với lần đo CPU trước (vd hybrid 77,9 ms trên CPU). Latency TB của rerank lớn hơn p95, vì lần gọi đầu tiên phải nạp model cross-encoder.

**Nhận xét:**
- **Các metric nhị phân gần như không đổi so với ground truth cũ** (ví dụ hybrid Recall 0.996 → 0.997, rerank nDCG 0.982 → 0.978). Thứ tự các cấu hình giữ nguyên, vì sản phẩm đúng cũ vẫn mang nhãn 3.
- **nDCG đa mức giờ có ý nghĩa:** khoảng 0.80–0.90, thay vì 0.31–0.36 với cách đoán theo category. Nó phân biệt được các cấu hình mà nDCG nhị phân không thấy:
  - **Hybrid có nDCG đa mức cao nhất (0.901)**, cao hơn cả hybrid + rerank (0.892), dù nDCG nhị phân của rerank cao hơn (0.978 so với 0.958). Tức là cross-encoder đưa đúng sản phẩm cần tìm lên đầu, nhưng xếp các sản phẩm liên quan một phần (phụ kiện, cùng nhóm, nhãn 1) thấp hơn hybrid, và còn làm rơi một số sản phẩm đúng (Recall 0.997 → 0.981).
  - BM25 vượt Dense ở cả hai loại nDCG (0.861 so với 0.798 ở bản đa mức), nhất quán với việc 200 câu sinh từ mẫu chứa nguyên văn caption.
- **Hybrid** có Recall cao nhất (0.997) và nDCG nhị phân cao nhất trong 3 cấu hình chưa rerank. Với dense = SigLIP, RRF không còn "làm loãng" thứ hạng BM25 như lúc dùng CLIP (khi đó hybrid nDCG 0.855, thấp hơn bm25 only). Đây là bằng chứng cụ thể cho quyết định đổi sang SigLIP (mục 6.5).
- **Rerank** đưa nDCG@10 nhị phân (0.978) và MRR (0.982) lên cao nhất, đổi lại Recall giảm và latency tăng khoảng 6,5 lần (14,4 → 93,6 ms trên GPU; khoảng 11 lần trên CPU).

### 6.2 Ablation theo trường index (`step9_ablation_index_fields.py`, 250 câu, dense = SigLIP, qrels)

| Cấu hình | Recall@10 | nDCG@10 | MRR |
|---|---|---|---|
| BM25 trên title (caption) | 0.964 | 0.946 | 0.944 |
| BM25 trên title + description | 0.972 | 0.953 | **0.950** |
| BM25 (title + description) + Dense ảnh | **0.997** | **0.958** | 0.946 |

- Thêm mô tả vào BM25 chỉ tăng nhẹ (Recall +0.008), vì phần lớn câu sinh từ mẫu đã khớp ngay với title.
- Dòng cuối trùng với dòng "hybrid" của mục 6.1, dùng để đối chiếu. Với SigLIP, thêm nhánh Dense tăng cả Recall (+0.025) lẫn nDCG (+0.005); khác lúc dùng CLIP, khi đó Recall tăng nhưng nDCG giảm. MRR giảm nhẹ (0.950 → 0.946).

### 6.3 Sweep tham số RRF (`step10_sweep_rrf.py`, 250 câu, hybrid, không rerank)

> Mục này **chưa đo lại theo qrels**; số liệu dưới đây dùng ground truth cũ (1 sản phẩm đúng / câu). Ở mục 6.1, đổi ground truth chỉ làm các metric nhị phân lệch ≤ 0.005, nên kết luận định tính ở đây nhiều khả năng giữ nguyên.

| rrf_k | top_n | Recall@10 | nDCG@10 | MRR |
|---|---|---|---|---|
| 10 | 20 | 0.996 | **0.963** | **0.952** |
| 10 | 50 | 0.996 | 0.962 | 0.951 |
| 10 | 100 | 0.996 | 0.962 | 0.950 |
| 60 | 20 | 0.996 | 0.963 | 0.951 |
| 60 | 50 (mặc định) | 0.996 | 0.954 | 0.940 |
| 60 | 100 | 0.980 | 0.944 | 0.932 |
| 100 | 20 | 0.996 | 0.963 | 0.951 |
| 100 | 50 | 0.996 | 0.954 | 0.940 |
| 100 | 100 | 0.976 | 0.942 | 0.930 |

**Số liệu đo lại với dense = SigLIP** (thay bảng CLIP trước đó — cùng kết luận định tính, nhưng khoảng cách giữa các ô đã THU HẸP LẠI đáng kể vì dense giờ chính xác hơn nhiều):
- Cormack et al. (2009) ghi rằng k không quan trọng. Trên bộ dữ liệu này, kết quả **vẫn có** thay đổi theo cả k lẫn top_n, dù ít rõ rệt hơn so với lúc dùng CLIP:
  - Ở cùng top_n=20/50, k=10/60/100 cho Recall/nDCG/MRR gần như GIỐNG HỆT nhau (vd top_n=20: cả 3 giá trị k đều ra 0.996/0.963/0.95x) — khác hẳn bảng CLIP trước đây (k=10 vượt trội rõ ràng so với k=60/100 ở mọi top_n).
  - Chỉ ở top_n=100, k lớn (60, 100) mới bắt đầu làm Recall giảm rõ (0.996→0.980/0.976) — hiện tượng "pha loãng" vẫn còn nhưng cần pool RỘNG HƠN (top_n=100) mới lộ ra, thay vì đã thấy rõ từ top_n=50 như lúc dùng CLIP.
- Cấu hình mặc định (k=60, top_n=50) **không phải** cấu hình tốt nhất trong lưới này (kém hơn k=10/top_n=20 khoảng 0.009 nDCG), nhưng khoảng cách với "tốt nhất" đã nhỏ đi nhiều so với lúc dùng CLIP (0.063 nDCG). Code chưa đổi mặc định — chênh lệch giờ đủ nhỏ để KHÔNG BẮT BUỘC phải đổi, nhưng vẫn có thể cân nhắc đổi `rrf_k=10` nếu muốn tối ưu thêm.
- Giả thuyết không đổi: top_n lớn đưa thêm ứng viên hạng thấp, nhiễu vào RRF; k nhỏ giữ chênh lệch lớn giữa các hạng đầu, nên thứ hạng tốt của một nhánh ít bị "pha loãng" — nhưng dense càng chính xác (SigLIP so với CLIP) thì "ứng viên hạng thấp" của dense càng ít nhiễu hơn, nên hiệu ứng pha loãng cũng giảm theo.

### 6.4 So sánh reranker (`step_ablation_rerank.py`, pool 30 ứng viên hybrid, qrels)

| Cấu hình | nDCG@1 | nDCG@5 | nDCG@10 | Recall@10 | Thời gian rerank / câu |
|---|---|---|---|---|---|
| 250 câu — không rerank | 0.924 | 0.961 | 0.964 | **0.998** | — |
| 250 câu — cross-encoder MiniLM | **0.980** | 0.977 | 0.978 | 0.981 | 0.026 s (GPU) |
| 250 câu — DeBERTa distill (A2) | 0.976 | **0.978** | **0.980** | 0.986 | 0.179 s (GPU) |
| 50 câu viết tay — không rerank | 0.820 | 0.905 | 0.915 | **0.990** | — |
| 50 câu viết tay — cross-encoder MiniLM | **0.900** | 0.902 | 0.906 | 0.925 | 0.33 s (CPU) |
| 50 câu viết tay — DeBERTa distill (A2) | 0.880 | **0.906** | **0.916** | 0.950 | 2.02 s (CPU) |

Dòng 250 câu chạy trên Kaggle (GPU); dòng 50 câu viết tay chạy local (CPU), cùng code và cùng qrels, ngày 10/10/2026.

**Nhận xét:**
- **Rerank chủ yếu cải thiện hạng 1:** trên 50 câu viết tay, nDCG@1 tăng từ 0.82 lên 0.90 (MiniLM) và 0.88 (DeBERTa). Ở nDCG@5 và @10 thì gần như không khác "không rerank" (0.902–0.916 so với 0.905–0.915).
- **Rerank làm giảm Recall@10** (0.990 xuống 0.925 với MiniLM, 0.950 với DeBERTa): reranker đẩy một số sản phẩm đúng ra khỏi top-10 (mục 8.1, nhóm 7).
- **DeBERTa distill (A2) ngang hoặc nhỉnh hơn MiniLM ở nDCG@5/@10 và Recall**, MiniLM nhỉnh hơn ở nDCG@1. Chênh lệch trên 50 câu chỉ 1–3 câu, chưa đủ để kết luận. DeBERTa chậm hơn khoảng 6–7 lần.
- So với ground truth cũ, hàng "không rerank" nhích lên (250 câu: nDCG@1 0.916 → 0.924; 50 câu: 0.780 → 0.820). Nguyên nhân là 3 câu đã sửa và vài câu có thêm sản phẩm liên quan (vd câu #27, mục 8.1).

### 6.5 So sánh embedding model (Kaggle, dense-only, 250 câu)

Số liệu lấy từ `data/embedding_model_ablation_results.json`. Chạy trên GPU T4 của Kaggle, đo lại ngày 10/10/2026 theo qrels (mục 2.2), có thêm cột nDCG đa mức.

| Model | Dim | Recall@10 | nDCG@10 | nDCG đa mức@10 | MRR | Latency TB (ms) |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 (baseline A1 gốc) | 512 | 0.657 | 0.532 | 0.492 | 0.500 | 9.7 |
| CLIP ViT-L/14 | 768 | 0.758 | 0.635 | 0.595 | 0.600 | 9.4 |
| SigLIP base patch16-224 (đang dùng) | 768 | **0.920** | **0.841** | **0.789** | **0.819** | **7.6** |

- **So với ground truth cũ, các metric nhị phân gần như không đổi** (ví dụ CLIP B/32 Recall 0.656 → 0.657, SigLIP nDCG 0.838 → 0.841), nên kết luận chọn model giữ nguyên.
- **nDCG đa mức cho cùng kết luận với nDCG nhị phân:** SigLIP 0.789 so với CLIP B/32 0.492 (gấp khoảng 1,6 lần, +0.297). Ở nDCG nhị phân là 0.841 so với 0.532 (cũng gấp khoảng 1,6 lần, +0.309). Như vậy khi tính cả các sản phẩm liên quan một phần (nhãn 1), thứ hạng và khoảng cách giữa các model vẫn giữ nguyên.
- Recall dense-only của SigLIP ở đây (0.920) lệch nhẹ so với dòng "dense only" ở mục 6.1 (0.917). Lý do là script này tải lại ảnh từ Hugging Face và encode lại trên Kaggle, thay vì dùng `dense.index` có sẵn.
- SigLIP vẫn vượt rõ cả hai model CLIP, đúng xu hướng đã thấy ở lần chạy 30 câu. **Đã chuyển production sang SigLIP** (xem mục 4.1) — `data/dense.index`/`image_embeddings.npy` hiện tại được build bằng `kaggle/kaggle_build_A1_data_siglip.py`, mọi số liệu ablation từ mục 6.1 trở đi đã đo lại trên bản SigLIP này.
- **Cả 3 model đều giảm điểm so với lần chạy 30 câu** (vd CLIP B/32: Recall 0.767→0.656, nDCG 0.608→0.529) — ngược chiều với BM25 ở mục 6.1 (BM25 tăng điểm khi thêm 220 câu). Nguyên nhân: 220 câu mới được bọc trong khung câu như "looking for...", "price of...", "do you have..." — các từ đệm này không ảnh hưởng BM25 (chỉ đếm từ khoá khớp) nhưng LÀM LOÃNG embedding câu hỏi của CLIP/SigLIP (encode cả câu thành 1 vector, từ đệm kéo vector ra xa embedding caption gốc). Đây là bằng chứng cụ thể cho thấy **retrieval dựa trên embedding nhạy với cách diễn đạt câu hỏi hơn BM25** — một điểm cần nêu khi bảo vệ.
- Latency ở bảng này đo trên GPU, không so sánh được với latency CPU ở mục 6.1.
- **Lỗi đã gặp khi chạy lần đầu (30 câu):** SigLIP cho mọi metric = 0.000.
  - Nguyên nhân: SigLIP lấy biểu diễn ở token cuối của chuỗi đã pad tới 64 token. Pad động như CLIP làm embedding gần như ngẫu nhiên.
  - Cách sửa: dùng `padding="max_length", max_length=64`.

### 6.6 Truy vấn ảnh (`step14_eval_image_queries.py`, 100 câu, hybrid)

Với query ảnh, BM25 tự bỏ qua, nên kết quả thực chất là dense-only (SigLIP + FAISS).

**Số liệu đo lại với dense = SigLIP** (thay bảng CLIP trước đó):

| Biến đổi | n | Recall@10 | nDCG@10 | MRR |
|---|---|---|---|---|
| crop 15% viền | 34 | **1.000** | **1.000** | **1.000** |
| rotate 15° | 33 | 1.000 | 1.000 | 1.000 |
| brightness 55% | 33 | 1.000 | 1.000 | 1.000 |
| **Tổng** | **100** | **1.000** | **1.000** | **1.000** |

- **100/100 ảnh tìm ra đúng sản phẩm ở hạng 1** — hoàn hảo tuyệt đối, kể cả nhóm crop (trước đây với CLIP: 97/100, cả 3 lỗi đều ở crop — id 516 "Handy Ceiling Fan Installation" rớt khỏi top-10 hoàn toàn, id 65 và id 204 đứng hạng 2). SigLIP không chỉ mạnh hơn CLIP với query TEXT (mục 6.1) mà còn robust hơn với các phép biến đổi ẢNH đã thử.
- **Giới hạn của phép đo (không đổi dù đổi model):**
  - Ảnh truy vấn được tạo từ chính ảnh trong index, và mức biến đổi còn nhẹ. Kết quả chỉ cho thấy SigLIP ổn định với các biến đổi này; chưa phản ánh ảnh do người dùng tự chụp (góc khác, nền khác, sản phẩm khác mẫu).
  - Vì đã đạt 100/100, phép đo này ĐÃ BÃO HOÀ (ceiling effect) với mức biến đổi hiện tại (crop 15%, xoay 15°, giảm sáng 55%) — không còn phân biệt được model tốt hơn model kém bao nhiêu nữa. Muốn tiếp tục so sánh định lượng, cần THỬ MỨC BIẾN ĐỔI MẠNH HƠN (crop >50%, xoay >45°, kết hợp nhiều phép biến đổi) để tìm điểm SigLIP bắt đầu thất bại.
  - Chưa thử biến đổi mạnh hơn hoặc kết hợp nhiều biến đổi.

---

## 7. Case study: truy vết từng tầng (`step12_case_study_trace.py`)

Mỗi dòng là hạng của sản phẩm đúng ở từng tầng. Mỗi nhánh lấy top-50; rerank nhận 50 ứng viên đầu sau RRF.

**Số liệu đo lại với dense = SigLIP** (thay bảng CLIP trước đó — 4 câu này được CHỌN vì lúc dùng CLIP chúng minh hoạ rõ hiện tượng "RRF làm loãng"; với SigLIP, câu chuyện đã khác hẳn ở 3/4 câu):

| Query | id | BM25 | Dense | RRF | Rerank |
|---|---|---|---|---|---|
| "rubber bag you fill with hot water to stay warm" | 888 | 6 | **1** | **1** | 1 |
| "rug for stairs so people don't slip" | 241 | không có | 1 | 5 | 5 |
| "device that helps someone relearn to walk after an injury" | 1681 | không có | 12 | 27 | 36 |
| "round basket for serving bread" (caption tiếng Hà Lan) | 1396 | không có | **1** | 9 | 24 |

Quan sát (khác hẳn kết luận với CLIP ở 2 điểm quan trọng):
- **Câu "rubber bag..." (id 888) đổi chiều hoàn toàn:** với CLIP, Dense KHÔNG tìm thấy sản phẩm này, RRF fusion kéo hạng BM25 từ 6 xuống 15 ("làm loãng"). Với SigLIP, Dense một mình đã xếp hạng 1, nên RRF giờ GIỮ NGUYÊN hạng 1 (2 nhánh đồng thuận, không còn cạnh tranh nhau) — **RRF không còn làm loãng câu này nữa vì Dense giờ đủ mạnh để tự tìm đúng**, chứ không phải vì sửa công thức RRF.
- **Câu "device that helps someone relearn to walk..." (id 1681)** — ca khó nhất — Dense cải thiện rõ (hạng 31→12 khi đổi CLIP→SigLIP) nhưng **vẫn chưa đủ tốt để lọt top-10** sau RRF (hạng 27) hay rerank (hạng 36, còn TỆ HƠN). Đây vẫn là ca khó thực sự — thuật ngữ y tế chuyên ngành ("gait trainer" ↔ "relearn to walk") nằm ngoài phân bố huấn luyện của CẢ 2 model CLIP lẫn SigLIP, không phải vấn đề riêng của CLIP.
- **2 câu còn lại** ("rug for stairs...", "round basket...") giữ nguyên mô hình cũ: BM25 không tìm thấy (không trùng từ/khác ngôn ngữ), Dense một mình xếp hạng 1 hoàn hảo, nhưng RRF fusion với BM25 (rỗng) vẫn kéo hạng xuống (1→5, 1→9) — hiện tượng "làm loãng" VẪN CÒN xảy ra khi MỘT nhánh hoàn toàn không tìm thấy gì, chỉ là ít gặp hơn vì SigLIP tìm thấy nhiều câu hơn CLIP (xem mục 6.1).
- **Rerank vẫn có thể cứu hoặc làm hỏng:** giữ nguyên hạng 1 cho id 888 (không đổi vì đã hoàn hảo), nhưng đẩy id 1396 (caption tiếng Hà Lan) từ hạng 9 xuống 24, và id 1681 từ hạng 27 xuống 36 — cross-encoder MiniLM (huấn luyện trên văn bản tiếng Anh) không đánh giá tốt các trường hợp này.

---

## 8. Phân tích kết quả tốt và xấu (error analysis)

Mục 1 (rubric) yêu cầu phân tích cả kết quả **tốt** lẫn **xấu**, không chỉ ablation study. Phần dưới đây trình bày 4 ví dụ tốt (mỗi ví dụ ứng với đúng 1 tầng trong kiến trúc — BM25, Dense, RRF, Rerank — cho thấy tầng đó phát huy đúng vai trò thiết kế), rồi mới tới taxonomy lỗi (kết quả xấu).

### 8.0 Kết quả tốt — mỗi tầng đóng góp đúng vai trò thiết kế

Truy vết bằng cách gọi trực tiếp `_run_bm25`/`_run_dense`/`rrf_fusion`/`rerank` (giống `step12_case_study_trace.py`) trên 4 câu dưới đây, lấy từ 50 câu viết tay:

| Query | id | BM25 | Dense | RRF | Rerank | Tầng nào "cứu" |
|---|---|---|---|---|---|---|
| "fly tying nippers with comfortable grip" → "Loon - Rogue Nippers w/ Comfy Grip" | 176 | **1** | 6 | 1 | — | **BM25**: từ khoá "nippers"/"grip" khớp chính xác, không cần ngữ nghĩa |
| "rug for stairs so people don't slip" → "Cobblestone Jute Stair Tread" | 241 | không có | **1** | 5 (vẫn trong top-10) | — | **Dense**: không từ nào trùng ("rug"≈"tread", "stairs"≈"stair"), SigLIP vẫn hiểu đúng ý nghĩa hình ảnh/công dụng |
| "essential oil diffuser with decorative tray" → "Clear Quartz Oil Diffuser set with Tray" | 1158 | 2 | 2 | **1** | — | **RRF**: cả 2 nhánh đều xếp hạng 2 (không hoàn hảo riêng lẻ), fusion kết hợp 2 tín hiệu ĐỒNG THUẬN đưa lên hạng 1 — đúng lý thuyết RRF, không phải trùng hợp |
| "fast fold portable projection screen"¹ → "Da-Lite Fast-Fold Deluxe..." | 1758 | 40 (rất yếu — "fast-fold" bị `lower().split()` gộp thành 1 token, không khớp "fast"/"fold" riêng) | 1 | 3 | **1** | **Rerank**: RRF bị nhánh BM25 yếu kéo tụt xuống hạng 3, cross-encoder đọc lại (query, caption) đồng thời và xác nhận đúng, đưa về hạng 1 |

¹ Truy vết với câu gốc, trước khi câu được sửa thành "replacement screen surface for a fast fold projection screen" khi xây qrels (sản phẩm là tấm màn thay thế, không phải nguyên bộ màn chiếu; mục 2.2).

**Nhận xét:** đây chính là bằng chứng "mỗi nhánh bổ khuyết điểm yếu cho nhau" đã nêu ở mục 4.2/4.3 — không phải khẳng định suông. Đặc biệt ví dụ id=1158 minh hoạ đúng bản chất RRF: fusion có thể cho kết quả TỐT HƠN cả 2 nhánh riêng lẻ khi 2 nhánh đồng thuận (khác với mục 7, nơi RRF làm TỆ HƠN khi chỉ 1 nhánh đồng ý).

### 8.1 Kết quả xấu (error analysis)

Phân tích trên **50 câu viết tay**. 200 câu sinh từ mẫu gần như không sinh lỗi mới vì chứa nguyên văn caption. Một câu tính là **lỗi** ở một cấu hình nếu không có sản phẩm nào nhãn ≥ 2 trong top-10. Đếm lại ngày 10/10/2026 theo qrels, dense = SigLIP.

**Tổng quan:** 5/50 câu lỗi ở ít nhất một cấu hình (CLIP + ground truth cũ: 16/50; SigLIP + ground truth cũ: 6/50). **Không còn câu nào lỗi ở cả 4 cấu hình** (trước đây là câu #27). Số câu lỗi theo cấu hình:

| Cấu hình | bm25 | dense | hybrid | hybrid + rerank |
|---|---|---|---|---|
| Số câu lỗi / 50 (qrels, SigLIP) | 5 | **0** | **0** | 3 |
| SigLIP, ground truth cũ | 6 | 1 | 1 | 4 |
| CLIP, ground truth cũ | 6 | 11 | 3 | 4 |

Hai thay đổi so với ground truth cũ đều có lý do cụ thể:
- **Câu #27** *"device that helps someone relearn to walk after an injury"*: qrels có thêm xe tập đi, nạng và gậy chống (nhãn 2), bên cạnh "Gait Trainer" (nhãn 3). Dense xếp gậy chống ở hạng 1, nên câu không còn bị tính là lỗi ở dense, hybrid và hybrid + rerank. Gait Trainer thì **vẫn chưa** vào top-10 ở cấu hình nào (dense hạng 12, hybrid 27, rerank rơi khỏi top-50). nDCG đa mức phạt đúng điểm này, còn chỉ số "lỗi / không lỗi" thì không.
- **Câu #28** (màn chiếu fast-fold) được sửa nội dung, nên BM25 không còn trượt do tách từ "fast-fold".

**Taxonomy lỗi** (một câu có thể thuộc nhiều nhóm):

| Nhóm lỗi | Số câu | Ví dụ (query → caption sản phẩm đúng) | Nguyên nhân | Đề xuất |
|---|---|---|---|---|
| **1. Dense bỏ sót, BM25 tìm được** | **0** (CLIP: 10) | — | SigLIP nắm được các liên hệ ảnh–từ khoá đơn giản mà CLIP B/32 bỏ lỡ (vd "wheelbarrow handle", "guitar capo"); bằng chứng trực tiếp cho quyết định đổi model ở mục 6.5 | Đã giải quyết bằng đổi embedding |
| **2. BM25 bỏ sót: diễn đạt khác từ** | 2 | "rug for stairs so people don't slip" → "Cobblestone Jute Stair Tread" (#6, id 241); "device that helps someone relearn to walk..." → gait trainer, xe tập đi, nạng, gậy (#27) | Không có từ nào trùng ("rug" và "tread", "relearn to walk" và "gait trainer") | Giữ nhánh Dense; thêm stemming; từ đồng nghĩa theo domain |
| **3. BM25 bỏ sót: caption không phải tiếng Anh** | 2 | "round basket for serving bread" → "APS Brood- en Fruitmand..." (#26, id 1396, tiếng Hà Lan); "long track speed skates" → "Zandstra 525 Lengdeløp Fastskøyte" (#47, id 1493, tiếng Na Uy) | Dataset gốc lẫn nhiều ngôn ngữ; BM25 không khớp được từ tiếng Anh | Phát hiện ngôn ngữ và dịch khi build catalog, hoặc dùng embedding đa ngôn ngữ |
| **4. BM25 bỏ sót: caption thiếu thông tin** | 1 | "inflatable towable tube for boating" → "Ace Racing" (#13, id 748; BM25 hạng 14) | Caption chỉ là tên thương mại, không mô tả công dụng | Đưa thêm `category` vào `search_text` |
| **5. BM25 bỏ sót: cách tách từ** | 0 (trước đây 1) | Câu "fast fold portable projection screen" (id 1758) đã được sửa khi xây qrels; lỗi tách "fast-fold" thành 1 token vẫn còn trong code | `lower().split()` giữ nguyên "fast-fold" thành một từ | Tách thêm theo dấu câu và gạch nối |
| **6. RRF làm mất kết quả đúng của BM25** | **0** (CLIP: 2) | — | Dense (SigLIP) tự tìm đúng ngay từ đầu nên 2 nhánh đồng thuận thay vì cạnh tranh (mục 7) | Không cần đề xuất thêm |
| **7. Rerank làm mất kết quả đúng** | 3 | #13 (id 748: hybrid hạng 2 → rerank ngoài top-50), #26 (id 1396: hạng 9 → 40), #47 (id 1493: hạng 6 → 16) | Cross-encoder huấn luyện trên văn bản tiếng Anh MS MARCO, chấm thấp caption ngoại ngữ hoặc caption không mô tả công dụng; không liên quan tới việc đổi CLIP → SigLIP | Chỉ rerank khi caption đủ thông tin và là tiếng Anh; hoặc kết hợp điểm rerank với điểm RRF thay vì thay thế hoàn toàn |
| **8. Sai ở mọi cấu hình** | **0** (trước đây 1) | Câu #27 không còn thuộc nhóm này nhờ qrels (xem trên), nhưng sản phẩm phù hợp nhất (gait trainer) vẫn ngoài top-10 | Thuật ngữ chuyên ngành, không trùng từ | Từ điển đồng nghĩa theo domain; embedding chuyên biệt hơn cho domain y tế |

Kết luận của phần phân tích lỗi:
- **Đổi CLIP → SigLIP giải quyết gần như hoàn toàn 2 nhóm lỗi:** nhóm 1 (Dense bỏ sót, 10 câu xuống 0) và nhóm 6 (RRF làm loãng, 2 câu xuống 0). Đây là bằng chứng định lượng rõ nhất cho quyết định đổi embedding model.
- **Các nhóm lỗi không liên quan tới embedding vẫn còn:** BM25 hỏng khi không trùng từ, khác ngôn ngữ hoặc caption thiếu thông tin (nhóm 2, 3, 4), và rerank tự gây lỗi mới trên caption ngoại ngữ hoặc thiếu thông tin (nhóm 7). Các nhóm này cần giải pháp khác (chuẩn hoá dữ liệu, cải thiện tokenize, cải thiện rerank), không sửa được bằng cách đổi dense model.
- **Hybrid không lỗi câu nào trên 50 câu viết tay, còn hybrid + rerank lỗi 3 câu**, dù nDCG nhị phân trung bình của rerank cao hơn. Khớp với mục 6.1: rerank tốt hơn ở hạng đầu nhưng làm rơi sản phẩm đúng và có nDCG đa mức thấp hơn hybrid.
- **Qrels làm lộ ra một kiểu lỗi mà ground truth cũ không phân biệt được** (câu #27): hệ thống tìm được sản phẩm dùng được, nhưng bỏ sót sản phẩm phù hợp nhất.

---

## 9. Hạn chế

**Về dữ liệu đánh giá:**
- 200/250 câu sinh từ mẫu, chứa nguyên văn caption, nên kết quả trên 250 câu lạc quan hơn thực tế.
- Qrels do LLM chấm, người rà soát các nhãn 2–3; nhãn 0–1 chưa được rà toàn bộ, chưa đo độ đồng thuận giữa người chấm. Pool chỉ lấy top-10 của 3 cách tìm hiện có, nên sản phẩm liên quan mà cả 3 cùng bỏ sót bị coi là nhãn 0.
- Bộ truy vấn ảnh (100 câu) dùng biến đổi nhẹ trên chính ảnh trong index, chưa có ảnh do người dùng tự chụp.

**Về kết quả đã đo:**
- Đã chuyển production từ CLIP B/32 sang SigLIP dựa trên ablation dense-only (mục 6.5); mọi số liệu hybrid/ablation trong báo cáo (mục 6.1 trở đi) đã đo lại trên SigLIP. **Đánh đổi:** encode 1 câu query bằng SigLIP chậm hơn CLIP B/32 khoảng 3 lần trên CPU (77ms so với 25ms, mục 6.1) — chấp nhận được ở quy mô demo nhưng cần lưu ý nếu scale traffic thật.
- Cấu hình RRF mặc định (k=60, top_n=50) không phải tốt nhất theo mục 6.3, dù khoảng cách với cấu hình tốt nhất đã thu hẹp lại sau khi đổi sang SigLIP.
- Latency: lần đo trên CPU cho thấy rerank thêm khoảng 0,8 s mỗi truy vấn; lần đo ngày 10/10 (mục 6.1, 6.4) chạy trên GPU nên nhanh hơn nhiều. Hai lần đo không so sánh trực tiếp được.

**Về code (chi tiết trong `docs/CAU_TRUC_DU_AN.md` mục 7):**
- `_run_dense` dùng vị trí trong FAISS làm id.
- `ndcg_at_k_graded` trả 0 thay vì None khi thiếu ground truth.
- `step12` gắn nhãn hạng RRF > 50 là "rerank bỏ sót".

---

## 10. Đối chiếu rubric

| Tiêu chí | Đáp ứng ở đâu |
|---|---|
| Cài đặt đúng cơ chế (BM25, RRF, metric tự viết, tái lập được) | Mục 4; `search_core.py`, `metrics.py`; seed cố định; mọi số liệu tái lập bằng script ở mục 11 |
| Tập đánh giá hợp lệ kèm qrels | Mục 2.2, 2.3; `data/eval_set.jsonl` có qrels đa mức 0–3 cho 5.064 cặp (câu hỏi, sản phẩm); `data/eval_set_image.jsonl`; hạn chế nêu ở mục 9 |
| Ablation có kiểm soát | Mục 5, 6: mỗi script đổi đúng 1 biến; có đủ 4 cấu hình tối thiểu BM25 / Dense / Hybrid / Hybrid + rerank |
| Phân tích kết quả tốt/xấu, error analysis | Mục 7, 8: 4 ví dụ tốt (mục 8.0, mỗi tầng 1 ví dụ) + taxonomy 8 nhóm lỗi (mục 8.1), đếm trên 50 câu viết tay theo qrels, có đề xuất cho từng nhóm |
| Frontend + API | Mục 3.2; trang `/search`, bật/tắt component và rerank |

---

## 11. Cách chạy lại

```bash
pip install -r requirements.txt
python step1_build_catalog.py && python step2_build_embeddings.py && python step3_build_index.py
python step16_build_qrels.py pool && python step16_build_qrels.py merge   # qrels (mục 2.2), cần nhãn trong data/qrels_judgments/
#   hoặc chạy step8/9/10/rerank trên Kaggle GPU: kaggle/kaggle_run_retrieval_ablation.py
python step8_run_ablation.py            # mục 6.1
python step9_ablation_index_fields.py   # mục 6.2
python step10_sweep_rrf.py              # mục 6.3
python step_ablation_rerank.py          # mục 6.4 (cần models/deberta-10k-rank_net cho dòng DeBERTa)
python step13_build_image_eval_set.py && python step14_eval_image_queries.py   # mục 6.6
python step12_case_study_trace.py       # mục 7
uvicorn app:app --port 8000             # giao diện: http://localhost:8000/search
```

Giải thích chi tiết từng file nằm trong `docs/CAU_TRUC_DU_AN.md`.
