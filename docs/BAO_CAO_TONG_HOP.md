# BÁO CÁO A1 — Hệ tìm kiếm đa phương thức (text + ảnh)
### Môn: Thiết kế Hệ thống Thông tin Thông minh dựa trên Truy hồi (IS6303)

> **Nguồn số liệu:** mọi kết quả trong báo cáo được đo lại ngày 24–28/09/2026 trên code hiện tại, bằng các script ghi ở từng mục. Latency đo trên CPU của máy chạy thử nên sẽ dao động giữa các lần chạy. Kết quả so sánh embedding model (mục 6.5) được chạy trên Kaggle (GPU T4), trên bộ eval 250 câu hiện tại.
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

Mỗi dòng gồm `query`, `query_type="text"`, `relevant_ids` (đúng 1 sản phẩm) và `answer_should_mention` (dùng cho A3). Thành phần theo `step15_expand_eval_set.py`:

| Nhóm | Số câu | Cách tạo |
|---|---|---|
| A1 gốc | 30 | Viết tay; trộn câu có từ khoá chính xác, câu diễn đạt lại và câu mơ hồ |
| A3 gốc | 20 | Viết tay (ban đầu tiếng Việt, viết lại bằng tiếng Anh cho khớp catalog) |
| Sinh từ mẫu | 200 | 10 mẫu câu, điền 8 từ đầu của caption và brand của sản phẩm chọn ngẫu nhiên (seed=42) |

**Hạn chế quan trọng:** nhóm 200 câu sinh từ mẫu chứa nguyên văn một phần caption, nên BM25 dễ khớp trúng. Vì vậy mục 6 báo cáo kết quả trên **cả 250 câu** và **riêng 50 câu viết tay** khi số liệu khác nhau đáng kể.

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
- **Lý do đổi từ CLIP B/32 sang SigLIP:** mục 6.5 đo thật trên 250 câu (dense-only) cho thấy SigLIP vượt CLIP B/32 rất rõ (Recall@10 0.920 so với 0.656) — chênh lệch đủ lớn để chuyển sang production thay vì chỉ dừng ở so sánh lý thuyết. Ảnh hưởng dây chuyền: hybrid (bm25+dense) từ 0.988 Recall (CLIP) lên **0.996** (SigLIP, mục 6.1).
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
| nDCG@k đa mức | rel = 2 nếu đúng id, 1 nếu cùng category với id đúng, 0 nếu khác |
| MRR | 1 / hạng của kết quả đúng đầu tiên (0 nếu không có trong top-k) |
| Latency | `time.perf_counter()` quanh mỗi lần gọi `search()`; báo trung bình và p95 |

- **Ví dụ tính tay:** đúng = {3, 7}, trả về [3, 1, 7, 2, 9]. Khi đó Recall@5 = 1.0; DCG = 1 + 1/log2(4) = 1.5; IDCG = 1 + 1/log2(3) = 1.63; nDCG@5 ≈ 0.92.
- Mỗi câu có đúng 1 sản phẩm đúng, nên nDCG@1 bằng tỷ lệ câu có sản phẩm đúng đứng hạng 1.
- nDCG đa mức luôn thấp hơn nhiều so với bản nhị phân, vì IDCG giả định cả top-k đều là sản phẩm cùng category. Đây là do cách đo chặt hơn, không phải hệ thống kém đi.

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

### 6.1 Ablation theo component (`step8_run_ablation.py`, 250 câu, k=10)

| Cấu hình | Recall@10 | nDCG@10 | nDCG đa mức@10 | MRR | Latency TB (ms) | Latency p95 (ms) |
|---|---|---|---|---|---|---|
| bm25 only | 0.972 | 0.952 | 0.349 | 0.945 | 3.8 | 8.7 |
| dense only | **0.916** | 0.842 | 0.312 | 0.818 | 77.1 | 100.9 |
| hybrid (bm25 + dense) | **0.996** | **0.954** | 0.351 | 0.940 | 77.9 | 89.0 |
| hybrid + rerank | 0.984 | **0.982** | **0.360** | **0.981** | 867.2 | 1094.2 |

**Số liệu đo LẠI sau khi đổi dense từ CLIP B/32 sang SigLIP (mục 4.1/6.5)** — thay hoàn toàn bảng trước đó (CLIP: dense only Recall 0.652, hybrid nDCG chỉ 0.855, thấp hơn cả bm25 only). Nhận xét:
- **Dense only** tăng vọt so với CLIP (Recall 0.652→0.916, nDCG 0.535→0.842) — đúng như dự đoán từ mục 6.5, và giờ **dense only đã mạnh hơn hẳn** so với lần chạy CLIP cũ, dù vẫn thấp hơn bm25 only trên bộ 250 câu này (vẫn còn hiện tượng câu sinh từ mẫu chứa nguyên văn caption, có lợi cho BM25 hơn dense — xem mục 2.2/9).
- **Hybrid** giờ có Recall cao nhất (0.996) **VÀ** nDCG cao nhất trong 3 cấu hình chưa rerank (0.954, nhỉnh hơn cả bm25 only 0.952) — khác hẳn kết quả với CLIP trước đây (hybrid nDCG 0.855, THẤP hơn bm25 only). Vì dense (SigLIP) giờ đã đủ chính xác, RRF không còn "làm loãng" thứ hạng BM25 đúng bằng cách kéo vào nhiều ứng viên dense sai như trước (hiện tượng đã ghi ở mục 7/8 khi dùng CLIP) — **đây là bằng chứng cụ thể nhất cho thấy đổi sang SigLIP là quyết định đúng**, không chỉ cải thiện dense only mà còn sửa luôn nhược điểm của hybrid.
- **Latency dense only tăng ~3 lần** so với CLIP (24.8ms→77.1ms) — SigLIP base nặng hơn CLIP B/32 khi encode text query trên CPU (chuỗi luôn pad cố định 64 token, xem mục 4.1), đánh đổi chấp nhận được vì vẫn dưới 100ms.
- **Rerank** vẫn đưa nDCG@10 lên cao nhất (0.982) và MRR cao nhất (0.981), nhưng Recall giảm nhẹ (0.996→0.984) — đánh đổi latency lớn nhất trong bảng (77.9ms→867.2ms, ~11 lần) trên CPU.

### 6.2 Ablation theo trường index (`step9_ablation_index_fields.py`, 250 câu, dense = SigLIP)

| Cấu hình | Recall@10 | nDCG@10 | MRR |
|---|---|---|---|
| BM25 trên title (caption) | 0.964 | 0.946 | 0.940 |
| BM25 trên title + description | 0.972 | 0.952 | 0.945 |
| BM25 (title + description) + Dense ảnh | **0.996** | **0.954** | 0.940 |

- Thêm mô tả vào BM25 chỉ tăng nhẹ (Recall +0.008), vì phần lớn câu sinh từ mẫu đã khớp ngay với title.
- Dòng cuối trùng với dòng "hybrid" của mục 6.1, dùng để đối chiếu — với SigLIP, thêm nhánh Dense giờ tăng CẢ Recall lẫn nDCG (khác CLIP trước đây: tăng Recall nhưng giảm nDCG).

### 6.3 Sweep tham số RRF (`step10_sweep_rrf.py`, 250 câu, hybrid, không rerank)

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

### 6.4 So sánh reranker (`step_ablation_rerank.py`, pool 30 ứng viên hybrid)

| Cấu hình | nDCG@1 | nDCG@5 | nDCG@10 | Recall@10 | Thời gian rerank / câu |
|---|---|---|---|---|---|
| 250 câu — không rerank | 0.916 | 0.958 | 0.959 | **0.996** | — |
| 250 câu — cross-encoder MiniLM | **0.980** | **0.982** | **0.982** | 0.984 | 0.25 s |
| 250 câu — DeBERTa distill (A2) | 0.972 | 0.981 | 0.983 | 0.992 | 2.00 s |
| 50 câu viết tay — không rerank | 0.780 | 0.881 | 0.888 | **0.980** | — |
| 50 câu viết tay — cross-encoder MiniLM | **0.900** | 0.909 | 0.909 | 0.920 | — |
| 50 câu viết tay — DeBERTa distill (A2) | 0.860 | 0.906 | **0.914** | 0.960 | — |

**Số liệu đo lại với dense = SigLIP** (thay bảng CLIP trước đó): hàng "không rerank" tăng mạnh ở mọi metric (250 câu: nDCG@1 từ 0.772→0.916, Recall từ 0.992→0.996; 50 câu viết tay: nDCG@1 từ 0.580→0.780) — vì pool hybrid đưa vào rerank giờ đã chính xác hơn nhiều ngay từ đầu (không cần rerank "cứu" nhiều như trước). Rerank vẫn cải thiện nDCG@1 (nhạy nhất với thứ hạng đầu) nhưng biên độ cải thiện **thu hẹp lại** so với lúc dùng CLIP — trên 50 câu viết tay, không rerank giờ đã đạt 78% câu đúng ở hạng 1 (trước đây chỉ 58%), rerank nâng lên 86–90% (tương tự trước). Đây là hệ quả trực tiếp của việc đổi dense: **dense tốt hơn làm giảm bớt (nhưng không xoá bỏ) giá trị gia tăng của rerank**, vì rerank chỉ sửa được cái mà retrieval đã tìm ra nhưng xếp sai hạng — retrieval càng tốt, càng ít việc để rerank "sửa".
- DeBERTa distill (từ paper A2) không hơn MiniLM về nDCG, và chậm hơn khoảng 9 lần trên CPU.

### 6.5 So sánh embedding model (Kaggle, dense-only, 250 câu)

Số liệu lấy từ `data/embedding_model_ablation_results.json`. Chạy trên GPU T4 của Kaggle, **sau khi** eval set đã mở rộng lên 250 câu (bản chạy trước dùng 30 câu, số liệu dưới đây thay thế hoàn toàn bản đó).

| Model | Dim | Recall@10 | nDCG@10 | MRR | Latency TB (ms) |
|---|---|---|---|---|---|
| CLIP ViT-B/32 (đang dùng) | 512 | 0.656 | 0.529 | 0.489 | 10.9 |
| CLIP ViT-L/14 | 768 | 0.756 | 0.631 | 0.592 | 10.1 |
| SigLIP base patch16-224 | 768 | **0.920** | **0.838** | **0.811** | 7.9 |

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
| "fast fold portable projection screen" → "Da-Lite Fast-Fold Deluxe..." | 1758 | 40 (rất yếu — "fast-fold" bị `lower().split()` gộp thành 1 token, không khớp "fast"/"fold" riêng) | 1 | 3 | **1** | **Rerank**: RRF bị nhánh BM25 yếu kéo tụt xuống hạng 3, cross-encoder đọc lại (query, caption) đồng thời và xác nhận đúng, đưa về hạng 1 |

**Nhận xét:** đây chính là bằng chứng "mỗi nhánh bổ khuyết điểm yếu cho nhau" đã nêu ở mục 4.2/4.3 — không phải khẳng định suông. Đặc biệt ví dụ id=1158 minh hoạ đúng bản chất RRF: fusion có thể cho kết quả TỐT HƠN cả 2 nhánh riêng lẻ khi 2 nhánh đồng thuận (khác với mục 7, nơi RRF làm TỆ HƠN khi chỉ 1 nhánh đồng ý).

### 8.1 Kết quả xấu (error analysis)

Phân tích trên **50 câu viết tay**. 200 câu sinh từ mẫu gần như không sinh lỗi mới vì chứa nguyên văn caption. Một câu tính là **lỗi** ở một cấu hình nếu sản phẩm đúng không có trong top-10.

**Số liệu đo lại với dense = SigLIP** (thay bảng CLIP trước đó — bức tranh đổi khác RÕ RỆT, không chỉ đổi vài con số):

**Tổng quan:** 6/50 câu lỗi ở ít nhất một cấu hình (trước đây với CLIP: 16/50); vẫn đúng 1/50 câu lỗi ở cả 4 cấu hình — cùng 1 câu như trước. Số câu lỗi theo cấu hình:

| Cấu hình | bm25 | dense | hybrid | hybrid + rerank |
|---|---|---|---|---|
| Số câu lỗi / 50 | 6 | **1** | **1** | 4 |

(so với CLIP trước đây: bm25=6, dense=**11**, hybrid=**3**, hybrid+rerank=4 — dense và hybrid giảm mạnh số câu lỗi, bm25/hybrid+rerank gần như không đổi vì không phụ thuộc dense)

**Taxonomy lỗi** (một câu có thể thuộc nhiều nhóm):

| Nhóm lỗi | Số câu | Ví dụ (query → caption sản phẩm đúng) | Nguyên nhân | Đề xuất |
|---|---|---|---|---|
| **1. Dense bỏ sót, BM25 tìm được** | **1** (trước đây 10) | Không còn ví dụ "point khoá thuần" nào — nhóm này gần như BIẾN MẤT khi đổi CLIP→SigLIP; câu duy nhất còn lại (id 1681) đã thuộc nhóm 8 | SigLIP nắm bắt tốt các mối liên hệ ảnh-từ khoá đơn giản mà CLIP B/32 bỏ lỡ (vd "wheelbarrow handle", "guitar capo") — bằng chứng trực tiếp cho quyết định đổi model ở mục 6.5 | Không cần đề xuất thêm — đã giải quyết bằng đổi embedding |
| **2. BM25 bỏ sót: diễn đạt khác từ** | 1 | "rug for stairs so people don't slip" → "Cobblestone Jute Stair Tread" (id 241) | Không có từ nào trùng ("rug" và "tread", "stairs" và "stair") | Giữ nhánh Dense; thêm stemming |
| **3. BM25 bỏ sót: caption không phải tiếng Anh** | 2 | "round basket for serving bread" → "APS Brood- en Fruitmand..." (id 1396, tiếng Hà Lan); "long track speed skates" → "Zandstra 525 Lengdeløp Fastskøyte" (id 1493, tiếng Na Uy) | Dataset gốc lẫn nhiều ngôn ngữ; BM25 không khớp được từ tiếng Anh | Phát hiện ngôn ngữ và dịch khi build catalog, hoặc dùng embedding đa ngôn ngữ |
| **4. BM25 bỏ sót: caption thiếu thông tin** | 1 | "inflatable towable tube for boating" → "Ace Racing" (id 748) | Caption chỉ là tên thương mại, không mô tả công dụng | Đưa thêm `category` vào `search_text` |
| **5. BM25 bỏ sót: cách tách từ** | 1 | "fast fold portable projection screen" → "Da-Lite Fast-Fold Deluxe..." (id 1758) | `lower().split()` giữ nguyên "fast-fold" thành một từ | Tách thêm theo dấu câu và gạch nối |
| **6. RRF làm mất kết quả đúng của BM25** | **0** (trước đây 2) | Không còn ví dụ nào — id 888 ("rubber bag...") và "16 inch chainsaw guide bar" trước đây bị RRF làm rớt hạng, giờ Dense (SigLIP) tự tìm đúng ngay từ đầu (hạng 1, xem mục 7) nên 2 nhánh ĐỒNG THUẬN thay vì cạnh tranh | Nhóm lỗi này biến mất KHÔNG PHẢI vì sửa công thức RRF, mà vì Dense giờ đủ mạnh để không còn "thua" BM25 ở các câu này | Không cần đề xuất thêm |
| **7. Rerank làm mất kết quả đúng** | 3 | id 1396, 1493 (nhóm 3) và id 748 (nhóm 4): hybrid tìm được (hạng 1–9), rerank đẩy ra ngoài top-10 | Cross-encoder huấn luyện trên văn bản tiếng Anh MS MARCO, chấm thấp caption ngoại ngữ hoặc caption không mô tả công dụng — VẤN ĐỀ NÀY KHÔNG LIÊN QUAN tới việc đổi CLIP→SigLIP (rerank là tầng độc lập, dùng caption text, không dùng embedding ảnh) | Chỉ rerank khi caption đủ thông tin/tiếng Anh; hoặc kết hợp điểm rerank với điểm RRF thay vì thay thế hoàn toàn |
| **8. Sai ở mọi cấu hình** | 1 | "device that helps someone relearn to walk after an injury" → "Drive Medical tk 1000 Trekker Gait Trainer" (id 1681) | Thuật ngữ chuyên ngành ("gait trainer"), không trùng từ; Dense cải thiện đáng kể (CLIP hạng 31 → SigLIP hạng 12) nhưng VẪN chưa đủ lọt top-10 sau RRF | Từ điển đồng nghĩa theo domain; hoặc embedding chuyên biệt hơn cho domain y tế |

Kết luận của phần phân tích lỗi:
- **Đổi CLIP→SigLIP giải quyết được 2 nhóm lỗi gần như hoàn toàn:** nhóm 1 (Dense bỏ sót, từ 10 câu xuống 1) và nhóm 6 (RRF làm loãng, từ 2 câu xuống 0). Đây là bằng chứng định lượng rõ ràng nhất cho quyết định đổi embedding model.
- **3 nhóm lỗi KHÔNG liên quan tới embedding model** vẫn còn nguyên: BM25 hỏng khi không trùng từ/khác ngôn ngữ/tách từ sai (nhóm 2, 3, 5), và rerank tự nó gây lỗi mới trên caption ngoại ngữ/thiếu thông tin (nhóm 7) — các nhóm này cần giải pháp KHÁC (chuẩn hoá dữ liệu, cải thiện tokenize, cải thiện rerank), không thể sửa bằng cách đổi dense model.
- Ca khó nhất (nhóm 8, thuật ngữ y tế chuyên ngành) vẫn tồn tại dù đã đổi model — cho thấy đây là hạn chế sâu hơn (miền dữ liệu huấn luyện của cả 2 model, không phải lựa chọn model cụ thể).
- RRF và rerank đều có kiểu lỗi riêng (nhóm 6 và 7). Hybrid + rerank lỗi 4 câu, nhiều hơn hybrid (3 câu), dù nDCG trung bình cao hơn nhiều.

---

## 9. Hạn chế

**Về dữ liệu đánh giá:**
- 200/250 câu sinh từ mẫu, chứa nguyên văn caption, nên kết quả trên 250 câu lạc quan hơn thực tế. Mỗi câu chỉ có 1 sản phẩm đúng; các sản phẩm tương tự khác không được tính là đúng.
- Bộ truy vấn ảnh (100 câu) dùng biến đổi nhẹ trên chính ảnh trong index, chưa có ảnh do người dùng tự chụp.

**Về kết quả đã đo:**
- Đã chuyển production từ CLIP B/32 sang SigLIP dựa trên ablation dense-only (mục 6.5); mọi số liệu hybrid/ablation trong báo cáo (mục 6.1 trở đi) đã đo lại trên SigLIP. **Đánh đổi:** encode 1 câu query bằng SigLIP chậm hơn CLIP B/32 khoảng 3 lần trên CPU (77ms so với 25ms, mục 6.1) — chấp nhận được ở quy mô demo nhưng cần lưu ý nếu scale traffic thật.
- Cấu hình RRF mặc định (k=60, top_n=50) không phải tốt nhất theo mục 6.3, dù khoảng cách với cấu hình tốt nhất đã thu hẹp lại sau khi đổi sang SigLIP.
- Latency đo trên một máy CPU; rerank thêm khoảng 0,8 s mỗi truy vấn.

**Về code (chi tiết trong `docs/CAU_TRUC_DU_AN.md` mục 7):**
- `_run_dense` dùng vị trí trong FAISS làm id.
- `ndcg_at_k_graded` trả 0 thay vì None khi thiếu ground truth.
- `step12` gắn nhãn hạng RRF > 50 là "rerank bỏ sót".

---

## 10. Đối chiếu rubric

| Tiêu chí | Đáp ứng ở đâu |
|---|---|
| Cài đặt đúng cơ chế (BM25, RRF, metric tự viết, tái lập được) | Mục 4; `search_core.py`, `metrics.py`; seed cố định; mọi số liệu tái lập bằng script ở mục 11 |
| Tập đánh giá hợp lệ kèm qrels | Mục 2.2, 2.3; `data/eval_set.jsonl`, `data/eval_set_image.jsonl`; hạn chế nêu ở mục 9 |
| Ablation có kiểm soát | Mục 5, 6: mỗi script đổi đúng 1 biến; có đủ 4 cấu hình tối thiểu BM25 / Dense / Hybrid / Hybrid + rerank |
| Phân tích kết quả tốt/xấu, error analysis | Mục 7, 8: 4 ví dụ tốt (mục 8.0, mỗi tầng 1 ví dụ) + taxonomy 8 nhóm lỗi (mục 8.1), đếm trên 50 câu viết tay, có đề xuất cho từng nhóm |
| Frontend + API | Mục 3.2; trang `/search`, bật/tắt component và rerank |

---

## 11. Cách chạy lại

```bash
pip install -r requirements.txt
python step1_build_catalog.py && python step2_build_embeddings.py && python step3_build_index.py
python step8_run_ablation.py            # mục 6.1
python step9_ablation_index_fields.py   # mục 6.2
python step10_sweep_rrf.py              # mục 6.3
python step_ablation_rerank.py          # mục 6.4 (cần models/deberta-10k-rank_net cho dòng DeBERTa)
python step13_build_image_eval_set.py && python step14_eval_image_queries.py   # mục 6.6
python step12_case_study_trace.py       # mục 7
uvicorn app:app --port 8000             # giao diện: http://localhost:8000/search
```

Giải thích chi tiết từng file nằm trong `docs/CAU_TRUC_DU_AN.md`.
