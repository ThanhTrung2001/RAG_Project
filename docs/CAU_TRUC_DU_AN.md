# Cấu trúc dự án và giải thích chi tiết code

Tài liệu này giải thích **từng file trong project làm gì, chạy theo thứ tự nào, và vì sao được viết như vậy**. Comment trong code được giữ ngắn; phần giải thích chi tiết nằm ở đây.

Mục lục:
1. Tổng quan: các phần A1, A2, A3, BPM
2. Cây thư mục
3. Luồng chạy
4. Dữ liệu
5. Giải thích từng file
6. Cách chạy
7. Các vấn đề đã biết trong code (chưa sửa)

---

## 1. Tổng quan

| Phần | Môn | Nội dung | File chính |
|---|---|---|---|
| **A1** | IS6303 | Hệ tìm kiếm đa phương thức (text + ảnh): BM25 + Dense (CLIP/FAISS) + RRF, rerank cross-encoder tuỳ chọn | `step1`–`step3`, `search_core.py`, `metrics.py`, `app.py`, `step8`–`step14` |
| **A2** | IS6303 | Đọc paper RankGPT (Sun et al., EMNLP 2023); áp dụng model distill `deberta-10k-rank_net` (mục 4, 7 của paper) làm reranker và so với cross-encoder của A1 | `search_core.py` (`distilled_rerank`), `step_ablation_rerank.py` |
| **A3** | IS6303 | RAG: dùng `search()` của A1 (tuỳ chọn rerank bằng MiniLM hoặc DeBERTa), rồi LLM (Ollama) viết câu trả lời có trích dẫn; có Naive RAG, Agentic RAG, Memory | `rag_core.py`, `rag_metrics.py`, `step_ablation_rag.py`, `step15_expand_eval_set.py` |
| **BPM** | IS6003 | Mô phỏng event log quy trình "Sales tư vấn sản phẩm" (cấu trúc Loop) | `bpm_event_log_simulation.py` |

Quan hệ giữa các phần:
- A3 **gọi lại** `search()` của A1, không viết lại phần tìm kiếm.
- Rerank cross-encoder MiniLM (A1) bật được trên trang `/search` và trong chatbot `/chat`.
- DeBERTa distill (từ paper A2) là một lựa chọn reranker **tuỳ chọn** của chatbot A3 (mặc định tắt).
- Đồ án không cài RankGPT dùng LLM để xếp hạng (lý do ở `BAO_CAO_A2.md` mục 5.1).

---

## 2. Cây thư mục

```
a1_multimodal_search/
├── step1_build_catalog.py         A1  tải dataset → data/catalog.jsonl + data/images/
├── step2_build_embeddings.py      A1  CLIP embedding → data/image_embeddings.npy, text_embeddings.npy
├── step3_build_index.py           A1  FAISS + BM25 → data/dense.index, data/bm25.pkl
├── search_core.py                 A1  lõi truy hồi: BM25, Dense, RRF, search(), rerank()
│                                  A2  distilled_rerank() (DeBERTa distill)
├── metrics.py                     A1  Recall@k, nDCG@k, nDCG đa mức, MRR, latency
├── app.py                         A1+A3  FastAPI: /api/v1/search, /search_by_image, /components, /ask, /reset_session
├── frontend/
│   ├── search.html                A1  trang tìm kiếm (GET /search)
│   ├── chat.html                  A3  trang chatbot RAG (GET /chat)
│   └── style.css                  CSS dùng chung, phục vụ ở /static/style.css
├── step8_run_ablation.py          A1  ablation: bm25 / dense / hybrid / hybrid+rerank
├── step9_ablation_index_fields.py A1  ablation: trường dữ liệu đưa vào BM25
├── step10_sweep_rrf.py            A1  sweep rrf_k × top_n
├── step12_case_study_trace.py     A1  truy vết hạng của 1 sản phẩm qua từng tầng
├── step13_build_image_eval_set.py A1  tạo bộ truy vấn ảnh bằng crop/rotate/brightness
├── step14_eval_image_queries.py   A1  đánh giá bộ truy vấn ảnh
├── step15_expand_eval_set.py      A1+A3  mở rộng data/eval_set.jsonl lên 250 câu (chạy 1 lần)
├── step_ablation_rerank.py        A2  ablation rerank: không rerank / MiniLM / DeBERTa distill
├── rag_core.py                    A3  Naive RAG, Agentic RAG, Memory
├── rag_metrics.py                 A3  Context Precision, Faithfulness, Answer Relevancy
├── step_ablation_rag.py           A3  ablation: Naive RAG vs Agentic RAG
├── bpm_event_log_simulation.py    BPM mô phỏng event log 3 kịch bản
├── kaggle/
│   ├── kaggle_build_A1_data.py            chạy step1–3 trên Kaggle (GPU) bằng CLIP B/32, xuất zip (bản gốc, không dùng nữa)
│   ├── kaggle_build_A1_data_siglip.py     như trên nhưng bằng SigLIP -- ĐANG DÙNG cho data/ hiện tại
│   └── kaggle_ablation_embedding_models.py so CLIP B/32, CLIP L/14, SigLIP (dense-only)
├── models/deberta-10k-rank_net/   A2  checkpoint tải từ repo RankGPT (không commit)
├── data/                          dữ liệu, index, eval set, kết quả thực nghiệm
└── docs/                          báo cáo và tài liệu
```

---

## 3. Luồng chạy

### 3.1 Offline: chuẩn bị dữ liệu (chạy 1 lần)

```
HF dataset Shopify/product-catalogue (split train)
   │  step1: shuffle(seed=42), lấy 2000 dòng, lọc dòng thiếu ảnh/thiếu cả title lẫn description
   ▼
data/catalog.jsonl + data/images/*.jpg
   │  step2: CLIP ViT-B/32 encode ảnh và caption, chuẩn hoá L2
   ▼
data/image_embeddings.npy (+ text_embeddings.npy khi chạy bản local)
   │  step3
   ├─► FAISS IndexFlatIP trên embedding ẢNH  → data/dense.index
   └─► BM25Okapi trên search_text            → data/bm25.pkl
```

Có thể chạy cả 3 bước trên Kaggle bằng `kaggle/kaggle_build_A1_data.py`. Bản này không tạo `text_embeddings.npy`. Thư mục `data/` hiện tại được tạo theo cách này, nên không có file đó; các bước sau cũng không dùng tới nó.

### 3.2 Online: tìm kiếm (A1): `GET /api/v1/search`

```
query (text hoặc ảnh)
   ├─ bm25 : tách từ lower().split(), chấm BM25 toàn catalog, lấy top_n   (ảnh → trả [])
   └─ dense: CLIP encode query → FAISS tìm top_n ảnh gần nhất
   ▼
RRF fusion: score(doc) = Σ 1/(rrf_k + hạng), hạng tính từ 1, rrf_k=60
   ▼
(tuỳ chọn, rerank=true, chỉ với text) cross-encoder MiniLM chấm lại toàn bộ pool
   ▼
top-k kết quả {id, image_path, caption, score}
```

Trang `/search` có công tắc "Rerank bằng cross-encoder MiniLM", gửi tham số `rerank=true` tới `/api/v1/search`. Tìm bằng ảnh không có rerank (cross-encoder cần cặp text). API không mở `top_n` và `rrf_k`; hai tham số này chỉ đổi trong script ablation.

### 3.3 Online: hỏi đáp (A3): `GET /api/v1/ask`

```
câu hỏi + session_id (+ agentic=true/false, reranker=none|cross_encoder|deberta)
   │
   ├─ Naive RAG (answer):
   │    1. lấy lịch sử hội thoại (tối đa 3 lượt)
   │    2. retrieve(câu hỏi, k=5, reranker)
   │         reranker=none: search(k=5) của A1 (bm25+dense+RRF)
   │         reranker khác: search(top_n=30) lấy 30 ứng viên → rerank → giữ 5
   │    3. không có kết quả → trả câu cố định, KHÔNG gọi LLM
   │    4. build_context: "[i] caption (score=...)"
   │    5. build_prompt: lịch sử + context + luật (chỉ dùng context, trích dẫn [n], trả lời tiếng Việt)
   │    6. call_llm → Ollama llama3.1       (1 lượt gọi LLM)
   │    7. lưu lịch sử, trả {answer, sources}
   │
   └─ Agentic RAG (answer_agentic, max_iters=2):
        lặp tối đa 2 lần:
          retrieve(query hiện tại, k=5, reranker)
          judge_sufficiency(câu hỏi GỐC, context) → "CÓ" hoặc "KHÔNG: lý do"   (1 lượt)
          nếu "CÓ" hoặc là lần cuối → dừng
          rewrite_query(query hiện tại, lý do)                               (1 lượt)
        build_prompt(câu hỏi GỐC, context cuối) → call_llm                   (1 lượt)
        trả {answer, sources, final_query_used}
```

Số lượt gọi LLM mỗi câu hỏi: Naive gọi 1 lượt. Agentic gọi 2 lượt nếu lần đánh giá đầu là "CÓ", ngược lại 4 lượt. Reranker không gọi LLM.

Trang `/chat` có công tắc "Rerank trước khi đưa vào context"; chỉ khi bật mới chọn được Cross-encoder MiniLM (A1) hoặc DeBERTa distill (A2). Danh sách lấy từ `/api/v1/rerankers`; DeBERTa chỉ xuất hiện khi đã có `models/deberta-10k-rank_net`.

### 3.4 Thực nghiệm so sánh reranker (`step_ablation_rerank.py`)

```
search(k=30, top_n=30, bm25+dense)   → pool 30 ứng viên
   ├─ không rerank
   ├─ rerank(): cross-encoder MiniLM, chấm từng cặp (query, caption)
   └─ distilled_rerank(): DeBERTa-v3-base học từ thứ tự ChatGPT, chấm từng cặp
   ▼
nDCG@1/5/10, Recall@10, thời gian rerank
```

---

## 4. Dữ liệu

### 4.1 `data/catalog.jsonl` (2000 dòng)

| Trường | Nguồn | Dùng ở đâu |
|---|---|---|
| `id` | chỉ số sau khi shuffle (0..1999) | khoá chung; trùng với vị trí hàng trong FAISS |
| `image_path` | `data/images/{id:06d}.jpg` | CLIP embedding, hiển thị |
| `caption` | `product_title`, hoặc `product_description[:200]` nếu không có title | CLIP text, cross-encoder, RAG context |
| `search_text` | `title + " " + description` | BM25 |
| `category` | `ground_truth_category` | nDCG đa mức, error analysis |
| `brand` | `ground_truth_brand` | sinh eval set, error analysis |

Dataset gốc `Shopify/product-catalogue` có 48,3k dòng (train 38,6k, test 9,66k), giấy phép apache-2.0. Project chỉ dùng split train.

### 4.2 `data/eval_set.jsonl` (250 câu, dùng chung cho A1 và A3)

Mỗi dòng: `{"query", "query_type": "text", "relevant_ids": [id], "answer_should_mention": [...]}`. Mỗi câu có đúng 1 sản phẩm đúng.

Thành phần (theo `step15_expand_eval_set.py`):

| Nhóm | Số câu | Cách tạo |
|---|---|---|
| 1 | 30 | Câu A1 gốc, viết tay; giữ ở đầu file để thứ tự sản phẩm mà `step13` lấy không đổi |
| 2 | 20 | Câu A3 viết tay (ban đầu tiếng Việt, đã viết lại bằng tiếng Anh) |
| 3 | 200 | Sinh từ 10 mẫu câu, điền 8 từ đầu của caption và brand của sản phẩm chọn ngẫu nhiên (seed=42) |

- `answer_should_mention` của nhóm 1 và 3 = brand + 3 từ đầu caption. Nhóm 2 có sẵn trong script.
- **Hạn chế:** nhóm 3 chứa nguyên văn một phần caption, nên BM25 dễ khớp trúng. Kết quả trên 250 câu vì vậy lạc quan hơn so với chỉ 50 câu viết tay.

### 4.3 Các file khác trong `data/`

| File | Tạo bởi | Nội dung |
|---|---|---|
| `eval_set_image.jsonl`, `query_images/` | `step13` | 100 truy vấn ảnh của 100 sản phẩm khác nhau (34 crop, 33 rotate, 33 brightness) |
| `embedding_model_ablation_results.json` | `kaggle_ablation_embedding_models.py` | Kết quả so CLIP B/32, L/14, SigLIP |
| `rerank_ablation_results.json` | `step_ablation_rerank.py` | Kết quả ablation reranker |
| `eval_set_template.jsonl`, `rag_eval_set_template.jsonl` | viết tay | Mẫu định dạng ban đầu (không còn dùng) |

---

## 5. Giải thích từng file

### 5.1 `step1_build_catalog.py`

- **Vì sao cần một lớp catalog:** `catalog.jsonl` có schema cố định, nên từ bước 2 trở đi không phụ thuộc định dạng dataset Hugging Face. Đổi dataset chỉ cần sửa bước này.
- **Vì sao shuffle trước khi cắt:** dataset gốc có thể được sắp theo nguồn thu thập; lấy 2000 dòng đầu dễ bị lệch về 1–2 category. `seed=42` cho cùng 2000 sản phẩm ở mọi lần chạy.
- **Vì sao 2 trường text:** `caption` ngắn vì CLIP được huấn luyện trên caption ngắn; `search_text` dài vì BM25 có lợi khi có nhiều từ (có từ khoá chỉ xuất hiện trong mô tả).
- Ảnh được chuyển sang RGB vì một số ảnh gốc là CMYK/RGBA.
- Cuối script in phân bố 15 category cấp đầu và cảnh báo nếu một category vượt 40%. Đây là kiểm tra đơn giản, không phải stratified sampling.
- `load_dataset` cache vào `~/.cache/huggingface/datasets/`; lần sau chạy không cần mạng.

### 5.2 `step2_build_embeddings.py`

- **SigLIP** (`google/siglip-base-patch16-224`, đổi từ CLIP B/32 sau ablation `BAO_CAO_TONG_HOP.md` mục 6.5) đưa ảnh và text vào cùng một không gian vector 768 chiều, nên một model làm được cả tìm ảnh bằng text lẫn tìm ảnh bằng ảnh.
- **Chuẩn hoá L2:** khi đó tích vô hướng bằng cosine similarity. `IndexFlatIP` ở bước 3 cần điều kiện này.
- Batch 32 để tận dụng phần cứng mà không tràn bộ nhớ. Text dùng `padding="max_length", max_length=64` (không phải `padding=True` như CLIP) vì SigLIP lấy biểu diễn ở token cuối của chuỗi đã pad cố định — xem `search_core.py::_encode_text()`.
- Embedding catalog được tính 1 lần và lưu ra `.npy`; embedding của query được tính lúc tìm kiếm.

### 5.3 `step3_build_index.py`

- **`IndexFlatIP`:** so sánh chính xác với mọi vector (brute-force). Với 2000 vector là đủ nhanh và không sai số; các index xấp xỉ (HNSW, IVF) chỉ cần khi có hàng triệu vector.
- **Vì sao cần cả hai index:** Dense bắt được nghĩa và từ đồng nghĩa nhưng yếu với từ hiếm chính xác (mã sản phẩm, tên thương hiệu); BM25 thì ngược lại.
- Tách từ chỉ là `lower().split()`: không xử lý dấu câu, stopword hay stemming. Đây là điểm có thể cải thiện.
- `BM25Okapi` không có hàm lưu, nên được pickle cùng danh sách id.

### 5.4 `search_core.py`

**Khởi tạo khi import:** SigLIP, FAISS index, BM25 và catalog được nạp 1 lần lúc import module (vài giây), không nạp lại mỗi request.

**Registry pattern:** mỗi kỹ thuật tìm kiếm là một hàm, đăng ký vào `COMPONENT_REGISTRY` bằng decorator `@component("tên")`.
- `search()` chỉ gọi các tên được yêu cầu, không cần biết bên trong làm gì.
- Thêm kỹ thuật mới chỉ cần viết thêm 1 hàm. `search()`, `app.py` và frontend không đổi, vì frontend hỏi `/api/v1/components` để tự sinh checkbox.
- Ablation script và API dùng chung một cơ chế bật/tắt.
- Tên component không hợp lệ bị bỏ qua, không gây lỗi.

**BM25 trả `[]` với truy vấn ảnh:** ảnh không có từ để khớp. `_fuse_all` bỏ qua danh sách rỗng, nên tìm bằng ảnh thực chất là dense-only.

**RRF** (Cormack et al., 2009):
- Không cộng thẳng điểm, vì điểm BM25 (0 đến 15+) và cosine (0 đến 1) khác thang đo.
- RRF chỉ dùng thứ hạng: `score = Σ 1/(k + hạng)`.
- `k=60` là giá trị paper gốc chọn; paper ghi kết quả gần như không đổi với k từ 20 đến 100.
- `_fuse_all` gộp N danh sách lần lượt từng cặp, để có thể thêm component thứ ba.

**`search(query, query_type, k=10, components=None, top_n=50, use_rerank=False, rrf_k=60, reranker=None, rerank_pool=None)`:**
- `top_n` > k để fusion có đủ ứng viên.
- Khi `use_rerank=True`, toàn bộ pool sau fusion được đưa vào rerank (không cắt còn k trước), vì rerank trên top-10 đã cắt chỉ sắp lại 10 cái đó, không cứu được ứng viên tốt bị fusion xếp thấp.
- Rerank chỉ áp dụng cho truy vấn text.
- `reranker`: `None`, `"cross_encoder"` hoặc `"deberta"` (tra trong `RERANKERS`); tên lạ gây `ValueError`. `use_rerank=True` tương đương `reranker="cross_encoder"`, giữ cho code cũ (`step8`).
- `rerank_pool`: chỉ đưa N ứng viên đầu sau RRF vào rerank; `None` = toàn bộ pool.
- Mặc định không rerank, giữ nguyên hành vi ban đầu của A1.
- `available_rerankers()` trả các reranker dùng được; `"deberta"` chỉ có khi thư mục checkpoint tồn tại.

**`rerank()`:** cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (Nogueira & Cho, 2019).
- Model đọc đồng thời query và caption, nên chính xác hơn bi-encoder nhưng chậm hơn nhiều: phải chạy model cho từng cặp, không tính sẵn được như FAISS.
- Chỉ dùng cho vài chục ứng viên đã lọc. Model được nạp khi gọi lần đầu.

**Phần A2: DeBERTa distill** (`distilled_rerank`):
- Checkpoint `deberta-10k-rank_net` từ mục "Download data and model" của repo RankGPT.
- Theo `config.json`: DeBERTa-v3-base, `hidden_size` 768, 12 layer, 1 đầu ra, ~184M tham số.
- Được huấn luyện bằng RankNet loss để bắt chước thứ tự do `gpt-3.5-turbo` xếp cho 10K query MS MARCO (paper mục 4, Appendix D.1; lệnh huấn luyện mẫu trong README của repo).
- Tên `deberta-10k-rank_net` chỉ có trên GitHub, không có trong paper. Table 13 có hai dòng cùng cấu hình base + RankNet + nhãn ChatGPT ("ChatGPT RankNet" và "deberta-v3-base (184M)"); paper không nói checkpoint công bố ứng với dòng nào.
- Khi chạy, model chấm **từng cặp** (query, caption) độc lập, `max_length=500`, điểm là logit. Không gọi LLM.
- Tokenizer nạp từ thư mục model; nếu thiếu thì dùng `microsoft/deberta-v3-base`.

### 5.5 `metrics.py`

| Hàm | Công thức | Ghi chú |
|---|---|---|
| `recall_at_k` | \|top-k ∩ đúng\| / \|đúng\| | None nếu không có ground truth (bị loại khỏi trung bình) |
| `ndcg_at_k` | DCG = Σ rel_i / log2(i+2), rel ∈ {0,1}; IDCG = Σ_{i<min(\|đúng\|,k)} 1/log2(i+2) | Nhị phân |
| `mrr` | 1 / hạng của kết quả đúng đầu tiên | 0 nếu không có; tính trên danh sách trả về (tối đa k) |
| `ndcg_at_k_graded` | rel = 2 nếu đúng id, 1 nếu cùng `category` với id đúng, 0 nếu khác | IDCG giả định lấp đầy top-k bằng rel=1 sau các rel=2 |
| `evaluate_all` | Gọi `search_fn` cho từng câu, đo latency bằng `perf_counter` | Trả trung bình, latency avg/p50/p95 |

- **Ví dụ tính tay:** đúng = {3, 7}, trả về [3, 1, 7, 2, 9]. Khi đó Recall@5 = 1.0; DCG = 1 + 0.5 = 1.5; IDCG = 1 + 0.63 = 1.63; nDCG@5 ≈ 0.92. Recall không phân biệt được việc id 7 đứng hạng 3 thay vì hạng 2, còn nDCG thì có.
- **nDCG đa mức luôn thấp hơn nhiều so với bản nhị phân**, vì IDCG giả định cả top-k đều là sản phẩm cùng category. Đây là do cách đo chặt hơn, không phải hệ thống kém đi.
- Metric được tự viết (không dùng sklearn); `evaluate_all` nhận `search_fn` làm tham số để có thể test riêng.

### 5.6 `app.py`

| Endpoint | Tham số | Trả về |
|---|---|---|
| `GET /api/v1/components` | — | `{components: [...]}` |
| `GET /api/v1/search` | `q`, `k=10`, `components` (chuỗi phân cách bằng dấu phẩy, rỗng = tất cả), `rerank=false` | `{query, components_used, rerank, latency_ms, results}` |
| `POST /api/v1/search_by_image` | file ảnh (multipart), `k=10`, `components` | như trên, `rerank` luôn false |
| `GET /api/v1/rerankers` | — | `{rerankers: [...]}` |
| `GET /api/v1/ask` | `q`, `k=5`, `agentic=false`, `session_id`, `reranker=none` | `{answer, sources, reranker}` (+ `final_query_used` nếu agentic). 400 nếu reranker không hợp lệ, 503 nếu không gọi được Ollama |
| `POST /api/v1/reset_session` | `session_id` | `{status, session_id}` |
| `GET /` | — | chuyển hướng tới `/search` |
| `GET /search` | — | `frontend/search.html` (A1) |
| `GET /chat` | — | `frontend/chat.html` (A3) |

- Model và index chỉ nằm ở server; trình duyệt chỉ gửi HTTP và nhận JSON.
- Tìm bằng ảnh dùng POST vì file nằm trong body.
- Giao diện A1 và A3 là hai trang riêng; A1 không chứa chatbot.

### 5.7 Các script ablation của A1

| Script | So sánh gì | Ghi chú |
|---|---|---|
| `step8_run_ablation.py` | `bm25` / `dense` / `bm25+dense` / `bm25+dense + rerank` | Gọi đúng `search()` như API; mỗi cấu hình khác đúng 1 thành phần; K=10 |
| `step9_ablation_index_fields.py` | BM25 trên `caption` / trên `search_text` / `search_text` + dense | Build BM25 tạm trong RAM, không ghi đè `bm25.pkl`; tách từ giống step3 |
| `step10_sweep_rrf.py` | `rrf_k ∈ {10, 60, 100}` × `top_n ∈ {20, 50, 100}` | Hybrid, không rerank; kiểm tra nhận định "RRF không nhạy với k" của Cormack et al. |
| `step12_case_study_trace.py` | Hạng của 1 sản phẩm đúng qua BM25 → Dense → RRF → Rerank | 4 case: id 888, 241, 1681, 1396 |
| `step13` + `step14` | Truy vấn ảnh: crop 15% viền, xoay 15° nền trắng, giảm sáng còn 55% | 100 sản phẩm khác nhau lấy theo thứ tự eval set text, mỗi ảnh 1 phép biến đổi; `step14` in bảng theo loại biến đổi, tổng Recall/nDCG/MRR, danh sách câu sai và câu không ở hạng 1 |

Trong `step10`: rrf_k nhỏ làm chênh lệch điểm giữa các hạng lớn hơn (fusion tin thứ hạng gốc hơn); top_n lớn cho fusion nhiều ứng viên hơn nhưng cũng nhiều nhiễu hơn.

### 5.8 `kaggle/`

- `kaggle_build_A1_data.py`: chạy step1–3 trên Kaggle GPU bằng CLIP B/32, cùng hằng số với bản local (2000 sản phẩm, seed=42, batch 32) — **bản gốc, không còn dùng cho `data/` hiện tại**.
  - Không tạo `text_embeddings.npy`.
  - Kết quả nén vào `a1_data_output.zip`; đường dẫn trong zip bắt đầu bằng `data/`, nên giải nén ở thư mục gốc project.
  - Cần cài `transformers==4.57.1`. Nếu viết `<` không có ngoặc kép trong lệnh shell, `<` bị hiểu là chuyển hướng input và ràng buộc phiên bản bị bỏ qua.
- `kaggle_build_A1_data_siglip.py`: giống hệt bản trên nhưng dùng `google/siglip-base-patch16-224` (`AutoModel`/`AutoProcessor` thay `CLIPModel`/`CLIPProcessor`) — **đang dùng để build `data/` hiện tại**, xuất `a1_data_output_siglip.zip`. Cần thêm `sentencepiece` khi cài thư viện (tokenizer SigLIP dùng SentencePiece).
- `kaggle_ablation_embedding_models.py`: so CLIP ViT-B/32, CLIP ViT-L/14 và SigLIP base patch16-224.
  - Đánh giá dense-only (Recall@10, nDCG@10, MRR, latency).
  - Tạo lại catalog trong RAM với cùng seed, nên id khớp với eval set local.
  - **Chi tiết quan trọng với SigLIP:** model lấy biểu diễn ở token cuối của chuỗi đã pad tới độ dài 64 và không dùng attention mask. Nếu pad động như CLIP (`padding=True`), embedding gần như ngẫu nhiên; lần chạy đầu cho kết quả 0.000 vì lỗi này. Code dùng `padding="max_length", max_length=64` cho SigLIP.
  - `EVAL_SET_PATH` trỏ tới dataset Kaggle của một tài khoản cụ thể; cần sửa khi dùng tài khoản khác.

### 5.9 `step15_expand_eval_set.py`

- Đọc `data/eval_set.jsonl` hiện có (30 câu A1), thêm 20 câu A3 viết tay, sinh thêm cho đủ 250 câu, rồi **ghi đè** lên chính file đó.
- Toàn bộ bằng tiếng Anh, vì catalog là tiếng Anh và BM25 so khớp từ vựng.
- Dùng chung một file cho A1 và A3, vì A3 là phần mở rộng của A1; A3 chỉ cần thêm `answer_should_mention`.
- **Chỉ chạy được một lần:** xem mục 7.

### 5.10 `rag_core.py`

- Không có logic tìm kiếm; `retrieve()` chỉ gọi `search_core.search()`.
- `retrieve(query, k, components, reranker)`: không có reranker thì giống hệt `search(k)` của A1. Có reranker thì gọi `search(top_n=30, reranker=..., rerank_pool=30)`, đúng pool đã đo trong `step_ablation_rerank.py`, rồi giữ k kết quả đầu.
- Phần sinh câu trả lời không đăng ký vào `COMPONENT_REGISTRY`, vì registry dành cho các nhánh chạy song song rồi fusion, còn sinh câu trả lời chạy tuần tự sau search.
- **Grounding:** context đánh số `[1]..[k]`, prompt yêu cầu chỉ dùng context, phải trích dẫn `[n]`, phải nói rõ nếu không có sản phẩm phù hợp. Response trả kèm `sources` để người dùng tự đối chiếu.
- **Ollama:** chạy local, không cần API key. Đổi sang API khác chỉ cần sửa `call_llm()`. `stream=False` để đơn giản.
- Không gọi LLM khi search không có kết quả, vì context rỗng dễ khiến LLM bịa câu trả lời. Chỉ `answer()` có kiểm tra này.
- **Memory:** dict trong RAM, giữ 3 lượt gần nhất. Mất khi restart server và không chia sẻ được giữa nhiều server; production cần Redis hoặc database.
- **Agentic:** `max_iters` bắt buộc có để tránh lặp vô hạn. Đánh giá luôn so với câu hỏi gốc. Câu trả lời cuối dùng câu hỏi gốc. `final_query_used` để debug.

### 5.11 `rag_metrics.py` và `step_ablation_rag.py`

- **Vì sao cần metric riêng cho RAG:** Retrieval và Generation có thể sai độc lập. Tìm đúng sản phẩm nhưng LLM vẫn bịa, hoặc không bịa nhưng lạc đề.
- **Vì sao tự viết, không dùng thư viện `ragas`:** `ragas` mặc định dùng OpenAI làm giám khảo, khó cấu hình với Ollama.

| Metric | Cách tính | Cần |
|---|---|---|
| Context Recall@5 | `recall_at_k` của A1 | — |
| Context Precision@5 | Σ (hits/i tại mỗi hạng i đúng) / min(\|đúng\|, k) | — |
| Faithfulness | LLM tách câu trả lời thành claim, gắn SUPPORTED/NOT_SUPPORTED; tỷ lệ SUPPORTED | 1 lượt LLM (llama3.1) |
| Answer Relevancy | cosine(embedding câu hỏi, embedding câu trả lời), `all-MiniLM-L6-v2` | embedding model |
| Mention hit rate | tỷ lệ từ khoá `answer_should_mention` xuất hiện trong câu trả lời | — |
| Citation rate | tỷ lệ câu trả lời có ít nhất một `[n]` | — |
| Latency | thời gian gọi `answer()` / `answer_agentic()` | — |

- Faithfulness trả None (không phải 0) khi không chấm được: 0 nghĩa là có claim nhưng sai hết, khác với không có claim nào.
- Answer Relevancy là bản đơn giản hoá của RAGAS; RAGAS gốc sinh câu hỏi giả định từ câu trả lời, tốn nhiều lượt LLM hơn.
- `step_ablation_rag.py` dùng K=5 và chạy 4 cấu hình: `Naive RAG`, `Naive + MiniLM`, `Naive + DeBERTa` (bỏ qua nếu chưa có model), `Agentic RAG`. So `Naive` với hai cấu hình rerank cho biết tác dụng của reranker lên context và câu trả lời.
- Số lượt gọi LLM mỗi câu (gồm cả chấm Faithfulness): Naive 2 (có hay không rerank), Agentic 3 hoặc 5. Với 250 câu và 4 cấu hình, tổng khoảng 2.250–2.750 lượt. `SAMPLE_SIZE` cho phép chạy thử ít câu trước.

### 5.12 `bpm_event_log_simulation.py`

- **Quy trình:** Khách nêu yêu cầu → Sales tìm sản phẩm → trình bày → "Khách hài lòng?" (Không: quay lại tìm; Có: sang lập đơn). Đây là cấu trúc Loop, với `CT = T / (1 - p)`.

| Kịch bản | T mỗi lần tìm (phút, ngẫu nhiên đều) | p (xác suất phải tìm lại) |
|---|---|---|
| A: không hỗ trợ | 4.0–6.0 | 0.40 |
| B: có A1 | 0.5–1.0 | 0.15 |
| C: có A1 + A3 Agentic | 1.0–1.5 | 0.03 |

- Mọi giá trị T và p là **giả định cố định trong code**, không đo từ hệ thống. Từ `data/catalog.jsonl` chỉ lấy caption của 8 sản phẩm đầu để đặt nhãn cho case.
- Mỗi kịch bản 8 case; mỗi lần tìm là 1 dòng event log; dừng khi `random() >= p`, tối đa 10 lần.
- CT mô phỏng = trung bình tổng thời gian mỗi case. CT lý thuyết = `((lo+hi)/2)/(1-p)`: A 8.33, B 0.88, C 1.29.
- Output: `bpm_event_log.csv` và bảng in ra màn hình.

---

## 6. Cách chạy

```bash
pip install -r requirements.txt

# A1 — chuẩn bị dữ liệu (hoặc chạy kaggle/kaggle_build_A1_data.py rồi giải nén zip ở thư mục gốc)
python step1_build_catalog.py
python step2_build_embeddings.py
python step3_build_index.py

# A1 — thực nghiệm
python step8_run_ablation.py
python step9_ablation_index_fields.py
python step10_sweep_rrf.py
python step12_case_study_trace.py
python step13_build_image_eval_set.py
python step14_eval_image_queries.py

# A2 — cần models/deberta-10k-rank_net (tự bỏ qua dòng DeBERTa nếu không có)
python step_ablation_rerank.py

# A3 — cần Ollama: ollama pull llama3.1
python step_ablation_rag.py

# BPM
python bpm_event_log_simulation.py

# Demo web (A1 + A3)
uvicorn app:app --reload --port 8000     # A1: http://localhost:8000/search   A3: http://localhost:8000/chat
```

---

## 7. Các vấn đề đã biết trong code (chưa sửa)

Các điểm dưới đây được phát hiện khi rà soát code và chưa được sửa.

**A1**
- `_run_dense` dùng vị trí hàng trong FAISS làm `id` của catalog. Chỉ đúng khi id là 0..N-1 theo thứ tự embedding (đúng với dữ liệu hiện tại).
- `ndcg_at_k_graded` trả 0.0 (không phải None) khi không có ground truth, khác `ndcg_at_k`.
- `evaluate_all`: `n_queries` đếm cả câu bị bỏ qua. p95 dùng `int(n*0.95)`, gần giá trị lớn nhất khi n nhỏ.
- `app.py`: `components_used` lặp lại cả tên component không hợp lệ đã bị bỏ qua. Latency đo bằng `time.time()`; lần rerank đầu tiên sau khi khởi động server tính cả thời gian tải model.
- `step12`: danh sách RRF không bị cắt nên hạng RRF có thể vượt 50, còn rerank chỉ nhận `fused[:50]`. Ứng viên ở hạng RRF > 50 sẽ bị báo là "rerank bỏ sót" dù rerank chưa thấy nó.
- `step9`: nDCG đa mức được tính nhưng không in ra.
- `step15`: đọc và ghi đè cùng một file. Chạy lần hai sẽ báo lỗi `ValueError` vì số câu cần sinh bị âm.
- `kaggle_build_A1_data.py` không tạo `text_embeddings.npy`, khác `step2` bản local.

**A3**
- `judge_sufficiency` kiểm tra `verdict.upper().startswith("CÓ")`. Nếu LLM trả `"CÓ"` có ngoặc kép (như chính prompt minh hoạ), `**CÓ**` hoặc `Yes`, kết quả bị coi là "chưa đủ" và gây thêm một lượt viết lại + search.
- Với `max_iters=2`, lượt đánh giá ở vòng cuối luôn bị bỏ qua kết quả (tốn 1 lượt LLM).
- `rewrite_query` nhận query hiện tại chứ không phải câu hỏi gốc, dù tham số tên là `original_query`.
- `answer_agentic` vẫn gọi LLM khi search không có kết quả (`answer` thì không).
- **Lệch ngôn ngữ:** eval set và catalog tiếng Anh, prompt yêu cầu trả lời tiếng Việt.
  - `all-MiniLM-L6-v2` là model tiếng Anh, nên Answer Relevancy giữa câu hỏi tiếng Anh và câu trả lời tiếng Việt kém tin cậy.
  - `rewrite_query` có thể sinh query tiếng Việt, bất lợi cho BM25/CLIP trên catalog tiếng Anh.
- Faithfulness được chấm bởi **chính model** đã sinh câu trả lời (llama3.1), nên có thể thiên vị. Dòng output viết `NOT SUPPORTED` (có dấu cách) không được nhận, có thể làm điểm tăng.
- `call_llm` không đặt timeout.
- `_format_history` luôn thêm `...` kể cả khi câu trả lời ngắn hơn 150 ký tự.

**BPM**
- Không đặt random seed, nên mỗi lần chạy cho số khác nhau. 8 case mỗi kịch bản là ít.
- Giới hạn 10 lần tìm cắt đuôi phân phối hình học, làm CT mô phỏng thấp hơn một chút.
