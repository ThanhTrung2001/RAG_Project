# BÁO CÁO A3 — Chatbot RAG mở rộng từ hệ tìm kiếm A1
### Môn: Thiết kế Hệ thống Thông tin Thông minh dựa trên Truy hồi (IS6303)

> **Trạng thái số liệu (ngày 11/10/2026):**
> - Bản này là bản tóm tắt. Bản đầy đủ là `docs/BAO_CAO_A3_IS6303.md`; hai bản dùng cùng số liệu.
> - Ground truth: `data/eval_set.jsonl` với qrels đa mức 0–3 (mục 5.1).
> - Đã chạy `step_ablation_rag.py` đủ **250 câu × 6 cấu hình** với `llama3.1` trên Kaggle (2 GPU T4) ngày 10/10/2026, 0 lỗi còn lại. Một câu bị timeout đã được chạy lại riêng ở máy local.
> - Đã chạy `step18_ablation_image_rag.py`: 100 ảnh × 3 cấu hình (chỉ chữ / chỉ ảnh / ảnh + chữ).
>
> **Căn cứ yêu cầu:** rubric và lưu ý của giảng viên chỉ có bản ghi lại trong `docs/KE_HOACH_TONG_HOP.md` và bản báo cáo A3 trước; repo không có bản gốc.

Mục lục
1. Yêu cầu
2. Quan hệ với A1 và A2
3. Kiến trúc và luồng xử lý từng bước
4. Grounding: giảm bịa thông tin
5. Phương pháp đánh giá
6. Kết quả
7. Chẩn đoán lỗi: Retrieval hay Generation
8. Đạo đức AI
9. Hạn chế đã biết
10. Đối chiếu yêu cầu
11. Cách chạy

---

## 1. Yêu cầu

Đồ án cuối kỳ (50% điểm môn), theo bản ghi trong repo:

| Tiêu chí | Điểm | Nội dung |
|---|---|---|
| Đề tài baseline | 5đ | Hệ RAG cơ bản |
| Hệ thống hoàn chỉnh | 2.5đ | Front-end, back-end, dữ liệu phù hợp |
| Nâng cấp engineering/research | 2.5đ | Cải tiến so với baseline, **chứng minh bằng ablation** |

Các lưu ý khác được ghi lại:
- A3 phải "nâng cấp" hệ search A1 bằng LLM.
- Phải có dạng chatbot.
- Đánh giá bằng đủ metric RAG (faithfulness, context precision...).
- Ground truth tối thiểu 50 câu/người (nhóm 5 người: 250 câu), dùng chung với A1.
- Phải chẩn đoán được vì sao tốt/xấu, tách lỗi Retrieval và Generation.
- Learning outcomes: G2.1 (thiết kế kiến trúc) và G3.2 (đạo đức AI khi vận hành).

---

## 2. Quan hệ với A1 và A2

- **A1:** A3 **không viết lại** phần tìm kiếm. `rag_core.py` gọi `search_core.search()` của A1 (BM25 + Dense SigLIP + RRF). Dữ liệu (catalog, index) là dữ liệu A1 đã chuẩn bị.
- **Rerank (tuỳ chọn, mặc định tắt):** trước khi đưa sản phẩm cho LLM, có thể xếp lại bằng một trong hai reranker:
  - **Cross-encoder MiniLM:** reranker của A1.
  - **DeBERTa distill** (`deberta-10k-rank_net`): model từ paper A2 (Sun et al., EMNLP 2023, mục 4 và 7). Nó được huấn luyện để bắt chước thứ tự do ChatGPT xếp.
- Không có bước nào dùng LLM để xếp hạng; LLM chỉ dùng để sinh câu trả lời và (ở chế độ Agentic) tự đánh giá, viết lại query.

**Cải tiến so với A1:**
1. **Ground truth qrels đa mức 0–3** thay cho "1 sản phẩm đúng / câu" (mục 5.1).
2. **Hỏi bằng ảnh**, có thể kèm chữ, trong chatbot (mục 3.3).

---

## 3. Kiến trúc và luồng xử lý từng bước

### 3.1 Thành phần

| Thành phần | File | Vai trò |
|---|---|---|
| Giao diện chatbot | `frontend/chat.html` (GET `/chat`) | Khung chat, nút đính kèm ảnh, công tắc rerank và Agentic, danh sách nguồn |
| API | `app.py`: `GET /api/v1/ask`, `POST /api/v1/ask_image`, `GET /api/v1/rerankers`, `POST /api/v1/reset_session` | Nhận câu hỏi (chữ hoặc ảnh), gọi `rag_core` |
| Lõi RAG | `rag_core.py` | `retrieve`, `build_context`, `build_prompt`, `call_llm`, Naive, Agentic, Memory |
| Tìm kiếm | `search_core.py` (A1) | `search()`, `rerank()`, `distilled_rerank()` |
| LLM | Ollama, `http://localhost:11434/api/generate`; model mặc định `llama3.1`, đổi bằng biến môi trường `OLLAMA_MODEL` | Sinh câu trả lời, tự đánh giá, viết lại query |

### 3.2 Luồng Naive RAG (baseline): `answer()`

1. Trình duyệt gửi `GET /api/v1/ask?q=...&k=5&agentic=false&reranker=none|cross_encoder|deberta&session_id=...`.
   - Tên reranker không hợp lệ trả lỗi 400.
2. **Memory:** nếu có `session_id`, lấy tối đa 3 lượt hỏi-đáp gần nhất của phiên (dict trong RAM).
3. **Retrieval**, `retrieve(q, k=5, reranker)`:
   - `reranker=none`: gọi `search(q, k=5)` của A1, tức BM25 + Dense + RRF, mỗi nhánh top-50.
   - Có reranker: `search(top_n=30)` lấy 30 ứng viên sau RRF, reranker chấm lại, giữ 5 sản phẩm đầu.
4. Nếu không có kết quả: trả câu cố định "Không tìm thấy sản phẩm nào phù hợp...", **không gọi LLM**.
5. **Augmentation:**
   - `build_context()` tạo mỗi sản phẩm một dòng: `[i] <caption> (score=<điểm RRF>)`.
   - `build_prompt()` ghép lịch sử, context và các luật ở mục 4.
6. **Generation:** `call_llm()` gọi Ollama **1 lượt**.
7. Lưu lượt hỏi-đáp vào Memory. Trả `{answer, sources, reranker}`. `sources` là 5 sản phẩm thật, được hiển thị kèm ảnh.

### 3.3 Hỏi bằng ảnh: `POST /api/v1/ask_image`

| Đầu vào | Cách tìm | Rerank |
|---|---|---|
| Chỉ ảnh | `search(image, query_type="image")` của A1: SigLIP mã hoá ảnh, FAISS tìm ảnh sản phẩm gần nhất | Không |
| Ảnh + chữ | 50 ứng viên tìm theo ảnh và 50 ứng viên tìm theo chữ (BM25 + Dense), gộp bằng RRF | Không |

- **LLM vẫn chỉ đọc chữ.** Ảnh chỉ dùng ở tầng Retrieval. Prompt có thêm một dòng báo rằng người dùng đã gửi ảnh và danh sách được tìm theo độ giống với ảnh.
- **Không rerank khi có ảnh**, vì reranker chỉ đọc chữ nên bỏ mất tín hiệu ảnh. Thử 3 ảnh kèm câu ngắn: cả 3 lần, bật cross-encoder đều đẩy sản phẩm đúng xuống.
- Đường đi của câu hỏi chữ không đổi (đã kiểm tra prompt và kết quả tìm trùng khớp với code trước khi thêm ảnh).

### 3.4 Luồng Agentic RAG (nâng cấp): `answer_agentic(max_iters=2)`

1. Bước 1–3 như Naive, với query hiện tại (ban đầu là câu hỏi gốc).
2. `judge_sufficiency(câu hỏi gốc, context)`: LLM trả "CÓ" hoặc "KHÔNG: <lý do>". Tốn **1 lượt LLM**.
3. Nếu "CÓ" hoặc đã là vòng cuối: dừng lặp.
4. Ngược lại, `rewrite_query(query hiện tại, lý do)` cho LLM viết query mới (**1 lượt LLM**), rồi quay lại bước 1.
5. Sau vòng lặp: `build_prompt(câu hỏi gốc, context cuối)` rồi `call_llm()` (**1 lượt LLM**).
6. Trả thêm `final_query_used` (query cuối dùng để search). Giao diện hiển thị nó nếu khác câu hỏi gốc.

Số lượt gọi LLM mỗi câu hỏi:

| | Naive | Agentic |
|---|---|---|
| Không cần search lại | 1 | 2 |
| Có search lại | — | 4 |
| Reranker | không gọi LLM | không gọi LLM |

### 3.5 Memory

- Lưu trong RAM theo `session_id`, giữ 3 lượt gần nhất. Mỗi câu trả lời cũ được cắt 150 ký tự khi chèn vào prompt. Lượt có ảnh được ghi là `[ảnh] <câu hỏi>`.
- Giao diện sinh `session_id` ngẫu nhiên khi mở trang. Nút "Mới" gọi `/api/v1/reset_session` rồi đổi sang `session_id` mới.
- Giới hạn: mất khi khởi động lại server; không dùng chung được giữa nhiều server.

### 3.6 Giao diện `/chat`

- **Nút "Ảnh":** đính kèm một ảnh sản phẩm cho lượt hỏi tiếp theo, có ô xem trước và nút bỏ ảnh.
- **Công tắc "Rerank trước khi đưa vào context":** chỉ khi bật mới chọn được "Cross-encoder MiniLM (A1)" hoặc "DeBERTa distill từ ChatGPT (A2)". DeBERTa chỉ hiện khi server có checkpoint.
- **Công tắc "Agentic":** mặc định tắt.
- **Khung nguồn** bên phải: 5 sản phẩm đưa vào context, kèm số `[n]`, ảnh, điểm RRF và điểm rerank (nếu có).
- **Sản phẩm trong câu trả lời:** dưới mỗi câu trả lời hiện thẻ nhỏ cho những sản phẩm được LLM trích dẫn `[n]`.
- **Chi tiết sản phẩm:** bấm vào thẻ hoặc một nguồn thì cột phải hiện ảnh lớn, tên, brand, category, mô tả (`GET /api/v1/products/{id}`).
- Khi không gọi được Ollama, API trả lỗi 503 và khung chat hiện thông báo lỗi.

---

## 4. Grounding: giảm bịa thông tin

Ba biện pháp có trong code:
1. **Context đánh số `[1]..[5]`** để LLM trích dẫn đúng nguồn.
2. **Luật trong prompt** (`build_prompt`), nguyên văn: *"CHỈ được dùng thông tin trong danh sách sản phẩm dưới đây để trả lời. KHÔNG được bịa thêm chi tiết (giá, tính năng...) không có trong danh sách. Nếu danh sách không có sản phẩm phù hợp, hãy nói rõ là không tìm thấy thay vì đoán."*; và yêu cầu trả lời tiếng Việt, trích dẫn `[n]` khi nhắc tới sản phẩm.
3. **Trả kèm `sources`** để người dùng tự kiểm chứng.

**Giới hạn thông tin trong context:** mỗi sản phẩm chỉ có **caption** (tên sản phẩm) và điểm RRF. Context không có mô tả, brand riêng, category hay giá, và **catalog không có trường giá**. Vì vậy:
- Câu hỏi về giá hoặc thuộc tính không nằm trong tên sản phẩm không thể trả lời đúng.
- Mention hit có mức trần dưới 1: 0.708 trên 250 câu, 0.760 trên 50 câu viết tay (tỷ lệ từ khoá `answer_should_mention` có trong tên sản phẩm đúng).

---

## 5. Phương pháp đánh giá

### 5.1 Dữ liệu

`data/eval_set.jsonl`, 250 câu, dùng chung với A1:
- 30 câu A1 viết tay, 20 câu A3 viết tay, 200 câu sinh từ mẫu (chi tiết ở `BAO_CAO_TONG_HOP.md` mục 2.2).
- Mỗi câu có `qrels` (nhãn 0–3 cho khoảng 20 sản phẩm trong pool ứng viên, `step16_build_qrels.py`), `relevant_ids` (các sản phẩm nhãn ≥ 2; 240/250 câu chỉ có 1) và `answer_should_mention` (brand và vài từ đầu của tên sản phẩm).
- **Cải tiến so với A1:** A1 chỉ có 1 sản phẩm đúng mỗi câu, nhãn nhị phân. A3 nâng lên qrels đa mức kiểu TREC: pooling top-10 BM25/Dense/Hybrid (5.064 cặp câu hỏi–sản phẩm), gán nhãn 0–3 theo dữ liệu catalog, rà soát lại, và sửa 3 truy vấn mâu thuẫn với sản phẩm đúng. Chi tiết ở `BAO_CAO_A3_IS6303.md` mục 3.2.1.
- **Hỏi bằng ảnh:** 100 ảnh của `data/eval_set_image.jsonl` (ảnh catalog bị cắt, xoay hoặc đổi độ sáng). Mỗi ảnh được ghép với câu chữ, qrels và `answer_should_mention` của cùng sản phẩm trong `eval_set.jsonl`.
- Toàn bộ câu hỏi bằng tiếng Anh; câu trả lời của hệ thống bằng tiếng Việt.

### 5.2 Metric (`rag_metrics.py`, `step_ablation_rag.py`)

| Tầng | Metric | Cách tính | Cần |
|---|---|---|---|
| Retrieval | Context Recall@5 | Sản phẩm đúng có trong 5 nguồn không (`recall_at_k` của A1) | — |
| Retrieval | Context Precision@5 | Σ (hits/i tại mỗi hạng i đúng) / min(\|đúng\|, 5). Với 1 sản phẩm đúng thì bằng 1/hạng nếu nằm trong top-5 | — |
| Retrieval | Context nDCG@5 đa mức (mới) | DCG/IDCG với gain = nhãn 0–3 trong `qrels` (`metrics.ndcg_at_k_graded`) | — |
| Generation | Faithfulness | LLM tách câu trả lời thành các claim, gắn SUPPORTED hoặc NOT_SUPPORTED so với context; tỷ lệ SUPPORTED | 1 lượt LLM |
| Generation | Answer Relevancy | cosine(embedding câu hỏi, embedding câu trả lời) bằng `all-MiniLM-L6-v2` | embedding model |
| Generation | Mention hit rate | Tỷ lệ từ khoá `answer_should_mention` xuất hiện trong câu trả lời | — |
| Generation | Citation rate | Tỷ lệ câu trả lời có ít nhất một `[n]` | — |
| Hệ thống | Latency | Thời gian gọi `answer()` / `answer_agentic()` | — |

- Faithfulness và Answer Relevancy là phiên bản **đơn giản hoá** theo ý tưởng của RAGAS, tự viết để chạy được với Ollama local; không dùng thư viện `ragas`.
- Answer Relevancy của RAGAS gốc sinh câu hỏi giả định từ câu trả lời; bản này chỉ so trực tiếp embedding.

### 5.3 Các cấu hình ablation

| Cấu hình | Khác baseline ở | Đo được gì |
|---|---|---|
| Naive RAG | — (baseline) | |
| Naive + MiniLM | Rerank cross-encoder trên 30 ứng viên | Tác dụng của reranker A1 lên context và câu trả lời |
| Naive + DeBERTa | Rerank DeBERTa distill trên 30 ứng viên | Tác dụng của kỹ thuật từ paper A2 |
| Agentic RAG | Tự đánh giá và search lại | Tác dụng của vòng lặp agentic |
| Agentic + MiniLM | Agentic, mọi lượt tìm đều rerank MiniLM | Tương tác giữa rerank và Agentic |
| Agentic + DeBERTa | Agentic, mọi lượt tìm đều rerank DeBERTa | Tương tác giữa rerank và Agentic |

- 4 cấu hình đầu khác baseline đúng một thành phần. 2 cấu hình cuối hoàn thiện bảng 2 × 3: {Naive, Agentic} × {không rerank, MiniLM, DeBERTa}.
- Hỏi bằng ảnh (`step18_ablation_image_rag.py`): 3 cấu hình Naive trên cùng 100 sản phẩm: chỉ chữ / chỉ ảnh / ảnh + chữ.

---

## 6. Kết quả

### 6.1 Tầng Retrieval: 5 sản phẩm đưa cho LLM

Số liệu lấy từ các nguồn thực sự đưa cho LLM trong lần chạy 6.2 (`rag_core.retrieve(q, k=5, reranker)`, dense = SigLIP). "Không có trong 5 nguồn" nghĩa là không nguồn nào có nhãn ≥ 2. Thời gian là của riêng bước `retrieve()` trên CPU, đo ngày 02/10/2026.

| Reranker | Tập câu | Ctx Recall@5 | Ctx Precision@5 | Ctx nDCG@5 | Đúng ở [1] | Không có trong 5 nguồn | Thời gian / câu |
|---|---|---|---|---|---|---|---|
| Không rerank | 250 | **0.983** | 0.942 | 0.878 | 229/250 (91.6%) | 3 | 0.08 s |
| Cross-encoder MiniLM (A1) | 250 | 0.979 | 0.974 | 0.895 | **245/250 (98.0%)** | 3 | 0.29 s |
| DeBERTa distill (A2) | 250 | 0.981 | **0.975** | **0.902** | 244/250 (97.6%) | 3 | 3.16 s |
| Không rerank | 50 viết tay | **0.945** | 0.877 | 0.820 | 41/50 (82%) | **2** | 0.07 s |
| Cross-encoder MiniLM (A1) | 50 viết tay | 0.915 | 0.892 | 0.828 | **45/50 (90%)** | 3 | 0.24 s |
| DeBERTa distill (A2) | 50 viết tay | 0.925 | **0.896** | **0.846** | 44/50 (88%) | 3 | 3.24 s |
| Không rerank | 200 sinh từ mẫu | 0.993 | 0.958 | 0.892 | 188/200 (94%) | 1 | — |
| Cross-encoder MiniLM (A1) | 200 sinh từ mẫu | **0.995** | **0.995** | 0.911 | **200/200** | **0** | — |
| DeBERTa distill (A2) | 200 sinh từ mẫu | **0.995** | **0.995** | **0.916** | **200/200** | **0** | — |

**Nhận xét:**
- **Không rerank, SigLIP đã đưa được sản phẩm liên quan vào 5 nguồn ở 48/50 câu viết tay.** Hai câu không có là *"round basket for serving bread"* và *"long track speed skates"* (caption tiếng Hà Lan và tiếng Na Uy).
- **Rerank cải thiện thứ hạng, không cải thiện độ phủ.** Trên 50 câu viết tay, tỷ lệ đúng ở `[1]` tăng từ 82% lên 88–90%, nDCG từ 0.820 lên 0.828–0.846; nhưng Recall giảm từ 0.945 xuống 0.915–0.925 (thêm 1 câu mất sản phẩm đúng).
- **MiniLM và DeBERTa chênh nhau 1 câu** ở metric nhị phân. DeBERTa có nDCG đa mức cao hơn (0.846 so với 0.828), tức xếp các sản phẩm liên quan một phần cao hơn, nhưng chậm hơn khoảng 11 lần trên CPU.
- 200 câu sinh từ mẫu gần như đạt tuyệt đối khi có rerank. Đây là dấu hiệu nhóm câu này quá dễ, không nên dùng riêng để so sánh.

### 6.2 Tầng Generation: 250 câu × 6 cấu hình

`llama3.1` (8B) qua Ollama, `k = 5`, có chấm Faithfulness, Kaggle 2 GPU T4, `--workers 4`. 4 cấu hình đầu và 2 cấu hình Agentic + rerank chạy ở hai lượt Kaggle khác nhau cùng ngày 10/10/2026. Latency chỉ dùng để so sánh trong cùng lượt chạy, vì 4 câu được xử lý song song trên 2 GPU.

**Bảng 6.2a: 250 câu**

| Cấu hình | Ctx Recall@5 | Ctx Precision@5 | Ctx nDCG@5 | Faithfulness | Answer Relevancy | Mention hit | Citation | Latency (s) | Rewrite |
|---|---|---|---|---|---|---|---|---|---|
| Naive RAG | **0.983** | 0.942 | 0.878 | **0.725** | 0.573 | **0.674** | 0.728 | **3.9** | — |
| Naive + MiniLM | 0.979 | 0.974 | 0.895 | 0.707 | **0.590** | **0.674** | **0.760** | 8.4 | — |
| Naive + DeBERTa | 0.981 | **0.975** | **0.902** | 0.700 | 0.582 | **0.674** | 0.728 | 9.7 | — |
| Agentic RAG | 0.919 | 0.857 | 0.799 | 0.718 | 0.573 | 0.660 | 0.660 | 14.0 | 0.524 |
| Agentic + MiniLM | 0.948 | 0.939 | 0.852 | 0.664 | 0.582 | 0.660 | 0.716 | 13.2 | 0.604 |
| Agentic + DeBERTa | 0.957 | 0.946 | 0.874 | 0.657 | 0.574 | 0.654 | 0.656 | 16.2 | 0.536 |

**Bảng 6.2b: 50 câu viết tay**

| Cấu hình | Ctx Recall@5 | Ctx Precision@5 | Ctx nDCG@5 | Faithfulness | Answer Relevancy | Mention hit | Citation | Latency (s) | Rewrite |
|---|---|---|---|---|---|---|---|---|---|
| Naive RAG | **0.945** | 0.877 | 0.820 | **0.765** | 0.498 | **0.640** | 0.780 | **2.9** | — |
| Naive + MiniLM | 0.915 | 0.892 | 0.828 | 0.656 | 0.516 | 0.630 | 0.800 | 4.2 | — |
| Naive + DeBERTa | 0.925 | **0.896** | **0.846** | 0.760 | **0.520** | **0.640** | **0.860** | 9.7 | — |
| Agentic RAG | 0.835 | 0.711 | 0.685 | 0.694 | 0.513 | 0.580 | 0.700 | 9.9 | 0.640 |
| Agentic + MiniLM | 0.880 | 0.834 | 0.766 | 0.560 | 0.499 | 0.580 | 0.760 | 10.9 | 0.700 |
| Agentic + DeBERTa | 0.885 | 0.834 | 0.797 | 0.608 | 0.493 | 0.610 | 0.740 | 19.9 | 0.480 |

**Độ dao động giữa hai lần chạy:** 4 cấu hình đầu đã chạy hai lần (06/10 trên ground truth cũ, 10/10 trên qrels), cùng code, cùng model. Các metric Generation không phụ thuộc ground truth, vậy mà vẫn chênh nhau tới **±3 điểm phần trăm** (ví dụ Faithfulness của Naive 0.701 → 0.725, Citation của DeBERTa 0.756 → 0.728). Vì vậy mọi chênh lệch dưới khoảng 3 điểm giữa các cấu hình không thể coi là khác biệt thật.

**Nhận xét:**
1. **Agentic RAG kém hơn Naive, ở cả hai lần chạy.** Trên 250 câu: Recall 0.983 → 0.919, nDCG 0.878 → 0.799, Citation 0.728 → 0.660, chậm hơn khoảng 3,6 lần. Mức giảm ở tầng Retrieval vượt xa độ dao động, nên đây là kết quả âm chắc chắn.
2. **Rerank cải thiện Retrieval** (Precision 0.942 → 0.975, nDCG 0.878 → 0.902) **nhưng không đo được cải thiện ở Generation:** Mention hit bằng nhau (0.674), các chênh lệch khác nằm trong ngưỡng dao động.
3. **Bảng 2 × 3: rerank giảm thiệt hại của Agentic nhưng không cứu được nó.** Agentic + rerank có Recall 0.948–0.957, cao hơn Agentic (0.919) nhưng vẫn thấp hơn Naive cùng reranker (0.979–0.981). Faithfulness của Agentic + rerank giảm 5–6 điểm (0.657–0.664), vượt ngưỡng dao động; chưa rõ nguyên nhân.
4. **Điểm nghẽn nằm ở Generation:** Recall của Naive là 0.983, nhưng Mention hit chỉ 0.674 (mức trần 0.708), và 68/250 câu trả lời (27%) không có trích dẫn `[n]`.

**Vì sao Agentic kém** (đếm từ `data/rag_ablation_details.jsonl`):

| | Agentic | Agentic + MiniLM | Agentic + DeBERTa |
|---|---|---|---|
| Giám khảo vòng 1 trả "KHÔNG" | 131 | 151 | 134 |
| Trong đó sản phẩm đúng đã ở `[1]` | 117 | 147 | 130 |
| Truy vấn viết lại có ngoặc kép | 118 | 136 | 117 |
| Câu mất sản phẩm đúng so với Naive cùng reranker | 17 | 9 | 6 |
| Câu được cứu | 2 | 0 | 0 |

- **Giám khảo đánh giá sai:** đánh "chưa đủ" ở hơn nửa số câu, kể cả khi sản phẩm đúng đã nằm ở `[1]`. Rerank không làm giám khảo đánh giá đúng hơn, vì lỗi nằm ở phần ra quyết định chứ không nằm ở context.
- **Truy vấn viết lại kém hơn truy vấn gốc:** bị bọc trong ngoặc kép, nên BM25 nhận `"wheelbarrow` thay vì `wheelbarrow`; một phần bị viết sang tiếng Việt.
- **Không có cơ chế quay về kết quả lượt đầu:** mỗi lần viết lại sai làm mất hẳn sản phẩm đúng. Rerank giảm được số câu bị mất (17 → 9 và 6), vì reranker đọc truy vấn viết lại tốt hơn BM25.

### 6.3 Hỏi bằng ảnh: 100 ảnh × 3 cấu hình

| Cấu hình | Ctx Recall@5 | Ctx nDCG@5 | Đúng ở [1] | Faithfulness | Answer Relevancy | Mention hit | Citation | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| Chỉ chữ (mốc) | 0.973 | 0.868 | 88/100 | 0.649 | **0.562** | **0.660** | 0.840 | **4.2** |
| Chỉ ảnh | 0.988 | 0.858 | **100/100** | 0.694 | 0.327 | 0.560 | **0.890** | 7.7 |
| **Ảnh + chữ** (cấu hình chính) | **0.995** | **0.895** | 98/100 | **0.733** | 0.527 | 0.625 | 0.820 | 4.4 |

- **Ảnh + chữ cho context tốt nhất**, và cứu được đúng 2 câu viết tay mà luồng chữ tìm trượt (caption tiếng Hà Lan và tiếng Na Uy), đưa sản phẩm đúng lên `[2]`.
- **Chỉ ảnh tìm đúng 100/100 nhưng câu trả lời kém dứt khoát.** LLM thường viết *"Tôi không thể xem được ảnh mà người dùng gửi, nhưng..."* rồi liệt kê nhiều sản phẩm (câu trả lời trung bình 305 ký tự, so với 138 khi chỉ dùng chữ). Đây là hệ quả của dòng "Bạn KHÔNG xem được ảnh" trong prompt.
- Answer Relevancy so câu trả lời với câu chữ đã ghép, kể cả ở cấu hình chỉ ảnh.
- **Giới hạn:** ảnh truy vấn là ảnh catalog bị biến đổi nhẹ, nên tầng Retrieval dễ hơn thực tế.

---

## 7. Chẩn đoán lỗi: Retrieval hay Generation

A3 gồm hai tầng. Khi câu trả lời sai, phải xác định tầng gây lỗi trước khi sửa:

| Dấu hiệu | Tầng lỗi | Cách xác nhận |
|---|---|---|
| Sản phẩm đúng **không có** trong `sources` | Retrieval (A1 + reranker) | Context Recall@5 = 0 cho câu đó |
| Sản phẩm đúng có trong `sources` nhưng câu trả lời không nhắc tới, nhắc sai, hoặc trích dẫn sai `[n]` | Generation | Mention hit thấp; đối chiếu `[n]` với danh sách nguồn |
| Câu trả lời có thông tin không nằm trong context | Generation (bịa) | Faithfulness thấp |
| Câu trả lời không có `[n]` | Generation (không tuân thủ grounding) | Citation rate |

**Ví dụ tầng Retrieval:** câu #27 *"device that helps someone relearn to walk after an injury"*. Qrels có 4 sản phẩm liên quan: gait trainer (nhãn 3), xe tập đi, nạng, gậy chống (nhãn 2). 5 nguồn chỉ chứa gậy chống, nên Context Recall@5 = 0.25. Câu hỏi dùng ngôn ngữ đời thường, tên sản phẩm dùng thuật ngữ chuyên ngành, nên BM25 không tìm thấy. Cần sửa ở search (từ đồng nghĩa theo domain), không phải ở prompt.

**Ví dụ tầng Generation, từ chối dù context đúng:** câu #34 *"brake drum for a Jeep"*. Sản phẩm đúng "Crown Automotive 52005350 Brake Drum" nằm ở `[1]`, nhưng `llama3.1` trả lời *"Không tìm thấy sản phẩm phù hợp..."* vì tên sản phẩm không có chữ "Jeep". Hiện tượng này xảy ra ở khoảng 50/250 câu của mỗi cấu hình Naive. Hướng sửa: sửa prompt để LLM chọn sản phẩm gần nhất kèm mức độ chắc chắn, và đưa thêm mô tả vào context.

**Ví dụ lỗi ra quyết định của Agentic:** câu #12 *"replacement handle for wheelbarrow"*. Lượt đầu có sản phẩm đúng ở `[2]`, nhưng giám khảo trả "KHÔNG". LLM viết lại thành `"wheelbarrow replacement handle"` (có ngoặc kép), lượt tìm thứ hai mất sản phẩm đúng.

---

## 8. Đạo đức AI

| Rủi ro | Ví dụ | Biện pháp trong hệ thống |
|---|---|---|
| Bịa thông tin | LLM tự thêm giá hoặc tính năng không có trong danh sách | Luật trong prompt (mục 4); context không chứa giá nên mọi con số giá đều là bịa và kiểm được |
| Trích dẫn sai nguồn | Nói "[2]" nhưng mô tả sản phẩm [1] | Hiển thị `sources` kèm số `[n]` để người dùng đối chiếu |
| Trả lời khi không có dữ liệu | LLM cố trả lời dù search rỗng | Naive không gọi LLM khi search rỗng (Agentic chưa có kiểm tra này, xem mục 9) |
| Vòng lặp Agentic không dừng | LLM liên tục đánh giá "chưa đủ" | `max_iters=2` |
| Thiên lệch khi xếp hạng sản phẩm | Paper A2 khuyến cáo không dùng LLM để xếp hạng sản phẩm | Reranker DeBERTa (học từ ChatGPT) là tuỳ chọn, mặc định tắt; xếp hạng mặc định là BM25 + Dense + RRF |
| Lộ dữ liệu hội thoại và ảnh | Lịch sử chat và ảnh tải lên | Lịch sử chỉ trong RAM, giữ 3 lượt, xoá được bằng nút "Mới"; ảnh chỉ xử lý trong bộ nhớ, không lưu; LLM chạy local, không gửi ra dịch vụ ngoài |

---

## 9. Hạn chế đã biết

**Đánh giá:**
- Mỗi cấu hình mới chạy tối đa 2 lần; metric Generation dao động khoảng ±3 điểm giữa các lần chạy (mục 6.2).
- Faithfulness do **chính model sinh câu trả lời** (`llama3.1`) chấm, nên có thể thiên vị; 50–58/250 câu mỗi cấu hình không chấm được (None).
- **Lệch ngôn ngữ:** câu hỏi và catalog tiếng Anh, câu trả lời tiếng Việt. Answer Relevancy dùng model embedding tiếng Anh; Mention hit dựa trên so khớp chuỗi.
- 200/250 câu sinh từ mẫu, dễ hơn câu thực tế (mục 6.1).
- Qrels do LLM chấm, người rà soát các nhãn 2–3; pool chỉ lấy top-10 của 3 cách tìm hiện có.
- Bộ ảnh truy vấn là ảnh catalog bị biến đổi nhẹ, chưa có ảnh người dùng tự chụp.

**Code** (chi tiết trong `docs/CAU_TRUC_DU_AN.md` mục 7):
- `judge_sufficiency` kiểm tra câu trả lời bắt đầu bằng "CÓ". Nếu LLM trả `"CÓ"` có ngoặc kép, `**CÓ**` hoặc "Yes", kết quả bị coi là chưa đủ.
- Với `max_iters=2`, kết quả đánh giá ở vòng cuối luôn bị bỏ qua (tốn 1 lượt LLM).
- `rewrite_query` nhận query hiện tại chứ không phải câu hỏi gốc, và code không làm sạch ngoặc kép hay tiếng Việt trong query viết lại.
- Agentic luôn dùng kết quả của lượt tìm cuối, không giữ kết quả tốt hơn ở lượt đầu.
- `answer_agentic` vẫn gọi LLM khi search rỗng.
- Memory chỉ dùng cho câu trả lời cuối; không dùng khi search hay khi tự đánh giá.

**Phạm vi:**
- Context chỉ có tên sản phẩm (mục 4).
- Khi hỏi bằng ảnh, LLM không xem được ảnh; ảnh và chữ được gộp bằng RRF trọng số bằng nhau.
- Khối Recommendation trong kiến trúc tổng quát của môn học chưa được xây.

---

## 10. Đối chiếu yêu cầu

| Yêu cầu | Đáp ứng | Ở đâu |
|---|---|---|
| Baseline RAG | Có | Naive RAG, mục 3.2 |
| Hệ thống hoàn chỉnh (front/back/dữ liệu) | Có | `/chat` hỏi bằng chữ hoặc ảnh, `/api/v1/ask`, `/api/v1/ask_image`, dữ liệu và index của A1 |
| Nâng cấp | Có | Agentic RAG (mục 3.4); rerank MiniLM hoặc DeBERTa (mục 2); hỏi bằng ảnh (mục 3.3) |
| Chứng minh nâng cấp bằng ablation | Có | 250 câu × 6 cấu hình (mục 6.1, 6.2); 100 ảnh × 3 cấu hình (mục 6.3). Kết quả: rerank cải thiện Retrieval nhưng không đo được cải thiện ở Generation; Agentic **không** cải thiện; ảnh + chữ cho context tốt nhất |
| Dạng chatbot, nhiều lượt | Có | Memory theo session, mục 3.5–3.6 |
| Đủ metric RAG | Có | 8 metric, mục 5.2 |
| Ground truth ≥ 250 câu, dùng chung A1 | Có, nâng lên qrels 0–3 | `data/eval_set.jsonl` (mục 5.1) |
| Chẩn đoán Retrieval và Generation | Có, có ví dụ thật | Mục 6.2, 7 |
| G3.2 đạo đức AI | Có | Mục 8 |

---

## 11. Cách chạy

```bash
# Cài Ollama (https://ollama.com), rồi:
ollama pull llama3.1              # hoặc model khác, rồi đặt biến môi trường:
# PowerShell: $env:OLLAMA_MODEL = "qwen2.5:3b"

uvicorn app:app --port 8000
# Chatbot: http://localhost:8000/chat     Tìm kiếm A1: http://localhost:8000/search

# Ablation A3 (cần Ollama; DeBERTa cần models/deberta-10k-rank_net)
python step_ablation_rag.py --sample 3          # chạy thử
python step_ablation_rag.py                     # 250 câu x 6 cấu hình
python step17_eval_image_rag_retrieval.py       # hỏi bằng ảnh, chỉ Retrieval, không cần Ollama
python step18_ablation_image_rag.py             # hỏi bằng ảnh, cả hai tầng

# Chạy đủ trên Kaggle GPU (cách đã dùng cho mục 6.2, 6.3)
python kaggle/make_kaggle_rag_bundle.py         # tạo dist/kaggle_rag_bundle.zip, upload làm Kaggle Dataset
#    Notebook Kaggle (GPU T4 x2, Internet On):
#    !python kaggle_run_rag_ablation.py --resume --with-image
```
