# BÁO CÁO A3 — Chatbot RAG mở rộng từ hệ tìm kiếm A1
### Môn: Thiết kế Hệ thống Thông tin Thông minh dựa trên Truy hồi (IS6303)

> **Trạng thái số liệu:**
> - Tầng **Retrieval** (mục 6.1) đã đo ngày 25/09/2026, không cần LLM.
> - Tầng **Generation** (Faithfulness, Answer Relevancy, citation, latency...) **chưa đo**: `step_ablation_rag.py` cần Ollama, mà máy thử không có. Báo cáo không đưa ra con số nào cho tầng này.
>
> **Căn cứ yêu cầu:** rubric và lưu ý của giảng viên chỉ có bản ghi lại trong `docs/KE_HOACH_TONG_HOP.md` và bản báo cáo A3 trước; repo không có bản gốc.

Mục lục
1. Yêu cầu
2. Quan hệ với A1 và A2
3. Kiến trúc và luồng xử lý từng bước
4. Grounding: giảm bịa thông tin
5. Phương pháp đánh giá
6. Kết quả
7. Khung chẩn đoán lỗi: Retrieval hay Generation
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

- **A1:** A3 **không viết lại** phần tìm kiếm. `rag_core.py` gọi `search_core.search()` của A1 (BM25 + Dense CLIP + RRF). Dữ liệu (catalog, index) là dữ liệu A1 đã chuẩn bị.
- **Rerank (tuỳ chọn, mặc định tắt):** trước khi đưa sản phẩm cho LLM, có thể xếp lại bằng một trong hai reranker:
  - **Cross-encoder MiniLM:** reranker của A1.
  - **DeBERTa distill** (`deberta-10k-rank_net`): model từ paper A2 (Sun et al., EMNLP 2023, mục 4 và 7). Nó được huấn luyện để bắt chước thứ tự do ChatGPT xếp.
- Không có bước nào dùng LLM để xếp hạng; LLM chỉ dùng để sinh câu trả lời và (ở chế độ Agentic) tự đánh giá, viết lại query.

---

## 3. Kiến trúc và luồng xử lý từng bước

### 3.1 Thành phần

| Thành phần | File | Vai trò |
|---|---|---|
| Giao diện chatbot | `frontend/chat.html` (GET `/chat`) | Khung chat, công tắc rerank và Agentic, danh sách nguồn |
| API | `app.py`: `GET /api/v1/ask`, `GET /api/v1/rerankers`, `POST /api/v1/reset_session` | Nhận câu hỏi, gọi `rag_core` |
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

### 3.3 Luồng Agentic RAG (nâng cấp): `answer_agentic(max_iters=2)`

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

### 3.4 Memory

- Lưu trong RAM theo `session_id`, giữ 3 lượt gần nhất. Mỗi câu trả lời cũ được cắt 150 ký tự khi chèn vào prompt.
- Giao diện sinh `session_id` ngẫu nhiên khi mở trang. Nút "Mới" gọi `/api/v1/reset_session` rồi đổi sang `session_id` mới.
- Giới hạn: mất khi khởi động lại server; không dùng chung được giữa nhiều server.

### 3.5 Giao diện `/chat`

- **Công tắc "Rerank trước khi đưa vào context":** chỉ khi bật mới chọn được "Cross-encoder MiniLM (A1)" hoặc "DeBERTa distill từ ChatGPT (A2)". Danh sách lấy từ `/api/v1/rerankers`; DeBERTa chỉ hiện khi server có checkpoint.
- **Công tắc "Agentic":** mặc định tắt.
- **Khung nguồn** bên phải: 5 sản phẩm đưa vào context, kèm số `[n]`, ảnh, điểm RRF và điểm rerank (nếu có), để người dùng đối chiếu với trích dẫn trong câu trả lời.
- **Sản phẩm trong câu trả lời:** dưới mỗi câu trả lời hiện thẻ nhỏ cho những sản phẩm được LLM trích dẫn `[n]`. Chúng được tách từ câu trả lời và ánh xạ sang đúng `sources` của lượt đó.
- **Chi tiết sản phẩm:** bấm vào thẻ trong câu trả lời hoặc một mục trong danh sách nguồn thì cột phải hiện ảnh lớn, tên, brand, category, mô tả. Dữ liệu lấy từ catalog A1 qua `GET /api/v1/products/{id}`.
- Khi không gọi được Ollama, API trả lỗi 503 và khung chat hiện thông báo lỗi.

---

## 4. Grounding: giảm bịa thông tin

Ba biện pháp có trong code:
1. **Context đánh số `[1]..[5]`** để LLM trích dẫn đúng nguồn.
2. **Luật trong prompt** (`build_prompt`), nguyên văn: *"CHỈ được dùng thông tin trong danh sách sản phẩm dưới đây để trả lời. KHÔNG được bịa thêm chi tiết (giá, tính năng...) không có trong danh sách. Nếu danh sách không có sản phẩm phù hợp, hãy nói rõ là không tìm thấy thay vì đoán."*; và yêu cầu trả lời tiếng Việt, trích dẫn `[n]` khi nhắc tới sản phẩm.
3. **Trả kèm `sources`** để người dùng tự kiểm chứng.

**Giới hạn thông tin trong context:** mỗi sản phẩm chỉ có **caption** (tên sản phẩm) và điểm RRF. Context không có mô tả, brand riêng, category hay giá, và **catalog không có trường giá**. Vì vậy:
- Câu hỏi về giá hoặc thuộc tính không nằm trong tên sản phẩm không thể trả lời đúng. Theo prompt, LLM phải trả lời là không tìm thấy thông tin đó.
- Eval set hiện tại không có câu hỏi về giá. Các câu đều là mô tả sản phẩm cần tìm, nên chưa đo được hệ thống xử lý loại câu này ra sao.

---

## 5. Phương pháp đánh giá

### 5.1 Dữ liệu

`data/eval_set.jsonl`, 250 câu, dùng chung với A1:
- 30 câu A1 viết tay, 20 câu A3 viết tay, 200 câu sinh từ mẫu (chi tiết ở `BAO_CAO_TONG_HOP.md` mục 2.2).
- Mỗi câu có `relevant_ids` (1 sản phẩm đúng) và `answer_should_mention` (brand và vài từ đầu của tên sản phẩm).
- Toàn bộ câu hỏi bằng tiếng Anh; câu trả lời của hệ thống bằng tiếng Việt.

### 5.2 Metric (`rag_metrics.py`, `step_ablation_rag.py`)

| Tầng | Metric | Cách tính | Cần |
|---|---|---|---|
| Retrieval | Context Recall@5 | Sản phẩm đúng có trong 5 nguồn không (`recall_at_k` của A1) | — |
| Retrieval | Context Precision@5 | Σ (hits/i tại mỗi hạng i đúng) / min(\|đúng\|, 5). Với 1 sản phẩm đúng thì bằng 1/hạng nếu nằm trong top-5 | — |
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

Mỗi cấu hình khác baseline đúng một thành phần. Tổng số lượt gọi LLM cho 250 câu (gồm chấm Faithfulness) khoảng 2.250–2.750.

---

## 6. Kết quả

### 6.1 Tầng Retrieval: 5 sản phẩm đưa cho LLM (đã đo, không cần LLM)

Đo bằng chính hàm `rag_core.retrieve(q, k=5, reranker)` mà chatbot dùng, trên 250 câu.

- "Không rerank" gọi `search(k=5)` mặc định: mỗi nhánh top-50, RRF, lấy 5.
- "MiniLM" và "DeBERTa" lấy 30 ứng viên sau RRF (mỗi nhánh top-30), xếp lại, giữ 5.
- Thời gian là thời gian của cả bước `retrieve()` trên CPU, chưa gồm thời gian gọi LLM.

| Reranker | Tập câu | Context Recall@5 | Context Precision@5 | Sản phẩm đúng ở [1] | Thời gian retrieve / câu |
|---|---|---|---|---|---|
| Không rerank | 250 | 0.940 | 0.806 | 182/250 (72.8%) | 0.03 s |
| Cross-encoder MiniLM (A1) | 250 | 0.984 | **0.981** | **245/250 (98.0%)** | 0.33 s |
| DeBERTa distill (A2) | 250 | **0.988** | 0.979 | 243/250 (97.2%) | 2.53 s |
| Không rerank | 50 viết tay | 0.840 | 0.680 | 29/50 (58%) | 0.02 s |
| Cross-encoder MiniLM (A1) | 50 viết tay | 0.920 | **0.905** | **45/50 (90%)** | 0.29 s |
| DeBERTa distill (A2) | 50 viết tay | **0.940** | 0.894 | 43/50 (86%) | 2.45 s |
| Không rerank | 200 sinh từ mẫu | 0.965 | 0.838 | 153/200 (76.5%) | 0.03 s |
| Cross-encoder MiniLM (A1) | 200 sinh từ mẫu | 1.000 | 1.000 | 200/200 (100%) | 0.35 s |
| DeBERTa distill (A2) | 200 sinh từ mẫu | 1.000 | 1.000 | 200/200 (100%) | 2.73 s |

**Nhận xét (chỉ về tầng Retrieval):**
- **Không rerank:** trên 50 câu viết tay, **16% câu (8/50) không có sản phẩm đúng trong 5 nguồn đưa cho LLM**, và chỉ 58% câu có sản phẩm đúng ở vị trí `[1]`. Với các câu đó, LLM không thể nêu đúng sản phẩm, dù prompt tốt đến đâu.
- **Có rerank:** cả hai reranker đều tăng mạnh chất lượng context.
  - Context Recall@5 trên 50 câu viết tay tăng từ 0.84 lên 0.92–0.94.
  - Sản phẩm đúng ở `[1]` tăng từ 58% lên 86–90%.
  - Trên 200 câu sinh từ mẫu, cả hai đạt tuyệt đối. Đây là dấu hiệu nhóm câu này quá dễ, không nên dùng riêng để so sánh.
- **MiniLM và DeBERTa gần tương đương:**
  - DeBERTa đưa được nhiều sản phẩm đúng vào top-5 hơn một chút (0.940 so với 0.920 trên 50 câu viết tay, tức hơn 1 câu).
  - MiniLM xếp sản phẩm đúng lên `[1]` nhiều hơn (45 so với 43 câu) và nhanh hơn khoảng 8 lần.
  - Chênh lệch chỉ vài câu trên 50, chưa đủ để kết luận model nào tốt hơn.
- **Cái giá:** thời gian bước tìm kiếm tăng từ ~0.03 s lên ~0.3 s (MiniLM) hoặc ~2.5 s (DeBERTa) mỗi câu. Thời gian này cộng thêm vào thời gian LLM sinh câu trả lời.
- Tác dụng của các thay đổi này lên **câu trả lời** (Faithfulness, Mention hit...) chưa đo (mục 6.2).

### 6.2 Tầng Generation: chưa đo

`step_ablation_rag.py` đã có đủ 4 cấu hình và 7 metric nhưng **chưa chạy**, vì máy thử không có Ollama. Báo cáo không đưa ra số liệu hay nhận định nào về Faithfulness, Answer Relevancy, citation rate, mention hit rate hay latency của các cấu hình RAG.

Cách chạy khi có Ollama: đặt `SAMPLE_SIZE = 20` trong `step_ablation_rag.py` để chạy thử, sau đó đặt `None` để chạy đủ 250 câu, rồi điền bảng:

| Cấu hình | Ctx Recall@5 | Ctx Precision@5 | Faithfulness | Answer Relevancy | Mention hit | Citation | Latency (s) |
|---|---|---|---|---|---|---|---|
| Naive RAG | | | | | | | |
| Naive + MiniLM | | | | | | | |
| Naive + DeBERTa | | | | | | | |
| Agentic RAG | | | | | | | |

---

## 7. Khung chẩn đoán lỗi: Retrieval hay Generation

A3 gồm hai tầng. Khi câu trả lời sai, phải xác định tầng gây lỗi trước khi sửa:

| Dấu hiệu | Tầng lỗi | Cách xác nhận |
|---|---|---|
| Sản phẩm đúng **không có** trong `sources` | Retrieval (A1 + reranker) | Context Recall@5 = 0 cho câu đó |
| Sản phẩm đúng có trong `sources` nhưng câu trả lời không nhắc tới, nhắc sai, hoặc trích dẫn sai `[n]` | Generation | Mention hit thấp; đối chiếu `[n]` với danh sách nguồn |
| Câu trả lời có thông tin không nằm trong context | Generation (bịa) | Faithfulness thấp |
| Câu trả lời không có `[n]` | Generation (không tuân thủ grounding) | Citation rate |

**Ví dụ tầng Retrieval (có số đo thật):** câu *"device that helps someone relearn to walk after an injury"*, sản phẩm đúng "Drive Medical tk 1000 Trekker Gait Trainer".
- Ở A1, sản phẩm này không có trong top-10 ở cả 4 cấu hình (`BAO_CAO_TONG_HOP.md` mục 7–8): BM25 không tìm thấy, Dense xếp hạng 31, sau RRF là hạng 62.
- Vì vậy sản phẩm không có trong 5 nguồn đưa cho LLM. Dù prompt tốt đến đâu, LLM không thể nêu đúng sản phẩm này.
- Đây là lỗi tầng Retrieval: sửa ở search (từ điển đồng nghĩa, embedding tốt hơn), không phải ở prompt.

**Ví dụ tầng Generation:** chưa có, vì tầng Generation chưa được chạy (mục 6.2). Sau khi chạy ablation, lấy các câu có Context Recall@5 = 1 nhưng Mention hit = 0 hoặc Faithfulness thấp để phân tích.

---

## 8. Đạo đức AI

| Rủi ro | Ví dụ | Biện pháp trong hệ thống |
|---|---|---|
| Bịa thông tin | LLM tự thêm giá hoặc tính năng không có trong danh sách | Luật trong prompt (mục 4); context không chứa giá nên mọi con số giá đều là bịa và kiểm được |
| Trích dẫn sai nguồn | Nói "[2]" nhưng mô tả sản phẩm [1] | Hiển thị `sources` kèm số `[n]` để người dùng đối chiếu |
| Trả lời khi không có dữ liệu | LLM cố trả lời dù search rỗng | Naive không gọi LLM khi search rỗng (Agentic chưa có kiểm tra này, xem mục 9) |
| Vòng lặp Agentic không dừng | LLM liên tục đánh giá "chưa đủ" | `max_iters=2` |
| Thiên lệch khi xếp hạng sản phẩm | Paper A2 khuyến cáo không dùng LLM để xếp hạng sản phẩm | Reranker DeBERTa (học từ ChatGPT) là tuỳ chọn, mặc định tắt; xếp hạng mặc định là BM25 + Dense + RRF |
| Lộ dữ liệu hội thoại | Lịch sử chat lưu trên server | Chỉ trong RAM, giữ 3 lượt, xoá được bằng nút "Mới"; LLM chạy local, không gửi ra dịch vụ ngoài |

---

## 9. Hạn chế đã biết

**Đánh giá:**
- Tầng Generation chưa được đo (mục 6.2).
- Faithfulness do **chính model sinh câu trả lời** (llama3.1) chấm, nên có thể thiên vị. Dòng output ghi `NOT SUPPORTED` (có dấu cách) không được nhận.
- **Lệch ngôn ngữ:** câu hỏi và catalog tiếng Anh, câu trả lời tiếng Việt.
  - `all-MiniLM-L6-v2` là model tiếng Anh, nên Answer Relevancy kém tin cậy.
  - Mention hit dựa trên so khớp chuỗi, nên tên sản phẩm bị dịch sẽ không khớp.
- 200/250 câu sinh từ mẫu, dễ hơn câu thực tế (xem mục 6.1).

**Code** (chi tiết trong `docs/CAU_TRUC_DU_AN.md` mục 7):
- `judge_sufficiency` kiểm tra câu trả lời bắt đầu bằng "CÓ". Nếu LLM trả `"CÓ"` có ngoặc kép, `**CÓ**` hoặc "Yes", kết quả bị coi là chưa đủ và gây thêm một lượt search.
- Với `max_iters=2`, kết quả đánh giá ở vòng cuối luôn bị bỏ qua (tốn 1 lượt LLM).
- `rewrite_query` nhận query hiện tại chứ không phải câu hỏi gốc. Prompt tiếng Việt có thể sinh query tiếng Việt, bất lợi khi tìm trên catalog tiếng Anh.
- `answer_agentic` vẫn gọi LLM khi search rỗng.
- Memory chỉ dùng cho câu trả lời cuối; không dùng khi search hay khi tự đánh giá.

**Phạm vi:**
- Context chỉ có tên sản phẩm (mục 4).
- Chatbot chỉ nhận câu hỏi bằng text.
- Khối Recommendation trong kiến trúc tổng quát của môn học chưa được xây.

---

## 10. Đối chiếu yêu cầu

| Yêu cầu | Đáp ứng | Ở đâu |
|---|---|---|
| Baseline RAG | Có | Naive RAG, mục 3.2 |
| Hệ thống hoàn chỉnh (front/back/dữ liệu) | Có | `/chat`, `/api/v1/ask`, dữ liệu và index của A1 |
| Nâng cấp | Có | Agentic RAG (mục 3.3); rerank MiniLM hoặc DeBERTa (mục 2, 3.2) |
| Chứng minh nâng cấp bằng ablation | **Một phần** | Tầng Retrieval đã đo cho cả 3 lựa chọn reranker (mục 6.1); tầng Generation chưa chạy (mục 6.2) |
| Dạng chatbot, nhiều lượt | Có | Memory theo session, mục 3.4–3.5 |
| Đủ metric RAG | Có trong code | 7 metric, mục 5.2; chưa có số cho tầng Generation |
| Ground truth ≥ 250 câu, dùng chung A1 | Có | `data/eval_set.jsonl`; hạn chế ở mục 9 |
| Chẩn đoán Retrieval và Generation | Có khung, có ví dụ Retrieval | Mục 7 |
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
python step_ablation_rag.py
```
