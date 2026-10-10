# BÁO CÁO A2 — Đọc và Phân tích Paper
### Môn: Thiết kế Hệ thống Thông tin Thông minh dựa trên Truy hồi (IS6303)

**Paper:** Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents
**Tác giả:** Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, Zhaochun Ren — Shandong University, Baidu Inc., Leiden University
**Nguồn:** Proceedings of EMNLP 2023, trang 14918–14937. DOI: 10.18653/v1/2023.emnlp-main.923
**Code:** github.com/sunnweiwei/RankGPT (tên kỹ thuật: RankGPT)

> Quy ước: mọi con số trong mục 1–4 lấy từ paper, kèm số bảng/mục để đối chiếu. Thông tin lấy từ repo GitHub (không có trong paper) được ghi rõ là "theo repo". Nhận xét của nhóm được tách riêng ở mục 4.3.

---

## 1. Bài toán

Theo paper, các nghiên cứu trước chủ yếu dùng khả năng **sinh văn bản** của LLM cho IR (sinh query, sinh passage), chưa dùng LLM để **xếp hạng trực tiếp** một nhóm passage. Hai cách dùng LLM để rerank đã có trước đó:

- **Query generation** (Sachan et al., 2022 — UPR): điểm liên quan là log-probability LLM sinh lại query từ passage.
- **Relevance generation** (Liang et al., 2022 — HELM): điểm liên quan dựa trên xác suất LLM trả lời "Yes"/"No".

Theo paper, hai cách này có hiệu quả hạn chế và **cần log-probability** của output. ChatCompletion API của OpenAI lúc đó không cung cấp log-probability, nên không áp dụng được cho ChatGPT/GPT-4 (Appendix B.2). Paper cũng nêu khác biệt giữa mục tiêu pretrain của LLM và mục tiêu ranking.

Hai câu hỏi nghiên cứu (mục 1):
- **RQ1:** ChatGPT làm rerank passage tốt đến đâu?
- **RQ2:** Làm sao chuyển khả năng ranking của ChatGPT sang một model nhỏ, chuyên biệt?

---

## 2. Phương pháp

### 2.1 Instructional permutation generation (mục 3.1, Appendix A.5, B.3)

Mỗi passage được gắn một số thứ tự `[1]`, `[2]`, ... và đưa vào LLM **cùng một lúc**. LLM được yêu cầu trả về **hoán vị** của các số thứ tự theo độ liên quan giảm dần, dạng `[2] > [3] > [1] > ...`. Phương pháp này xếp hạng trực tiếp, **không sinh điểm liên quan trung gian**, nên không cần log-probability.

Với `gpt-3.5-turbo` và `gpt-4`, prompt dạng chat nhiều lượt (Appendix A.5): system message "You are RankGPT...", mỗi passage là một lượt user và assistant trả lời "Received passage [i]", lượt cuối nhắc lại query và yêu cầu chỉ trả về thứ tự.

### 2.2 Sliding window (mục 3.2, 6.1)

Do giới hạn token, LLM không xếp được nhiều passage trong một lần. Với M passage từ bước retrieval đầu tiên, paper xếp lại theo thứ tự **từ cuối lên đầu** với 2 siêu tham số: window size `w` và step `s`. Cửa sổ đầu tiên gồm passage thứ (M − w) đến M, sau đó trượt lên `s` vị trí và lặp lại cho đến hết danh sách. Nhờ đó, passage tốt ở cuối danh sách được đẩy dần lên đầu (Figure 3).

Thiết lập chính: rerank **top-100 BM25**, `w = 20`, `s = 10` (mục 6.1). Theo Appendix H, cách này cần 10 lượt gọi API mỗi query. QG và RG cần 100 lượt.

### 2.3 Permutation distillation (mục 4, Appendix D)

Mục tiêu là giảm chi phí và độ trễ khi triển khai:

- Lấy ngẫu nhiên **10.000 query** từ tập train MS MARCO. Mỗi query lấy **20 passage** bằng BM25, rồi cho `gpt-3.5-turbo` xếp hạng bằng permutation generation. Chi phí khoảng **$40** (Appendix D.1).
- Dùng thứ tự do ChatGPT sinh làm nhãn trực tiếp, huấn luyện model học sinh bằng **RankNet loss**:
  `L = Σᵢ Σⱼ 1[rᵢ < rⱼ] · log(1 + exp(sᵢ − sⱼ))`. Mỗi query cho M(M−1)/2 cặp.
- Hai kiến trúc học sinh (mục 4.3):
  - **Cross-encoder DeBERTa**: ghép query và passage bằng `[SEP]`, lấy biểu diễn `[CLS]`.
  - **LLaMA-7B** với instruction relevance generation zero-shot (Appendix A.3). Điểm là xác suất sinh token liên quan.
- Cấu hình huấn luyện (Appendix D.1): batch 32, AdamW, learning rate cố định 5×10⁻⁵, 2 epoch, 8 GPU A100-40G.

---

## 3. Thiết kế thực nghiệm

### 3.1 Dữ liệu (mục 5)

| Bộ dữ liệu | Mô tả |
|---|---|
| TREC-DL19 / DL20 | 43 / 54 query |
| BEIR (8 bộ) | Covid, NFCorpus, Touche, DBPedia, SciFact, Signal, News, Robust04 |
| Mr.TyDi | 10 ngôn ngữ ít tài nguyên (Arabic, Bengali, Finnish, Indonesian, Japanese, Korean, Russian, Swahili, Telugu, Thai); 100 mẫu test đầu tiên của mỗi ngôn ngữ |
| **NovelEval-2306** (mới) | 21 câu hỏi thuộc 4 lĩnh vực (Sport, Tech, Movie, Other), về sự kiện xảy ra sau khi GPT-4 ra mắt; mỗi câu 20 passage lấy từ Google Search, gán nhãn tay 0/1/2 (tổng 420 passage: 290 / 40 / 90). Các LLM được thử, gồm `gpt-4-0314` và `gpt-4-0613`, đều trả lời đúng 0% (Appendix C) |

NovelEval được tạo để giảm lo ngại **data contamination**: LLM có thể đã thấy dữ liệu của các benchmark cũ khi huấn luyện.

### 3.2 Thiết lập và metric (mục 6.1)

- Rerank top-100 passage do BM25 (pyserini) trả về. Với NovelEval, 20 passage được xáo trộn ngẫu nhiên trước khi rerank.
- Metric: **nDCG@1, nDCG@5, nDCG@10**.
- Trên BEIR và Mr.TyDi, `gpt-4` chỉ xếp lại **top-30** trong kết quả của `gpt-3.5-turbo` để giảm chi phí (chú thích Table 1, Appendix H).

### 3.3 Baseline (mục 6.2, Appendix E)

- **Có giám sát:** monoBERT (340M), monoT5 (220M và 3B), Cohere Rerank-v2, mmarcoCE (chỉ dùng cho Mr.TyDi).
- **Không giám sát:** UPR (FLAN-T5-XL), InPars (monoT5-3B), Promptagator++ (few-shot).
- Mọi phương pháp trừ InPars và Promptagator++ đều rerank cùng một top-100 BM25 (chú thích Table 1).

### 3.4 Các phân tích

| Phân tích | Nội dung | Vị trí |
|---|---|---|
| So sánh instruction và model | QG / RG / PG trên `curie-001` và `davinci-003`; `gpt-3.5` và `gpt-4` chỉ chạy PG; trên DL19 và DL20 | Table 4 |
| Ablation | Thứ tự passage đầu vào (BM25 / ngẫu nhiên / đảo ngược); số lần rerank (1–3); GPT-4 rerank top-30; `gpt-3.5-turbo` trên DL19 | Table 5 |
| LLM khác ngoài OpenAI | Rerank top-20 BM25 trên DL19 bằng model của Cohere, Anthropic, Google và 3 LLM mã nguồn mở | Table 6 |
| Model chuyên biệt | Nhãn MS MARCO so với nhãn ChatGPT; các loss khác nhau; kích thước model 70M–435M; lượng dữ liệu 500–10K query | Table 7, 13; Figure 4 |
| Độ ổn định | Số lần lặp ID (repetition), thiếu ID (missing), từ chối xếp hạng (rejection), và RBO giữa hai lần xếp cùng nhóm passage | Table 10 |
| Kích thước sliding window | w = 20/40/60/80, s = w/2, dùng `gpt-3.5-turbo-16k` | Table 12 |
| Chi phí | Token, số request, USD mỗi query | Table 11 |

---

## 4. Kết quả và hạn chế

### 4.1 Kết quả chính

**TREC và BEIR — nDCG@10 (Table 1)**

| Phương pháp | DL19 | DL20 | BEIR (TB 8 bộ) |
|---|---|---|---|
| BM25 | 50.58 | 47.96 | 43.42 |
| monoT5 (3B) | 71.83 | 68.89 | 51.36 |
| Cohere Rerank-v2 | 73.22 | 67.08 | 49.45 |
| gpt-3.5-turbo | 65.80 | 62.91 | 49.37 |
| gpt-4 | **75.59** | **70.56** | **53.68** |

So với monoT5 (3B), GPT-4 hơn trung bình **2.7** nDCG@10 trên TREC và **2.3** trên BEIR. Trên BEIR, ChatGPT vượt phần lớn baseline có giám sát (mục 6.2).

**Mr.TyDi — nDCG@10 (Table 2):** trung bình BM25 38.10, mmarcoCE 60.28, gpt-3.5 55.96, gpt-4 62.93. GPT-4 hơn mmarcoCE 2.65 điểm, nhưng **kém hơn** ở Bengali, Telugu và Thai.

**NovelEval (Table 3):** gpt-4 đạt 85.71 / 87.49 / 90.45 (nDCG@1/5/10), cao hơn monoT5 (3B) (83.33 / 78.38 / 84.62). gpt-3.5 (76.19 / 74.15 / 75.71) ngang monoBERT (78.57 / 70.65 / 77.27).

**So sánh instruction (Table 4):** PG tốt hơn QG và RG trên cùng model. Ví dụ `davinci-003` trên DL19, nDCG@10: RG 50.58, QG 45.37, PG 61.50.

**Ablation (Table 5, DL19, nDCG@10 gốc 65.80):**
- Thứ tự đầu vào ngẫu nhiên: 25.17. Thứ tự BM25 đảo ngược: 32.77. Kết quả **rất nhạy với thứ tự ban đầu**.
- Rerank 2 lần: 66.62. Rerank 3 lần: 66.97. nDCG@10 tăng nhẹ nhưng nDCG@1 giảm từ 82.17 xuống 78.29.
- GPT-4 rerank lại top-30: 73.64.

**LLM khác ngoài OpenAI (Table 6, top-20 BM25, DL19, nDCG@10):**

| Model | nDCG@10 |
|---|---|
| BM25 (thứ tự ban đầu; số liệu lấy từ Table 1) | 50.58 |
| gpt-4 | 65.68 |
| Cohere rerank-english-v2.0 (có giám sát) | 64.78 |
| claude-instant-1 / claude-2 | 62.23 / 55.91 |
| gpt-3.5-turbo | 60.89 |
| bard-2023.10.21 / text-bison-001 | 60.11 / 58.67 |
| flan-t5-xxl | 50.26 |
| ChatGLM-6B | 50.58 |
| Vicuna-13B | 49.08 |

Ba LLM mã nguồn mở gần như **không cải thiện so với BM25**. Paper cho rằng một lý do có thể là việc sinh hoán vị của 20 passage quá khó với các model này (mục 6.6).

**Model chuyên biệt (Table 7, nDCG@10):**

| Nhãn huấn luyện | Model | DL19 | DL20 | BEIR |
|---|---|---|---|---|
| MS MARCO | DeBERTa-Large | 68.89 | 61.38 | 42.64 |
| MS MARCO | LLaMA-7B | 69.24 | 58.97 | 47.71 |
| ChatGPT | DeBERTa-Large (435M) | 70.66 | **67.15** | **53.03** |
| ChatGPT | LLaMA-7B | **71.78** | 66.89 | 51.68 |

Theo mục 7.1:
- Học từ thứ tự của ChatGPT tốt hơn học từ nhãn MS MARCO.
- DeBERTa distill (435M) vượt monoT5 (3B) trên BEIR (53.03 so với 51.36).
- Model distill vượt cả ChatGPT, chính model thầy (49.37 trên BEIR).

Theo Figure 4, model distill tốt hơn model học nhãn MS MARCO ở mọi kích thước và lượng dữ liệu, và tăng kích thước model có lợi hơn tăng dữ liệu.

**Độ ổn định (Table 10, 97 query TREC):**

| Model | Repetition ↓ | Missing ↓ | Rejection | RBO ↑ |
|---|---|---|---|---|
| text-davinci-003 | 0 | 280 | 0 | 72.30 |
| gpt-3.5-turbo | 14 | 153 | 7 | 81.49 |
| gpt-4 | 0 | 1 | 11 | 82.08 |

Theo footnote 9, passage bị thiếu được nối vào cuối theo thứ tự ban đầu.

**Kích thước window (Table 12):** w = 20 cho nDCG@10 cao nhất (67.05). w = 40 cho nDCG@1 và nDCG@5 cao nhất (78.30 / 71.32).

**Chi phí (Table 11, Appendix H):**
- Chi phí mỗi query trên TREC: `gpt-3.5-turbo` $0.040, `gpt-4` $0.596, GPT-4 chỉ rerank top-30 $0.098.
- Độ trễ trung bình mỗi lượt gọi: 1.1 s (gpt-3.5) và 3.2 s (gpt-4). Tính ra khoảng 11 s và 32 s mỗi query với 10 lượt gọi.
- Tổng chi phí GPT-4 cho toàn bộ thực nghiệm: $556.

### 4.2 Hạn chế do tác giả nêu (mục Limitations, Ethics Statement)

- Phân tích chính dựa trên model độc quyền (ChatGPT, GPT-4). Các model mã nguồn mở đã thử (FLAN-T5, ChatGLM-6B, Vicuna-13B) cho kết quả kém xa ChatGPT. Cách khai thác tốt model mã nguồn mở là câu hỏi còn mở.
- Paper chỉ xét bước rerank, nên giới hạn trên của kết quả phụ thuộc recall của bước retrieval đầu tiên.
- Kết quả rất nhạy với thứ tự passage ban đầu, vốn do retrieval đầu tiên (như BM25) quyết định.
- Về đạo đức: LLM có thể thiên lệch và hallucinate. Tác giả **không khuyến nghị** dùng LLM cho các tác vụ xếp hạng có tác động xã hội, *"such as ranking job candidates or ranking products"*.
- Model distill chịu điều khoản của OpenAI vì dùng output của ChatGPT. Model LLaMA distill chịu thêm giấy phép phi thương mại của LLaMA.

### 4.3 Nhận xét của nhóm

- **Một số chi tiết trong paper không thống nhất**, cần lưu ý khi trích dẫn:
  - Mục 4.3.1 nói model học sinh là DeBERTa-large, còn Appendix D.1 ghi DeBERTa-V3-base.
  - Abstract ghi "440M", thân bài ghi "435M".
  - Mục 7.2 nói model distill chỉ với 1K query đã vượt monoT5 (3B) trên BEIR, nhưng Table 13 ghi 51.26 so với 51.36.
  - Table 12 dùng `gpt-3.5-turbo-16k` nên số liệu khác Table 1.
- **NovelEval nhỏ** (21 câu hỏi, 420 passage), nên kết quả trên bộ này có độ bất định cao hơn các benchmark lớn.
- **Chi phí và độ trễ:** theo Table 11 và Appendix H, RankGPT với GPT-4 mất khoảng 32 s mỗi query. Con số này khó chấp nhận cho ứng dụng cần phản hồi ngay. Hướng distillation ở mục 4 của paper chính là để giải quyết vấn đề này.
- **Liên quan domain của đồ án:** phần đạo đức của paper nhắc trực tiếp việc xếp hạng sản phẩm là tác vụ không khuyến nghị dùng LLM. Catalog của A1 là sản phẩm.
  - Đồ án không dùng LLM để xếp hạng.
  - DeBERTa distill (học từ output ChatGPT) chỉ là một lựa chọn reranker **tuỳ chọn, mặc định tắt** trong chatbot A3.
---

## 5. Liên hệ với đồ án

### 5.1 Áp dụng phần nào của paper, và vì sao

Paper có hai đóng góp chính (mục 1): **RQ1** (instructional permutation generation, dùng LLM trực tiếp để xếp hạng) và **RQ2** (permutation distillation, chuyển khả năng xếp hạng của ChatGPT sang một model nhỏ). **Đồ án chỉ áp dụng RQ2**, bằng checkpoint model distill mà tác giả công bố.

**Không cài RankGPT dùng LLM (RQ1):**
- Kết quả tốt của RQ1 trong paper đến từ model độc quyền qua OpenAI API (`gpt-3.5-turbo`, `gpt-4`).
- Với LLM mã nguồn mở, chính paper cho kết quả kém. Table 6 cho nDCG@10 trên DL19 lần lượt là flan-t5-xxl 50.26, ChatGLM-6B 50.58, Vicuna-13B 49.08, so với thứ tự BM25 ban đầu 50.58. Mục 6.6 viết: *"As for the open-source LLMs, we observed a significant performance gap compared to ChatGPT."*
- Paper **không mô tả cách chạy** RankGPT với model mã nguồn mở, và **không khuyến nghị** cách dùng đó. Mục Limitations chỉ ghi: *"How to further exploit the open-source models is a question worth exploring."*
- Hướng paper đề xuất để giảm chi phí và độ trễ khi triển khai là distillation (mục 4): *"we can distill the re-ranking capability of large language models into a small model by specialization."*

**Thành phần trong code:**

| Thành phần | Vị trí | Dùng ở đâu |
|---|---|---|
| `distilled_rerank()`, nạp checkpoint `deberta-10k-rank_net` | [search_core.py](../search_core.py) | Thực nghiệm [step_ablation_rerank.py](../step_ablation_rerank.py); chatbot A3 khi bật rerank và chọn "DeBERTa" (mặc định tắt); cấu hình "Naive + DeBERTa" của `step_ablation_rag.py` |

- Tìm kiếm A1 (`/search`) không dùng thành phần nào của paper này. Rerank của A1 là cross-encoder MiniLM (Nogueira & Cho, 2019).
- Checkpoint `deberta-10k-rank_net` lấy từ mục "Download data and model" của repo (*"Specialized Deberta model trained with RankNet loss"*).
  - Theo `config.json`: **DeBERTa-v3-base** (`hidden_size` 768, 12 layer, 1 đầu ra; ~184M tham số khi nạp).
  - **Tên checkpoint không xuất hiện trong paper.** Paper chỉ mô tả cách huấn luyện (mục 4, Appendix D.1: DeBERTa-V3-base, 10K query, 20 passage BM25 mỗi query, thứ tự do `gpt-3.5-turbo` xếp, RankNet, batch 32, lr 5×10⁻⁵, 2 epoch). Lệnh huấn luyện mẫu trong README của repo dùng `--model microsoft/deberta-v3-base --loss rank_net ... --save_path out/deberta-10k-rank_net`, khớp với mô tả đó.
  - Table 13 có **hai dòng** cùng cấu hình này, với kết quả khác nhau:
    - "ChatGPT RankNet": DL19 65.75, DL20 59.34, BEIR 50.53.
    - "deberta-v3-base (184M)": DL19 70.28, DL20 62.52, BEIR 50.57.
  - Paper và repo **không nói** checkpoint công bố ứng với dòng nào. Chắc chắn duy nhất là nó **không phải** bản DeBERTa-large 435M, bản cho kết quả tốt nhất ở Table 7 (BEIR 53.03).

### 5.2 Cài đặt so với paper

| Thành phần | Paper / repo | Đồ án |
|---|---|---|
| Model | DeBERTa cross-encoder học bằng RankNet từ thứ tự của ChatGPT trên 10K query MS MARCO (mục 4, Appendix D.1) | Dùng nguyên checkpoint `deberta-10k-rank_net` của repo, **không huấn luyện lại** |
| Cách chấm | Ghép (query, passage), điểm là logit, `max_length=500` (`specialization.py`) | Như repo; passage là **caption sản phẩm** |
| Ứng viên đầu vào | BM25 top-100 | **Khác:** 30 ứng viên hybrid BM25 + Dense (RRF) của A1 |
| Dữ liệu đánh giá | TREC / BEIR, nhãn nhiều mức | **Khác:** 250 câu của `data/eval_set.jsonl` (dùng chung A1/A3), mỗi câu một sản phẩm đúng (nhãn nhị phân) |
| Metric | nDCG@1/5/10 | nDCG@1/5/10, thêm Recall@10 và thời gian rerank |
| So sánh với | Model có giám sát (monoBERT, monoT5...) và ChatGPT | Không rerank, và cross-encoder MiniLM của A1 |

Do dữ liệu, domain và model khác, **số liệu của đồ án không so sánh trực tiếp được với số liệu của paper**. Chỉ nên so **xu hướng** giữa các cấu hình trong cùng một thực nghiệm.

### 5.3 Kết quả

Chạy ngày 24–25/09/2026 bằng `step_ablation_rerank.py`, kết quả 250 câu lưu trong `data/rerank_ablation_results.json`.
- Mọi cấu hình dùng cùng pool 30 ứng viên hybrid; thời gian rerank đo trên CPU.
- Dòng "50 viết tay" đo riêng trên 50 dòng đầu của eval set, với cùng pool và cùng hàm rerank, vì 200 câu còn lại sinh từ mẫu và dễ hơn.

| Cấu hình | Tập câu | nDCG@1 | nDCG@5 | nDCG@10 | Recall@10 | Thời gian rerank / câu |
|---|---|---|---|---|---|---|
| Không rerank | 250 | 0.772 | 0.879 | 0.888 | 0.992 | — |
| Cross-encoder MiniLM (A1) | 250 | **0.980** | **0.982** | **0.982** | 0.984 | 0.23 s |
| DeBERTa distill (paper A2) | 250 | 0.972 | 0.981 | 0.981 | 0.988 | 2.20 s |
| Không rerank | 50 viết tay | 0.580 | 0.748 | 0.774 | 0.960 | — |
| Cross-encoder MiniLM (A1) | 50 viết tay | **0.900** | **0.909** | **0.909** | 0.920 | — |
| DeBERTa distill (paper A2) | 50 viết tay | 0.860 | 0.906 | 0.906 | 0.940 | — |

**Nhận xét:**
- DeBERTa distill cải thiện mạnh thứ hạng so với không rerank. Trên 50 câu viết tay, tỷ lệ câu có sản phẩm đúng ở hạng 1 tăng từ 58% lên 86%.
- **DeBERTa distill không hơn cross-encoder MiniLM** trên dữ liệu này (nDCG@1 thấp hơn 0.04 trên 50 câu viết tay, gần bằng ở nDCG@5/10), và chậm hơn khoảng 9 lần trên CPU (~184M tham số so với ~22M).
  - Điều này không mâu thuẫn với paper. Paper so model distill với model có giám sát và với ChatGPT trên TREC/BEIR, không so với MiniLM trên caption sản phẩm.
  - Cả hai model đều học trên văn bản web MS MARCO, khác domain sản phẩm của đồ án.
- Recall@10 giảm nhẹ khi rerank (0.992 xuống 0.984–0.988): một vài sản phẩm đúng bị đẩy ra khỏi top-10 (xem nhóm lỗi 7 trong `BAO_CAO_TONG_HOP.md`).

Chạy lại: `python step_ablation_rerank.py` (cần `models/deberta-10k-rank_net`; nếu thiếu, script bỏ qua dòng DeBERTa).

### 5.4 Thông tin từ repo GitHub (không có trong paper)

- Repo cho biết paper nhận giải Outstanding Paper Award của EMNLP 2023.
- Từ 08/2023, repo hỗ trợ gọi Azure, Claude, Cohere và Llama2 qua LiteLLM.
- Từ 11/2023, repo giới thiệu hướng Instruction Distillation, là công trình tiếp theo dành cho LLM mã nguồn mở.
- Mục "Download data and model" cung cấp 10K và 100K query MS MARCO, thứ tự do ChatGPT sinh, và checkpoint `deberta-10k-rank_net`.
