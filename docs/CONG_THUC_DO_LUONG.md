# Phụ lục — Công thức tính các con số trong báo cáo

Phụ lục này liệt kê công thức đằng sau mọi con số trong các bảng của báo cáo A1, A2, A3 và BPM, kèm ý nghĩa của từng con số. Mỗi công thức đều khớp với code của project (tên file ghi trong ngoặc).

**Ký hiệu chung** cho một câu hỏi:
- **G**: tập id sản phẩm đúng. Trong mọi bộ đánh giá của đồ án, |G| = 1.
- **R = (r₁, r₂, …)**: danh sách id hệ thống trả về, theo thứ hạng.
- **rel(i)** = 1 nếu rᵢ ∈ G, ngược lại 0.

---

## 0. Quy tắc lấy trung bình

- Mọi metric được **tính cho từng câu hỏi**, rồi **lấy trung bình cộng** trên các câu.
- Câu có giá trị `None` (không tính được, ví dụ không có ground truth, hoặc giám khảo LLM không trả claim nào) bị **bỏ qua** khi lấy trung bình.
- Riêng ablation RAG (A3): câu bị lỗi khi gọi LLM, sau 3 lần thử lại, không được tính vào trung bình mà được đếm riêng ở cột "Lỗi".

---

## 1. Điểm dùng để xếp hạng

Đây là các cột "score", "rrf", "rerank" trên giao diện và trong context đưa cho LLM.

| Điểm | Công thức | Ý nghĩa |
|---|---|---|
| **BM25** (`rank_bm25.BM25Okapi`, mặc định k₁ = 1.5, b = 0.75) | score(q, d) = Σ_{từ t ∈ q} IDF(t) · f(t,d)·(k₁+1) / (f(t,d) + k₁·(1 − b + b·\|d\|/avgdl)), với IDF(t) = ln((N − n(t) + 0.5) / (n(t) + 0.5)) | Khớp từ khoá: từ hiếm và xuất hiện nhiều trong tài liệu thì điểm cao. f(t,d) là số lần từ t xuất hiện trong d; \|d\| là độ dài d; avgdl là độ dài trung bình; N = 2.000 sản phẩm; n(t) là số tài liệu chứa t |
| **Dense** (SigLIP + FAISS `IndexFlatIP`) | score(q, d) = v_q · v_d, với các vector đã chuẩn hoá L2, nên bằng cos(v_q, v_d) ∈ [−1, 1] | Độ gần về nghĩa giữa câu hỏi và ảnh sản phẩm |
| **RRF** (Cormack et al., 2009) | score(d) = Σ_{danh sách L chứa d} 1 / (k + hạng_L(d)), với k = 60, hạng tính từ 1 | Gộp BM25 và Dense theo thứ hạng, vì điểm của hai nhánh khác thang đo. Tối đa 2/61 ≈ 0,0328 (đứng hạng 1 ở cả hai danh sách) |
| **Rerank** (cross-encoder MiniLM hoặc DeBERTa distill) | Logit của cross-encoder trên cặp (câu hỏi, tên sản phẩm), không chuẩn hoá | Càng lớn càng liên quan. Chỉ dùng để **so sánh trong cùng một câu hỏi**; không so điểm giữa hai model |

---

## 2. Metric tầng tìm kiếm (`metrics.py`)

Dùng ở: báo cáo A1 mục 6.1–6.6 và 7; báo cáo A2 mục 5.3; bảng so sánh embedding model.

| Metric | Công thức | Ý nghĩa | Khi \|G\| = 1 |
|---|---|---|---|
| **Recall@k** | \|{r₁, …, r_k} ∩ G\| / \|G\| | Sản phẩm đúng có nằm trong top-k không | Bằng 1 hoặc 0; trung bình là **tỷ lệ câu tìm thấy** |
| **nDCG@k** (nhị phân) | DCG@k / IDCG@k, với DCG@k = Σ_{i=1..k} rel(i) / log₂(i+1); IDCG@k = Σ_{i=1..min(\|G\|,k)} 1 / log₂(i+1) | Tìm thấy **và** xếp càng cao càng tốt | Bằng 1/log₂(hạng+1): hạng 1 → 1; hạng 2 → 0,631; hạng 3 → 0,5; hạng 10 → 0,289; không có → 0 |
| **nDCG@1** | như trên với k = 1 | | Bằng 1 nếu đúng ở hạng 1; trung bình là **tỷ lệ câu đúng ngay hạng 1** |
| **nDCG đa mức@k** (A1 mục 6.1) | rel(i) = 2 nếu rᵢ ∈ G; 1 nếu rᵢ cùng `category` với sản phẩm đúng; 0 nếu khác. DCG tính như trên. IDCG = Σ_{i=1..m} 2/log₂(i+1) + Σ_{i=m+1..k} 1/log₂(i+1), với m = min(\|G\|, k) | Thưởng một phần cho sản phẩm cùng loại | IDCG giả định top-k được lấp đầy bằng sản phẩm cùng loại, nên giá trị luôn thấp hơn bản nhị phân. Chỉ dùng để so giữa các cấu hình |
| **MRR** | 1 / hạng của sản phẩm đúng đầu tiên trong danh sách trả về; bằng 0 nếu không có | Người dùng phải kéo xuống bao xa mới thấy kết quả đúng | Danh sách chỉ dài k nên thực chất là MRR@k. Khi \|G\| = 1: MRR = 1/hạng |
| **Latency trung bình** (A1 mục 6.1) | (1/n) · Σ thời gian mỗi lần gọi `search()`, tính bằng ms, đo bằng `time.perf_counter` | Tốc độ phản hồi của hệ tìm kiếm | Gồm cả rerank nếu bật; đo trên CPU |
| **Latency p95** | Sắp các thời gian tăng dần, lấy phần tử thứ ⌊0,95·n⌋ (đánh số từ 0) | 95% truy vấn nhanh hơn mức này | Với n = 250 là phần tử thứ 237 |

**Bảng so sánh embedding model** (A1 mục 6.5, `kaggle/kaggle_ablation_embedding_models.py`):
- Recall@10, nDCG@10, MRR tính như trên, nhưng chỉ dùng nhánh Dense.
- Latency = thời gian encode câu hỏi + tìm trong FAISS (ms), đo trên GPU Kaggle, nên không so được với latency CPU ở mục 6.1.

**Bảng so sánh reranker** (A2 mục 5.3, A1 mục 6.4, `step_ablation_rerank.py`):
- nDCG@1/5/10 và Recall@10 tính trên 30 ứng viên hybrid sau khi rerank.
- **Thời gian rerank** = chỉ thời gian gọi hàm rerank trên 30 ứng viên (giây/câu), **không** gồm thời gian tìm kiếm.

---

## 3. Bảng đếm

| Con số | Cách tính | Ý nghĩa | Dùng ở |
|---|---|---|---|
| **Hạng ở từng tầng** (`step12_case_study_trace.py`) | Vị trí (từ 1) của sản phẩm đúng trong danh sách BM25 top-50, Dense top-50, sau RRF, và sau rerank (rerank chỉ nhận 50 ứng viên đầu sau RRF); "không có" nếu không xuất hiện | Tầng nào kéo sản phẩm đúng lên hoặc đẩy xuống | A1 mục 7 |
| **Số câu lỗi theo cấu hình** | Số câu viết tay mà sản phẩm đúng **không có trong top-10** của cấu hình đó | Mỗi nhánh tìm kiếm hỏng ở loại câu nào | A1 mục 8 |
| **"Sản phẩm đúng ở [1]"** | Số câu có r₁ ∈ G, kèm tỷ lệ % trên số câu | Tương đương nDCG@1 | A3 mục 7.1 |
| **"Không có trong 5 nguồn"** | Số câu có Context Recall@5 = 0 | Số câu mà LLM chắc chắn không trả lời đúng được | A3 mục 7.1 |

---

## 4. Metric RAG (`rag_metrics.py`, `step_ablation_rag.py`)

Dùng ở: báo cáo A3 mục 6.2 và 7. Ở đây **S = (s₁, …, s₅)** là 5 nguồn đưa vào context, **A** là câu trả lời của LLM.

| Metric | Công thức | Ý nghĩa | Lưu ý |
|---|---|---|---|
| **Context Recall@5** | \|S ∩ G\| / \|G\| (dùng lại `recall_at_k` của A1) | LLM có được sản phẩm đúng để trả lời không | |
| **Context Precision@5** | (1 / min(\|G\|, 5)) · Σ_{i: sᵢ ∈ G} P@i, với P@i = (số sản phẩm đúng trong i nguồn đầu) / i | Sản phẩm đúng có đứng cao trong context không | Khi \|G\| = 1: bằng 1/hạng nếu có trong top-5, ngược lại 0. Ví dụ sản phẩm đúng ở `[3]` thì được 1/3 ≈ 0,333 |
| **Faithfulness** | (số claim được gắn SUPPORTED) / (tổng số claim đọc được) | Câu trả lời có bịa thông tin ngoài context không | LLM giám khảo (cùng model sinh câu trả lời) tách claim theo định dạng `CLAIM: … \| SUPPORTED` hoặc `\| NOT_SUPPORTED`. Bằng `None` nếu không đọc được claim nào |
| **Answer Relevancy** | cos(e(câu hỏi), e(A)), với e là embedding `sentence-transformers/all-MiniLM-L6-v2` đã chuẩn hoá | Câu trả lời có đúng chủ đề câu hỏi không | Trong [−1, 1]. Model embedding tiếng Anh, còn câu trả lời là tiếng Việt, nên chỉ dùng để so giữa các cấu hình |
| **Mention hit** | (số từ khoá trong `answer_should_mention` xuất hiện trong A, so chuỗi không phân biệt hoa thường) / (số từ khoá) | Câu trả lời có nhắc đúng sản phẩm không | Bị chặn bởi **mức trần** (dòng cuối bảng) |
| **Citation** | 1 nếu A khớp biểu thức `\[\d+\]` ít nhất một lần, ngược lại 0; trung bình là tỷ lệ % | Có tuân thủ yêu cầu trích dẫn không | Chỉ kiểm định dạng, không kiểm `[n]` có trỏ đúng sản phẩm |
| **Latency** (A3) | Thời gian gọi `answer()` hoặc `answer_agentic()`, tính bằng giây (`time.time`) | Thời gian người dùng chờ một câu trả lời | **Không** gồm thời gian chấm Faithfulness và Answer Relevancy. Lần gọi đầu sau khi khởi động gồm cả thời gian nạp model |
| **Rewrite rate** (chỉ Agentic) | Tỷ lệ câu có `final_query_used` ≠ câu hỏi gốc | Agentic phải tìm lại bao nhiêu lần | |
| **Lỗi** (`n_errors`) | Số câu vẫn lỗi sau 3 lần thử gọi Ollama | Độ ổn định khi chạy | Câu lỗi không tính vào trung bình |
| **Mức trần Mention hit** (A3 mục 3.3) | Trung bình trên các câu của (số từ khoá `answer_should_mention` có trong **tên sản phẩm đúng**) / (số từ khoá) | Mention hit cao nhất có thể đạt, vì context chỉ có tên sản phẩm | 0,708 (250 câu); 0,760 (50 câu viết tay); 0,695 (200 câu sinh từ mẫu) |

---

## 5. Báo cáo BPM (mục 6–7, `bpm_event_log_simulation.py`)

Mỗi kịch bản có thời gian một lần tìm T ~ phân phối đều trên [T_min, T_max], và xác suất phải tìm lại p. Mọi giá trị T và p đều là **giả định**, nên các con số dưới đây chỉ minh hoạ cách tính.

| Con số | Công thức | Ý nghĩa |
|---|---|---|
| **CT lý thuyết** (công thức Loop) | CT = T̄ / (1 − p), với T̄ = (T_min + T_max) / 2 | Cycle time kỳ vọng khi phải lặp với xác suất p. Ví dụ kịch bản A: 5 / 0,6 = 8,33 phút |
| **Số lần lặp kỳ vọng** | 1 / (1 − p) | Kỳ vọng của phân phối hình học. Ví dụ kịch bản A: 1,67 lần |
| **CT mô phỏng** | (1 / số case) · Σ_case Σ_lần thử (thời gian lần thử) | Trung bình thời gian thật của các case trong mô phỏng |
| **Số lần lặp trung bình** (mô phỏng) | (số dòng event log của kịch bản) / (số case) | Trung bình số lần tìm mỗi case |
| **% giảm so với kịch bản A** | (CT_A − CT_X) / CT_A | Ví dụ B: (8,33 − 0,88) / 8,33 = 89,4% |
| **C so với B** | CT_C / CT_B − 1 | 1,289 / 0,882 − 1 = +46,1% |
| **Ngưỡng hoà vốn p_B\*** | Giải T̄_B / (1 − p_B) = CT_C, được p_B\* = 1 − T̄_B / CT_C | Với giả định hiện tại: 1 − 0,75 / 1,289 = 41,8% |
