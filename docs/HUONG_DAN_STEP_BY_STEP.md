# HƯỚNG DẪN STEP BY STEP — A1 → A2 → A3
### Đọc file này để LÀM, đọc các file `BAO_CAO_*.md` để HIỂU SÂU/VIẾT BÁO CÁO
### Ước tính thời gian mỗi bước có ghi kèm — tổng khoảng 1-2 ngày làm việc thật (không tính viết báo cáo/quay video)

---

## BƯỚC 0 — Chuẩn bị môi trường (làm 1 lần duy nhất, ~15 phút)

```bash
# Giải nén project, vào đúng thư mục
cd a1_multimodal_search

# Cài toàn bộ thư viện cần thiết cho cả A1 + A3
pip install -r requirements.txt --break-system-packages
```

**Kiểm tra đã cài đúng chưa:**
```bash
python3 -c "import torch, transformers, faiss, rank_bm25, fastapi, sentence_transformers; print('Tất cả thư viện OK')"
```

Nếu máy không có GPU hoặc mạng không truy cập được Hugging Face ổn định → dùng `kaggle_build_A1_data.py` (xem mục "Phụ lục — Chạy trên Kaggle" cuối file này) thay vì chạy Bước 1-3 dưới đây trên máy.

---

## PHẦN A1 — HỆ THỐNG SEARCH (BM25 + Dense + RRF + Rerank)

### Bước 1 — Tải dataset, build catalog (~5-10 phút, cần internet)

```bash
python step1_build_catalog.py
```

**Kỳ vọng thấy:** log tiến trình, cuối cùng in ra bảng phân bố category (kiểm tra đa dạng dữ liệu — không category nào chiếm quá 40%).

**Kiểm tra kết quả:**
```bash
ls data/                    # phải thấy catalog.jsonl và thư mục images/
wc -l data/catalog.jsonl    # phải ra khoảng 2000 (hoặc gần đó, tuỳ số bị bỏ qua)
```

### Bước 2 — Sinh embedding CLIP (~5-20 phút tuỳ máy, KHÔNG cần internet)

```bash
python step2_build_embeddings.py
```

**Kỳ vọng thấy:** log "đã encode X/2000" chạy tới hết. Nếu máy không có GPU, bước này chạy trên CPU — chấp nhận chậm hơn (không sai, chỉ lâu).

**Kiểm tra:**
```bash
ls -lh data/image_embeddings.npy   # phải có, vài MB
```

### Bước 3 — Build FAISS + BM25 index (~1 phút)

```bash
python step3_build_index.py
```

**Kiểm tra:**
```bash
ls data/    # phải có đủ: catalog.jsonl, images/, image_embeddings.npy, dense.index, bm25.pkl
```

**→ Nếu đủ 5 thứ trên trong `data/`, Bước 1-3 đã HOÀN TẤT.**

### Bước 4 — Viết bộ câu hỏi đánh giá thật (~1-2 tiếng, việc KHÔNG code sẵn được)

```bash
cp data/eval_set_template.jsonl data/eval_set.jsonl
```

M�� `data/catalog.jsonl` bằng bất kỳ text editor nào, xem sản phẩm thật (mỗi dòng có `id`, `caption`). Chọn 20-40 sản phẩm khác nhau, tự nghĩ câu hỏi tự nhiên tương ứng, viết vào `data/eval_set.jsonl` theo đúng format:

```json
{"query": "câu hỏi tự nhiên bạn nghĩ ra", "query_type": "text", "relevant_ids": [id đúng]}
```

**Mẹo chọn câu hỏi đa dạng độ khó** (để ablation ở Bước 5 có ý nghĩa):
- Vài câu chứa **từ khoá/mã số chính xác** (test BM25) — vd tên thương hiệu
- Vài câu **diễn đạt tự do, đồng nghĩa** (test Dense) — vd mô tả công dụng thay vì tên
- Vài câu **mơ hồ** (test khả năng cả hệ thống, không câu nào dễ)

### Bước 5 — Chạy ablation A1 (~2-5 phút, model rerank tự tải lần đầu ~80MB)

```bash
python step8_run_ablation.py
```

**Kỳ vọng thấy:** bảng 4 dòng (bm25 only / dense only / hybrid / hybrid + rerank), mỗi dòng có **Recall@10, nDCG@10, nDCG đa mức (0/1/2), MRR, Latency avg/p95**. Cấu hình cuối (+rerank) sẽ **chậm hơn rõ rệt** (Latency tăng hàng chục lần) — đây là bình thường (đã giải thích trong `BAO_CAO_A2.md`).

**Việc cần làm với số liệu này:** copy bảng in ra, thay vào bảng minh hoạ trong `BAO_CAO_TONG_HOP.md` mục 11.

### Bước 5b — Ablation nâng cao (tuỳ chọn, ~10-15 phút, phục vụ mục 5.2/5.4/6 của template báo cáo)

Chạy thêm 3 script sau nếu template báo cáo yêu cầu sweep tham số / so sánh trường index / case study theo từng tầng — cả 3 đều dùng lại `data/eval_set.jsonl` đã viết ở Bước 4, không cần chuẩn bị gì thêm:

```bash
# So sánh BM25 index theo trường: title-only / title+description / +ảnh
python step9_ablation_index_fields.py

# Sweep tham số RRF: rrf_k x top_n (N)
python step10_sweep_rrf.py

# Truy vết 1 câu qua từng tầng BM25 -> Dense -> RRF -> Rerank
python step12_case_study_trace.py
```

**Việc cần làm với số liệu này:** copy 3 bảng in ra vào `BAO_CAO_TONG_HOP.md` mục 11.1/11.2/11.3 (đã có sẵn khung + số liệu mẫu, thay bằng số bạn chạy ra nếu khác).

### Bước 5c — Bộ truy vấn ẢNH bằng biến đổi (tuỳ chọn, ~2 phút, phục vụ mục 2.3/4.3 template)

```bash
# Dựng 12 ảnh query bằng crop/xoay/đổi sáng từ ảnh thật, chọn từ eval_set.jsonl
python step13_build_image_eval_set.py

# Đánh giá — in bảng Recall/nDCG/MRR theo từng loại transform
python step14_eval_image_queries.py
```

**Kỳ vọng thấy:** bảng 3 dòng (crop/rotate/brightness) — nếu chạy trên đúng `data/` hiện tại, Recall/nDCG/MRR đều = 1.000 cả 3 loại (đã kiểm chứng thật, CLIP rất bền với mức biến đổi vừa phải này). Copy bảng vào `BAO_CAO_TONG_HOP.md` mục 9 (phần "Mục 2.3/4.3 template").

### Bước 5d — Ablation embedding model: CLIP B/32 vs L/14 vs SigLIP (tuỳ chọn, nặng — chạy trên Kaggle)

CLIP L/14 encode 2000 ảnh trên CPU mất ~37 phút CHỈ CHO 1 MODEL, và SigLIP cần thêm thư viện `sentencepiece` — thay vì chờ trên máy local, chạy `kaggle/kaggle_ablation_embedding_models.py` trên Kaggle (GPU T4 miễn phí, vài phút/model). Làm theo đúng 7 bước trong docstring đầu file đó (cần upload `data/eval_set.jsonl` làm Kaggle Dataset input), copy bảng kết quả vào `BAO_CAO_TONG_HOP.md` mục 11.4.

### Bước 6 — Error analysis (~1-2 tiếng, việc KHÔNG code sẵn được)

Từ bảng Bước 5, chọn ra các câu hỏi có Recall thấp nhất ở mỗi cấu hình. Với mỗi câu:
1. Mở ảnh sản phẩm liên quan trong `data/images/`
2. Đọc lại `category`/`brand` trong `catalog.jsonl`
3. Tự hỏi: lỗi do đồng nghĩa? do mã số bị bỏ lỡ? do category chồng chéo?
4. Gom nhóm các câu lỗi theo nguyên nhân — điền vào bảng mẫu trong `BAO_CAO_TONG_HOP.md` mục 12.

### Bước 7 — Chạy thử giao diện (~5 phút)

```bash
uvicorn app:app --reload --port 8000
```

M�� `http://localhost:8000` — thử search bằng text và ảnh, tick/bỏ tick checkbox BM25/Dense để tự kiểm chứng ablation trực tiếp trên UI.

**→ PHẦN A1 HOÀN TẤT khi xong đủ Bước 1-7.**

---

## PHẦN A2 — ĐỌC PAPER

### Bước 8 — Đọc paper gốc (~1-2 tiếng)

Paper đã chọn: **Sun et al. (EMNLP 2023), "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents"** (tên kỹ thuật: RankGPT) — tải PDF tại [aclanthology.org/2023.emnlp-main.923](https://aclanthology.org/2023.emnlp-main.923.pdf).

Đọc `BAO_CAO_A2.md` **trước** để có khung tham chiếu (bối cảnh, phương pháp, kết quả, liên hệ code), sau đó đọc paper gốc để tự xác nhận/mở rộng hiểu biết — không chỉ chép lại báo cáo đã có.

### Bước 9 — Cá nhân hoá báo cáo A2 (~30 phút)

`BAO_CAO_A2.md` đã viết sẵn đầy đủ 6 phần, nhưng nên tự đọc lại và:
- Thêm 1-2 câu nhận xét CÁ NHÂN sau khi đọc paper gốc (không có trong bản mẫu)
- Kiểm tra lại mục 5 ("Liên hệ với đồ án A1") có khớp đúng với `rerank()` trong `search_core.py` bạn đang có hay không

### Bước 9b — Chạy ablation reranker: MiniLM vs DeBERTa distill (~10 phút trên CPU, không cần Ollama)

1. Tải `deberta-10k-rank_net.zip` ở mục "Download data and model" của github.com/sunnweiwei/RankGPT, giải nén sao cho có `models/deberta-10k-rank_net/config.json`.
2. Chạy:

```bash
python step_ablation_rerank.py
```

**Kỳ vọng thấy:** bảng 3 dòng (không rerank / cross-encoder MiniLM / DeBERTa distill), cột nDCG@1/5/10, Recall@10, thời gian rerank. Kết quả lưu vào `data/rerank_ablation_results.json`; đối chiếu với bảng ở `BAO_CAO_A2.md` mục 5.3.

**→ PHẦN A2 HOÀN TẤT khi đã đọc paper gốc + chạy xong ablation reranker + xác nhận nội dung báo cáo.**

---

## PHẦN A3 — RAG (Naive + Agentic + Chatbot)

### Bước 10 — Cài Ollama (~10 phút, cần internet)

```bash
curl -fsSL https://ollama.com/install.sh | sh      # Linux/Mac
# Windows: tải installer tại https://ollama.com

ollama pull llama3.1        # ~4.7GB, cần máy khá mạnh
# HOẶC nếu máy yếu:
ollama pull qwen2.5:3b      # ~2GB, nhẹ hơn
```

Nếu dùng model khác `llama3.1`, sửa `LLM_MODEL` ở đầu file `rag_core.py` cho khớp.

**Kiểm tra Ollama đã chạy:**
```bash
curl http://localhost:11434
# Nếu không phản hồi, chạy: ollama serve
```

### Bước 11 — Chạy thử chatbot (~10 phút)

```bash
uvicorn app:app --reload --port 8000
```

M�� `http://localhost:8000`, cuộn xuống khu vực **"Trợ lý tư vấn (Chatbot RAG)"**. Thử hỏi liên tiếp 2-3 câu **có liên quan tới nhau** để kiểm chứng Memory hoạt động, ví dụ:
```
Câu 1: "áo đen khoảng 200 nghìn"
Câu 2: "còn màu khác không"        <- kiểm tra hệ thống có hiểu "màu khác" của cái áo vừa nói
```

Bấm nút **"Cuộc trò chuyện mới"** để xác nhận lịch sử được xoá đúng.

### Bước 12 — Ground truth cho RAG dùng CHUNG với A1 (không có file riêng nữa)

A3 = "upgrade A1 bằng LLM", nên KHÔNG cần bộ qrels riêng — `data/eval_set.jsonl` (Bước 4/6) đã được mở rộng để dùng chung cho cả A1 và A3:

```bash
python step15_expand_eval_set.py
```

Script này gộp: 30 câu gốc A1 (giữ nguyên) + 20 câu tay dành riêng cho A3 (đã dịch sang tiếng Anh, khớp quy ước ngôn ngữ của 30 câu A1) + 200 câu sinh bằng template tiếng Anh từ dữ liệu sản phẩm thật — tổng **250 câu** (đúng yêu cầu nhóm 5 người x 50 câu/người), MỖI câu đều có sẵn field `answer_should_mention`:
```json
{"query": "10-in-1 air fryer toaster oven combo", "relevant_ids": [272], "answer_should_mention": ["NuWave", "Air Fryer"]}
```

`answer_should_mention` dùng để tự kiểm tra sau này xem lỗi nằm ở Retrieval hay Generation (xem `BAO_CAO_A3.md` mục 10c). **Lưu ý:** chạy lại script này sẽ GHI ĐÈ `data/eval_set.jsonl` — nếu đã tự sửa tay thêm câu nào, backup trước khi chạy lại.

### Bước 13 — Chạy ablation A3: Naive vs Agentic, đủ 6 metric RAG (~vài chục phút đến vài giờ với 250 câu, gọi LLM thật nên chậm hơn A1)

```bash
python step_ablation_rag.py
```

Mở file, sửa `SAMPLE_SIZE = 20` (thay vì `None`) để chạy thử nhanh vài chục câu trước khi chạy full 250 câu chính thức — 250 câu x 2 cấu hình x ~2 lượt gọi LLM/câu có thể mất hàng giờ với model nhỏ chạy CPU.

**Kỳ vọng thấy:** bảng so sánh Naive RAG vs Agentic RAG — Context Recall/Precision, Faithfulness, Answer Relevancy, Mention hit rate, Citation rate, Latency trung bình.

**Việc cần làm với số liệu này:** copy vào `BAO_CAO_A3.md` mục 10c Bước 4 và `BAO_CAO_BPM.md` mục 6 (thay số ước lượng), đối chiếu với ngưỡng ~42% đã tính trong mục 7 của báo cáo đó.

### Bước 14 — Chẩn đoán lỗi theo khung Retrieval vs Generation (~1-2 tiếng)

Dùng đúng bảng chẩn đoán trong `BAO_CAO_A3.md` mục 10c: với mỗi câu trả lời sai từ Bước 13, xác định:
1. `sources` có chứa sản phẩm đúng không? (Không → lỗi Retrieval, xem lại Bước 6 của A1)
2. `sources` đúng nhưng `answer` không nhắc tới `answer_should_mention`? (→ lỗi Generation)
3. Viết ra: hiện tượng → tầng lỗi → nguyên nhân gốc → giải pháp (theo đúng 2 ví dụ mẫu đã có trong báo cáo)

**→ PHẦN A3 HOÀN TẤT khi xong đủ Bước 10-14.**

---

## BƯỚC CUỐI — Đóng gói nộp bài (không code sẵn được)

```
□ Convert 4 file BAO_CAO_*.md thành Word/PDF theo đúng khuôn nộp bài của trường/môn
□ Làm slide trình bày (dựa theo cấu trúc mục lục từng báo cáo)
□ Quay demo video — nên quay đủ 3 phần:
    1. A1: search bằng text + ảnh, tick/bỏ checkbox demo ablation trực tiếp
    2. A3: hỏi liên tiếp nhiều câu trong chatbot, demo Memory hoạt động
    3. (nếu có) so sánh nút "Hỏi" vs "Hỏi (agentic)" trên cùng 1 câu hỏi mơ hồ
□ Cả nhóm đọc Phụ lục Q&A trong 4 báo cáo, tự hỏi-đáp chéo trước buổi bảo vệ
```

---

## PHỤ LỤC — Chạy Bước 1-3 trên Kaggle (nếu máy không đủ mạnh/không có internet ổn định tới Hugging Face)

Xem chi tiết đầy đủ trong `kaggle_build_A1_data.py` (phần docstring đầu file). Tóm tắt nhanh:

1. Tạo Kaggle Notebook mới, **Settings → Accelerator: GPU**, **Internet: ON**
2. Upload `kaggle_build_A1_data.py` làm Dataset, gắn vào Notebook qua **Add Input**
3. Chạy:
```python
!pip install -q datasets transformers==4.57.1 faiss-cpu rank_bm25
!python -c "import transformers; print(transformers.__version__)"   # phải ra 4.57.1
!python /kaggle/input/<tên-dataset>/build-shopify-data-rag.py
```
4. Tải `a1_data_output.zip` từ tab **Output**, giải nén đè vào `data/` trên máy
5. Từ đây tiếp tục **Bước 4** ở trên bình thường (không cần chạy lại Bước 1-3 trên máy)
