# Học lý thuyết môn IS6303 — từ A1 đến A3

**Môn:** Thiết kế Hệ thống Thông tin Thông minh dựa trên Truy hồi · GV: TS. Phạm Đỗ Kim Chi

Tài liệu này tổng hợp **toàn bộ kiến thức trong 3 bộ slide** (Chương 0: Giới thiệu, Chương 1: Tìm kiếm thông minh, Chương 2: RAG), giải thích lại từng khái niệm, tính tay các ví dụ trên slide, trả lời gợi ý các câu hỏi thảo luận, rồi nối vào code của project.

## Cách đọc

- **Quy ước trích dẫn:** `[C1-s26]` = Chương 1, slide số 26 (số in ở góc slide). `[C2-s35]` = Chương 2, slide 35. `[C0-s8]` = Chương 0, slide 8.
- Câu in nghiêng trong khung `> Slide:` là **trích nguyên văn** từ slide. Phần còn lại là giải thích thêm.
- Chỗ nào ghi **(bổ sung)** là kiến thức ngoài slide, thêm vào để hiểu cho trọn. Không cần học thuộc phần này.
- Ký hiệu 📌 đánh dấu ý hay bị hỏi khi bảo vệ.

## Mục lục

- **Phần 0.** Bản đồ môn học (Chương 0)
- **Phần I.** Tìm kiếm thông minh (Chương 1) → nền tảng cho **A1**
  - I.1 HTTT và kiến trúc · I.2 Truy vấn bảng · I.3 Offline/online · I.4 Tiền xử lý
  - I.5 Chỉ mục · I.6 TF-IDF · I.7 BM25
  - I.8 Embedding · I.9 Multimodal/CLIP · I.10 Dense retrieval và ANN
  - I.11 Hybrid và RRF · I.12 Rerank và kiến trúc 2 tầng
  - I.13 Đánh giá: qrels, Precision/Recall, Recall@k, MRR, nDCG, ablation
  - I.14 Serving, API, vận hành · I.15 Bài A1
- **Phần II.** RAG (Chương 2) → nền tảng cho **A2, A3**
  - II.1 Ôn tập và vì sao cần tạo sinh · II.2 LLM · II.3 Prompt engineering
  - II.4 Naive RAG · II.5 Đánh giá RAG · II.6 Các điểm lỗi và Advanced RAG
  - II.7 Tối ưu ingestion/indexing · II.8 Tối ưu truy vấn · II.9 Tối ưu truy hồi
  - II.10 Tối ưu hậu truy hồi · II.11 Cung cấp RAG qua API · II.12 Bài A2
- **Phần III.** Nối lý thuyết vào project A1 → A3 (III.0: luồng search text / search ảnh và lý do build 2 index)
- **Phần IV.** Câu hỏi ôn tập để bảo vệ
- **Phụ lục A.** Lộ trình code A1 từng bước (bản cũ, giữ nguyên)

---

# PHẦN 0 — BẢN ĐỒ MÔN HỌC (Chương 0)

## 0.1 Môn học giải bài toán gì

> Slide [C0-s4]: *"Các hệ thống trên cùng giải một bài toán truy hồi (retrieval): tìm nội dung liên quan trong kho dữ liệu lớn."*

Google (gõ từ khoá → trang web đã xếp hạng), Shopee/Tiki (tìm sản phẩm trong hàng triệu mặt hàng), Netflix/TikTok (không cần truy vấn, hệ thống tự gợi ý), ChatGPT/Claude/Gemini (hỏi câu tự nhiên → tìm nguồn → LLM viết câu trả lời). Bốn loại sản phẩm này trông khác nhau nhưng đều có chung lõi: **cho một nhu cầu thông tin, lấy ra một số ít phần tử liên quan nhất trong kho rất lớn.**

Khung WHAT / WHY / HOW [C0-s5]:
- **WHAT:** phương pháp xây hệ tìm kiếm, hỏi đáp trên tài liệu (RAG), khuyến nghị.
- **WHY:** dữ liệu quá nhiều để con người tự duyệt.
- **HOW:** đi từ SQL cơ bản → tìm theo ngữ nghĩa → RAG → khuyến nghị.

Ba sản phẩm đích [C0-s6]: **Tìm kiếm ngữ nghĩa**, **Hỏi đáp (RAG)**, **Khuyến nghị**. Ngoài ra môn đòi hai kỹ năng: *engineer* (dựng được hệ chạy thật) và *research* (đo, so sánh, chứng minh bằng số).

## 0.2 Chuẩn đầu ra (CLO) [C0-s7]

| Mã | Nội dung | Thể hiện ở bài nào |
|---|---|---|
| G1.1 | Tư duy hệ thống, tự nghiên cứu về HTTT dựa trên truy hồi, gồm embedding text và cross-modal | A1 (CLIP), A2 |
| G2.1 | Phân tích yêu cầu, thiết kế kiến trúc Search, RAG, Recommendation | A1, A3 |
| G2.2 | Vẽ sơ đồ data pipeline (online/offline) và kiến trúc hai tầng | A1, A3 |
| G2.3 | Cài đặt thành phần cốt lõi, tích hợp dịch vụ AI thành hệ chạy được, cung cấp qua API | A1, A3 |
| G3.1 | Thiết kế thí nghiệm, ablation, error analysis | A1, A3 |
| G3.2 | Tự cập nhật công nghệ qua paper, đánh giá đạo đức AI | A2, A3 |
| G3.3 | Vận dụng tổng hợp để xây một HTTT thông minh hoàn chỉnh | A3 |

## 0.3 Đánh giá [C0-s8 → s11]

| Bài | Tỷ trọng | Nội dung |
|---|---|---|
| A1 | 25% | Hệ tìm kiếm multimodal + đánh giá trên tập dữ liệu nhỏ. Báo cáo ≤ 8 trang, bắt buộc: dataset, phân tích kết quả tốt/xấu, ablation các model. Nhóm lên báo cáo được +1 [C0-s9] |
| A2 | 25% | Đọc paper, báo cáo ≤ 4 trang, slide 7 phút [C0-s10] |
| A3 | 50% | Đồ án: baseline (5đ) + hệ hoàn chỉnh front/back/DB (2.5đ) + nâng cấp engineer/research (2.5đ). Có cải tiến, **chứng minh bằng ablation**, cung cấp qua API [C0-s8, s11] |
| Cộng | ≤ +10% | Góp ý, thảo luận, trả lời câu hỏi |

📌 Cả ba bài đều có câu: *"Các thành viên phải trả lời được các câu hỏi về hệ thống đã xây dựng."* Vì vậy Phần IV có bộ câu hỏi ôn.

Lịch [C0-s12]: B1-2 truy hồi, embedding, đánh giá, serving (giao A1) · B3-4 LLM, prompt, naive/advanced RAG, hosting, LoRA (giao A2) · B5-6 RAG nâng cao, Agentic RAG, vận hành (nộp A1, A2) · B7 khuyến nghị · B8 báo cáo A1/A2 · B9-11 bảo vệ A3.

Tài liệu [C0-s14]: Manning et al. *Introduction to Information Retrieval* (2008) · Huyen *AI Engineering* (2025) · Aggarwal *Recommender Systems* (2016) · Karpukhin et al. 2020 (DPR) · Lewis et al. 2020 (RAG) · pgvector.

---

# PHẦN I — TÌM KIẾM THÔNG MINH (Chương 1)

## I.1 Hệ thống thông tin và kiến trúc

**5 thành phần của HTTT** [C1-s4, theo Laudon & Laudon]: Con người (trung tâm, người ra quyết định), Quy trình, Dữ liệu (nguyên liệu thô), Phần mềm, Phần cứng. Các HTTT doanh nghiệp [C1-s5]: TMĐT, core banking, ERP, CRM, DSS. Mỗi ngày thế giới sinh ra khoảng 402,74 triệu terabyte dữ liệu; Google xử lý hơn 8 tỷ lượt tìm kiếm/ngày [C1-s5, s6].

**Kiến trúc 3 tầng phổ biến** [C1-s9]:

```
Tầng giao diện (web, mobile)
        ↓
Tầng ứng dụng (nghiệp vụ + API)  ←── trả kết quả ──┐
        ↓                                           │
Tầng dữ liệu (DBMS) ──── nạp dữ liệu ────→ Search service
                                          (lập chỉ mục, xử lý truy vấn, xếp hạng)
```

Ý chính: **search service là một khối tách riêng**, nhận dữ liệu từ DB, tự lập chỉ mục, và tầng ứng dụng gọi nó. Không nhét logic tìm kiếm vào câu SQL của tầng ứng dụng.

**DBMS** [C1-s10]: phần mềm lưu trữ, truy vấn, bảo mật, giữ toàn vẹn. Quan hệ (SQL): MySQL, PostgreSQL (+ pgvector để tìm vector), Oracle. NoSQL: MongoDB, Redis, Cassandra.

## I.2 Vì sao truy vấn bảng không đủ

Bảng `SanPham` 200.000 dòng [C1-s8].

**Toán tử `=`** [C1-s11]:
```sql
SELECT * FROM SanPham WHERE ten = 'laptop chơi game giá rẻ';
```
Trả **0 dòng**, vì bảng chỉ có "Laptop Gaming Acer Nitro 5". `=` so khớp **toàn chuỗi**, sai một ký tự là trượt.

**Toán tử `ILIKE '%laptop%'`** [C1-s12]: linh hoạt hơn (không phân biệt hoa thường, khớp chuỗi con), bắt được D1, D2.

> Slide [C1-s12]: *"ILIKE linh hoạt hơn toán tử =, nhưng vẫn chỉ đối sánh mặt chữ, không xử lý được từ đồng nghĩa. + không thể viết hàng nghìn câu lệnh OR để liệt kê hết mọi từ đồng nghĩa trong thực tế."*

Ngoài ra (bổ sung): `ILIKE '%x%'` không dùng được B-tree index nên phải quét cả bảng, và nó **không xếp hạng**: dòng nào khớp đều ngang nhau. Hai thiếu sót này (không hiểu nghĩa, không xếp hạng) là lý do cả chương tồn tại.

## I.3 Sơ đồ offline / online của môn học [C1-s13]

Đây là sơ đồ quan trọng nhất của chương, và là yêu cầu G2.2.

```
OFFLINE (batch: ingestion, indexing) — chạy trước, chạy lại khi dữ liệu đổi
  Nguồn dữ liệu → Ingestion + tiền xử lý → Biểu diễn (embedding) → Chỉ mục
  (bảng, tài liệu)  (chunking, làm sạch)    (multimodal, ngữ nghĩa)  (B-tree, inverted, vector DB)

ONLINE (serving, querying) — chạy mỗi lần người dùng hỏi
  Front end + API gateway → Truy hồi (tầng 1) → Xếp hạng (tầng 2) → Tạo sinh (RAG, agent) / Khuyến nghị
                            Boolean, top-k      BM25, dense, rerank
```

> Slide [C1-s13]: *"→ việc tính toán nhiều phức tạp về offline, online chỉ query"*

**Vì sao chia hai pha:** encode 1 triệu ảnh bằng CLIP mất hàng giờ; người dùng chỉ chờ được vài trăm mili-giây. Nên mọi thứ tốn kém và không phụ thuộc câu hỏi (encode tài liệu, build index) làm trước. Lúc online chỉ encode **một** câu hỏi và tra index.

Trong project: `step1` → `step3` là offline; `app.py` + `search_core.search()` là online.

## I.4 Tiền xử lý dữ liệu [C1-s14 → s18]

Lợi ích [C1-s14]: dữ liệu sạch hơn, bớt nhiễu, model học/chạy trên đầu vào đáng tin, tăng độ chính xác, ra quyết định tốt hơn. Các bước chung: clean → organize → filter → transform.

**Văn bản** [C1-s15]:
1. Bóc nội dung: bỏ HTML, menu, quảng cáo.
2. Làm sạch: sửa encoding, loại trang trùng.
3. Chuẩn hoá: **Unicode NFC** (tiếng Việt), chữ thường.
4. Chunking: chia theo đoạn hoặc câu.

Giải thích NFC (bổ sung): chữ "ệ" có thể lưu là **1 ký tự** (dạng dựng sẵn) hoặc **3 ký tự** ("e" + dấu mũ + dấu nặng, dạng tổ hợp). Hai chuỗi trông giống hệt nhau nhưng so byte thì khác, nên BM25 coi là hai từ khác nhau. NFC ép về một dạng duy nhất.

**Dữ liệu bảng** [C1-s16]: loại trùng, xử lý thiếu (mỗi sản phẩm đúng một bản ghi) → chuẩn hoá (ví dụ M/F → 1/0, scale số) → chọn cách biểu diễn (chia train/validation/test).

**Ảnh** [C1-s17]:
1. Loại ảnh hỏng, trùng: so **hash** để bắt ảnh trùng.
2. Resize về kích thước model nhận, **224×224 với CLIP** (trên slide ảnh được pad viền đen thành hình vuông để không méo).
3. Chuẩn hoá pixel về khoảng giá trị dùng khi huấn luyện model (CLIP trừ mean, chia std theo từng kênh màu).
4. Gắn metadata: caption, mã sản phẩm.

> Slide [C1-s15..17]: *"Tuỳ theo bài toán ta có cách xử lý phù hợp"*

**Thảo luận [C1-s18] — gợi ý trả lời:**

*A. Tìm ứng viên trong CV PDF, query "tìm liên lạc ứng viên kinh nghiệm Python 3 năm":*
- Tiền xử lý: parse PDF (có thể OCR), tách các mục (liên hệ, kinh nghiệm, kỹ năng), **trích xuất trường có cấu trúc**: email, SĐT, danh sách kỹ năng, số năm kinh nghiệm theo từng kỹ năng.
- Lập chỉ mục: trường có cấu trúc vào bảng SQL (`years_python >= 3` lọc chính xác, không cần AI); phần mô tả kinh nghiệm vào BM25 + vector để bắt "Django", "backend Python".
- Lưu ý dữ liệu cá nhân: phân quyền theo vai trò HR (xem II.7 về che dữ liệu nhạy cảm).

*B. Tìm xe vượt đèn đỏ từ video camera Võ Văn Tần:*
- Tiền xử lý: cắt frame, phát hiện đèn đỏ + vạch dừng + phương tiện vượt vạch, đọc biển số (OCR).
- Lập chỉ mục: mỗi vi phạm là một bản ghi `{camera_id, giao_lo, timestamp, bien_so, loai_xe, frame_path}`. Query "VVT-TĐ lúc 9:00 ngày dd-mm-yyyy" là truy vấn **có cấu trúc thuần** → B-tree trên `(giao_lo, timestamp)`. Có thể thêm embedding ảnh xe để tìm "xe tải màu trắng".
- Bài học: không phải bài nào cũng cần embedding; **chọn chỉ mục theo loại câu hỏi.**

## I.5 Chỉ mục (index) [C1-s20 → s23]

Dataset ví dụ UITShop gồm 8 sản phẩm D1-D8 [C1-s20], dùng xuyên suốt chương:

| ID | Tên | Loại | Đặc trưng |
|---|---|---|---|
| D1 | Máy tính xách tay gaming Acer Nitro 5 | laptop | RTX 5050 |
| D2 | Máy tính xách tay gaming Asus TUF F15 | laptop | RTX 5060 |
| D3 | Máy tính xách tay văn phòng Vivobook | laptop | mỏng nhẹ |
| D4 | Card đồ họa MSI RTX 5060 Ventus 2X | vga | 8GB |
| D5 | Card đồ họa Asus Dual RTX 5050 | vga | 6GB |
| D6 | Màn hình gaming LG UltraGear 27GP850 | monitor | 2K 165Hz |
| D7 | Chuột gaming Logitech G502 Hero | mouse | 25600 DPI |
| D8 | Bàn phím cơ gaming Keychron K8 | keyboard | switch đỏ |

**Hai loại trường** [C1-s21]:
- **Có cấu trúc** (category, brand, price): *"lọc và sắp xếp chính xác, không cần AI"* → `WHERE`, `ORDER BY`.
- **Văn bản tự do** (name, description): *"BM25 đếm từ và embedding trích nghĩa"*.

📌 Hệ tốt dùng **cả hai**: lọc cứng bằng trường cấu trúc, xếp hạng mềm bằng trường văn bản.

**Chỉ mục** [C1-s22]: không có chỉ mục phải quét từng dòng. Bảng dùng **B-tree** trên cột (id, giá): 10 triệu dòng chỉ cần khoảng log₂(10⁷) ≈ 23 bước so sánh. Văn bản dùng **inverted index**.

**Inverted index** [C1-s23]: thay vì "tài liệu → các từ", lưu ngược "từ → các tài liệu chứa nó" (posting list):

```
gaming → D1 D2 D4 D6 D7 D8
rtx    → D1 D2 D4 D5
5060   → D2 D4
laptop → (không có)
```

Query "rtx 5060": lấy giao/hợp 2 posting list thay vì đọc 8 tài liệu. Đây là lõi của Lucene, Elasticsearch.

> Slide [C1-s23]: *"Inverted index chỉ trả lời tài liệu nào CHỨA từ, không trả lời tài liệu nào liên quan HƠN."* → cần hàm chấm điểm: TF-IDF, BM25.

Để ý: "laptop" không có trong tên nào, dù D1-D3 đều là laptop. Đây là mầm mống của lỗi vocabulary mismatch ở I.7.

(Đính chính nhỏ: bảng trên slide ghi `gaming → D1 D2 D4 D6 D7 D8`, nhưng tên D4 không chứa "gaming". Theo bảng tên thì đúng là D1 D2 D6 D7 D8, và slide 27 cũng dùng df(gaming) = 5.)

## I.6 TF-IDF [C1-s24, s25]

Hai quan sát [C1-s25]:
1. Tài liệu nhắc một từ càng nhiều thì càng có khả năng nói về từ đó → **TF(t,d)** = số lần t xuất hiện trong d.
2. Từ càng hiếm trong kho thì mỗi lần xuất hiện càng mang nhiều thông tin → **IDF(t) = ln(N / df)**, với N là số tài liệu, df là số tài liệu chứa t.

Kết hợp: **score(q, d) = Σ trên các từ t trong q của TF(t,d) × IDF(t).**

Trực giác IDF: từ "của", "và" có trong gần hết tài liệu, df ≈ N, ln(1) = 0, gần như không đóng góp. Từ hiếm như "G502" có df = 1, IDF lớn. **IDF chính là trọng số "độ phân biệt" của từ.** Dùng log để điểm không phình quá khi N rất lớn [C1-s25, ghi chú].

**Ví dụ slide 25 (tính lại):**
- score("Chuột", D7) = 1 × ln(8/1) = ln 8 = **2,079** (slide làm tròn thành 2).
- score("game", D7) = 1 × ln(8/4) = ln 2 = **0,693**. "Chuột" hiếm hơn nên nặng gấp 3.

**Ví dụ slide 24** dùng TF chuẩn hoá theo độ dài và log cơ số 10:
- TF("won", d1) = 1/7 = 0,14; TF(d2) = 0/7 = 0; TF(d3) = 2/10 = 0,2.
- IDF("won") = log₁₀(3/2) = 0,176.
- TF-IDF: d1 = 0,025; d2 = 0; d3 = 0,035 → **d3 > d1 > d2**.

Có nhiều biến thể TF-IDF (TF thô hay chia độ dài, log cơ số e hay 10). Thứ hạng thường giống nhau, chỉ khác thang điểm. Khi trình bày nhớ nói rõ mình dùng biến thể nào.

## I.7 BM25 [C1-s26 → s30]

**Hai vấn đề của TF-IDF** [C1-s26]:
1. Điểm tăng **tuyến tính** theo tần suất: từ xuất hiện 10 lần được gấp 10 lần điểm, dù thực tế lần thứ 10 không cho thêm bao nhiêu thông tin. Tài liệu spam lặp từ khoá sẽ thắng.
2. **Thiên vị tài liệu dài**: dài thì tự nhiên chứa nhiều từ hơn.

BM25 ("Best Match 25") [C1-s26, Robertson & Zaragoza 2009], mặc định trong Lucene, Elasticsearch, OpenSearch từ 2016:

$$
\text{BM25}(D,Q)=\sum_{i=1}^{n}\text{IDF}(q_i)\cdot\frac{f(q_i,D)\,(k_1+1)}{f(q_i,D)+k_1\left(1-b+b\cdot\frac{|D|}{\text{avgdl}}\right)}
$$

Giải từng phần:

| Thành phần | Ý nghĩa |
|---|---|
| IDF(qᵢ) | Từ hiếm nặng hơn từ phổ biến, kế thừa TF-IDF. Bản dùng trong slide 27 (giống Lucene): IDF = ln(1 + (N − df + 0,5)/(df + 0,5)). Số "+1" giữ IDF luôn dương |
| f(qᵢ, D) | Số lần từ xuất hiện trong D |
| k₁ (thường 1,2 → 2,0) | **Bão hoà tần suất.** Phân số f(k₁+1)/(f + k₁·…) tiến tới giới hạn (k₁+1) khi f → ∞. Lần xuất hiện đầu cho nhiều điểm nhất, các lần sau cho ít dần |
| b (thường 0,75) | **Chuẩn hoá độ dài.** b = 1: phạt độ dài tối đa. b = 0: tắt phạt |
| \|D\| / avgdl | Tài liệu dài hơn trung bình (tỷ lệ > 1) → mẫu số lớn → bị chia bớt điểm |

**Đồ thị bão hoà** [C1-s28]: k₁ nhỏ (0,5) thì 1 lần hay 10 lần gần như bằng nhau; k₁ lớn (5,0) thì tần suất tiếp tục được thưởng. Mặc định k₁ = 1,2, b = 0,75 tốt cho đa số corpus, chỉ tinh chỉnh khi có tập đánh giá.

### Tính tay ví dụ "máy gaming" [C1-s27]

Tham số: k₁ = 1,5, b = 0,75, N = 8, chỉ tính cột tên, tách từ theo khoảng trắng.

**Bước 1 — avgdl.** Đếm số từ: D1 = 8, D2 = 8, D3 = 7, D4 = 8, D5 = 7, D6 = 6, D7 = 5, D8 = 6. Tổng 55 → avgdl = 55/8 = **6,875** ✓.

**Bước 2 — IDF.**
- "gaming" có trong D1, D2, D6, D7, D8 → df = 5 → IDF = ln(1 + 3,5/5,5) = ln(1,636) = **0,492**.
- "máy" có trong D1, D2, D3 → df = 3 → IDF = ln(1 + 5,5/3,5) = ln(2,571) = **0,944**.

**Bước 3 — từng tài liệu** (f = 1 nên tử số = IDF × 2,5):

| Doc | \|D\| | 1 − b + b·\|D\|/avgdl | Mẫu = 1 + 1,5 × (…) | "máy" | "gaming" | Tổng |
|---|---|---|---|---|---|---|
| D7 | 5 | 0,25 + 0,545 = 0,795 | 2,193 | 0 | 1,231/2,193 = **0,561** | 0,561 |
| D6, D8 | 6 | 0,25 + 0,655 = 0,905 | 2,357 | 0 | **0,522** | 0,522 |
| D3 | 7 | 0,25 + 0,764 = 1,014 | 2,520 | 2,360/2,520 = **0,936** | 0 | 0,936 |
| D1, D2 | 8 | 0,25 + 0,873 = 1,123 | 2,684 | **0,879** | **0,458** | 1,338 |

Khớp đúng bảng slide. Để ý: cùng từ "gaming" (f = 1) nhưng D7 (5 từ) được 0,561, còn D1 (8 từ) chỉ được 0,458. Đó là tác dụng của b.

### Thất bại kinh điển: Q1 = "laptop gaming" [C1-s29]

| Hạng | Doc | BM25 |
|---|---|---|
| 1 | D7 Chuột | 0,561 |
| 2 | D6 Màn hình | 0,522 |
| 3 | D8 Bàn phím | 0,522 |
| 4 | D2 Laptop | 0,458 |
| 5 | D1 Laptop | 0,458 |

Hai nguyên nhân:
1. "laptop" không có trong tên nào (tên ghi "máy tính xách tay") → chỉ "gaming" được tính. **Vocabulary mismatch.**
2. Chỉ còn "gaming" thì tài liệu ngắn thắng, đẩy laptop xuống.

> Slide [C1-s29]: *"Cần so sánh theo nghĩa, không chỉ theo chuỗi ký tự → embedding."*

### Ưu, nhược điểm BM25 [C1-s30]

| Ưu | Nhược |
|---|---|
| Khớp chính xác mã sản phẩm, tên riêng, số hiệu | Vocabulary mismatch: khác từ ngữ là không tìm thấy |
| Rẻ, rất nhanh, không cần GPU, không huấn luyện | Không hiểu đồng nghĩa, viết tắt, diễn đạt tự do |
| Explainable: truy được điểm về từng từ khoá | Không dùng được ngữ cảnh cả câu |
| Có sẵn (Lucene, Postgres FTS) | Phụ thuộc khâu tách từ, **đặc biệt tiếng Việt** |

Vì sao tiếng Việt khó tách từ (bổ sung): tiếng Việt viết theo âm tiết, "máy tính xách tay" là **một** từ gồm 4 âm tiết. Tách theo khoảng trắng thì thành 4 token, và "máy" khớp cả "máy giặt". Có thể dùng bộ tách từ (underthesea, pyvi) để giữ cụm "máy_tính_xách_tay".

## I.8 Embedding [C1-s32 → s40]

**Vấn đề** [C1-s32]: "cách hủy đơn hàng" và "quy trình bãi bỏ giao dịch" cùng nghĩa, không trùng từ nào → BM25 trả 0 kết quả.

**Định nghĩa** [C1-s33]:
> *"Cách biểu diễn một đoạn thông tin (text, image,...) thành một VECTOR — nội dung gần nghĩa → vector gần nhau trong không gian."* *"Quy nghĩa về vector, việc so sánh nghĩa trở thành phép đo khoảng cách giữa 2 vector."*

**Lịch sử ý tưởng** [C1-s34]:
- **Distributional Hypothesis:** từ xuất hiện trong ngữ cảnh giống nhau thì nghĩa giống nhau. "chó" và "mèo" đều đi với "cho ăn", "nuôi", "lông" nên vector gần nhau.
- **word2vec (2013):** học vector từ ngữ cảnh. Phép toán vector mang nghĩa: *vua − đàn ông + phụ nữ ≈ nữ hoàng* (hướng "giới tính" là một hướng cố định trong không gian).
- **BERT (2018):** vector **theo ngữ cảnh**, cùng một từ ở mỗi câu có một vector khác ("bank" ngân hàng khác "bank" bờ sông).
- **CLIP (2021):** mở rộng sang ảnh, nền cho multimodal.

> Slide [C1-s34]: *"Giá trị cốt lõi: biểu diễn nghĩa bằng vector chuyển bài toán so sánh ngữ nghĩa về phép toán số học mà máy tính xử lý hiệu quả."*

**Ví dụ** [C1-s35]: "laptop chơi game" → [0,82, 0,13, −0,25, …], "máy tính chơi game" → [0,79, 0,15, −0,22, …] (gần), "nồi cơm điện" → [−0,44, 0,91, 0,37, …] (xa).

Câu hỏi trên slide: *"biến đổi thông tin (hình ảnh, văn bản) thành vector như thế nào?"* → bằng một mạng nơ-ron (encoder) huấn luyện bằng **contrastive learning**, xem bên dưới.

**k-NN** [C1-s36]: query cũng thành vector, top-k = k điểm gần query nhất. Câu hỏi slide: *"Tìm các vector gần nhau như thế nào?"* → đo bằng cosine (dưới đây), tìm nhanh bằng ANN (I.10).

### Cosine similarity [C1-s37]

$$\cos\theta=\frac{A\cdot B}{\|A\|\,\|B\|}\in[-1,1]$$

Gần 1: cùng hướng, gần nghĩa ("laptop" với "máy tính xách tay"). Gần 0: vuông góc, không liên quan ("laptop" với "nồi chiên không dầu").

**Ví dụ tính tay (bổ sung):** A = [3, 4], B = [4, 3].
A·B = 12 + 12 = 24; ‖A‖ = ‖B‖ = 5 → cos = 24/25 = **0,96**.

📌 **Câu hỏi slide: "tại sao dùng cosine mà không dùng Euclid?"**
1. **Cosine chỉ đo hướng, bỏ qua độ dài.** Độ dài vector thường mang thông tin phụ (độ dài văn bản, độ "tự tin" của model), không phải nghĩa. Vector [3,4] và [30,40] cùng nghĩa, cosine = 1, nhưng khoảng cách Euclid = 45.
2. **Khi đã chuẩn hoá về độ dài 1, hai cách cho cùng thứ hạng.** ‖A − B‖² = ‖A‖² + ‖B‖² − 2A·B = 2 − 2cos θ. Euclid nhỏ nhất ⇔ cosine lớn nhất. Vì vậy project chuẩn hoá vector (`feat / feat.norm()`) rồi dùng `IndexFlatIP` (tích vô hướng) = cosine.
3. Model được **huấn luyện bằng cosine** (loss contrastive dùng cosine), nên đo bằng cosine là nhất quán với lúc huấn luyện.
4. Ở không gian nhiều chiều (512, 768), khoảng cách Euclid giữa các điểm có xu hướng xấp xỉ nhau, kém phân biệt.

Slide 37 còn minh hoạ **static vs contextual embedding**: static (word2vec) thì "Apple" chỉ có một vector nằm lưng chừng giữa trái cây và công nghệ; contextual (BERT) thì "Apple" trong câu hỏi cổ phiếu nằm gần "Technology", trong câu hỏi dinh dưỡng nằm gần "fruit".

### Huấn luyện contrastive [C1-s38, s39]

> Slide [C1-s38]: *"Loss function kéo vector cặp đúng lại gần, đẩy cặp sai ra xa. Hard negative (tài liệu sai nhưng trông giống đúng) quyết định chất lượng."*

Công thức phổ biến (InfoNCE, bổ sung): với query q, tài liệu đúng d⁺ và các tài liệu sai d⁻ⱼ, τ là temperature:

$$\mathcal{L}=-\log\frac{e^{\cos(q,d^+)/\tau}}{e^{\cos(q,d^+)/\tau}+\sum_j e^{\cos(q,d^-_j)/\tau}}$$

Đây là softmax "chọn đúng d⁺ trong đám đông". Loss nhỏ khi cos(q, d⁺) lớn hơn hẳn các cos(q, d⁻).

**Vì sao hard negative quan trọng:** negative ngẫu nhiên ("nồi cơm điện" cho query "RTX 5060") quá dễ, model không học được gì mới. Hard negative ("RTX 5050" cho query "RTX 5060") buộc model phân biệt chi tiết. Slide 39: *"negative lấy ngẫu nhiên trong batch (in-batch negatives) cộng thêm hard negative do BM25 tìm ra"*. BM25 tìm ra tài liệu trùng nhiều từ nhưng sai, đúng loại "trông giống đúng".

**In-batch negatives:** batch có N cặp (query, doc), thì với mỗi query, N−1 doc của các cặp khác là negative miễn phí. Batch càng lớn, càng nhiều negative.

**Nguồn dữ liệu cặp** [C1-s39]: câu hỏi và đoạn chứa đáp án (Natural Questions, MS MARCO), tiêu đề và nội dung bài viết, cặp câu dịch, hỏi đáp diễn đàn, cặp paraphrase.

> Slide [C1-s39]: *"Học viên có thể tự huấn luyện mô hình (advantage cho project cuối kỳ)"*

**Kết quả** [C1-s40]: "laptop", "máy tính xách tay", "notebook" vào cùng một cụm. "card đồ họa", "GPU" một cụm. Phụ kiện một cụm. *"Nếu so trùng vector thay vì chuỗi ký tự, Q1 sẽ tìm ra laptop."*

## I.9 Embedding ảnh và Multimodal (CLIP) [C1-s41 → s46]

**Image embedding thuần** [C1-s41]: image encoder (ViT, ResNet) biến ảnh thành vector. Ảnh giống nội dung thì vector gần. Ứng dụng: tìm ảnh giống ảnh (Google Lens), gom cụm ảnh.

> *"Ở bước này ảnh và văn bản vẫn là HAI không gian riêng"* — không so được vector ảnh với vector câu.

**Multimodal** [C1-s42]:
> *"Khác image embedding thuần: hai encoder được huấn luyện CHUNG để ánh xạ ảnh và văn bản về CÙNG một không gian."*

Dữ liệu: hàng trăm triệu cặp (ảnh, caption) từ web (CLIP gốc dùng khoảng 400 triệu cặp, bổ sung).

**Cách huấn luyện CLIP** [C1-s43, Radford et al. 2021]: mỗi batch N cặp → ma trận N×N cosine giữa N ảnh và N caption.

```
              "máy tính xách tay gaming"  "card đồ họa hai quạt"  "chuột gaming không dây"
ảnh laptop     KÉO GẦN (cặp đúng)          đẩy xa                  đẩy xa
ảnh card       đẩy xa                      KÉO GẦN                 đẩy xa
ảnh chuột      đẩy xa                      đẩy xa                  KÉO GẦN
```

> *"Loss thưởng cosine cao trên đường chéo và phạt cosine cao ngoài đường chéo."*

Cụ thể (bổ sung): lấy cross-entropy theo **hàng** (mỗi ảnh phải chọn đúng caption) và theo **cột** (mỗi caption phải chọn đúng ảnh), cộng lại chia đôi. Đây là InfoNCE đối xứng, với N−1 negative mỗi chiều.

**Kết quả** [C1-s44]: vector ảnh laptop nằm gần vector câu "máy tính xách tay gaming". *"Hai loại dữ liệu khác hẳn nhau nhưng so sánh được bằng cosine."* Ví dụ [C1-s46]: query "laptop chơi game màu đen" → ảnh Acer Nitro 5 cos = 0,81, card MSI 0,14, chuột 0,06.

**Hệ quả cho A1:** một model CLIP làm được 4 chiều: text→ảnh, ảnh→ảnh, ảnh→text, text→text. Chỉ cần encode catalog một lần.

**Thảo luận [C1-s45] — gợi ý trả lời:**
1. *Phân loại mềm:* thay vì gán cứng "laptop", tính cos(ảnh, "ảnh laptop"), cos(ảnh, "ảnh chuột")… rồi softmax thành xác suất. Đây là **zero-shot classification** của CLIP: không cần huấn luyện bộ phân loại, thêm lớp mới chỉ cần thêm một câu mô tả.
2. *Một vector hay nhiều vector?* Gộp một vector (trung bình ảnh + mô tả + review) thì rẻ, nhưng pha loãng ý: review "pin tệ" kéo vector lệch đi. Multi-vector (mỗi ảnh/đoạn một vector; hoặc ColBERT: mỗi token một vector, điểm = tổng MaxSim) giữ chi tiết, nhưng tốn bộ nhớ gấp nhiều lần và tính điểm phức tạp hơn. Giải pháp trung gian: nhiều vector cho mỗi sản phẩm, gộp điểm bằng max hoặc RRF.
3. *Thiết kế "tìm bằng ảnh":* offline: tiền xử lý ảnh (I.4) → CLIP image encoder → vector DB kèm metadata `product_id`. Online: ảnh người dùng → **cùng** encoder → ANN tầng 1 (top-50) → tầng 2 lọc theo tồn kho/danh mục, rerank → top-k. Có thể thêm nhánh BM25 trên caption sinh tự động từ ảnh query.

## I.10 Dense retrieval và ANN [C1-s47 → s56]

**Định nghĩa** [C1-s47]:
> *"Truy hồi bằng cách so sánh VECTOR EMBEDDING của truy vấn và tài liệu, thay vì so trùng từ. Gọi là 'dense' vì vector dày đặc (mọi chiều đều có giá trị), đối lập biểu diễn thưa theo từ của BM25."*

**Sparse vs dense** [C1-s48]:

| | Sparse (BM25, TF-IDF) | Dense (embedding) |
|---|---|---|
| Số chiều | ≈ 50.000 (bằng cỡ từ vựng) | 384 / 512 / 768 / 1024 |
| Mỗi chiều | ứng với **một từ** | một **đặc trưng học được** |
| Giá trị | hầu hết bằng 0 | mọi chiều đều khác 0 |
| Chỉ mục | inverted index | vector index (ANN) |

Pipeline dense: Query → **Encoder (bi-encoder)** → **ANN trên vector DB** → Top-k.

📌 **Câu hỏi slide: "giá trị của từng chiều trong embedding có ý nghĩa thế nào?"** Không chiều nào có nghĩa riêng mà con người đọc được. Nghĩa được **phân tán** trên nhiều chiều (distributed representation). Cái có nghĩa là **hướng** và **khoảng cách tương đối** giữa các vector. Đổi model thì giá trị từng chiều đổi hoàn toàn, nên **không được trộn vector của hai model** trong cùng một index.

**Bi-encoder** (bổ sung): encode query và tài liệu **độc lập** bằng encoder. Vì độc lập nên vector tài liệu tính trước lúc offline được, online chỉ encode query. Đây là điều kiện để dense retrieval chạy nhanh (so với cross-encoder ở I.12).

### ANN: IVF và HNSW [C1-s49]

Brute-force (so query với từng vector) tốn O(N·d). Với 100 triệu vector là quá chậm. ANN (Approximate Nearest Neighbor) **đánh đổi một chút độ chính xác để lấy tốc độ.**

| | IVF (Inverted File) | HNSW (Hierarchical Navigable Small World) |
|---|---|---|
| Ý tưởng | Gom cụm (k-means) lúc indexing. Query chỉ quét cụm gần nhất | Đồ thị nhiều tầng nối các vector. Tìm từ tầng thưa trên cùng đi xuống tầng dày |
| Tương tự | Tra từ điển theo chữ cái đầu | Đi từ đường cao tốc xuống đường nhỏ |
| Ưu | Nhanh, dễ xây, tốn ít RAM | Chính xác cao, mặc định của đa số vector DB |
| Nhược | Kém chính xác ở **biên cụm** (hàng xóm thật nằm ở cụm bên cạnh) | Tốn RAM hơn |

Project A1 dùng `faiss.IndexFlatIP` = brute-force chính xác 100%, đủ nhanh ở vài nghìn ảnh. Khi lên triệu ảnh mới cần IVF/HNSW.

### Dense chạy trên ví dụ UITShop

**Q1 "laptop gaming"** [C1-s50]: D1 ✓, D2 ✓, D6, D3. *"Hai laptop gaming D1, D2 lên đầu dù không chứa 'laptop'."* D3 (laptop văn phòng) vào top vì cùng cụm "máy tính xách tay" nhưng xếp sau D6 vì thiếu nghĩa "gaming".

**Q2 "RTX 5060"** [C1-s51]: D4 ✓, **D5 ✗ (5050)**, D2 ✓, **D1 ✗ (5050)**.
> *"Với embedding, 'RTX 5050' và 'RTX 5060' gần như cùng một nghĩa: chỉ khác một chữ số. Mã sản phẩm, số hiệu, tên riêng không phải 'nghĩa': đúng chỗ dense yếu và BM25 mạnh."*

Vì sao (bổ sung): tokenizer cắt "5060" thành các mảnh con, và trong dữ liệu huấn luyện "5050" với "5060" xuất hiện trong ngữ cảnh gần như giống hệt nhau ("card đồ hoạ tầm trung…"). Theo distributional hypothesis, chúng phải có vector gần nhau.

📌 **Hai ví dụ Q1, Q2 là cặp đối xứng chủ đạo của cả chương:** BM25 thua ở Q1 (đồng nghĩa), dense thua ở Q2 (mã số chính xác). Đây là lý do cần hybrid.

**Ưu, nhược dense** [C1-s52]:

| Ưu | Nhược |
|---|---|
| Hiểu đồng nghĩa, diễn đạt tự do, so theo nghĩa cả câu | Mờ chi tiết chính xác (5050 ≈ 5060) |
| Đa ngữ, đa modality (+ảnh) với model phù hợp | Cần model và GPU, tốn chi phí encode cả kho lúc offline |
| ANN co giãn tới hàng trăm triệu vector | Điểm khó giải thích, không truy được về từng từ |
| | Chất lượng phụ thuộc domain dữ liệu huấn luyện |

**Chọn model** [C1-s53]:

| Lựa chọn | Ví dụ | Ưu | Cân nhắc |
|---|---|---|---|
| API thương mại | text-embedding-3-large, Cohere embed-v4, voyage-3.5, gemini-embedding | Chất lượng cao, không cần hạ tầng | Phí theo token, dữ liệu rời hệ thống |
| Mã nguồn mở đa ngữ | BGE-M3, multilingual-e5, Qwen3-Embedding | Tự host, hỗ trợ tiếng Việt | Cần GPU, tự vận hành |
| Chuyên tiếng Việt | bkai bi-encoder, halong-embedding | Huấn luyện trên dữ liệu Việt | Cộng đồng nhỏ |
| Fine-tune | BGE-M3 + dữ liệu nội bộ | Sát domain (mã SP, thuật ngữ) | Cần dữ liệu cặp |
| Ảnh, đa modality | CLIP, SigLIP 2, jina-clip-v2, Cohere embed-v4 | Ảnh và text chung không gian | *"Text thuần vẫn nên dùng model text"* |

**Công cụ** [C1-s54]: Keyword: Postgres tsvector + GIN (GIN chính là inverted index), MySQL FULLTEXT, Elasticsearch/OpenSearch/Solr. Dense: pgvector, Qdrant, Milvus, Weaviate, Pinecone. Hybrid: tự ghép + RRF, hoặc Elasticsearch 8+, OpenSearch, Weaviate có sẵn.

> Slide [C1-s54]: *"Bài A1 yêu cầu tự cài để hiểu cơ chế trước khi dùng tool."* *"Một Postgres đáp ứng cả hai họ: tsvector + pgvector. Đủ cho đồ án."*

**Mở rộng** [C1-s55]:
- **SPLADE** (learned sparse): model học ra vector sparse, giữ ưu điểm từ khoá, lại **mở rộng từ** được (tài liệu "máy tính xách tay" được gán thêm trọng số cho "laptop").
- Postgres **tsvector không phải BM25 chuẩn** (xếp hạng theo tần suất giản lược). Cần BM25 thật trên Postgres: pg_search (ParadeDB).

**Embedding không chỉ để tìm kiếm** [C1-s56]: Spotify Discover Weekly, Google Photos (CLIP), TikTok For You (two-tower user-video), Face ID, phát hiện gần trùng lặp, semantic cache cho LLM. Bốn câu hỏi thiết kế cho mọi hệ embedding: **encoder nào, chỉ mục gì, vector cập nhật khi nào, đo chất lượng bằng metric nào?**

## I.11 Hybrid search và RRF [C1-s58 → s64]

> Slide [C1-s58]: *"BM25 sai ở các query về ngữ nghĩa, dense sai ở các query cần tìm chính xác → kết hợp 2 phương pháp → điều kiện kết hợp là gì?"*

**Vấn đề thang điểm** [C1-s59]: BM25 thang mở từ 0 đến +∞, cosine thường nằm trong [0,5; 1]. **Không cộng thẳng được.**

### Cách 1: Min-max rồi cộng có trọng số [C1-s60]

$$s'=\frac{s-\min}{\max-\min}\in[0,1],\qquad \text{score}=\alpha\cdot\text{BM25}'+(1-\alpha)\cdot\cos'$$

Ba hạn chế:
1. min/max **đổi theo từng truy vấn**, nên điểm chuẩn hoá không ổn định giữa các query.
2. **Outlier nén phần còn lại:** BM25 gốc [0,24 … 1,79] với top-1 vượt trội, sau min-max top-1 = 1, bốn tài liệu còn lại dồn sát 0 → *"mất khả năng phân biệt"*.
3. Thêm siêu tham số α phải tinh chỉnh và bảo trì.

### Cách 2: Reciprocal Rank Fusion [C1-s61, Cormack et al. 2009]

$$\text{RRF}(d)=\sum_{r\in\text{các danh sách}}\frac{1}{k+\text{rank}_r(d)}$$

- **Chỉ dùng thứ hạng, không dùng điểm** → không cần chuẩn hoá, miễn nhiễm outlier.
- Hạng 1 góp 1/(k+1), hạng 2 góp 1/(k+2): giảm dần, không rơi về 0 đột ngột.
- Tài liệu có mặt ở **nhiều** danh sách được cộng dồn → *"thưởng cho sự đồng thuận"*.
- Chỉ một siêu tham số k (thường 60).

> *"Dù công thức rất đơn giản, RRF vượt nhiều phương pháp kết hợp phức tạp hơn trong thực nghiệm (Cormack et al., 2009)."*

### Bài tập slide 62 — lời giải đầy đủ

Danh sách: BM25 = [D7, D8, D5, D2, D1]; Dense = [D1, D2, D3, D6, D5].

**k = 60:**

| Doc | Hạng BM25 | Hạng Dense | RRF | Hạng cuối |
|---|---|---|---|---|
| D1 | 5 | 1 | 1/65 + 1/61 = 0,015385 + 0,016393 = **0,031778** | 1 |
| D2 | 4 | 2 | 1/64 + 1/62 = 0,015625 + 0,016129 = **0,031754** | 2 |
| D5 | 3 | 5 | 1/63 + 1/65 = 0,015873 + 0,015385 = **0,031258** | 3 |
| D7 | 1 | – | 1/61 = **0,016393** | 4 |
| D8 | 2 | – | 1/62 = **0,016129** | 5 |
| D3 | – | 3 | 1/63 = **0,015873** | 6 |
| D6 | – | 4 | 1/64 = **0,015625** | 7 |
| D4 | – | – | 0 | – |

**k = 1** (hạng đầu áp đảo): D1 = 1/6 + 1/2 = 0,667 · D2 = 1/5 + 1/3 = 0,533 · **D7 = 1/2 = 0,500** · D5 = 1/4 + 1/6 = 0,417 · D8 = 0,333 · D3 = 0,250 · D6 = 0,200.
→ D7 (chỉ đứng nhất ở **một** danh sách) vượt lên trên D5 (có mặt ở **cả hai**).

**k = 120:** thứ tự giống k = 60, nhưng các điểm sát nhau hơn nữa (D1 = 0,016264, D2 = 0,016261).

📌 **Bài học về k:**
- **k nhỏ**: hạng cao được thưởng rất đậm, giống "ai đứng nhất ở đâu đó thì thắng".
- **k lớn**: chênh lệch giữa các hạng bị làm phẳng, "có mặt ở nhiều danh sách" quan trọng hơn "đứng cao".
- Để ý D5 (card 5050, vốn không liên quan tới Q1) lên hạng 3 chỉ vì có mặt ở cả hai danh sách. Đây chính là hạn chế "ưu tiên tài liệu xuất hiện trong nhiều danh sách" ở slide 64, và là lý do cần rerank [C1-s66].

**Practical RRF** [C1-s63]: với danh sách dense khác một chút (D1, D2, D3, D6, D7), hybrid ra D1 ✓, D2 ✓, D3, D7, D6. Laptop đúng lên đầu. *"Tuy nhiên nếu truy vấn là RTX 5060, cụm 5060 có thể bị bỏ qua."*

**Ưu, nhược hybrid** [C1-s64]:

| Ưu | Nhược |
|---|---|
| Recall cao: tập ứng viên tốt của cả hai | Chỉ dùng thứ hạng, bỏ qua điểm (mất thông tin "hơn nhiều hay hơn ít") |
| RRF không cần chuẩn hoá, gần như không phải tinh chỉnh | Mặc định trọng số ngang nhau giữa các hệ |
| Tính toán đơn giản | Phụ thuộc k và số lượng kết quả lấy từ mỗi nhánh (top-N) |
| Nhiều engine hỗ trợ sẵn | Ưu tiên tài liệu có mặt ở nhiều danh sách; không tự thích nghi theo loại truy vấn |

## I.12 Rerank và kiến trúc 2 tầng [C1-s66 → s72]

**Định nghĩa** [C1-s66]:
> *"Bước chấm điểm LẠI một danh sách ứng viên nhỏ bằng mô hình mạnh hơn để sửa trật tự. Không tìm thêm tài liệu mới, chỉ sắp xếp lại những gì tầng truy hồi đã đưa lên."*
> *"Người dùng chỉ nhìn 5-10 kết quả đầu nên trật tự của top quyết định trải nghiệm."*

**Bi-encoder vs cross-encoder** [C1-s67, Nogueira & Cho 2019]:

```
Bi-encoder:     [Query] → BERT → vec_q ┐
                [Doc]   → BERT → vec_d ┴→ cosine        ("5050" ≈ "5060" sau khi nén)
Cross-encoder:  [CLS] Query [SEP] Doc [SEP] → BERT (attention chéo giữa mọi token) → điểm liên quan
```

> *"Token '5060' của truy vấn đối chiếu trực tiếp token '5050' của tài liệu → phát hiện được."*

Bi-encoder nén cả câu thành **một vector** trước khi so, nên chi tiết nhỏ bị mất. Cross-encoder cho query và tài liệu **nhìn nhau** qua attention ngay trong mạng, nên bắt được khác biệt từng token.

**Vì sao không rerank cả kho** [C1-s68]: 50 ms/cặp × 1.000.000 tài liệu = 50.000 giây ≈ **14 giờ cho một truy vấn**.

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Vector tài liệu tính trước | Có | **Không thể** (điểm phụ thuộc cặp) |
| Chi phí mỗi query | 1 lần encode + tra ANN | 1 lần chạy BERT cho **mỗi** tài liệu |
| Độ chính xác xếp hạng | Khá | Cao nhất |

**Kiến trúc 2 tầng** [C1-s69]:

```
Truy vấn → TẦNG 1: TRUY HỒI (recall)              → TẦNG 2: XẾP HẠNG (precision) → Top 10
           BM25 + Dense ANN → RRF                    Cross-encoder rerank
           hàng triệu tài liệu → 50 ứng viên          50 → 10
```

**Tối ưu từng tầng** [C1-s70]:

| Tầng 1 — RECALL | Tầng 2 — PRECISION |
|---|---|
| Câu hỏi: ứng viên tốt **CÓ** trong top-50 không? | Câu hỏi: kết quả **ĐỨNG ĐẦU** có đúng không? |
| Bỏ sót ở tầng 1 thì tầng 2 không khôi phục được | Người dùng chỉ nhìn 5-10 kết quả đầu |
| Chấp nhận lẫn nhiều kết quả xấu | Chấp nhận chi phí cao vì chỉ chấm 50 cặp |
| Đo bằng **Recall@50** | Đo bằng **nDCG@10, MRR** |

> 📌 Slide [C1-s70]: *"Chú ý: Chỉ tối ưu tầng 2 khi đảm bảo tầng 1 không bỏ sót ứng viên."*

**Tổ hợp theo bài toán** [C1-s71] (slide để trống, gợi ý đáp án):

| Đặc điểm bài toán | Nên dùng |
|---|---|
| Nhiều mã số, tên riêng, thuật ngữ trùng tuyệt đối | **BM25** (dense làm phụ) |
| Người dùng diễn đạt tự do, nhiều đồng nghĩa | **Dense** |
| Cả hai kiểu truy vấn cùng tồn tại | **Hybrid** |
| Top đầu cần chính xác cao nhất, chấp nhận thêm độ trễ | **Hybrid + Rerank** |
| Ngân sách hạ tầng nhỏ, cần giao nhanh | **BM25** (không cần GPU) |

**Công cụ** [C1-s72]: Full-text: Elasticsearch, OpenSearch, Solr. Vector DB: pgvector, Milvus, Qdrant, Pinecone, Weaviate. Thư viện ANN: FAISS, hnswlib. Dịch vụ: Algolia, Azure AI Search.

## I.13 Đánh giá [C1-s73 → s80]

### qrels — ground truth [C1-s73]

qrels (query relevance judgments) = bảng (truy vấn, tài liệu, nhãn liên quan). Thang **0-3**: 3 đúng hoàn toàn, 2 liên quan tốt, 1 một phần, 0 không liên quan.

| Truy vấn: "chuột gaming không dây pin lâu" | Nhãn |
|---|---|
| G304 (pin 250 giờ) | 3 |
| Basilisk (pin 285 giờ) | 3 |
| Katar Elite (pin 110 giờ) | 2 |
| M331 (văn phòng) | 1 |
| chuột có dây | 0 |

> *"Recall@k, MRR, nDCG@k so sánh danh sách xếp hạng với bảng qrels → ground truth của bài toán truy hồi."*

Rubric A1 [C1-s91] đòi *"Tập đánh giá hợp lệ (kèm qrels ảnh)"*: tức phải có cả query bằng ảnh kèm nhãn, không chỉ query text.

### Precision và Recall [C1-s74]

$$\text{Precision}=\frac{\text{kết quả đúng đã trả về}}{\text{tổng số kết quả trả về}}\qquad \text{Recall}=\frac{\text{kết quả đúng đã trả về}}{\text{số kết quả đúng trong kho}}$$

Ví dụ: trả 10, đúng 7 → Precision = 0,7. Kho có 20 liên quan, tìm được 7 → Recall = 0,35.
- Precision thấp: trả nhiều kết quả sai.
- Recall thấp: bỏ sót nhiều kết quả đúng.

**Thảo luận [C1-s75] — gợi ý trả lời:**
1. *Mô hình phát hiện bình luận độc hại có precision cao, recall thấp:* cái nó gắn cờ thì hầu như đúng là độc hại, nhưng **bỏ lọt nhiều** bình luận độc hại (false negative nhiều). Ngưỡng confidence đang đặt **quá cao**. Hạ ngưỡng → recall tăng, precision giảm. Chọn ngưỡng theo đường Precision-Recall trên tập validation.
2. *Tin giả: A (P 90%, R 60%) vs B (P 70%, R 85%):* tuỳ hành động sau khi phát hiện. Nếu **tự động gỡ bài**, chọn A (gỡ nhầm tin thật là tổn hại tự do ngôn luận và uy tín nền tảng). Nếu chỉ **gắn nhãn/đưa người kiểm duyệt**, chọn B (bỏ lọt tin giả lan truyền nguy hiểm hơn, còn con người sẽ lọc lại phần báo nhầm). Có thể so bằng F1 (A ≈ 0,72; B ≈ 0,77), nhưng quyết định thật dựa trên **chi phí của từng loại lỗi**.
3. *Spam: recall 95%, precision 60%:* gần như mọi spam đều bị chặn, nhưng **40% thư bị đưa vào spam là thư thật**. Người dùng có thể mất email quan trọng (hoá đơn, phỏng vấn). Thường **không chấp nhận được**: với lọc spam, false positive đắt hơn false negative.

### Recall@k [C1-s76]

$$\text{Recall@}k=\frac{|R\cap\text{Top}_k|}{|R|}$$

R = tập tài liệu liên quan (nhãn > 0). Ví dụ: top-5 = [G304 ✓, Basilisk ✓, chuột có dây ✗, Katar ✓, lót chuột ✗], kho có 4 tài liệu liên quan (G304, Basilisk, Katar, M331) → **Recall@5 = 3/4 = 0,75**.

> *"Không quan tâm thứ tự trong top k, chỉ quan tâm có GT trong top hay không → phù hợp cho tầng 1."*

### MRR [C1-s77]

$$\text{MRR}=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{\text{rank}_i}$$

rankᵢ là hạng của tài liệu **đúng đầu tiên** ở truy vấn thứ i.

| Truy vấn | Hạng đúng đầu tiên | RR |
|---|---|---|
| chuột gaming không dây pin lâu | 1 | 1,00 |
| bàn phím cơ im lặng | 2 | 0,50 |
| tai nghe chống ồn họp online | 4 | 0,25 |

MRR = (1 + 0,5 + 0,25)/3 = **0,583**.

> *"Phù hợp khi người dùng chỉ cần một kết quả đúng (tìm đúng sản phẩm, hỏi đáp). Chỉ nhìn kết quả đúng đầu tiên, bỏ qua toàn bộ phần còn lại."*

**Bài tập slide 78:** Q1 → {D1, D2}; BM25 xếp Q1: D7, D8, D5, D2, D1.
- Recall@2: top-2 = {D7, D8}, trúng 0/2 = **0,0**.
- Recall@5: top-5 chứa cả D2 và D1 = **1,0**.
- RR: tài liệu đúng đầu tiên (D2) ở hạng 4 → **0,25**.
- Bài học: cùng một danh sách, Recall@5 nói "hoàn hảo" còn Recall@2 và MRR nói "tệ". **Chọn k và metric theo tầng đang đo.**

### nDCG@k [C1-s79, Järvelin & Kekäläinen 2002]

Bốn bước:
1. **Gain:** nhãn liên quan phân bậc (0-3) tại mỗi hạng.
2. **Discount:** chia gain cho log₂(hạng + 1). Xếp càng sâu càng mất giá trị.
3. **DCG@k** = Σᵢ₌₁ᵏ gainᵢ / log₂(i+1).
4. **Chuẩn hoá:** nDCG@k = DCG@k / IDCG@k, với IDCG là DCG của **thứ tự lý tưởng** (xếp nhãn giảm dần).

Kết quả trong [0, 1], so sánh được giữa các truy vấn và các hệ. *"Là metric tiêu chuẩn cho tầng xếp hạng, thứ tự và mức liên quan đều quan trọng."*

**Tính tay (bổ sung), dùng top-5 của slide 76 với nhãn slide 73:** nhãn theo hạng = [3, 3, 0, 2, 0].

| Hạng i | log₂(i+1) | Gain | Gain/log |
|---|---|---|---|
| 1 | 1,000 | 3 | 3,000 |
| 2 | 1,585 | 3 | 1,893 |
| 3 | 2,000 | 0 | 0 |
| 4 | 2,322 | 2 | 0,861 |
| 5 | 2,585 | 0 | 0 |

DCG@5 = **5,754**. Thứ tự lý tưởng trong kho: [3, 3, 2, 1, 0] → IDCG@5 = 3 + 1,893 + 2/2 + 1/2,322 = 3 + 1,893 + 1 + 0,431 = **6,324**.
**nDCG@5 = 5,754 / 6,324 = 0,910.** Mất điểm vì "chuột có dây" chen vào hạng 3 đẩy Katar xuống hạng 4, và M331 (nhãn 1) không được lấy.

Ghi chú (bổ sung): một số thư viện dùng gain = 2^rel − 1 (nhấn mạnh tài liệu rất liên quan). Slide dùng gain = rel. Báo cáo cần ghi rõ dùng công thức nào.

**Tóm tắt chọn metric:**

| Metric | Đo gì | Nhạy thứ tự? | Dùng nhãn bậc? | Hợp cho |
|---|---|---|---|---|
| Recall@k | Có tìm đủ không | Không | Không | Tầng 1 |
| MRR | Kết quả đúng đầu tiên cao cỡ nào | Có (chỉ cái đầu) | Không | Hỏi đáp, tìm đúng 1 món |
| nDCG@k | Thứ tự + mức liên quan | Có | Có | Tầng 2 |

### Ablation [C1-s80]

> *"Nguyên tắc: mỗi thí nghiệm chỉ thay đổi một thành phần, giữ nguyên các điều kiện còn lại. So sánh trên cùng tập đánh giá, cùng metric, cùng tham số còn lại. Chênh lệch điểm số giữa hai cấu hình chính là đóng góp của thành phần bị tắt."*
> *"Ablation là bắt buộc trong nghiên cứu: bài toán ghép nhiều model hay thành phần chỉ thuyết phục khi tách được đóng góp từng phần."*

Ví dụ bảng ablation cho pipeline BM25 → Dense → RRF → Rerank:

| Cấu hình | BM25 | Dense | RRF | Rerank | So với dòng trên cho biết |
|---|---|---|---|---|---|
| BM25-only | ✓ | | | | baseline từ khoá |
| Dense-only | | ✓ | | | baseline ngữ nghĩa |
| Hybrid | ✓ | ✓ | ✓ | | đóng góp của fusion |
| Hybrid + Rerank | ✓ | ✓ | ✓ | ✓ | đóng góp của rerank |

Sai lầm hay gặp: vừa đổi model embedding vừa bật rerank rồi so với baseline → không biết cái nào tạo ra chênh lệch.

## I.14 Serving, API, vận hành [C1-s81 → s87]

**Ngân sách độ trễ** [C1-s81, s82]: gateway khoảng 15 ms + search khoảng 50 ms = **khoảng 65 ms**. Thêm rerank (lấy 100 ứng viên, chấm lại, trả top-10) khoảng 100 ms → **khoảng 165 ms**. Rerank là khoản tốn thời gian lớn nhất.

**API trả JSON** [C1-s83]:
```json
GET /api/v1/search?q=laptop chơi game&k=10
{
  "query": "laptop chơi game",
  "latency_ms": 142,
  "results": [
    {"id": 1, "ten": "Laptop Gaming Acer Nitro 5", "gia": 22990000, "score": 0.92}
  ]
}
```
- Endpoint có **version (/v1/)**: đổi schema không làm vỡ front end cũ.
- **k**: front end quyết định số kết quả.
- **score** kèm từng kết quả: front end tự quyết ngưỡng hiển thị.

> *"API là cổng giao tiếp front end và back end → đổi search service không ảnh hưởng front end."*

**Sequence diagram** [C1-s84]: Người dùng → Front end → API gateway (5 ms) → Tầng 1: BM25 + dense, RRF 200 ứng viên (35 ms) → Tầng 2: rerank 200 cặp, top-10 (190 ms) → lọc nghiệp vụ, trả JSON (15 ms). *"→ đo thời gian và báo cáo"*: báo cáo A1 nên có latency từng tầng.

**Hosting** [C1-s85, s86]:

| Self-hosting | Cloud (VM/container) | Dịch vụ ngoài (SaaS, API) |
|---|---|---|
| Máy chủ/GPU của đơn vị | Container trên VM thuê | Algolia, LLM API |
| Chủ động dữ liệu, tuỳ biến sâu | Scale ngang theo tải | Triển khai nhanh nhất |
| Tốn công vận hành, đầu tư lớn | Trả theo tài nguyên | Phụ thuộc nhà cung cấp, phí theo lượt |

Theo thành phần: chỉ mục + vector DB (Postgres + pgvector một máy; Elasticsearch cụm riêng khi lớn) · embedding (CPU đủ cho model nhỏ e5/bge; GPU khi cần thông lượng; hoặc API) · reranker (cross-encoder trên GPU hoặc Cohere Rerank). *"Có thể sử dụng Ollama local hoặc Groq cho LLM API call."* → trade-off **chất lượng, chi phí, độ trễ**.

**Vận hành sau triển khai** [C1-s87]:
- **Cache:** cache kết quả truy vấn phổ biến ở gateway; TTL ngắn; xoá cache khi chỉ mục cập nhật.
- **Giám sát:** đo **p95** riêng từng tầng (95% request nhanh hơn mức này; trung bình che giấu các request chậm); cảnh báo khi vượt ngân sách khoảng 300 ms; **log các truy vấn trả 0 kết quả** để phân tích.
- **Phiên bản chỉ mục:** build chỉ mục mới song song bản đang chạy, chuyển **alias** khi kiểm thử xong, rollback được (blue-green).
- **Sự cố:** timeout riêng từng tầng; quá tải thì **bỏ rerank, trả kết quả tầng 1** (graceful degradation); mã lỗi JSON thống nhất.

## I.15 Bài A1 [C1-s88 → s92]

**Thảo luận thiết kế [C1-s89] — gợi ý ngắn:**

*A. Review Google Maps chuỗi cà phê:*
1. Rating 4,2 chỉ là một con số. Embedding + gom cụm cho biết **khen/chê cái gì** (chờ lâu, ổ điện ít, nhân viên dễ thương) và **ở chi nhánh nào**.
2. Tiếng lóng Gen Z ("căng", "ổn áp", "xỉu"): model chuẩn có thể hiểu sai (ví dụ "căng" = tệ). Kiểm chứng: làm tập nhỏ có nhãn, đo cosine giữa câu lóng và câu chuẩn cùng nghĩa, xem cụm có đúng không.
3. "trà đào ngon nhưng chờ 30 phút" chứa hai ý trái chiều. Embed nguyên câu thì vector nằm lưng chừng, rơi sai cụm. Xử lý: **tách theo khía cạnh** (aspect-based: sản phẩm +, tốc độ −) trước khi embed.
4. 50 review một sao na ná nhau trong 2 ngày: embedding bắt được **gần trùng lặp** (cosine rất cao giữa các review) + mật độ thời gian bất thường → cờ nghi ngờ.

*B. Web tìm việc 200.000 tin:*
1. Thiết kế: trường có cấu trúc (lương, địa điểm, hình thức làm) lọc bằng SQL; query đi qua bước **trích xuất** ("12 củ" → lương ≥ 12 triệu, "gần Ba Son" → bán kính quanh toạ độ); phần còn lại hybrid BM25 + dense; rerank.
2. Tin giật tít "LƯƠNG KHỦNG 100 TRIỆU": vector bị kéo về các khái niệm lương cao, bằng cấp, nên hiện lên cho nhiều query không liên quan, làm giảm precision. Cần phát hiện spam/clickbait, embed **nội dung mô tả** thay vì tiêu đề, và phạt trong rerank.
3. Gợi ý part-time mãi dù đã tốt nghiệp: hồ sơ người dùng bị "đóng băng" theo một hành vi cũ. Cần **suy giảm theo thời gian** cho tín hiệu hành vi, ưu tiên hồ sơ khai báo, và cho người dùng tự sửa sở thích.

**Đề A1** [C1-s90]: xây hệ tìm ảnh bằng text và bằng ảnh.
1. Chọn dataset ảnh/text/metadata/multimodal.
2. Tìm kiếm ngữ nghĩa: input ảnh hoặc text tự do; output ảnh hoặc text.
3. Front end gọi API, hiển thị top-k.
4. Báo cáo ≤ 8 trang: dataset, phân tích kết quả tốt/xấu, **ablation các model**.
5. Slide và demo.

**Rubric A1** [C1-s91]:

| Tiêu chí | % | Để đạt điểm cao |
|---|---|---|
| Cài đặt đúng cơ chế | 30 | BM25, RRF, tìm ảnh/text chạy đúng, **tái lập được** |
| Đánh giá đúng phương pháp | 20 | Tập đánh giá hợp lệ **(kèm qrels ảnh)**, metric **đo đúng tầng** |
| Ablation có kiểm soát | 20 | Mỗi cấu hình khác **đúng một** thành phần, kết luận dựa trên số liệu |
| Error analysis và báo cáo | 20 | **Taxonomy** lỗi rõ, đề xuất cải tiến có căn cứ |
| Trả lời câu hỏi | 10 | |

"Metric đo đúng tầng" = Recall@k cho tầng truy hồi, nDCG/MRR cho tầng xếp hạng (I.12, I.13).

**Kiến trúc tổng quát** [C1-s92]: tầng dữ liệu và xử lý (kho tài liệu, hành vi & item, ETL & embedding) → tầng lưu trữ & chỉ mục (search index, vector DB, feature store) → tầng dịch vụ (semantic search, RAG pipeline, recommendation, tools, agent controller, memory) → API gateway → ứng dụng (search UI, chatbot, trang gợi ý, dashboard). A1 xây nhánh semantic search, A3 thêm RAG/agent/memory.

---

# PHẦN II — RETRIEVAL-AUGMENTED GENERATION (Chương 2)

## II.1 Ôn tập và vì sao cần tạo sinh [C2-s4, s5]

**Schema vector search trong DB quan hệ** [C2-s4]: bảng `chunks (id, doc_id, noi_dung, embedding vector(768), trang)`, `documents (id, tieu_de, dept_id, date, nguon)`, `departments`, `doc_access (doc_id, vai_tro)`.

```sql
SELECT c.noi_dung, d.tieu_de, c.trang
FROM chunks c
JOIN documents d   ON c.doc_id = d.id
JOIN departments p ON d.dept_id = p.id
JOIN doc_access a  ON a.doc_id = d.id
WHERE p.ten = 'Chăm sóc khách hàng'
  AND a.vai_tro = 'nhan_vien_cskh'
ORDER BY c.embedding <=> :q      -- <=> là khoảng cách cosine trong pgvector
LIMIT 5
```

- **Pre-filter:** WHERE thu hẹp ứng viên bằng metadata trước, rồi mới so vector.
- **Phân quyền:** JOIN `doc_access` đảm bảo chỉ truy hồi tài liệu người hỏi được phép đọc.
- **Trích nguồn:** tiêu đề và trang lấy từ `documents`, trả kèm câu trả lời.

> *"Cùng một câu hỏi, hai người ở hai vai trò khác nhau nhận về hai tập ngữ cảnh khác nhau → vector database thuần túy khó làm gọn bằng một câu lệnh duy nhất."* Đây là lý do slide nghiêng về Postgres + pgvector.

**Hệ tìm kiếm chỉ trả danh sách tài liệu** [C2-s5]. Với "quy định nghỉ phép năm nay là gì?", người dùng vẫn phải tự mở, đọc, tổng hợp. Hệ tìm kiếm không thể: (1) **tổng hợp nhiều nguồn**, (2) **diễn đạt thành câu trả lời**, (3) **hiểu câu hỏi tự nhiên** dài, nhiều ý, tham chiếu ngữ cảnh trước. → *"Cần bước tạo sinh (Generation) của LLM."*

## II.2 LLM — giới thiệu ngắn [C2-s6 → s16]

### Ngữ cảnh quyết định nghĩa [C2-s6]

"Alice deposited money in the **bank**" vs "The fisherman sat on the **bank** of the river" → *"Meaning of a word often depends on context."* Đây là động lực của cơ chế attention.

### Transformer và attention [C2-s7 → s9]

> Slide [C2-s8]: *"Simply put, attention allows the model to decide which other words (tokens) in the input are the most relevant to a given word when producing the next representation."*

Công thức (bổ sung, từ paper "Attention is all you need"):

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Hiểu nôm na: mỗi token tạo ra ba vector. **Query** ("tôi đang tìm gì"), **Key** ("tôi chứa gì"), **Value** ("nội dung tôi đưa ra"). Tích Q·K cho biết token này nên chú ý token kia bao nhiêu. Softmax biến thành trọng số cộng lại bằng 1. Biểu diễn mới = trung bình có trọng số của các Value. Chia √d_k để tích vô hướng không quá lớn khi số chiều cao (softmax bị bão hoà). "Multi-head" = chạy nhiều bộ Q, K, V song song, mỗi head học một loại quan hệ (cú pháp, đồng tham chiếu…).

Ví dụ [C2-s9, Vig 2019]: trong "The animal didn't cross the street because **it** was too scared", ở layer 6 head 0, token "it" chú ý mạnh vào "The animal". Model tự giải được đại từ chỉ ai.

**Encoder vs decoder** [C2-s7]:

| BERT (encoder-only, Google) | GPT (decoder-only, OpenAI) |
|---|---|
| Nhìn mọi token trong câu (**2 phía**) | Dùng **causal mask**, chỉ nhìn các token trước |
| Huấn luyện bằng đoán token bị che (masked LM) | Dự đoán token tiếp theo |
| Hiểu, phân loại, trích xuất → dùng làm **embedding, cross-encoder** | **Tạo sinh** văn bản |

Liên hệ: embedding model và cross-encoder ở Chương 1 thuộc họ encoder. LLM trong RAG (GPT-4, Claude, LLaMA, Mistral) là decoder-only [C2-s11]. Có thể xem trực quan tại bbycroft.net/llm.

### Token [C2-s10]

> *"Mô hình không làm việc với ký tự hay từ, mà với token: các mảnh văn bản đã được mã hóa thành số nguyên."*

"Quy định nghỉ phép năm nay?" → 7 token. Ước lượng: tiếng Anh 1 token ≈ 4 ký tự ≈ 0,75 từ. **Tiếng Việt có dấu tốn token hơn**, khoảng 1 token cho 2-3 ký tự.

Token là **đơn vị tính tiền** và **đơn vị đo context window**:

$$\text{Cost}=n_{in}\cdot p_{in}+n_{out}\cdot p_{out}$$

Ví dụ tính (bổ sung, đơn giá giả định): một lượt RAG có system 300 + context 5 chunk × 300 + câu hỏi 50 = 1.850 token vào, 250 token ra. Với p_in = 3 USD/triệu token, p_out = 15 USD/triệu: 1.850 × 3/10⁶ + 250 × 15/10⁶ = 0,00555 + 0,00375 ≈ **0,0093 USD/lượt**, khoảng 9,3 USD cho 1.000 câu hỏi. Để ý **context chiếm phần lớn token vào**: đó là lý do chunking, top-k và nén context (II.10) ảnh hưởng trực tiếp tới chi phí.

### LLM dự đoán token kế tiếp [C2-s12]

"Quy định nghỉ phép sẽ" → phân phối: thay 62%, được 18%, tăng 9%, hết 6%, khác 5%. *"Chọn một token theo phân phối → nối vào chuỗi đầu vào → lặp lại cho token kế tiếp."* Cả câu trả lời dài chỉ là vòng lặp này (tự hồi quy).

Các task ngôn ngữ LLM làm được [C2-s13]: tạo sinh, tóm tắt, suy luận nhiều bước (multi-hop), dịch và chuyển văn phong (T5 đưa mọi task về dạng text → text), in-context learning / chain-of-thought.

### Softmax và temperature [C2-s14]

$$P(w_t=v\mid w_{<t})=\frac{\exp(z_v/T)}{\sum_{u\in V}\exp(z_u/T)}$$

z_v là điểm thô (logit), V là toàn bộ từ vựng, T là temperature.

**Tính tay (bổ sung):** hai token có logit z = [2, 1].
- T = 1: e²/(e² + e¹) = 7,39/(7,39 + 2,72) = **0,73** vs 0,27.
- T = 0,5 (chia logit cho 0,5 → [4, 2]): e⁴/(e⁴ + e²) = 54,6/(54,6 + 7,39) = **0,88** vs 0,12.
- T = 2 (→ [1; 0,5]): 2,72/(2,72 + 1,65) = **0,62** vs 0,38.

T nhỏ làm phân phối **nhọn** (gần như luôn chọn token điểm cao nhất), T lớn làm **phẳng** (đa dạng hơn, dễ sai hơn). T → 0 tương đương greedy.

> Slide [C2-s14]: *"Với RAG, đặt temperature thấp (0 đến 0.3): cần câu trả lời bám tài liệu. Lưu ý: nhiều API mô hình suy luận đời 2025-2026 đã bỏ tham số này → tự ràng buộc bằng prompt."*

📌 **Câu hỏi slide: "Tại sao cùng prompt, kết quả vẫn có thể khác giữa các lần gọi?"**
1. **Lấy mẫu ngẫu nhiên:** T > 0, top-p (chỉ lấy mẫu trong nhóm token có tổng xác suất p) → mỗi lần rút có thể khác.
2. **Ngay cả T = 0 vẫn không đảm bảo giống nhau** (slide: *"không đảm bảo deterministic/reproducible"*): phép cộng số thực trên GPU không có tính kết hợp, thứ tự cộng thay đổi theo cách gom batch trên server, sai số nhỏ có thể lật thứ hạng hai token sát điểm nhau; và một token khác đi thì cả phần sau khác theo.
3. Nhà cung cấp **cập nhật model** phía sau cùng một tên API.

→ Hệ quả cho đánh giá: chạy nhiều lần hoặc cố định seed nếu API cho phép, và báo cáo cấu hình (model, temperature).

### Vì sao LLM tạo thông tin sai (hallucination) [C2-s15, s16]

> *"Toàn bộ kiến thức của LLM được nén vào hàng tỷ tham số trong lúc huấn luyện. Khi trả lời, mô hình không tra cứu tài liệu nào, chỉ dùng lại những gì đã nén."*

Hai vùng mù: sự kiện **sau knowledge cutoff**, và **dữ liệu riêng của tổ chức** (không có trong tập huấn luyện).

Chuỗi nguyên nhân [C2-s15, Huang et al. 2025]:
1. Câu hỏi cần dữ liệu model không được học
2. \+ Model vẫn phải tạo sinh token kế tiếp
3. \+ Pretraining gốc không thưởng cho câu "tôi không biết"
4. → Chọn chuỗi token **có xác suất cao nhất**, tức là câu **nghe hợp lý nhất**, không phải câu đúng.

Ví dụ [C2-s16]: hỏi chính sách đổi trả UITShop 2026 vs 2025 → LLM bịa "7 ngày, 2026 tăng lên 10 ngày cho thành viên". Hoặc "Thủ đô sao Hoả là Muskland". *"Mô hình không nói 'tôi không biết': vẫn phải tạo sinh token kế tiếp, nên chọn con số nghe có vẻ hợp lý."*

Ghi chú slide: AI hiện đại được triển khai như **hệ thống** (LLM + luật điều khiển + công cụ), ở đây chỉ xét LLM đứng một mình.

## II.3 Prompt engineering [C2-s18 → s24]

**Ba cần gạt** khi gọi LLM qua API mà không sửa được trọng số [C2-s18]:
1. **Nội dung prompt:** chỉ dẫn, ngữ cảnh, ví dụ, câu hỏi. *"Phần chính, và là phần RAG tự động hóa."*
2. **Tham số tạo sinh:** temperature, top-p, max tokens (RAG: temperature thấp).
3. **Lựa chọn model:** mạnh hơn hay rẻ hơn, theo chất lượng/chi phí/độ trễ.

> *"So với fine-tuning: không đổi trọng số, chỉ đổi đầu vào: rẻ hơn, nhanh hơn, thử được ngay."*

**Cấu trúc prompt tốt** [C2-s19, theo hướng dẫn của Anthropic]:

```
SYSTEM PROMPT
  Role         → Bạn là [vai trò/chuyên môn]
  Objective    → Mục tiêu tổng quát
  Behavior     → Nguyên tắc làm việc, mức độ chủ động, phong cách
  Tool_rules   → Khi nào dùng công cụ, khi nào hỏi lại người dùng
  Safety       → Hành động cần xác nhận trước

USER PROMPT
  Documents        → Nguồn tham chiếu, nội dung tài liệu   ← RAG chèn vào đây
  Context          → Bối cảnh, đối tượng, lý do
  Examples         → [input1][output1], [input2][output2]
  Instructions     → Các yêu cầu
  Output_format    → Định dạng, cấu trúc, độ dài, ngôn ngữ
  Success_criteria → Tiêu chí đánh giá kết quả
  Task             → Câu hỏi cụ thể
```

Thảo luận [C2-s20]: tự viết prompt cho một task công việc hàng ngày theo khung trên.

**Zero / one / few-shot** [C2-s21]: không ví dụ / một ví dụ / 2+ ví dụ trước câu hỏi. Ví dụ slide: zero-shot phân loại "I think the vacation was okay" ra "Positive" (sai), one-shot có mẫu thì ra "Neutral" (đúng).
- **In-context learning:** model học mẫu ngay trong prompt, không đổi trọng số.
- > *"Đánh đổi: few-shot cho kết quả ổn định hơn nhưng tốn token hơn, và số token đó cạnh tranh trực tiếp với ngữ cảnh mà RAG muốn chèn vào."*

**Chain-of-thought** [C2-s22, Wei et al. 2022]: yêu cầu model viết các bước trung gian ("23 × 10 = 230, 23 × 7 = 161, 230 + 161 = 391"). Hữu ích cho bài suy luận phức tạp, **tăng token và độ trễ**. Lưu ý slide: các bước sinh ra *"không nhất thiết phản ánh đầy đủ quá trình tính toán bên trong mô hình"*, tức không nên coi CoT là lời giải thích trung thực.

Vì sao CoT giúp (bổ sung): model sinh từng token dựa trên các token trước. Viết ra bước trung gian nghĩa là cho model thêm "giấy nháp", mỗi bước nhỏ dễ đoán đúng hơn một bước nhảy thẳng tới đáp án.

**Grounding** [C2-s23]: chèn tài liệu vào prompt kèm câu lệnh ràng buộc.
```
[Context] "Từ 01/2026, đổi trả điện thoại và laptop tăng từ 7 lên 14 ngày,
           phụ kiện tăng từ 15 lên 30 ngày." (nguồn: CS 05/2026, trang 2)
→ LLM: "Năm 2026 có hai thay đổi: ... [nguồn: CS 05/2026, trang 2]"
```
*"Câu trả lời bám vào tài liệu và kèm trích dẫn nguồn, nên người dùng kiểm chứng lại được."* Câu hỏi mở tiếp: *"Tài liệu nào trong 10000 tài liệu nội bộ nên được input vào prompt?"*

**Grounding thủ công không mở rộng được** [C2-s24]: (1) không biết chọn tài liệu nào, (2) không chèn hết được vì vượt context window, (3) không tự động hoá được. → **Cần bước truy hồi tự động = RAG.**

## II.4 Naive RAG [C2-s26 → s33]

### RAG 2020 [C2-s26, Lewis et al., NeurIPS 2020]

> *"Thay vì để LLM trả lời từ tham số đã học, ta truy hồi các đoạn liên quan từ một kho tri thức bên ngoài, đưa chúng vào prompt, rồi mới yêu cầu LLM tạo sinh câu trả lời."*

Kiến trúc gốc: **Retriever p_η** (non-parametric: query encoder q(x) + document index d(z), dùng **MIPS**) → **Generator p_θ** (parametric: BART) → marginalize trên các tài liệu z. Huấn luyện **end-to-end** (backprop qua query encoder và generator).

- **MIPS** (Maximum Inner Product Search): tìm các vector d(z) có tích vô hướng lớn nhất với q(x). Với vector đã chuẩn hoá thì MIPS = tìm cosine lớn nhất.
- **Non-parametric memory:** tri thức nằm trong kho tài liệu (đổi được mà không huấn luyện lại). **Parametric memory:** tri thức nén trong trọng số.
- **"Marginalize"** (bổ sung): model không chọn cứng một tài liệu. Xác suất câu trả lời = Σ_z P(z | x) · P(y | x, z), tức trung bình có trọng số theo độ liên quan của từng tài liệu.

📌 **"RAG 2020 = huấn luyện end-to-end. RAG today?"** Hiện nay RAG thường là **ghép module có sẵn, không huấn luyện chung**: embedding model đóng băng + vector DB + LLM gọi qua API, nối bằng prompt. Nhờ vậy đổi từng khối độc lập, nhưng retriever và generator không được tối ưu cho nhau.

### Công cụ [C2-s27]

Truy hồi (Chương 1) + Tạo sinh (Chương 2) = RAG. Bốn khối: (1) Embedding model: Sentence-Transformers, BGE-M3, Qwen3-Embedding, CLIP, SigLIP; (2) Thư viện/index: pgvector, FAISS, Chroma, Qdrant; (3) LLM: GPT, Claude, open source; (4) Framework: LangChain, LlamaIndex, **tự viết bằng Python**.

### RAG vs fine-tuning [C2-s28]

| | RAG | Fine-tuning | Long-context (bổ sung, cột slide để trống) |
|---|---|---|---|
| Tri thức thay đổi thường xuyên | ✓ (chỉ cần re-index) | ✗ (phải train lại) | ✓ nhưng đắt mỗi lượt |
| Cần trích dẫn nguồn | ✓ | ✗ (tri thức tan vào trọng số) | ✓ một phần |
| Kho riêng dung lượng lớn | ✓ | ✗ | ✗ (vượt context window, chi phí token) |
| Đổi phong cách/định dạng đầu ra | ✗ | ✓ | ✗ |

📌 Tóm lại: **RAG dạy model "biết gì", fine-tuning dạy model "nói thế nào".** Hai cách bổ sung cho nhau.

### Pipeline naive RAG [C2-s29]

```
① Câu hỏi q → ② Truy hồi (top-k đoạn gần nhất từ vector DB) → ③ Augment (chèn vào Context) → ④ Generate (LLM trả lời + trích dẫn)
```

```python
# OFFLINE (batch: ingestion, indexing) — chạy khi có tài liệu mới
for doc in documents:
    text = load(doc)                            # PDF, web, DB -> văn bản
    chunks = split(text, size=c, overlap=o)
    for ch in chunks:
        v = embed(ch)                           # vector d chiều
        index.add(v, ch, metadata(doc))         # ghi vào vector DB

# ONLINE (serving) — chạy mỗi lần người dùng hỏi
def answer(query):
    q = embed(query)                            # CÙNG model embedding với offline
    topk = index.search(q, k=5)                 # cosine, ANN
    ctx = concat(topk.texts)                    # augment
    prompt = template(system, ctx, query)
    return llm.generate(prompt)                 # generate
```

> *"Không vòng lặp, không tự đánh giá lại kết quả → naive RAG."*

📌 Hai chi tiết hay bị hỏi: query phải encode bằng **cùng model** với tài liệu (khác model thì hai vector nằm ở hai không gian khác nhau, cosine vô nghĩa); **metadata** đi kèm từ lúc index để lọc và trích dẫn.

### Data ingestion offline [C2-s30, s31]

Load → **Chunk** → Embed → Index. Chi tiết xử lý tài liệu:
1. **Collection:** .pdf, .docx, .html, CSDL, wiki. *"Mỗi định dạng cần một bộ đọc riêng, chất lượng khác nhau rõ rệt."*
2. **Parsing:** *"PDF nhiều cột, bảng và hình là chỗ dễ vỡ nhất khi parse."*
3. **Cleaning:** bỏ header/footer lặp, mục lục, ký tự lỗi, đoạn trùng giữa các file.
4. **Chuẩn hoá + metadata:** `tieu_de`, `phong_ban`, `hieu_luc`, `nguon`. *"dùng để lọc theo điều kiện và trích dẫn nguồn khi trả lời."*

> *"Lỗi ở bước này sẽ ảnh hưởng lớn đến chất lượng truy hồi."* (garbage in, garbage out)

**Thảo luận tiền xử lý ảnh [C2-s32]:** kho ảnh D, q1 = "xe 4 bánh", q2 = một ảnh. Gợi ý: với q1 text, dùng CLIP text encoder trực tiếp, hoặc sinh caption/tag cho từng ảnh lúc ingest (bằng VLM) rồi BM25 trên caption ("ô tô", "xe hơi", "xe tải"); "xe 4 bánh" là khái niệm gộp nên cần query expansion. Với q2 ảnh: CLIP image encoder, tiền xử lý giống lúc index (resize, pad, chuẩn hoá pixel), loại ảnh trùng bằng hash.

### Prompt RAG [C2-s33]

```
SYSTEM:
Bạn là trợ lý của UITShop. Chỉ trả lời dựa trên tài liệu trong phần CONTEXT.
Nếu tài liệu không có thông tin, hãy trả lời "Tôi không tìm thấy".
Luôn ghi rõ nguồn trích dẫn.
CONTEXT:
[1] "Từ 01/2026, đổi trả điện thoại, laptop tăng từ 7 lên 14 ngày." (CS 05/2026, tr.2)
[2] "Phụ kiện: thời hạn đổi trả tăng từ 15 lên 30 ngày." (CS 05/2026, tr.2)
QUESTION:
Chính sách đổi trả của UITShop năm 2026 khác gì so với năm 2025?
```

- **Ràng buộc:** chỉ dùng tài liệu được cấp · cho phép trả lời "không tìm thấy" (sửa đúng nguyên nhân 3 của hallucination ở II.2) · bắt buộc ghi nguồn. *"Giảm hallucination, nhưng không xóa hết."*
- **CONTEXT là động**, các phần khác là template cố định.

## II.5 Đánh giá RAG [C2-s34 → s40, RAGAS — Es et al. 2024]

**Chia đôi pipeline để biết lỗi nằm ở đâu** [C2-s34]:

```
Câu hỏi → Truy hồi → Ngữ cảnh top-k → LLM tạo sinh → Câu trả lời
└──── Chất lượng truy hồi ───┘       └───── Chất lượng câu trả lời ─────┘
      Context Recall                        Faithfulness
      Context Precision                     Answer Relevance
```

### Context Recall [C2-s35]

$$\text{Context Recall@}k=\frac{|S\cap Z_k|}{|S|}$$

S = tập các **ý (claim)** trong đáp án chuẩn, Z_k = top-k đoạn truy hồi. Đo *"bao nhiêu phần của đáp án chuẩn có mặt trong ngữ cảnh"*.

Ví dụ: "Laptop Asus TUF F15 giá bao nhiêu và bảo hành mấy năm?". Ground truth tách thành 3 ý: (1) giá KM 19.990.000đ, (2) giá niêm yết 22.990.000đ, (3) bảo hành 2 năm. Top-4 = [C14 giá TUF F15, C32 trả góp 0%, C38 bảo hành Asus 2 năm, C445 chuột Logitech sale]. Cả 3 ý đều tìm thấy (ý 1, 2 ở C14; ý 3 ở C38) → **Context Recall = 3/3 = 1**.

> 📌 *"Luôn kiểm tra Context Recall trước. Nếu ngữ cảnh không chứa câu trả lời thì mọi cải tiến sau đều vô nghĩa."*

### Context Precision [C2-s36]

Công thức đơn giản trên slide: Precision@k = |Z_k⁺| / k, với Z_k⁺ là số đoạn trong top-k thực sự liên quan. Cùng ví dụ: relevance theo hạng = [✓, ✗, ✓, ✗].

| k | Relevant? | Số relevant tới k | Precision@k |
|---|---|---|---|
| 1 | ✓ | 1 | 1/1 |
| 2 | ✗ | 1 | 1/2 |
| 3 | ✓ | 2 | 2/3 |
| 4 | ✗ | 2 | 2/4 |

- Công thức đơn giản: 2/4 = **0,5**.
- Cách RAGAS dùng (ví dụ trên slide): **chỉ lấy trung bình Precision@k tại các hạng có đoạn liên quan** = (1 + 2/3)/2 = **0,833**. Đây là Average Precision: đoạn liên quan xếp càng cao, điểm càng cao. Đổi C38 lên hạng 2 thì điểm = (1 + 1)/2 = 1.

*"Context Precision thấp nghĩa là prompt bị nhiễu: tốn token và làm LLM dễ lạc hướng."*

### Faithfulness và Answer Relevance [C2-s37]

$$\text{Faithfulness}=\frac{|C^+|}{|C|}\qquad \text{Answer Relevance}=\frac{1}{n}\sum_{i=1}^{n}\cos(\vec q,\vec{\hat q}_i)$$

- **Faithfulness:** C = các ý tách từ **câu trả lời của hệ thống**, C⁺ = các ý được ngữ cảnh xác nhận. = 1 → mọi ý truy được về tài liệu; < 1 → model tự thêm thông tin. Ví dụ (bổ sung): câu trả lời có 4 ý, 3 ý có trong context, ý "được trả góp 0% 12 tháng" không có → 3/4 = 0,75.
- **Answer Relevance:** cho LLM **sinh ngược n câu hỏi** từ câu trả lời, đo cosine với câu hỏi gốc. Câu trả lời đúng trọng tâm thì câu hỏi sinh ngược sẽ giống câu hỏi gốc. Thấp → *"câu trả lời đúng tài liệu nhưng lạc đề"*.

> 📌 *"Faithfulness đo hallucination. Đây là con số cần báo cáo khi chứng minh RAG có thực sự giảm hallucination so với LLM only."*

Phân biệt hay nhầm: **Context Recall** so **ground truth** với **context**; **Faithfulness** so **câu trả lời** với **context**. Một câu trả lời có thể faithful 100% nhưng vẫn sai, nếu context sai.

**Khung chẩn đoán kết hợp (bổ sung):**

| Context Recall | Faithfulness | Chẩn đoán | Sửa ở đâu |
|---|---|---|---|
| Thấp | – | Lỗi **truy hồi** | chunking, embedding, hybrid, query rewriting |
| Cao | Thấp | Lỗi **tạo sinh** (bịa) | prompt, model, temperature, nén context |
| Cao | Cao, nhưng Answer Relevance thấp | Trả lời lạc đề | prompt, phân rã câu hỏi |

### Kiểm thêm ngoài RAGAS [C2-s38]

| Khả năng | Mô tả | Cách test |
|---|---|---|
| Noise robustness | Top-k luôn lẫn đoạn không liên quan, vẫn phải rút đúng ý | Chèn thêm đoạn nhiễu vào context rồi đo lại |
| Negative rejection | Kho không có câu trả lời thì nói không tìm thấy | Hỏi câu chắc chắn ngoài kho |
| Information integration | Ghép dữ kiện từ nhiều đoạn | Câu hỏi so sánh cần ≥ 2 tài liệu |
| Counterfactual robustness | Tài liệu lỗi thời hoặc mâu thuẫn | Để bản cũ và bản mới cùng trong kho |

### Sinh tập đánh giá từ tài liệu [C2-s39, ARES — Saad-Falcon et al. 2024]

1. **Sinh câu hỏi từ chunk:** LLM đặt 1-2 câu hỏi mà chunk trả lời được; chunk nguồn thành đáp án → qrels sơ bộ không cần gán tay (RAGAS có sẵn).
2. **LLM-as-judge:** LLM khác chấm theo context relevance, faithfulness, answer relevance. ARES huấn luyện judge nhẹ trên dữ liệu sinh ra.
3. **Human check:** rút ngẫu nhiên 30-50 câu cho người có chuyên môn soát. *"không thể bỏ bước này"*, vì đây là mốc để biết judge có đáng tin không.

> *"Cảnh báo: tập sinh tự động thiên về câu hỏi mà hệ vốn trả lời tốt → Bổ sung câu hỏi thật từ log người dùng."*

### Truy hồi là trần của RAG [C2-s40]

> *"Vì naive RAG dùng lại nguyên cơ chế truy hồi và chỉ thêm bước tạo sinh, chất lượng câu trả lời bị chặn trên bởi chất lượng của top-k đoạn đưa vào prompt."*

Việc cần làm: đầu tư truy hồi trước (embedding, chunking, hybrid, rerank) · đo Context Recall trước khi chỉnh prompt · *"Chọn sai chunk size hoặc k thì LLM mạnh đến đâu cũng không bù được."*

## II.6 Các điểm lỗi và Advanced RAG [C2-s41 → s43]

**Thảo luận RAG đa phương thức [C2-s41] — gợi ý:**
1. *Khách gửi ảnh sản phẩm lỗi + câu hỏi bảo hành:* đặt vision-language model (VLM) ở **tiền truy hồi**: VLM mô tả ảnh ("màn hình laptop có vết sọc dọc") → ghép vào câu hỏi → truy hồi chính sách bảo hành. Hoặc embed ảnh bằng CLIP để tìm sản phẩm/lỗi tương tự. Ở bước **tạo sinh**, dùng LLM đa phương thức nhận cả ảnh lẫn context.
2. *Tài liệu có bảng và hình:* không cắt ngang bảng; chuyển bảng sang markdown hoặc sinh mô tả text để embed, metadata giữ tham chiếu tới bảng gốc để đưa nguyên bảng vào prompt (xem [C2-s56]).
3. *Kho gồm text và ảnh sản phẩm:* truy hồi hai nhánh (text index + CLIP index), gộp bằng RRF. Prompt ghép: text chunk đánh số [1..n], ảnh đưa vào LLM đa phương thức dưới dạng ảnh kèm nhãn [ảnh k], hoặc thay bằng caption nếu LLM chỉ đọc text.

**Thảo luận các vấn đề của RAG [C2-s42] (bổ sung, gom theo pipeline):** parse sai (PDF hai cột) → chunk cắt ngang ý → embedding không bắt mã số → query mơ hồ → top-k lẫn tài liệu hết hiệu lực → đoạn đúng nằm giữa context bị bỏ qua (lost in the middle) → LLM bịa ngoài context → lộ dữ liệu, prompt injection → chậm, tốn tiền.

**Bản đồ Advanced RAG** [C2-s43, Kimothi 2024] — 4 vị trí tối ưu quanh pipeline naive:

| Vị trí | Kỹ thuật | Mục trong tài liệu này |
|---|---|---|
| **Index Optimization** (offline) | chunk theo cấu trúc, parent-child, thêm metadata, mô tả bảng và hình | II.7 |
| **Pre-retrieval** (xử lý query) | viết lại truy vấn, mở rộng từ đồng nghĩa, HyDE, phân rã câu hỏi | II.8 |
| **Retrieval Strategies** | hybrid BM25 + vector, lọc metadata trước, chọn lại k | II.9 |
| **Post-retrieval** | rerank cross-encoder, MMR đa dạng, nén ngữ cảnh, sắp thứ tự trong prompt | II.10 |

Chuỗi tên gọi: **Rewrite → Retrieve → Rerank → Read.**

## II.7 Tối ưu ingestion / indexing [C2-s45 → s62]

### Thảo luận chunking [C2-s45] — gợi ý

Ba tài liệu trên slide cần ba cách chunk khác nhau:
- **Quy chế học bổng** (có điều khoản đánh số + bảng xếp loại): chunk theo điều/khoản; **giữ nguyên bảng trong một chunk** cùng điều chứa nó, kèm tiêu đề điều.
- **Bài báo** về giảm thuế (văn xuôi liền mạch + ảnh minh hoạ): semantic chunking hoặc theo đoạn; gắn tiêu đề bài + ngày vào từng chunk; ảnh xử lý riêng (caption).
- **Chính sách UITShop** (Chương → Điều → Bảng): chunk theo cấu trúc, metadata chương/điều.

### Lỗi parsing [C2-s46]

| Định dạng | Vấn đề |
|---|---|
| PDF | Layout hai cột, header/footer làm sai thứ tự đọc: parse thành "câu 1, câu 4, câu 2, câu 5…" (trộn hai cột) |
| HTML | Menu, quảng cáo lẫn vào nội dung |
| DOCX / bảng | Flatten thành text, mất cấu trúc hàng cột |
| Scan / ảnh | OCR phụ thuộc chất lượng bản gốc |

Công cụ được nêu: Firecrawl AnyDoc (tài liệu → markdown), NVIDIA Nemotron.

### Nhược điểm chunking [C2-s47]

Bảng "Thời hạn đổi trả theo nhóm hàng **năm 2026**" bị ranh giới chunk cắt ngang:
- Chunk A có tiêu đề "năm 2026".
- Chunk B chỉ còn "Laptop 14 ngày", **mất năm áp dụng** → embedding không mang thông tin năm → query về 2026 không khớp chunk B, hoặc khớp nhầm bản 2025.

Đây là **ví dụ chủ đạo** của cả phần II.7; các kỹ thuật sau đều là cách sửa nó.

### Các chiến lược chunking

**1. Fixed-size** [C2-s48]: cắt cứng theo số token, thêm **overlap 10-20%**.
- Ưu: rẻ, nhanh, dễ cài; là **baseline** cho các phương án khác.
- Nhược: chỉ đếm độ dài, không xét nội dung. *"Overlap chỉ **giảm** tổn thất ở ranh giới, không xử lý tận gốc."*

**2. Semantic chunking** [C2-s49]: (1) embed từng câu, (2) đo tương đồng giữa hai câu liền kề, (3) **cắt tại chỗ tương đồng tụt xuống** (chủ đề đổi). Trade-off: thêm chi phí embedding lúc ingest, đổi lấy chunk không bị cắt giữa ý. Hợp với văn bản trôi chảy không có cấu trúc (bài viết, biên bản họp).

**3. Theo cấu trúc** [C2-s50]: chia theo heading/điều/mục, **không cắt ngang bảng** ("Điều 5 + Bảng 2" gộp một chunk). Metadata có sẵn: số điều, chương, tiêu đề.

**4. Parent-child** [C2-s51]:
```
Câu hỏi "laptop đổi trả mấy ngày"
  → khớp CHILD nhỏ "Laptop: đổi trả 14 ngày"           (vector của đoạn nhỏ sát query hơn)
  → tra ngược lên PARENT qua parent_id trong metadata
  → đưa cả PARENT (Điều 5 + Bảng 2, có "từ 01/01/2026") vào LLM   (đủ ngữ cảnh)
```
Giải quyết mâu thuẫn: **chunk nhỏ thì truy hồi chính xác, chunk lớn thì đủ ngữ cảnh.** Parent-child dùng chunk nhỏ để tìm, chunk lớn để đọc.

**5. RAPTOR** [C2-s52, Sarthi et al. ICLR 2024]: gom cụm các chunk gần nghĩa → **LLM tóm tắt** mỗi cụm thành node cha → lặp đệ quy thành cây nhiều tầng → **truy hồi trên mọi tầng** (câu hỏi chi tiết trúng leaf, câu hỏi tổng hợp trúng node cha). Hợp tài liệu dài, câu hỏi tổng hợp (báo cáo, hợp đồng). Chi phí build: một lượt LLM cho mỗi node tóm tắt. Cùng họ ý tưởng với "community" của GraphRAG.

**Chọn theo loại tài liệu** [C2-s53]:

| Loại tài liệu | Gợi ý khởi đầu | Ghi chú |
|---|---|---|
| FAQ, hỏi đáp ngắn | 128-256 token | Mỗi cặp hỏi-đáp một chunk |
| Quy định, hợp đồng | Theo điều khoản (256-512) | Giữ bảng chung chunk với điều |
| Báo cáo, tài liệu kỹ thuật dài | 512-1024 + parent-child | Overlap 10-15% nếu cắt cứng |
| Email, hội thoại | Theo luồng trao đổi | Metadata người gửi, ngày |

> 📌 *"Đây là gợi ý bắt đầu develop, không phải đáp án cho mọi document. → đo Recall@k và nDCG trên tập đánh giá của chính mình trước khi chốt. → Kích thước chunk là siêu tham số đầu tiên nên đưa vào ablation study."*

**Kích thước chunk** [C2-s58]:
- **Quá lớn:** 1 vector = trung bình của nhiều ý → truy hồi kém chính xác.
- **Vừa:** 1 vector = 1 ý trọn vẹn, overlap 10-20% ở ranh giới.
- **Quá nhỏ:** một ý bị chia nhiều mảnh, câu rời khỏi ngữ cảnh.

Slide cũng liệt kê 9 chiến lược: fixed-size, sentence-based, semantic, recursive, sliding window, hierarchical, topic-based, modality-specific, agentic. *"Chunking là siêu tham số, không phải hằng số."*

### Làm giàu chunk

**Context-enriched chunking / Contextual Retrieval** [C2-s54, Anthropic 2024]: dùng LLM viết một câu ngữ cảnh gắn vào đầu mỗi chunk trước khi embed.
- Raw: "Laptop: 14 ngày kể từ khi nhận hàng."
- Sau: "Trích Bảng 2, Điều 5, Chính sách đổi trả CS 05/2026 của UITShop, hiệu lực từ 01/01/2026. Laptop: 14 ngày kể từ khi nhận hàng."
- Giá: thêm lượt gọi LLM lúc ingest, chunk dài hơn (tốn dung lượng, token), dựng lại chỉ mục lâu hơn.

**Metadata filter** [C2-s55]: mỗi chunk có `{"tai_lieu": "CS 05/2026", "date": "03-06-2026", "loai": "chinh_sach", "dieu": 5, "hieu_luc": true}`. Câu hỏi có "2026" → lọc tài liệu năm 2026 → vector search trong tập đã lọc. SQL thêm `AND d.date >= '01-01-2026'`. Tách dứt điểm 2025 và 2026, thứ mà embedding làm không nổi (giống lỗi 5050/5060).

**Bảng và hình** [C2-s56]:
- Bảng gốc → LLM sinh **mô tả text** lúc ingest ("Bảng so sánh thời hạn đổi trả 2025 và 2026: điện thoại và laptop tăng từ 7 lên 14 ngày…") → **embed bản mô tả** để truy hồi.
- Metadata tham chiếu bảng gốc: khi trả lời, **đưa nguyên bảng** vào prompt thay vì bản tóm tắt (tránh sai số do tóm tắt).
- Hình, sơ đồ: vision model sinh mô tả → embed, hoặc embed ảnh trực tiếp.

**Metadata augmentation** [C2-s57]:
- Có sẵn lúc load: tên, số hiệu, ngày ban hành/hiệu lực, phòng ban, tác giả, loại, đường dẫn.
- LLM sinh thêm: tóm tắt một câu, chủ đề, nhóm hàng được nhắc, mức độ nhạy cảm (tốn 1 lượt LLM/chunk).
- Dùng để: lọc trước khi so vector · định tuyến sang đúng nguồn · hiện trích dẫn · dùng tóm tắt thay chunk thô để embed.

### Bảo mật và vận hành index

**Che dữ liệu nhạy cảm** [C2-s59]:
> *"Chỉ mục là một bản sao của kho tài liệu. Nếu không có cơ chế phân quyền phù hợp, dữ liệu nhạy cảm trong index có nguy cơ bị truy hồi ngoài ý muốn."*

| Cách | Làm gì | Khi nào |
|---|---|---|
| Redaction | Xoá hẳn trường nhạy cảm | Thông tin không cần cho việc trả lời |
| Masking | "Nguyễn Văn A" → "[TÊN KHÁCH]", "0901234567" → "[SĐT]" | Giữ cấu trúc câu, LLM vẫn hiểu |
| Khoá quyền | Giữ nguyên, gắn mức nhạy cảm vào metadata, lọc theo quyền lúc truy hồi | Dữ liệu cần dùng nhưng không phải ai cũng được xem |

> *"Ưu tiên loại bỏ hoặc pseudonymize dữ liệu không cần thiết càng sớm càng tốt; với dữ liệu cần giữ phải có access control ở retrieval. Ngoài ra, còn phải chặn data poisoning: tài liệu độc hại chèn vào kho sẽ làm lệch truy hồi về sau."*

**Chọn embedding model** [C2-s60]: Cloud API (chất lượng cao, trả theo lượt, dữ liệu rời hạ tầng) · self-host BGE/E5 (khá đến cao, tự vận hành GPU, dữ liệu ở lại nội bộ; tiếng Việt: BGE-M3, mE5, Qwen3-Embedding) · fine-tune nhẹ (tốt trên miền hẹp nếu có dữ liệu). *"Quyết định cuối dựa trên tập đánh giá của chính bài toán."* 📌 **Đổi embedding model thì phải re-embed toàn bộ** hoặc migrate sang index mới.

**Tài liệu đổi → chỉ mục phải đổi** [C2-s61]: (1) phát hiện thay đổi (hash, ngày sửa) → (2) parse + chunk lại phần đổi → (3) re-embed → (4) ghi đè theo `doc_id + version` → (5) gỡ bản cũ khỏi truy hồi.
- **Incremental** cho cập nhật hằng ngày; **full rebuild** theo chu kỳ dài hoặc khi đổi model embedding.
- **Version hoá:** bản 2026 thay bản 2025 nhưng bản cũ vẫn giữ để đối chiếu, chỉ đánh dấu hết hiệu lực trong metadata.
- *"Thiếu cơ chế cập nhật, hệ trả lời bằng tài liệu hết hiệu lực: đúng kịch bản lỗi của ví dụ chủ đạo."*

**Quy trình ingestion tổng kết** [C2-s62]: thu thập → làm sạch (bỏ header/footer, chuẩn hoá unicode) → chunking (200-500 token, giữ ranh giới mục) → embedding → lưu + metadata (nguồn, ngày, quyền). Một dòng bảng chunks: `chunk_id, doc_id, section, updated, acl, text, embedding`. Chạy định kỳ hoặc theo sự kiện; **tách khỏi online** (lỗi ingestion không làm sập tuyến trả lời). *"Chất lượng chunk quyết định trần chất lượng truy hồi."*

## II.8 Tối ưu truy vấn (pre-retrieval) [C2-s64 → s71]

**Truy vấn người dùng thường không tốt** [C2-s64]:
- "ĐT laptop 2026?": quá ngắn, viết tắt ("ĐT" = đổi trả hay điện thoại?).
- "cái chính sách mới ấy áp dụng chưa": mơ hồ, thiếu chủ thể.
- "đổi trả 2026 khác 2025 chỗ nào, laptop thì sao": hai câu hỏi trong một.

> *"Khoảng cách văn phong: câu hỏi là văn nói, tài liệu là văn quy phạm → có thể xuất hiện query–document representation gap."*

**1. Query rewriting** [C2-s66]: LLM (model nhỏ, rẻ) viết lại trước khi tìm. "ĐT laptop 2026?" → "thời hạn đổi trả laptop theo chính sách năm 2026". Mở rộng viết tắt, thêm chủ thể, chuẩn hoá về văn phong tài liệu. Trong hội thoại nhiều lượt: viết lại kèm ngữ cảnh ("nó" là gì, "năm ngoái" là năm nào). *"Viết lại giúp câu rõ hơn, nhưng từ vựng vẫn có thể lệch với tài liệu."*

**2. Query expansion** [C2-s67]: "đổi trả" → thêm "hoàn trả hàng", "trả hàng hoàn tiền", "return, refund", "đổi sản phẩm" (LLM sinh hoặc từ điển miền). Đặc biệt hiệu quả với **BM25** (sửa vocabulary mismatch). Phải kiểm soát mức mở rộng: thêm từ sai chủ đề làm nhiễu.

**3. HyDE** (Hypothetical Document Embeddings) [C2-s68, Gao et al. ACL 2023]:
```python
doan = llm("Viết đoạn trả lời giả định cho: " + cau_hoi)
chunks = vector_db.top_k(embed(doan), k=5)   # embed đoạn giả định, không embed câu hỏi
```
Đoạn giả định *có thể sai chi tiết* ("…đổi trả laptop trong **30** ngày…", thực tế là 14) nhưng không sao: nó chỉ dùng để **tìm**, không dùng để trả lời. *"Mục tiêu: tạo representation có khả năng nằm gần tài liệu liên quan hơn so với embedding trực tiếp câu hỏi."* Lý do: đoạn giả định có văn phong và từ vựng giống tài liệu (câu trả lời giống câu trả lời hơn là câu hỏi giống câu trả lời).

**4. Multi-query + RRF (RAG-Fusion)** [C2-s69]: LLM sinh 3 biến thể ("thời hạn đổi trả laptop 2026", "chính sách hoàn trả mới", "thay đổi quy định đổi trả 2026") → 3 lượt truy hồi song song → gộp bằng RRF (công thức ở I.11). Ví dụ slide: hạng 1 và hạng 3 ở hai danh sách → 1/61 + 1/63 ≈ 0,032. *"Chỉ dùng thứ hạng nên gộp được BM25 và vector dù hai thang điểm khác nhau."*

**5. Phân rã câu hỏi** [C2-s70]: "Chính sách đổi trả 2026 khác gì 2025?" → câu con A (2026) + câu con B (2025) → truy hồi riêng → Context A (CS 05/2026), Context B (CS 03/2025) → LLM tổng hợp so sánh. Vì sao cần: một lượt truy hồi với câu hỏi gốc dễ chỉ lấy được một phía (toàn chunk 2026).

**Chọn kỹ thuật** [C2-s71] (slide để trống, đáp án):

| Đặc điểm truy vấn | Kỹ thuật |
|---|---|
| Viết tắt, cụt, thiếu chủ thể, tham chiếu lượt trước | **Query rewriting** |
| Tài liệu dùng từ khác người hỏi (BM25 không khớp) | **Query expansion** |
| Câu hỏi ngắn văn nói, tài liệu dài văn quy phạm | **HyDE** |
| Một ý nhiều cách diễn đạt, cần recall cao | **Multi-query + RRF** |
| Nhiều câu hỏi con, so sánh nhiều nguồn | **Phân rã câu hỏi** |

Tất cả đều tốn thêm ít nhất một lượt gọi LLM trước khi truy hồi (tăng độ trễ, chi phí). Đưa vào ablation để chứng minh đáng tiền.

## II.9 Tối ưu truy hồi [C2-s73, s74]

**Truy vấn mới với pgvector** [C2-s73]:
```sql
SELECT c.text, c.doc_id
FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
WHERE d.hieu_luc = true
  AND c.nhom_hang = 'laptop'
ORDER BY c.embedding <=> :q LIMIT 2
```
Kết quả: c101 laptop 2026 (khoảng cách 0,06 ✓) · c102 phụ kiện 2026 (**bị lọc** vì nhom_hang) · c203 laptop 2025 (**bị lọc** vì hieu_luc). Câu trả lời: "Laptop mua tại UITShop được đổi trả trong 14 ngày… Nguồn: [CS 05/2026, Điều 5]."

**Các biến thể truy hồi** [C2-s74]:

| Biến thể | Dùng cột/bảng | Khác gì khi truy vấn |
|---|---|---|
| Vector thuần | embedding | Chỉ ORDER BY khoảng cách |
| Metadata filter | embedding + nam, nhom_hang, hieu_luc | Thêm WHERE trước khi so |
| Hybrid BM25 + vector | thêm chỉ mục từ khoá trên cột text | Hai danh sách, gộp RRF |
| Parent-child | thêm parent_id, bảng parents | Khớp child, JOIN lấy parent |
| **Text2SQL** | bảng nghiệp vụ (orders, products) | Không đọc bảng chunks; LLM sinh SQL |

## II.10 Tối ưu hậu truy hồi [C2-s76 → s78]

**Rerank bằng cross-encoder** [C2-s76]: Vector DB + BM25 → top-50 (tầng 1, recall) → cross-encoder chấm lại từng cặp → top-5 (tầng 2, precision) → LLM. Đây là kiến trúc 2 tầng của Chương 1 đặt vào trong RAG. *"Chi phí rerank thường nhỏ hơn số token LLM tiết kiệm được nhờ context gọn và đúng."*

**Lost in the middle** [C2-s77, Liu et al. TACL 2024]: 20 tài liệu (khoảng 4K token), đổi vị trí tài liệu chứa đáp án. Độ chính xác của gpt-3.5-turbo **cao nhất khi đáp án ở vị trí 1 (khoảng 75%)**, tụt xuống **khoảng 54% ở giữa (vị trí khoảng 10)**, hồi lại khoảng 63% ở vị trí cuối. Đường đồ thị hình chữ U. Mức closed-book (không đưa tài liệu) khoảng 56%, nghĩa là **đặt đáp án ở giữa còn tệ hơn không đưa tài liệu nào.**
- Đặt đoạn quan trọng nhất ở **đầu hoặc cuối** context.
- Bớt đoạn thừa thay vì lấp đầy context window.
- Dùng thứ hạng rerank để sắp xếp có chủ đích.

📌 Hệ quả: **context dài hơn không tự động tốt hơn.** Chọn k là một đánh đổi giữa recall (k lớn) và nhiễu/lost-in-middle/chi phí (k lớn).

**Nén context** [C2-s78]: 5 chunk (Điều 5 + Bảng 2 bản 2026, Điều 5 bản 2025, quy trình khiếu nại, hướng dẫn trả hàng, lịch sử sửa đổi) → bộ lọc nén → chỉ giữ các câu về thời hạn đổi trả 2025 và 2026 (khoảng **40% token**).
- **Extractive:** chấm điểm và giữ câu liên quan (an toàn, giữ nguyên văn).
- **Abstractive:** LLM tóm tắt lại (rủi ro sai lệch cao hơn).
- Trade-off: thêm một bước, có thể cắt nhầm câu cần thiết → *"Đo faithfulness khi bật, bằng RAGAS hoặc LLM-as-judge."*

**MMR** (Maximal Marginal Relevance, nêu ở [C2-s43], bổ sung): chọn lần lượt từng chunk, mỗi lần chọn chunk vừa **liên quan** tới query vừa **khác** các chunk đã chọn: MMR = λ·sim(q, d) − (1 − λ)·max sim(d, đã chọn). Tránh đưa 5 chunk gần giống nhau vào prompt.

## II.11 Cung cấp RAG qua API [C2-s80 → s86]

**Mọi client gọi qua API** [C2-s80]: web app, mobile, hệ khác (ERP, chatbot) → **API** (REST/FastAPI, xác thực, phân quyền) → **RAG service** → vector DB + LLM.
- Xác thực, phân quyền theo vai trò, giới hạn tần suất (rate limit).
- *"Client không biết gì về chunking hay mô hình: đổi embedding, đổi LLM không phải sửa app."*
- Mọi truy vấn qua một điểm: log và giám sát tập trung.

**Endpoint** [C2-s81]:

| Endpoint | Vai trò |
|---|---|
| `POST /documents` | Nạp tài liệu mới, **xử lý nền**, trả `job_id` (ingest lâu, không bắt client chờ) |
| `GET /documents/{id}/status` | Theo dõi tiến độ ingest |
| `POST /query` | Vào câu hỏi, ra câu trả lời + trích dẫn |
| `POST /feedback` | Người dùng chấm câu trả lời, làm giàu tập đánh giá |

```json
{
  "answer": "Năm 2026 thời hạn đổi trả tăng...",
  "citations": ["CS 05/2026, Điều 5", "CS 03/2025, Điều 5"],
  "usage": {"tokens": 1840, "latency_ms": 2900}
}
```

**Sequence POST /query** [C2-s82]: Client → API (xác thực, chuyển tiếp) → RAG service → Vector DB (truy hồi top-k, **filter quyền**) → trả chunks + metadata → LLM (prompt: câu hỏi + context) → **stream token** → client nhận qua **SSE** (Server-Sent Events) từng phần câu trả lời + trích dẫn. Streaming làm người dùng thấy chữ đầu tiên sớm (time-to-first-token thấp) dù cả câu trả lời mất khoảng 3 giây.

**Caching** [C2-s83]:
1. **Embedding cache:** văn bản đã embed không embed lại; khoá là hash nội dung.
2. **Semantic cache:** câu hỏi gần giống câu đã trả lời → trả kết quả cũ (ngưỡng tương đồng).
3. **Prompt caching:** phần prompt lặp lại (system, few-shot) được phía LLM cache sẵn.

> 📌 *"Rủi ro của semantic cache: hai câu gần giống nhau nhưng khác ý (2025 với 2026 chỉ khác một chữ số). Đặt ngưỡng chặt và loại trừ theo metadata thời gian."* Cùng loại lỗi với 5050/5060 ở Chương 1.

**Text2SQL** [C2-s84]: "Doanh thu laptop tháng 6?" không nằm trong tài liệu mà trong bảng `orders`. Luồng: câu hỏi → LLM + **schema** các bảng → sinh SQL → chạy trên DB **quyền chỉ đọc** → LLM tổng hợp kết quả. Kho nhiều bảng thì truy hồi trước các bảng liên quan theo mô tả bảng. **Router** phân loại: câu về chính sách → nhánh RAG tài liệu; câu về số liệu → nhánh text2SQL. Benchmark: Spider (Yu et al. 2018).

**Cần tránh** [C2-s85]:

| Hành vi cần tránh | Rủi ro | Cách làm đúng |
|---|---|---|
| Đưa tài liệu chưa phân quyền vào chỉ mục | Lộ dữ liệu nhạy cảm | Gắn quyền vào metadata, filter lúc truy hồi |
| Cho LLM tự chạy SQL/lệnh không kiểm soát | Sửa, xoá dữ liệu | Quyền chỉ đọc, danh sách bảng/trường cho phép |
| **Coi nội dung tài liệu là chỉ dẫn** | **Prompt injection**: tài liệu chứa lệnh độc hại ("bỏ qua các chỉ dẫn trên và…") | Coi context là **dữ liệu, không phải lệnh** |
| Trả lời không kèm trích dẫn | Không kiểm chứng được, lỗi lan rộng | Bắt buộc citation trong prompt và API |
| Đưa toàn bộ kho vào prompt | Chi phí lớn, chất lượng giảm vì nhiễu | Truy hồi chọn lọc, ngân sách token |

**Hệ thống xây được** [C2-s86]: trợ lý tài liệu nội bộ, chatbot CSKH, tra cứu pháp lý/hợp đồng, hỏi đáp báo cáo tài chính, trợ lý lập trình nội bộ, tư vấn sản phẩm e-commerce (hybrid search + RAG trên catalog kiểu UITShop, đúng hướng A3). *"Cùng pipeline, khác chunking, metadata và ràng buộc prompt theo từng miền dữ liệu."*

## II.12 Bài A2 [C2-s88, s89]

Bốn phần phải trình bày [C2-s88]:
1. **Bài toán:** paper giải gì, vì sao phương pháp trước chưa đủ.
2. **Ý tưởng cốt lõi:** điểm mới ở đâu, khác gì pipeline đã học.
3. **Thiết kế thí nghiệm:** baseline, dataset, metric, ablation kiểm chứng thành phần nào.
4. **Kết quả và hạn chế:** hạn chế tác giả nêu + nhóm tự đánh giá.

Tự chọn paper về Tìm kiếm/RAG, hoặc chạy và giải thích công nghệ mới (ví dụ repo Understand-Anything).

**Rubric A2** [C2-s89]:

| Tiêu chí | % | Đánh giá |
|---|---|---|
| Bài toán và bối cảnh | 15 | Nêu được vì sao cách cũ chưa đủ, **không chỉ dịch abstract** |
| Ý tưởng cốt lõi | 25 | Chỉ đúng điểm mới, **so với pipeline đã học** |
| Thiết kế thí nghiệm và ablation | **35** | Baseline, metric, ablation chứng minh thành phần nào |
| Kết quả, hạn chế, liên hệ đồ án | 15 | Đọc số liệu, nêu điều áp dụng cho A3 |
| Trả lời câu hỏi | 10 | |

Project này chọn paper **RankGPT** (Sun et al., EMNLP 2023, "Is ChatGPT Good at Search?"), xem [BAO_CAO_A2.md](BAO_CAO_A2.md). Nối vào lý thuyết: paper nằm ở vị trí **tầng 2 / post-retrieval rerank** (I.12, II.10). Điểm mới là dùng LLM sinh **hoán vị** thứ hạng thay vì chấm điểm từng cặp như cross-encoder, và **chưng cất** (distill) khả năng đó sang model nhỏ (DeBERTa) để chạy rẻ. Model DeBERTa distill này được A3 dùng làm một lựa chọn reranker.

---

# PHẦN III — NỐI LÝ THUYẾT VÀO PROJECT

## III.0 Các luồng của hệ A1 (theo code hiện tại)

Mục này mô tả **đúng những gì code đang chạy**, không phải hệ lý tưởng trên slide. Các con số lấy từ code: `step1/2/3_*.py`, `search_core.py`, `app.py`.

> Lưu ý: code đã **đổi từ CLIP sang SigLIP** (`google/siglip-base-patch16-224`, vector 768 chiều). Lý do ghi trong `search_core.py`: ablation embedding cho Recall@10 dense-only 0,920 (SigLIP) so với 0,656 (CLIP B/32) trên 250 câu. SigLIP cùng ý tưởng CLIP (ảnh và text chung một không gian, mục I.9), chỉ khác hàm loss: sigmoid trên từng cặp thay vì softmax trên cả hàng. Mọi chỗ nói "CLIP" ở Phần I vẫn đúng về nguyên lý.

### III.0.1 Bức tranh tổng: 1 pha offline, 2 luồng online

```
                    ┌──────────────── OFFLINE (chạy 1 lần, trước khi mở app) ────────────────┐
Shopify/product-    │ step1: catalog.jsonl + images/*.jpg                                     │
catalogue (HF)  ──► │ step2: SigLIP encode ẢNH  → image_embeddings.npy  (N × 768, chuẩn hoá L2) │
                    │ step3: FAISS IndexFlatIP(image_embeddings)  → dense.index               │
                    │        BM25Okapi(search_text đã tách từ)    → bm25.pkl                  │
                    └─────────────────────────────────────────────────────────────────────────┘
                                          │ search_core.py nạp 3 file + model SigLIP MỘT LẦN lúc import
                    ┌──────────────── ONLINE (mỗi request) ──────────────────────────────────┐
 Luồng TEXT  : GET  /api/v1/search?q=...      → search(q, "text")  → BM25 ┐                    │
                                                                  → Dense ┴→ RRF → [rerank] → top-k
 Luồng ẢNH   : POST /api/v1/search_by_image   → search(img, "image") → Dense ─────────────→ top-k
                    └─────────────────────────────────────────────────────────────────────────┘
```

### III.0.2 Pha offline: mỗi bước sinh ra cái gì

| Bước | Đầu vào | Làm gì | Đầu ra |
|---|---|---|---|
| step1 | Dataset `Shopify/product-catalogue` | Xáo trộn seed 42, lấy tối đa 2.000 sản phẩm, lưu ảnh ra file | `data/catalog.jsonl`: mỗi dòng `{id, image_path, caption (= tên SP), search_text (= tên + mô tả), category, brand}`, `data/images/*.jpg` |
| step2 | catalog + ảnh | SigLIP encode **ảnh** (và caption), chuẩn hoá L2 | `image_embeddings.npy` (N × 768). Có lưu thêm `text_embeddings.npy` nhưng **index không dùng file này** |
| step3 | `image_embeddings.npy`, catalog | Dựng 2 chỉ mục | `dense.index` (FAISS), `bm25.pkl` ({`bm25`, `ids`}) |

Ba điểm dễ bị hỏi:
- **Dense index chứa vector ẢNH, không chứa vector caption.** Vì vậy query text đi qua nhánh dense là so **chữ với hình** (cross-modal), đúng tinh thần "tìm ảnh bằng câu chữ" của đề A1 [C1-s90]. Phần chữ của sản phẩm đã có BM25 lo.
- **BM25 index dùng `search_text` (tên + mô tả)**, không chỉ tên, để có nhiều từ khoá hơn. Biến này được cô lập riêng trong `step9_ablation_index_fields.py`.
- `id` sản phẩm = **vị trí vector trong FAISS** (0..N−1). FAISS trả về vị trí, code dùng thẳng làm id. BM25 trả điểm theo vị trí tài liệu, nên cần thêm mảng `ids` trong `bm25.pkl` để đổi vị trí → id.

### III.0.3 Luồng search bằng TEXT, từng bước

Ví dụ: `GET /api/v1/search?q=folding pocket knife&k=10&rerank=true`

1. **`app.py`** nhận request, bấm giờ, gọi `search(q, query_type="text", k=10, use_rerank=True)`.
2. **`search()`** chạy lần lượt mọi component đã đăng ký (`bm25`, `dense`), mỗi nhánh lấy `top_n = 50` ứng viên.
3. **Nhánh BM25** (`_run_bm25`):
   - Tách từ query **đúng như lúc build**: `query.lower().split()` → `["folding", "pocket", "knife"]`.
   - `bm25.get_scores(tokens)` tính điểm BM25 cho cả 2.000 sản phẩm (công thức I.7, k₁ = 1,5, b = 0,75).
   - Sắp giảm dần, lấy 50 cặp `(id, điểm BM25)`.
4. **Nhánh Dense** (`_run_dense`):
   - `_encode_text(q)`: SigLIP text encoder, pad **cố định 64 token**, chuẩn hoá L2 → vector 1 × 768.
   - `dense_index.search(vec, 50)`: tích vô hướng với 2.000 vector **ảnh** = cosine (vì đều đã chuẩn hoá) → 50 cặp `(id, cosine)`.
5. **Gộp RRF** (`_fuse_all` → `rrf_fusion(k=60)`): bỏ điểm gốc, chỉ dùng thứ hạng, cộng 1/(60 + rank) ở mỗi danh sách. Sản phẩm có mặt ở cả hai danh sách được cộng hai lần (mục I.11). Pool sau gộp tối đa khoảng 100 id (hợp của hai danh sách 50).
6. **Rerank (tuỳ chọn)**: cross-encoder MiniLM chấm từng cặp `(query, caption)` cho **toàn bộ pool** (không chỉ top-10), để có thể kéo một ứng viên hạng thấp lên đầu, rồi sắp lại theo `rerank_score`.
7. Cắt `top k`, gắn `image_path`, `caption` từ catalog, trả JSON `{query, components_used, rerank, latency_ms, results}`.

### III.0.4 Luồng search bằng ẢNH, từng bước

Ví dụ: `POST /api/v1/search_by_image` với file ảnh một con dao.

1. **`app.py`** đọc bytes → `PIL.Image` RGB → `search(image, query_type="image")`.
2. **Nhánh BM25 trả `[]`**: ảnh không có từ nào để đếm, BM25 không áp dụng được.
3. **Nhánh Dense**: `_encode_image()` qua SigLIP **image encoder** (processor tự resize về 224×224, chuẩn hoá pixel, mục I.4) → vector 768 → FAISS so với 2.000 vector ảnh → top-50. Đây là so **ảnh với ảnh**.
4. **RRF không có tác dụng**: `_fuse_all` bỏ danh sách rỗng, chỉ còn một danh sách nên trả nguyên thứ tự dense.
5. **Không rerank**: cross-encoder cần cặp (text, text), query là ảnh nên bị bỏ qua (`do_rerank` chỉ bật khi `query_type == "text"`).
6. Trả top-k kèm `latency_ms`, `rerank: false`.

### III.0.5 So sánh hai luồng

| | Luồng TEXT | Luồng ẢNH |
|---|---|---|
| Endpoint | `GET /api/v1/search` | `POST /api/v1/search_by_image` (multipart) |
| Encoder query | SigLIP **text** encoder | SigLIP **image** encoder |
| BM25 | Có (trên `search_text`) | Không (trả rỗng) |
| Dense | text → **ảnh** (cross-modal) | ảnh → **ảnh** (cùng modality) |
| RRF | Gộp 2 danh sách thật | Chỉ còn 1 danh sách, không đổi thứ tự |
| Rerank | Tuỳ chọn (cross-encoder hoặc DeBERTa) | Không |
| Điểm mạnh | Kết hợp khớp từ khoá + khớp nghĩa | Tìm sản phẩm trông giống |
| Điểm yếu | Mã số/biến thể sát nhau vẫn có thể lẫn | Chỉ dựa vào hình: hai sản phẩm trông giống nhưng khác loại dễ bị nhầm; không có tín hiệu chữ để sửa |

Điều khiến hai luồng cùng chạy được trên **một** index dense: ảnh và text được SigLIP đưa vào **cùng một không gian** [C1-s42]. Nếu dùng hai encoder riêng (ResNet cho ảnh, BERT cho text) thì luồng text → ảnh không làm được, vì cosine giữa hai không gian khác nhau không có nghĩa.

### III.0.6 Vì sao phải build index cho BM25 và cho Dense

**Lý do chung: chuyển việc nặng sang offline** [C1-s13]. "Build index" nghĩa là tính sẵn những thứ **không phụ thuộc câu hỏi**, lưu xuống đĩa, để mỗi request chỉ làm phần nhỏ phụ thuộc câu hỏi.

**Nếu không có dense index:** mỗi request phải chạy SigLIP qua cả 2.000 ảnh. Trên CPU việc này mất nhiều phút, trong khi encode một câu query chỉ mất vài chục đến vài trăm mili-giây. Có index rồi thì online chỉ còn: encode **một** query + **một** phép nhân ma trận (1 × 768) · (768 × 2.000).

Dense index của project lưu gì: ma trận 2.000 × 768 vector ảnh đã chuẩn hoá, đặt trong `IndexFlatIP`. "Flat" nghĩa là **so hết, chính xác 100%**, chưa phải ANN (IVF/HNSW, mục I.10). Ở 2.000 sản phẩm thì so hết vẫn rất nhanh. Phần tốn kém mà index tiết kiệm được là **encode catalog**, không phải bước so sánh.

**Nếu không có BM25 index:** mỗi request phải tách từ lại 2.000 tài liệu, đếm từ, tính độ dài, avgdl và IDF của mọi từ. Các con số này không đổi giữa các query.

BM25 index của project lưu gì (`BM25Okapi` của thư viện `rank_bm25`): với mỗi tài liệu, bảng tần suất từ (`doc_freqs`) và độ dài (`doc_len`); cho cả kho, `avgdl` và bảng `idf` của từng từ. Online chỉ cần tra các con số đó cho mấy từ của query rồi ráp vào công thức.

> Nói cho chính xác: `rank_bm25` lưu **thống kê đã tính sẵn**, không phải inverted index đúng nghĩa (danh sách tài liệu theo từng từ, mục I.5). `get_scores` vẫn duyệt qua cả 2.000 tài liệu, nhưng mỗi tài liệu chỉ là một phép tra bảng nên vẫn nhanh. Ở quy mô triệu tài liệu thì phải dùng inverted index thật (Elasticsearch, Postgres GIN) để chỉ chạm tới các tài liệu có chứa từ của query.
>
> Biến thể IDF của thư viện là ln((N − df + 0,5)/(df + 0,5)), không có "+1" như slide 27. Từ xuất hiện trong hơn nửa kho sẽ cho IDF âm, nên thư viện thay bằng một mức sàn nhỏ (ε × IDF trung bình, ε = 0,25). Vì vậy điểm số tuyệt đối khác slide một chút, nhưng nguyên lý giống hệt.

**Vì sao cần HAI index chứ không một:**
1. **Hai biểu diễn khác bản chất, không index nào trả lời thay index kia được.** BM25 cần từ và số lần đếm (sparse). Dense cần vector nghĩa (dense). Không thể tính BM25 từ vector, cũng không thể so nghĩa từ bảng đếm từ [C1-s48].
2. **Hai kiểu lỗi bù nhau** [C1-s29, s51, s58]:
   - BM25 thua khi query khác chữ với tài liệu (đồng nghĩa, diễn đạt tự do; ví dụ query "knife for camping" nhưng tên sản phẩm ghi "folding blade"), và **không xem được ảnh**.
   - Dense thua khi cần khớp chính xác mã số, tên hãng, model ("G502", "RTX 5060"); nhánh dense lại so query với **ảnh**, mà ảnh thì không chứa mã model.
   - RRF gộp hai danh sách để giữ phần đúng của mỗi bên. Bằng chứng phải lấy từ số liệu ablation (`step8_run_ablation.py`: BM25-only, Dense-only, Hybrid, Hybrid + rerank), không chỉ lập luận.
3. **Luồng ảnh chỉ có dense dùng được**, nên dense index là bắt buộc cho đề A1. BM25 thì là bắt buộc theo rubric [C1-s91: "BM25, RRF … chạy đúng"].

**Điều kiện để index dùng được: online phải làm ĐÚNG như offline.**
- Tách từ: `step3.simple_tokenize` và `_run_bm25` đều dùng `lower().split()`. Lệch nhau (ví dụ offline có bỏ dấu câu, online thì không) thì từ "knife," và "knife" thành hai từ khác nhau, BM25 trượt.
- Encoder: cùng model SigLIP, cùng cách tiền xử lý. Code ghi lại một lỗi thật: encode text với padding động thay vì `padding="max_length", max_length=64` làm Recall@10 về **0,000**, vì SigLIP lấy biểu diễn ở token cuối của chuỗi đã pad cố định.
- Chuẩn hoá L2 ở cả hai phía, để tích vô hướng của FAISS bằng cosine (mục I.8).
- **Đổi model embedding thì phải build lại `dense.index`** (vector cũ và mới nằm ở hai không gian khác nhau, khác cả số chiều 512 và 768) [C2-s60].

### III.0.7 Vài câu hỏi bảo vệ riêng về luồng

- *Query ảnh có dùng hybrid không?* Không. BM25 trả rỗng, RRF không đổi thứ tự, không rerank. Thực chất là dense-only.
- *Muốn query ảnh cũng có tín hiệu chữ thì làm sao?* Sinh caption cho ảnh query (bằng VLM) rồi chạy thêm nhánh BM25; hoặc dùng `text_embeddings.npy` làm thêm một nhánh dense (ảnh query → vector caption). Cả hai đều là ablation hợp lệ cho A3.
- *Vì sao rerank cả pool mà không chỉ top-10?* Nếu chỉ rerank top-10 thì rerank chỉ đảo thứ tự trong 10 cái đó, không cứu được sản phẩm đúng đang nằm ở hạng 30 [C1-s70].
- *Vì sao `top_n = 50` mỗi nhánh?* Tầng 1 tối ưu recall: lấy rộng để tài liệu đúng có mặt trong pool. `step10_sweep_rrf.py` quét giá trị này cùng với `rrf_k`.
- *Vì sao dense index lưu vector ảnh mà không lưu vector caption?* Để câu chữ được so với **nội dung hình** (đúng mục tiêu "tìm ảnh bằng câu chữ"), còn chữ của sản phẩm thì BM25 đã phủ. Index caption là một lựa chọn khác, có thể đưa vào ablation.

## III.1 Bản đồ khái niệm → code

| Khái niệm | Slide | Code trong project |
|---|---|---|
| Chuẩn hoá dữ liệu, catalog | C1-s15..17, C2-s31 | `step1_build_catalog.py` |
| Offline encode bằng SigLIP (họ CLIP), chuẩn hoá vector | C1-s13, s37, s42 | `step2_build_embeddings.py` |
| Inverted index + BM25, index vector brute-force | C1-s23, s26, s49 | `step3_build_index.py` (`BM25Okapi`, `faiss.IndexFlatIP`) |
| BM25 / Dense / RRF lúc online | C1-s47, s61 | `search_core.py`: `_run_bm25`, `_run_dense`, `rrf_fusion(k=60)`, `search(top_n=50)` |
| Rerank cross-encoder | C1-s67, C2-s76 | `search_core.rerank()` |
| Rerank chưng cất từ LLM (paper A2) | A2 | `search_core.distilled_rerank()` |
| Recall@k, nDCG@k, nDCG phân bậc, MRR | C1-s76..79 | `metrics.py`: `recall_at_k`, `ndcg_at_k`, `ndcg_at_k_graded`, `mrr` |
| qrels text + ảnh | C1-s73, s91 | `eval_set`, `step13_build_image_eval_set.py`, `step14_eval_image_queries.py`, `step15_expand_eval_set.py` |
| Ablation có kiểm soát | C1-s80 | `step8_run_ablation.py`, `step9_ablation_index_fields.py`, `step10_sweep_rrf.py`, `step_ablation_rerank.py` |
| Error analysis, case study | C1-s91 | `Notebook.ipynb`, `step12_case_study_trace.py` |
| API có version, JSON kèm latency | C1-s83 | `app.py`: `/api/v1/search` |
| Naive RAG (retrieve → augment → generate) | C2-s29 | `rag_core.py`: `retrieve`, `build_context`, `build_prompt`, `call_llm`, `answer` |
| Prompt ràng buộc + trích dẫn [n] | C2-s33 | `rag_core.build_prompt()` |
| Query rewriting (dạng agentic, có tự đánh giá) | C2-s66 | `rag_core.judge_sufficiency()`, `rewrite_query()`, `answer_agentic()` |
| Memory hội thoại (viết lại kèm ngữ cảnh lượt trước) | C2-s66 | `rag_core.get_history/append_history` |
| Context Recall / Context Precision (dạng AP) / Faithfulness / Answer Relevancy | C2-s35..37 | `rag_metrics.py` |
| Ablation RAG | C1-s80, C2-s40 | `step_ablation_rag.py` |
| LLM local | C1-s86 | Ollama `llama3.1` |

Hai chỗ **khác với slide**, cần tự nói rõ khi bảo vệ:
- `rag_metrics.answer_relevancy` tính **cosine(câu hỏi, câu trả lời)** bằng all-MiniLM-L6-v2, đơn giản hơn cách RAGAS trên slide (sinh ngược n câu hỏi rồi so) [C2-s37].
- `context_precision_at_k` dùng dạng Average Precision chia cho min(số id đúng, k), khớp với ví dụ 0,833 của slide [C2-s36], không phải công thức đơn giản |Z⁺|/k.

## III.2 A1 → A2 → A3 đi theo lý thuyết thế nào

```
A1 (Chương 1)  : Tầng 1 hybrid (BM25 + CLIP + RRF) → Tầng 2 rerank → API → đo Recall/nDCG/MRR → ablation
A2 (paper)     : RankGPT — một cách làm Tầng 2 tốt hơn (LLM rerank → distill sang DeBERTa)
A3 (Chương 2)  : A1 làm bước Retrieve của RAG → Augment (context đánh số) → Generate (Ollama)
                 + nâng cấp: rerank (A1 hoặc A2), agentic rewrite, memory
                 + đo tách tầng: Context Recall/Precision (truy hồi) vs Faithfulness/Answer Relevance (tạo sinh)
```

Lý do thiết kế này hợp lý theo slide: *"Truy hồi quyết định chất lượng RAG"* [C2-s40], nên A3 tận dụng hệ truy hồi đã được đánh giá kỹ ở A1 thay vì làm lại.

---

# PHẦN IV — CÂU HỎI ÔN TẬP ĐỂ BẢO VỆ

Trả lời ngắn; chi tiết ở mục ghi trong ngoặc.

**Chương 1**
1. *Vì sao không dùng SQL ILIKE?* Chỉ khớp mặt chữ, không hiểu đồng nghĩa, không xếp hạng, không dùng được index (I.2).
2. *Offline và online khác nhau thế nào? Cái gì chạy ở đâu?* Offline: encode kho, build index. Online: encode một query, tra index, fusion, rerank (I.3).
3. *Inverted index lưu gì?* Từ → danh sách tài liệu chứa từ; chỉ trả lời "có chứa", không trả lời "liên quan hơn" (I.5).
4. *IDF để làm gì?* Trọng số độ hiếm; từ phổ biến gần 0 điểm (I.6).
5. *k₁ và b trong BM25 làm gì?* k₁ bão hoà tần suất, b chuẩn hoá độ dài (I.7).
6. *Vì sao BM25 xếp chuột trên laptop cho "laptop gaming"?* "laptop" không có trong tên + tài liệu ngắn được lợi (I.7).
7. *Vì sao cosine chứ không Euclid?* Chỉ đo hướng; với vector chuẩn hoá thì hai cách cho cùng thứ hạng: ‖a−b‖² = 2 − 2cos (I.8).
8. *Contrastive learning là gì? Hard negative là gì?* Kéo cặp đúng lại gần, đẩy cặp sai ra xa; hard negative = sai nhưng trông giống đúng (I.8).
9. *CLIP khác image embedding thường ở đâu?* Hai encoder huấn luyện chung vào một không gian, nên so ảnh với chữ được (I.9).
10. *Vì sao dense nhầm RTX 5050/5060?* Mã số không mang "nghĩa", ngữ cảnh huấn luyện gần giống nhau (I.10).
11. *IVF vs HNSW? Project dùng gì, vì sao?* Project dùng FlatIP brute-force, chính xác 100%, đủ nhanh ở quy mô vài nghìn (I.10).
12. *Vì sao không cộng điểm BM25 và cosine? Min-max có vấn đề gì?* Khác thang; min-max không ổn định giữa query, bị outlier nén (I.11).
13. *RRF tính thế nào? k lớn/nhỏ ảnh hưởng gì?* (I.11, bài tập slide 62).
14. *Vì sao cross-encoder chính xác hơn nhưng chỉ dùng ở tầng 2?* Attention chéo giữa token query và doc, nhưng không tính trước được; 1 triệu doc ≈ 14 giờ (I.12).
15. *Tầng 1 đo bằng gì, tầng 2 đo bằng gì?* Recall@50; nDCG@10, MRR (I.12, I.13).
16. *Recall@k, MRR, nDCG khác nhau thế nào?* (bảng tóm tắt I.13).
17. *Ablation là gì? Vì sao chỉ đổi một thành phần?* (I.13).
18. *Vì sao đo p95 chứ không đo trung bình?* Trung bình che giấu các request chậm mà người dùng vẫn gặp (I.14).

**Chương 2**
19. *Attention làm gì? BERT khác GPT thế nào?* (II.2).
20. *Token ảnh hưởng chi phí và thiết kế RAG ra sao?* Context chiếm phần lớn token vào (II.2).
21. *Temperature là gì? Vì sao RAG đặt thấp? Vì sao T = 0 vẫn có thể khác kết quả?* (II.2).
22. *Vì sao LLM hallucinate?* Phải sinh token tiếp, không được thưởng cho "không biết", chọn câu nghe hợp lý nhất (II.2).
23. *Few-shot đánh đổi gì với RAG?* Token của ví dụ cạnh tranh với context (II.3).
24. *RAG 2020 khác RAG hiện nay?* End-to-end vs ghép module đóng băng (II.4).
25. *Khi nào RAG, khi nào fine-tune?* Biết gì vs nói thế nào (II.4).
26. *Prompt RAG của nhóm có những ràng buộc nào? Vì sao cho phép "không tìm thấy"?* (II.4, `build_prompt`).
27. *Context Recall khác Faithfulness?* Recall: ground truth vs context; Faithfulness: câu trả lời vs context (II.5).
28. *Làm sao biết lỗi do truy hồi hay tạo sinh?* Bảng chẩn đoán ở II.5.
29. *Fixed-size, semantic, theo cấu trúc, parent-child khác nhau thế nào? Chunk quá lớn/nhỏ bị gì?* (II.7).
30. *Contextual retrieval sửa lỗi gì?* Chunk mất ngữ cảnh (năm áp dụng) (II.7).
31. *HyDE hoạt động thế nào? Đoạn giả định sai có sao không?* (II.8).
32. *Lost in the middle là gì, ảnh hưởng cách sắp context ra sao?* (II.10).
33. *Semantic cache rủi ro gì?* 2025 vs 2026 (II.11).
34. *Prompt injection là gì, phòng thế nào?* (II.11).
35. *Đổi embedding model phải làm gì với index?* Re-embed toàn bộ (II.7).

---

# PHỤ LỤC A — LỘ TRÌNH CODE A1 TỪNG BƯỚC

Mỗi bước dưới đây giải thích **tại sao cần bước này** trước, rồi mới nối vào **code thật** đã có trong project. Đọc theo thứ tự vì bước sau phụ thuộc bước trước.

---

## Bước 0 — Bài toán này khác gì search từ khoá thông thường?

**Lý thuyết:**
Search từ khoá (như Ctrl+F, hay SQL `LIKE '%dog%'`) chỉ khớp được khi chữ trong query trùng khớp chữ trong dữ liệu. Nếu ảnh có caption "a dog running" mà bạn gõ "chó đang chạy", search từ khoá thất bại hoàn toàn — không có chữ nào trùng.

**Semantic search** giải quyết việc này bằng cách biến câu chữ thành một **vector số** (embedding) sao cho hai câu *có nghĩa gần nhau* thì vector của chúng cũng *gần nhau trong không gian số*, bất kể chữ có trùng hay không. "a dog running" và "chó đang chạy" tuy chữ khác hẳn nhưng vector sẽ nằm gần nhau.

Đây là lý do toàn bộ 8 bước sau tồn tại: biến dữ liệu (ảnh + text) thành vector, rồi so vector thay vì so chữ.

---

## Bước 1 — Chuẩn hoá dữ liệu thành catalog

**Lý thuyết:**
Dataset tải từ Hugging Face có cấu trúc riêng của nó (object `Image`, list caption...). Nếu bạn để nguyên định dạng đó, mọi đoạn code sau (embedding, index, search, API) đều phải "biết" cấu trúc đặc thù của dataset đó. Thay vào đó, ta làm **một lớp trung gian** — catalog với schema cố định (`id`, `image_path`, `caption`) — để đổi dataset khác chỉ cần sửa `step1_build_catalog.py`, không phải sửa cả hệ thống.

**Thực hành:**
`step1_build_catalog.py` tải `Shopify/product-catalogue`, xáo trộn với seed 42, lấy tối đa 2.000 sản phẩm, lưu mỗi ảnh ra file `.jpg` thật trên đĩa, và ghi 1 dòng JSON cho mỗi sản phẩm (ví dụ minh hoạ):
```json
{"id": 0, "image_path": "data/images/000000.jpg", "caption": "<tên sản phẩm>",
 "search_text": "<tên sản phẩm> <mô tả>", "category": "...", "brand": "..."}
```
Từ giờ, mọi bước sau chỉ đọc file `catalog.jsonl` này.

---

## Bước 2 — Sinh embedding bằng CLIP / SigLIP

> Code hiện tại dùng **SigLIP** (`google/siglip-base-patch16-224`, 768 chiều) thay cho CLIP B/32 (512 chiều), xem III.0. Nguyên lý dưới đây giống nhau cho cả hai. Riêng SigLIP phải encode text với `padding="max_length", max_length=64`.

**Lý thuyết:**
CLIP là mô hình được huấn luyện để **ảnh và text mô tả đúng ảnh đó nằm gần nhau trong cùng một không gian vector** (gọi là "joint embedding space"). Đây là điều đặc biệt: encoder ảnh và encoder text tuy khác nhau, nhưng output của chúng có thể so sánh trực tiếp bằng cosine similarity.

Đây chính là lý do CLIP làm được cả 2 chiều: text→ảnh (encode câu hỏi, so với vector ảnh) và ảnh→ảnh/text (encode ảnh, so với vector khác).

Nếu không dùng model kiểu CLIP, bạn sẽ phải có 2 không gian vector riêng biệt (1 cho ảnh, 1 cho text) và không so được trực tiếp.

**Thực hành:**
`step2_build_embeddings.py` chạy từng batch ảnh qua `model.get_image_features()`, từng batch caption qua `model.get_text_features()`, rồi **chuẩn hoá vector về độ dài 1** (`feat / feat.norm()`) — bước này quan trọng vì sau đó cosine similarity = tích vô hướng (inner product), tính nhanh hơn nhiều.

Ví dụ cụ thể (giả lập, vector thật có 768 chiều với SigLIP, ở đây rút gọn 2 chiều để dễ hình dung):
```
"a dog running on the beach"  → [0.80, 0.10]
ảnh con chó chạy trên biển    → [0.78, 0.12]   (gần nhau vì cùng nghĩa)
"a red car parked"            → [0.05, 0.90]   (xa nhau vì khác nghĩa)
```

---

## Bước 3 — Build 2 index song song

**Lý thuyết:**
Có vector rồi, nhưng nếu dataset có 100.000 ảnh, so sánh tuần tự từng vector một (brute-force) sẽ chậm khi query nhiều. **FAISS** giải quyết bài toán "tìm k vector gần nhất" hiệu quả hơn (dù ở quy mô vài nghìn ảnh của A1, brute-force `IndexFlatIP` vẫn đủ nhanh — bạn chưa cần thuật toán ANN phức tạp hơn).

Song song đó, **BM25** là thuật toán xếp hạng dựa trên tần suất từ (không dùng embedding) — mạnh ở việc khớp chính xác từ/cụm từ hiếm gặp (tên riêng, mã số) mà dense embedding có thể bỏ lỡ vì embedding "làm mờ" nghĩa để bắt được similarity.

Đây là lý do ta build **cả 2 index**: chúng bổ khuyết điểm yếu cho nhau (đúng như bảng BM25/Dense/Hybrid bạn thấy trong slide).

**Thực hành:**
`step3_build_index.py`:
- `faiss.IndexFlatIP(dim)` trên **vector ảnh**: index dạng "so trực tiếp", trả điểm inner product (= cosine vì đã chuẩn hoá).
- `BM25Okapi(tokenized search_text)`: lưu sẵn thống kê BM25 (tần suất từ từng tài liệu, độ dài, avgdl, IDF). Nói chặt thì đây không phải inverted index đúng nghĩa, xem III.0.6.

---

## Bước 4 — Hàm search() và RRF fusion

**Lý thuyết — vì sao không cộng điểm trực tiếp?**
BM25 trả điểm kiểu 0–15 (tuỳ độ dài văn bản), còn cosine similarity trả điểm 0–1. Cộng thẳng hai loại điểm này lại là **so sánh táo với cam** — con số không cùng ý nghĩa.

**RRF (Reciprocal Rank Fusion)** né hoàn toàn vấn đề đó: thay vì dùng *điểm số*, nó chỉ dùng *thứ hạng* (rank) của mỗi tài liệu trong từng danh sách:
```
score(doc) = Σ  1 / (k + rank_trong_danh_sách)
```
Một tài liệu đứng hạng 1 ở cả BM25 lẫn Dense sẽ có điểm RRF cao nhất, dù điểm gốc của 2 hệ thống hoàn toàn khác thang đo.

**Thực hành — ví dụ tính tay:**
Query: "con chó đang ngủ", giả sử k=60 (hằng số RRF chuẩn):

BM25 trả về (theo caption khớp chữ):
```
rank 1: id=3 "dog sleeping in a park"
rank 2: id=1 "dog running on the beach"
```
Dense trả về (theo nghĩa gần):
```
rank 1: id=3 "dog sleeping in a park"
rank 2: id=2 "cat sleeping on a sofa"
```

Tính RRF:
```
id=3: 1/(60+1) + 1/(60+1) = 0.0164 + 0.0164 = 0.0328   ← cao nhất, đúng ý nghĩa
id=1: 1/(60+2) = 0.0161
id=2: 1/(60+2) = 0.0161
```
id=3 thắng vì nó **đồng thuận ở cả 2 hệ thống** — đây chính là giá trị của hybrid search.

Trong code, đúng hàm `rrf_fusion()` trong `search_core.py` làm phép tính này.

---

## Bước 5 — Vì sao phải tự viết Recall@k và nDCG@k?

**Lý thuyết:**
- **Recall@k** trả lời: "trong số các kết quả đúng tồn tại, hệ thống tìm được bao nhiêu trong top-k?" — không quan tâm thứ tự.
- **nDCG@k** khắt khe hơn: nó **phạt nặng** nếu kết quả đúng nằm ở hạng thấp (ví dụ hạng 8 thay vì hạng 1), vì người dùng thực tế ít khi kéo xuống xem hết top-10.

Công thức nDCG dùng `log2(rank+1)` làm mẫu số — rank càng lớn, mẫu số càng lớn, đóng góp của kết quả đó vào điểm càng nhỏ. Đây là lý do nDCG "nhạy" với thứ tự còn Recall thì không.

**Thực hành — ví dụ tính tay:**
Câu query có 2 ảnh đúng: `relevant_ids = {3, 7}`. Hệ thống trả về top-5: `[3, 1, 7, 2, 9]`.

Recall@5 = số đúng tìm thấy / tổng số đúng = 2/2 = **1.0** (tìm đủ cả 2, dù thứ tự không hoàn hảo)

nDCG@5:
```
DCG = 1/log2(1+1) + 0/log2(2+1) + 1/log2(3+1) + 0 + 0
    = 1/1.0 + 1/2.0
    = 1.0 + 0.5 = 1.5
IDCG (nếu 2 ảnh đúng nằm ở hạng 1,2) = 1/log2(2) + 1/log2(3) = 1.0 + 0.63 = 1.63
nDCG@5 = 1.5 / 1.63 ≈ 0.92
```
Recall nói "tìm đủ rồi", nhưng nDCG nói "gần đủ điểm tối đa, trừ một chút vì ảnh đúng thứ 2 (id=7) nằm ở hạng 3 thay vì hạng 2" — đây là thông tin Recall không thấy được.

---

## Bước 6 — Vì sao phải tự viết eval set + qrels?

**Lý thuyết:**
Metric ở bước 5 cần biết "đâu là đúng" để so sánh. Không ai (kể cả dataset gốc) biết trước bạn sẽ hỏi câu gì, nên **không có ground truth sẵn** cho bài toán search tự do. Bạn phải tự đóng vai người dùng: nghĩ ra câu hỏi, rồi tự trả lời "ảnh nào đúng cho câu này" — đó chính là qrels (query relevance judgments).

Đây cũng là lý do rubric chấm riêng tiêu chí "tập đánh giá hợp lệ" (20%) — nếu qrels sai hoặc không đủ đa dạng, mọi con số Recall/nDCG phía sau đều vô nghĩa.

**Thực hành:**
Mở `data/catalog.jsonl`, chọn ngẫu nhiên/có chủ đích một số ảnh, viết câu query mà bạn nghĩ người dùng thật sẽ gõ, note lại `id` ảnh đúng — theo đúng format `eval_set_template.jsonl`.

---

## Bước 7 — Vì sao tách riêng API thay vì để frontend gọi thẳng search()?

**Lý thuyết:**
Đây là nguyên tắc **tách lớp (separation of concerns)**: frontend không cần biết BM25/Dense/RRF là gì, nó chỉ cần biết "gửi câu hỏi, nhận JSON kết quả". Nếu sau này bạn đổi toàn bộ thuật toán search bên trong, frontend không cần sửa 1 dòng nào — miễn là API vẫn trả đúng format cũ.

**Thực hành:**
`app.py` bọc `search()` trong endpoint `/api/v1/search`, trả JSON có `query`, `latency_ms`, `results` — đúng format bạn thấy trong slide.

---

## Bước 8 — Ablation: cô lập từng biến để đo đóng góp

**Lý thuyết:**
Đây là tư duy thực nghiệm khoa học cốt lõi: nếu bạn đổi cùng lúc 2 thứ (ví dụ vừa tắt BM25 vừa đổi model embedding) rồi thấy điểm giảm, bạn **không biết cái nào gây ra** sự giảm đó. Ablation yêu cầu: mỗi lần chạy chỉ đổi **đúng 1 biến**, mọi thứ khác giữ nguyên — khi đó chênh lệch điểm số giữa 2 lần chạy chính là đóng góp thuần của biến đó.

**Thực hành:**
`step8_run_ablation.py` chạy 4 cấu hình (BM25-only, Dense-only, Hybrid, Hybrid+rerank) trên **cùng một eval_set**, in bảng Recall@k, nDCG@k, nDCG đa mức (0/1/2, xem `metrics.ndcg_at_k_graded()`), MRR (`metrics.mrr()`) và latency avg/p95. Nếu Hybrid > cả BM25-only và Dense-only, bạn có bằng chứng số liệu rằng fusion có giá trị thật — không phải suy đoán.

Có thêm 3 script ablation nâng cao (tuỳ chọn, đúng phạm vi mục 5.2/5.4/6 của template báo cáo):
- `step9_ablation_index_fields.py` — cô lập biến "trường dữ liệu đưa vào BM25 index" (title-only vs title+description vs +ảnh), cùng tư duy "chỉ đổi 1 biến" như trên nhưng áp cho tầng index thay vì tầng component.
- `step10_sweep_rrf.py` — cô lập 2 biến của công thức RRF (`rrf_k` và `top_n`/N) bằng cách sweep lưới (grid) thay vì chỉ thử 1 giá trị cố định — cho biết fusion có thật sự "không nhạy với k" như paper Cormack et al. (2009) khẳng định, hay chỉ đúng ở quy mô dữ liệu lớn của paper đó.
- `step12_case_study_trace.py` — mở rộng ý tưởng ablation xuống MỨC 1 CÂU QUERY: thay vì so trung bình cả eval_set, script này cho biết chính xác BM25/Dense/RRF/Rerank xếp hạng id đúng ở đâu cho 1 câu cụ thể — dữ liệu thô cần thiết để viết case study định tính (mục 6 template), không thể suy ra được nếu chỉ nhìn số Recall/nDCG trung bình.

---

## Bước 9 — Error analysis: nhìn vào từng ca sai để tìm khuôn mẫu

**Lý thuyết:**
Con số tổng (Recall=0.72) không nói cho bạn biết **loại câu hỏi nào** hệ thống yếu. Error analysis là việc lọc ra các câu bị sai, đọc từng câu, và **gom nhóm theo nguyên nhân** (ví dụ: "sai vì query dùng từ đồng nghĩa mà CLIP không phân biệt được", "sai vì ảnh có nhiều vật thể, embedding bị pha loãng"). Việc gom nhóm này biến một danh sách lỗi rời rạc thành **insight có thể hành động** — đúng yêu cầu "đề xuất cải tiến có căn cứ" trong rubric.

**Thực hành:**
`Notebook.ipynb` (ở gốc project) đã làm đúng việc này: chạy lại `search()` với cả 4 cấu hình, in ra *câu nào bị MISS* (không tìm thấy id đúng trong top-k) kèm caption đúng + top-3 kết quả sai để đối chiếu, thay vì chỉ in điểm trung bình. Muốn xem hạng cụ thể của 1 câu qua từng tầng BM25/Dense/RRF/Rerank (sâu hơn "đúng/sai" nhị phân), dùng `step12_case_study_trace.py` ở mục Bước 8 trên.

---

## Bước 10-11 — Frontend + báo cáo

Đây là bước "đóng gói" — không có lý thuyết mới, chỉ là trình bày lại kết quả từ bước 4-9 theo cách người khác (giảng viên) đọc được. Bảng ablation (bước 8) và taxonomy lỗi (bước 9) là 2 tài sản quan trọng nhất trong báo cáo — mọi thứ khác chỉ là bối cảnh dẫn tới 2 bảng đó.
