# BÁO CÁO — Áp dụng hệ thống truy hồi (IS6303) vào quy trình Sales tư vấn sản phẩm (IS6003)
### Môn: Hoạch định nguồn lực doanh nghiệp nâng cao (IS6003.11.CH) — GV Đỗ Phúc
### Liên kết: đồ án A1 (Search) và A3 (RAG) của môn IS6303

> **Tính chất số liệu:** toàn bộ thời gian (T) và xác suất lặp (p) trong báo cáo là **giả định** viết cố định trong `bpm_event_log_simulation.py`, **không đo** từ hệ thống hay từ người dùng thật. Mọi kết luận ở mục 6–7 là hệ quả của các giả định đó. Mô phỏng không đặt random seed, nên mỗi lần chạy cho số hơi khác nhau (mục 6).

Mục lục
1. Bối cảnh môn học
2. Bài toán
3. Chọn đúng quy trình
4. Mô hình BPMN
5. Cấu trúc Loop và công thức
6. Mô phỏng event log
7. So sánh ba kịch bản
8. Hạn chế và cách lấy số liệu thật

---

## 1. Bối cảnh môn học

| Buổi | Nội dung | Dùng ở mục |
|---|---|---|
| 1 | Hệ thống tích hợp doanh nghiệp, BPMN | 4 |
| 2 | Đánh giá quy trình theo Time/Cost: Sequence, XOR, AND, Loop | 5, 7 |
| 3 | Event log | 6 |

---

## 2. Bài toán

**Phía IS6303 (đã xây):**
- **A1:** tìm sản phẩm bằng câu tự nhiên hoặc ảnh, trên catalog 2.000 sản phẩm lấy từ dataset `Shopify/product-catalogue`.
- **A3:** chatbot RAG trả lời bằng câu tự nhiên, có trích dẫn sản phẩm nguồn.

**Phía IS6003 (báo cáo này):** xác định quy trình nghiệp vụ mà hệ thống thực sự tác động, mô hình hoá quy trình đó, rồi dùng khung Time/Cost và event log để ước lượng tác động.

---

## 3. Chọn đúng quy trình

- Bản nháp đầu gắn hệ thống vào bước "Process Order" (Authorize Payment → Process Order → Pack Goods → Ship Goods). Cách gắn này **không đúng**: quy trình đó diễn ra **sau khi** khách đã chốt đơn, không còn bước tư vấn sản phẩm.
- Hệ thống A1/A3 hỗ trợ việc **tìm và giới thiệu sản phẩm cho khách**, tức giai đoạn tiền bán hàng, **trước** Order Processing. Báo cáo chọn quy trình "Sales tư vấn sản phẩm".

---

## 4. Mô hình BPMN

```
[Start] Khách nêu yêu cầu (vd "waterproof hiking boots for winter")
     │
     ▼
 Sales tìm sản phẩm  ◄───────────────┐
     │                               │ Không
     ▼                               │ (Loop)
 Trình bày kết quả cho khách         │
     │                               │
     ▼                               │
 ◇ Khách hài lòng? ──────────────────┘
     │ Có
     ▼
 Chuyển sang lập đơn (quy trình Order Processing, ngoài phạm vi)
```

**Khâu bị ảnh hưởng:** "Sales tìm sản phẩm". Với giao diện tìm kiếm dạng form (tên, category, khoảng giá), những yêu cầu kiểu "không thấm nước, dùng cho mùa đông" khó nhập vào form. Giả định của báo cáo là điều này làm khách thường chưa hài lòng ở lần tìm đầu, dẫn đến vòng lặp. Đây là giả định về nghiệp vụ, chưa được khảo sát.

---

## 5. Cấu trúc Loop và công thức

Một hoạt động có thời gian T và xác suất phải lặp lại p (khách chưa hài lòng) có **cycle time kỳ vọng**:

```
CT = T / (1 - p)
```

Số lần thực hiện trung bình là 1/(1-p), tức kỳ vọng của phân phối hình học.

---

## 6. Mô phỏng event log

### 6.1 Tham số (giả định, trong code)

| Kịch bản | T mỗi lần tìm (phút, phân phối đều) | p | Lý do giả định |
|---|---|---|---|
| A — Không hỗ trợ (form nhiều ô) | 4.0 – 6.0 | 0.40 | Nhập nhiều ô, thiếu bộ lọc đúng ý |
| B — Có A1 | 0.5 – 1.0 | 0.15 | Gõ một câu tự nhiên; vẫn có thể trả sai |
| C — Có A1 + A3 Agentic | 1.0 – 1.5 | 0.03 | Mỗi lượt lâu hơn (LLM tự đánh giá, có thể search lại), nhưng hiếm phải lặp |

### 6.2 Cách mô phỏng (`bpm_event_log_simulation.py`)

1. Lấy caption của 8 sản phẩm đầu trong `data/catalog.jsonl` để đặt nhãn cho 8 case mỗi kịch bản. Sản phẩm không ảnh hưởng tới thời gian.
2. Mỗi case lặp hoạt động "Sales tìm sản phẩm". Mỗi lần lặp là **một dòng event log** gồm `case_id`, `attempt_number`, `start_time`, `end_time`, `duration_minutes`.
3. Thời gian mỗi lần lấy ngẫu nhiên đều trong khoảng T. Case dừng khi `random() >= p`, tối đa 10 lần.
4. CT mô phỏng là trung bình tổng thời gian mỗi case. CT lý thuyết là `((T_min + T_max)/2) / (1 - p)`.
5. Output: `bpm_event_log.csv` và bảng so sánh.

### 6.3 Kết quả ba lần chạy (25/09/2026)

| Kịch bản | CT lý thuyết (phút) | CT lần 1 | CT lần 2 | CT lần 3 | Số lần lặp TB (3 lần chạy) |
|---|---|---|---|---|---|
| A | 8.33 | 11.06 | 9.48 | 8.82 | 2.12 / 1.88 / 1.75 |
| B | 0.88 | 0.93 | 0.90 | 0.87 | 1.25 / 1.12 / 1.12 |
| C | 1.29 | 1.18 | 1.28 | 1.30 | 1.00 / 1.00 / 1.00 |

- CT mô phỏng dao động quanh CT lý thuyết. Kịch bản A lệch nhiều nhất (tới +33% ở lần 1), vì chỉ có 8 case và p lớn.
- Mô phỏng xác nhận đúng cách tính: giá trị trung bình tiến về công thức. Nó **không** tạo thêm thông tin về hệ thống thật, vì cả hai cùng dựa trên T và p giả định.
- Để số ổn định hơn cần tăng số case và đặt seed. Code hiện chưa làm.

---

## 7. So sánh ba kịch bản

Tính từ CT lý thuyết (không phụ thuộc lần chạy ngẫu nhiên):

| So sánh | CT | Chênh lệch |
|---|---|---|
| B so với A | 0.88 so với 8.33 | giảm 89.4% |
| C so với A | 1.29 so với 8.33 | giảm 84.5% |
| C so với B | 1.29 so với 0.88 | **tăng 46.1%** |

**Diễn giải trong khuôn khổ giả định:** cả A1 và A3 Agentic đều rút ngắn mạnh khâu tìm kiếm so với form nhiều ô. Nhưng với tham số đã giả định, **kịch bản C chậm hơn B**. Lý do: T mỗi lượt của C dài hơn (1.25 so với 0.75 phút), mà lợi ích giảm p (0.15 xuống 0.03) không bù được.

**Ngưỡng hoà vốn:** giữ nguyên CT_C = 1.29 và T_B = 0.75, rồi giải `0.75 / (1 - p_B) = 1.289`, được p_B ≈ 41.8%. Nghĩa là **nếu** các giá trị T là đúng, A3 Agentic chỉ nhanh hơn A1 khi A1 phải tìm lại hơn khoảng 42% số lần.

**Lưu ý:**
- Ngưỡng này thay đổi ngay khi bất kỳ giá trị T hoặc p nào thay đổi.
- Hiện **chưa có số đo** nào xác nhận T hay p của B và C. Ablation A3 (`step_ablation_rag.py`) chưa chạy, nên cũng chưa có latency thật của Agentic RAG.
- Kết luận ở mục này chỉ mang tính minh hoạ cho cách dùng công thức Loop.

**Liên hệ với giao diện:** chatbot A3 để chế độ Agentic là một công tắc riêng, mặc định tắt. Cách này phù hợp với nhận định trên: chỉ bật khi yêu cầu khó.

---

## 8. Hạn chế và cách lấy số liệu thật

**Hạn chế:**
1. T và p là giả định; 8 case mỗi kịch bản là ít; mô phỏng không có seed.
2. Giới hạn 10 lần lặp cắt đuôi phân phối hình học, làm CT mô phỏng thấp hơn lý thuyết một chút.
3. Mô hình chỉ có một hoạt động lặp. Chưa mô hình hoá nhánh "khách bỏ đi sau N lần" (một End Event khác).
4. T là **thời gian thao tác của nhân viên** (phút), khác latency của hệ thống (A1 hybrid khoảng 22 ms mỗi truy vấn trên CPU, xem `BAO_CAO_TONG_HOP.md`). Latency hệ thống chỉ là một phần nhỏ của T.

**Cách lấy số liệu thật:**
- **T:** bấm giờ thao tác của Sales trên cùng một bộ yêu cầu khách hàng, với từng giao diện (form cũ, `/search`, `/chat`).
- **p:** tỷ lệ phải tìm lại do khách chưa hài lòng, ghi nhận trong phiên thử. Có thể tham khảo gián tiếp từ số liệu retrieval, ví dụ tỷ lệ câu không có sản phẩm đúng trong top-5. Nhưng đó chỉ là cận dưới, vì khách có thể không hài lòng dù hệ thống trả đúng sản phẩm.
- Đưa T và p đo được vào công thức ở mục 5 và 7 để tính lại CT và ngưỡng.
