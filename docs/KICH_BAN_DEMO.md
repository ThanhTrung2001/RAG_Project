# Kịch bản demo chatbot A3 — câu hỏi và kết quả

Kết quả dưới đây là **lần chạy thật** ngày 29/10/2026: `llama3.1` qua Ollama, 5 nguồn mỗi lượt (như giao diện), không bật rerank.
Chạy lại bằng:

```
python tests/run_product_dialogue_live.py --k 5 --output out.json --scenario naive_airfryer_auto_focus --scenario agentic_thermostat_auto_focus --scenario image_text_switch_and_replace --scenario image_only_followup
```

**Đọc bảng:**
- **Luồng** là badge "Luồng" trên giao diện: *tìm sản phẩm* (search), *hỏi chi tiết* (detail), *cần chọn sản phẩm* (clarify).
- **Focus** là sản phẩm mà câu hỏi tiếp theo kiểu "cái này", "capacity?" sẽ nói tới.
- Điểm nguồn ghi theo dạng hiện trên thẻ: `BM25 #hạng · Dense #hạng · Ảnh #hạng`.
- Câu trả lời giữ nguyên tiếng Anh của LLM; câu dài được rút gọn (…).

> **LLM có tính ngẫu nhiên.** Luồng, nguồn và focus ổn định giữa các lần chạy, nhưng câu chữ có thể khác.
> Riêng lượt **"Show me more options like that"** có 2 kết quả đều đúng: LLM đề xuất được sản phẩm khác
> (focus chuyển sang sản phẩm mới), hoặc nói không có (giữ sản phẩm cũ). Xem mục "Hai kết quả của more options" ở cuối.

---

## 1. Naive — nồi chiên

Tắt Agentic, tắt Rerank.

| # | Câu hỏi | Luồng | Nguồn / Focus | Câu trả lời |
|---|---|---|---|---|
| 1 | `10-in-1 air fryer toaster oven combo` | tìm sản phẩm | [1] **Nuwave Bravo Pro** (BM25 #1 · Dense #1)<br>[2] Breville Smart Oven (BM25 #2 · Dense #3)<br>[3] Pyramid Silicone Mat · [4] Ovenkast Lyon · [5] Baking Paper<br>**Focus: Nuwave** (câu trả lời chỉ trích dẫn 1 sản phẩm) | *"We have the Nuwave Bravo Pro Convection Air Fryer Toaster Oven Combo, which is a 10-in-1 air fryer toaster oven combo ([1])."* |
| 2 | `What material is this made of?` | hỏi chi tiết | Nuwave (tra catalog theo ID, không search) | *"…made of stainless steel. According to the product description, it says: "Material Stainless Steel" ([1])."* |
| 3 | `How much capacity?` | hỏi chi tiết | Nuwave (câu cụt → dùng focus) | *"The capacity of the Nuwave Bravo Pro … is 21 Quarts. [1]"* |
| 4 | `Show me more options like that` | tìm sản phẩm | Câu tìm: *"air fryer toaster oven combo"*. **Nuwave bị loại.**<br>[1] Breville Smart Oven · [2] Pyramid Silicone Mat · [3] Ovenkast Lyon · [4] Sweejar Baking Pans · [5] Cooling Rack<br>**Focus: Breville** | *"We don't have many options like a 10-in-1 air fryer toaster oven combo. However, you might consider [1] Breville Smart Oven, which is a toaster oven, but I couldn't find any information about it having air fryer capabilities."* |

**Điểm nhấn khi nói:**
- Lượt 1: Nuwave đứng đầu cả 2 nhánh nên RRF xếp [1].
- Lượt 2–3: không search lại, trả lời từ bản ghi catalog của đúng Nuwave.
- Lượt 4: ghi chú dưới câu trả lời *"Đã bỏ sản phẩm đang hỏi (ID 272)…"*. LLM nói thật rằng Breville chỉ là lò nướng, không rõ có chiên không khí.

---

## 2. Agentic — thermostat

Bật Agentic, tắt Rerank.

| # | Câu hỏi | Luồng | Nguồn / Focus | Câu trả lời |
|---|---|---|---|---|
| 1 | `smart thermostat with air quality sensor` | tìm sản phẩm | Giám khảo: **YES** (không tìm lại)<br>[1] **ecobee Smart Thermostat Premium** (BM25 #1 · Dense #1)<br>[2] Kelon Air Conditioner · [3] SenseCAP Sensor Hub · [4] WaterBug · [5] RAINPOINT Water Timer<br>**Focus: ecobee** | *"The ecobee Smart Thermostat Premium with Smart Sensor and Air Quality Monitor [1] has an air quality sensor."* |
| 2 | `Show me details of this product` | hỏi chi tiết | ecobee | *"…a programmable WiFi thermostat that works with Siri, Alexa, or Google Assistant. It's ENERGY STAR certified and can save you up to 26% per year… built-in air quality monitor… smoke alarm detection… compatible with most 24VAC HVAC systems…"* |
| 3 | `How does it monitor air quality?` | hỏi chi tiết | ecobee | *"…a built-in air quality monitor that alerts you to poor air quality. This monitor provides tips on how to improve the air quality and reminds you when it's time to change your furnace's air filter. [1] Note that the catalog does not provide any additional information on how the air quality monitor works…"* |
| 4 | `Show me more options like that` | tìm sản phẩm | **ecobee bị loại.** Giám khảo vòng 1: *"NO: None of the products listed match…"* → viết lại thành *"Smart thermostat with indoor air quality sensor and air purification"* → vòng 2: YES<br>Nguồn: Kelon Air Conditioner · Air Purifier · SenseCAP · WaterBug · Moisture Meter<br>**Không đề xuất gì → giữ danh sách cũ, focus vẫn là ecobee** | *"I could not find other options like a programmable WiFi thermostat with an air quality sensor."* |

**Điểm nhấn khi nói:**
- Lượt 4 cho thấy đủ vòng Agentic: tự chấm "NO" → viết lại truy vấn → tìm lại.
- Catalog chỉ có 1 thermostat, nên LLM trả lời "không có" là đúng.
- Ghi chú dưới câu trả lời: *"Không có lựa chọn khác phù hợp — vẫn giữ sản phẩm đang hỏi (ID 443)…"*. Nếu hỏi tiếp `What is the price?` thì vẫn hỏi về ecobee (catalog không có giá nên LLM sẽ nói không có).

---

## 3. Ảnh + chữ, rồi đổi chủ đề

Naive. Bấm **Ảnh** để đính kèm, gõ chữ, rồi Gửi.

| # | Câu hỏi | Luồng | Nguồn / Focus | Câu trả lời |
|---|---|---|---|---|
| 1 | Ảnh `data/images/000272.jpg` + `10-in-1 air fryer toaster oven combo` | tìm sản phẩm (ảnh + chữ) | [1] **Nuwave** (Ảnh #1 · BM25 #1 · Dense #1)<br>[2] **Breville Smart Oven** (Ảnh #2 · BM25 #2 · Dense #3)<br>[3] Pyramid Silicone Mat · [4] Mini Fry Baskets · [5] Фритюрник Medion (nồi chiên, tên tiếng Bulgaria; BM25: ngoài top 50, nhưng Ảnh #3)<br>**Focus: Nuwave** | *"We have a 10-in-1 air fryer toaster oven combo option: [1] Nuwave Bravo Pro Convection Air Fryer Toaster Oven Combo."* |
| 2 | `How much capacity?` | hỏi chi tiết | Nuwave | *"…has a capacity of 21 Quarts. [1]"* |
| 3 | `What is the weight of product 2?` | hỏi chi tiết | "product 2" = **Breville** trong danh sách lượt 1. **Focus: Breville** | *"…the Breville Smart Oven weighs [1]. Unfortunately, the catalog does not provide the weight of the product."* |
| 4 | Ảnh `data/images/000443.jpg` + `smart thermostat with air quality sensor` | tìm sản phẩm (ảnh + chữ) | Chủ đề mới → danh sách mới<br>[1] **ecobee** (Ảnh #1 · BM25 #1 · Dense #1)<br>[2] Air Purifier · [3] Kelon Air Conditioner · [4] Smart Wellness Ring · [5] RAINPOINT<br>**Focus: ecobee** | *"We have a smart thermostat with an air quality sensor: [1] ecobee Smart Thermostat Premium with Smart Sensor and Air Quality Monitor."* |
| 5 | `How does it monitor air quality?` | hỏi chi tiết | ecobee | *"…has a built-in air quality monitor that alerts you to poor air quality, provides tips on how to improve it, and reminds you when it's time to change your furnace's air filter…"* |
| 6 | `Show me more options like that` | tìm sản phẩm | Câu tìm: *"smart thermostat"*. **ecobee bị loại.**<br>Nguồn: RAINPOINT · Wellness Ring · Kelon · SenseCAP · Bluetooth Scale<br>**Không đề xuất gì → giữ danh sách lượt 4, focus vẫn là ecobee** | *"…Since you're looking for more options like that, I could not find other similar products in the list."* |

**Điểm nhấn khi nói:**
- Lượt 1: ảnh kéo được *Фритюрник Medion* vào dù BM25 không khớp chữ nào (tên tiếng Bulgaria). Đây là lý do ảnh + chữ tốt hơn chỉ chữ.
- Lượt 3: số thứ tự "product 2" luôn tính theo **danh sách search gần nhất**, không theo số [1] trong câu trả lời chi tiết.
- Lượt 3: catalog không có cân nặng nên LLM nói không có, không bịa. Câu "weighs [1]" hơi lủng củng, đó là lỗi diễn đạt của LLM.
- Lượt 4: đổi chủ đề thì danh sách và focus được thay mới hoàn toàn.

---

## 4. Chỉ ảnh, hỏi tiếp bằng chữ

**Nên chạy Naive.** Ở một lần chạy Agentic, bước tự viết lại truy vấn đã kéo *bàn chải đánh răng* lên [1] cho ảnh nồi chiên.
Bấm **Ảnh**, để trống ô chữ, rồi Gửi.

| # | Câu hỏi | Luồng | Nguồn / Focus | Câu trả lời |
|---|---|---|---|---|
| 1 | Ảnh `data/images/000272.jpg`, không gõ chữ | tìm sản phẩm (chỉ ảnh) | Chỉ có nhánh ảnh:<br>[1] **Nuwave** (Ảnh #1, cosine 1.0) · [2] Breville (0.806) · [3] Фритюрник Medion (0.707) · [4] Mini Fry Baskets · [5] Estante para horno<br>**Focus: Nuwave** | *"Based on the product list, I found a product similar to the photo you sent: [1] Nuwave Bravo Pro Convection Air Fryer Toaster Oven Combo."* |
| 2 | `What material is this made of?` | hỏi chi tiết | Nuwave | *"…made of stainless steel. According to the product description, it says "Material Stainless Steel" (Catalog record [1])."* |
| 3 | `Show me more options like that` | tìm sản phẩm | Câu tìm: *"air fryer toaster oven"*. **Nuwave bị loại.**<br>[1] **Breville** · [2] Pyramid Silicone Mat · [3] Ovenkast Lyon · [4] Baking Paper · [5] Cooling Rack<br>**Focus: Breville** | *"I found one more product that is similar to the one you're looking for: [1] Breville Smart Oven. It's an oven and toaster combination, just like the one you're interested in."* |
| 4 | `What is the weight of product 2?` | hỏi chi tiết | "product 2" của **danh sách mới (lượt 3)** = **Pyramid Silicone Mat** | *"The weight of the Pyramid Silicone Mat Non Stick Fat Reducing - UK Seller is not specified in the catalog record [1]."* |

**Điểm nhấn khi nói:**
- Lượt 1: LLM không nhìn được ảnh, chỉ đọc danh sách tên mà SigLIP tìm ra theo độ giống ảnh.
- Lượt 4: "product 2" là tấm thảm silicone vì lượt 3 đã tạo danh sách mới. Logic đúng, nhưng nghe lạ khi demo.
  **Gợi ý:** đổi thành `What is the weight of product 1?` (→ Breville), hoặc hỏi "product 2" trước lượt "more options".

---

## Hai kết quả của "Show me more options like that"

| LLM trả lời | Hệ thống làm gì | Ví dụ |
|---|---|---|
| Đề xuất ≥ 1 sản phẩm (có `[n]`) | Danh sách mới thay danh sách cũ. Focus chuyển sang sản phẩm được đề xuất (nếu chỉ có 1). Ghi chú: *"Đã bỏ sản phẩm đang hỏi (ID …) khỏi danh sách lựa chọn khác."* | Kịch bản 1 và 4: đề xuất Breville |
| Không đề xuất gì (không có `[n]`) | **Giữ danh sách và focus cũ.** Cột nguồn vẫn hiện danh sách cũ. Ghi chú: *"Không có lựa chọn khác phù hợp — vẫn giữ sản phẩm đang hỏi…"* | Kịch bản 2 và 3: giữ ecobee |

Trong cả hai trường hợp, sản phẩm đang hỏi **không bao giờ** xuất hiện lại trong danh sách nguồn.
Nếu trước đó vừa so sánh 2 sản phẩm (*"Compare product 1 and product 2"*), cả 2 đều bị loại, và câu cụt như "How much capacity?" sẽ trả lời cho cả 2.

## Trước khi demo
- Khởi động lại Ollama. Một lần chạy thử bị crash với lỗi `CUDA error: shared object initialization failed`.
- Bấm **Mới** giữa các kịch bản để xoá memory của phiên.
- Mỗi câu trả lời mất khoảng 5–30 s (Agentic có thể tới khoảng 1 phút).
