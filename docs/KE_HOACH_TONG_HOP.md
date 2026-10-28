# KẾ HOẠCH TỔNG HỢP — A1, A2, A3

Cập nhật: 23/10/2026. Trạng thái dựa trên code và artifact trong repo; không đồng nghĩa toàn bộ dịch vụ đã được chạy lại trên máy cá nhân.

## 1. Trạng thái đã kiểm chứng

| Phần | Hiện có | Còn cần làm |
|---|---|---|
| A1 — Search | Catalog 2.000 sản phẩm; BM25 + dense ảnh SigLIP + RRF; rerank MiniLM/DeBERTa; API và frontend; báo cáo kết quả component, trường index, embedding, reranker | Củng cố eval tự nhiên; đo lại sweep theo qrels hiện tại; kiểm tra chạy trên máy demo |
| A2 — Paper | `BAO_CAO_A2.md`; tích hợp checkpoint DeBERTa distill từ RankGPT | Chuẩn bị giải thích khác biệt giữa checkpoint distill và RankGPT gọi LLM trực tiếp |
| A3 — RAG | Naive và Agentic tối giản; chat chữ/ảnh; memory RAM; ablation có log; báo cáo và phân tích lỗi | Hoàn thiện hội thoại chi tiết, kiểm chứng nâng cấp bằng bộ eval mới; củng cố metric |
| BPM — IS6003 | `bpm_event_log_simulation.py` | Kết quả mô phỏng là giả định, không dùng làm bằng chứng thực nghiệm RAG; không thấy báo cáo BPM riêng trong docs hiện tại |

### Artifact đánh giá hiện có

- `data/eval_set.jsonl`: 250 query khác nhau, đủ qrels 0–3 và `answer_should_mention`, dùng chung A1/A3. 50 câu viết tay + 200 câu template. 240 câu có một sản phẩm relevant, 9 câu có hai, 1 câu có bốn.
- `data/qrels_pool.jsonl`: 5.064 cặp query–sản phẩm. Qrels do LLM chấm trước, người rà nhãn 2–3 theo báo cáo; chưa rà toàn bộ 0–1 hay đo đồng thuận.
- `data/rag_ablation_details.jsonl`: 1.500 dòng = 250 × 6 cấu hình, không có dòng error. Query và relevant_ids khớp eval hiện tại; trung bình trong bảng tổng hợp khớp log.
- `data/multiturn_ablation_details.jsonl`: 500 hội thoại = 250 follow-up + 250 đổi chủ đề; chỉ đánh giá truy hồi với 4 câu follow-up cố định.
- `data/image_rag_ablation_details.jsonl`: 300 dòng = 100 × {text, image, image+text}. Ảnh catalog biến đổi nhẹ, chưa đại diện ảnh người dùng chụp.
- Có kết quả embedding và reranker trong `data/*ablation_results.json`. Các artifact trên là kết quả cũ, không phải kết quả nâng cấp ngày 23/10.

## 2. Model, kiến trúc và kết quả đang dùng

- Dense hiện tại: `google/siglip-base-patch16-224`, 768 chiều; CLIP là baseline ban đầu, không còn là model serving hiện tại.
- Ablation dense-only: Recall@10 CLIP B/32 = 0,657; CLIP L/14 = 0,758; SigLIP = 0,920. Nguồn: `data/embedding_model_ablation_results.json`. Số 0,967/0,767 trong plan cũ không dùng làm kết quả hiện tại.
- Ingestion: Shopify → catalog → embeddings → FAISS IndexFlatIP ảnh + BM25 title/description. Text embeddings có script tạo nhưng chưa là nhánh dense text trong search đang chạy.
- Naive: truy hồi → context → sinh câu trả lời. Agentic: truy hồi → judge → có thể rewrite và truy hồi lại, tối đa 2 lần. Đây là Agentic tối giản, không phải hệ agent đa công cụ hoàn chỉnh.
- Naive Recall@5 = 0,983, nDCG đa mức@5 = 0,878. Rerank cải thiện thứ hạng nhưng chưa chứng minh cải thiện generation.
- Agentic Recall@5 = 0,968; chưa chứng minh tốt hơn Naive. Rewrite có thể mất tên riêng, bỏ kết quả tốt lượt đầu; không nên chọn Agentic làm mặc định chỉ vì phức tạp hơn.
- Memory cũ có 3 lượt RAM, dùng rewrite và prompt, nhớ tên 3 sản phẩm. Memory nâng cấp bổ sung định danh, thứ tự nguồn của lượt mới nhất và sản phẩm đang tập trung; vẫn mất khi restart, chưa hỗ trợ chia sẻ state giữa nhiều worker.

## 3. Đối chiếu yêu cầu IS6303

| Yêu cầu | Kết luận và việc còn thiếu |
|---|---|
| Hệ thống đủ thành phần | Có dữ liệu, ingestion, retrieval, LLM, API, UI. Cần demo end-to-end trên máy nộp bài; không tự coi recommendation là bắt buộc khi đề không yêu cầu |
| A3 nâng từ A1 | Có tái sử dụng search A1, thêm grounding, generation, chat và memory |
| Cải tiến + ablation | Đã có kết quả rerank, memory và ảnh+text. Các nâng cấp hội thoại mới cần đánh giá riêng; chưa được ghi số cũ thành số mới |
| Ground truth ≥50 mẫu/người | Đủ 250 dòng nhưng 200 template chưa chứng minh 50 mẫu/người. Cần người tạo/chấm, rà chéo và câu tự nhiên |
| Đánh giá retrieval và generation | Có metric tự viết; chưa thể gọi là đã kiểm chứng đầy đủ chất lượng trả lời |
| Báo cáo, slide, video | Có Markdown; chưa thấy slide/video/file xuất nộp trong repo |
| Trách nhiệm cá nhân | Có tài liệu học và giải thích; cần phân công, luyện trả lời chéo và tự chạy demo |
| RAPTOR | Có lý thuyết trong `HOC_LY_THUYET.md`, chưa triển khai. Nếu là yêu cầu bắt buộc cần bổ sung; nếu là gợi ý cần giải thích lựa chọn và thử nghiệm phù hợp |

## 4. Nâng cấp hỏi chi tiết sản phẩm qua nhiều lượt — 23/10/2026

### Phạm vi triển khai

- `product_dialogue.py`: định tuyến `search`, `detail`, `clarify` dựa trên ý định và tham chiếu sản phẩm; quy tắc rõ ràng trước, LLM JSON cho trường hợp còn lại.
- `rag_core.py`: giữ IDs của tất cả nguồn theo lượt; giữ danh sách nguồn đánh số gần nhất và focus theo session; detail dùng `get_product(id)` lấy brand, category, mô tả từ catalog, không search lại toàn bộ.
- `app.py`: `/api/v1/ask` nhận `selected_product_id` và `product_dialogue` (mặc định true). `product_dialogue=false` giữ luồng cũ để ablation. Python `answer`/`answer_agentic` cũng nhận hai tùy chọn này.
- `frontend/chat.html`: bấm thẻ sản phẩm chọn ID để hỏi tiếp; hiện luồng xử lý; search mới/reset xóa selection. Ảnh mới vẫn đi luồng tìm kiếm.
- Luồng detail dùng chung cho Naive/Agentic, bỏ qua vòng judge/rewrite vì đã xác định sản phẩm. Không có giá/thông số thì prompt buộc nói thiếu dữ liệu, không chuyển sang sản phẩm khác để lấp chỗ trống.
- Context search baseline vẫn tên + score; detail dùng thêm dữ liệu catalog. Số thứ tự trong câu trả lời detail được đánh lại theo nguồn của chính lượt đó.

### Cách phân biệt

| Ví dụ | Luồng |
|---|---|
| “Tìm lều chống nước”, “recommend a waterproof tent” | Search mới |
| “Có sản phẩm khác/rẻ hơn không?” | Search theo nhu cầu nối tiếp |
| “Sản phẩm số 2 làm bằng gì?” | Detail, resolve ID theo nguồn mới nhất |
| “Cái này chất liệu gì?” sau khi bấm chọn | Detail trên selected ID |
| “Cái đó chất liệu gì?” khi có nhiều sản phẩm, chưa chọn | Clarify, không tự đoán sản phẩm đầu |
| “Tôi cần máy pha cà phê” sau khi đang xem lều | Search, không để focus cũ ép sang detail |
| Hỏi về sản phẩm ở thẻ cũ | Bấm thẻ cũ để gửi ID chính xác; không đoán thứ tự của lượt cũ |

### Kiểm chứng và giới hạn

- Đã pass 15 test offline (`python -m unittest discover -s tests -v`), gồm định tuyến EN/VI, ID=0, sản phẩm thứ 4–5, tham chiếu sai, hỏi nhiều lượt, đổi chủ đề, so sánh, reset, tách session và tích hợp HTTP API bằng FastAPI TestClient.
- Test dùng stub catalog/LLM kiểm tra luồng và context; không phải bằng chứng LLM trả lời tốt. Python/JavaScript đã kiểm tra cú pháp. Chưa kiểm tra giao diện tương tác bằng trình duyệt.
- Không kết nối được Ollama mặc định `localhost:11434` trong phiên kiểm tra 23/10, nên chưa chạy hội thoại LLM thật.
- Cần chạy Ollama thật và gán nhãn bộ hội thoại: intent, target_product_ids, facts/reference answer, expected abstention. Đo intent accuracy, target-ID accuracy, answer correctness, faithfulness, citation correctness, clarification và latency.
- Ablation: baseline (`product_dialogue=false`) / routing + IDs / routing + IDs + catalog detail. Bản hiện tại có công tắc baseline/full; cấu hình IDs-only cần thêm cho thí nghiệm ba mức.
- Router kết hợp rule/LLM vẫn có thể sai; chưa có độ chính xác thực nghiệm. So sánh chỉ các sản phẩm đã xác định trong ngữ cảnh, chưa phải bộ lọc giá/tồn kho chuẩn hóa.

## 5. Các điểm phải xử lý trong đánh giá

1. 32–47/250 mẫu mỗi cấu hình thiếu Faithfulness và bị bỏ khỏi trung bình. Công bố mẫu số, lý do thiếu; không đánh đồng 0 lỗi chạy với chấm đủ mẫu.
2. Model sinh và chấm giống nhau. Cần rà người hoặc judge độc lập, đo nhiều lần/khoảng tin cậy; ngưỡng “3 điểm” trong báo cáo chưa là kiểm định thống kê.
3. Answer relevancy hiện là cosine câu hỏi–trả lời, citation chỉ regex `[n]`, mention hit là substring. Chưa thay thế answer correctness/citation correctness.
4. Context recall hiện đo relevant product IDs, chưa đo phủ các fact để trả lời chi tiết.
5. RAG rerank và không rerank dùng pool mỗi nhánh 30/50, cần cố định pool khi cô lập tác động reranker.
6. Latency chạy song song trên Kaggle không đại diện serving máy cá nhân; đo lại cùng hardware, warm-up và số worker.
7. Tách dev/test để điều chỉnh prompt/router mà không tối ưu trực tiếp trên test.

## 6. Thứ tự công việc tiếp theo

- [x] Rà soát code, tài liệu và log; thay plan lỗi thời bằng trạng thái có bằng chứng.
- [x] Triển khai luồng hỏi chi tiết có định danh và tích hợp API/UI; xem kết quả kiểm tra trong phản hồi thực hiện, chưa coi là đánh giá chất lượng LLM.
- [ ] Chạy demo Ollama thật: search → chọn sản phẩm → hỏi thuộc tính → hỏi tiếp → đổi chủ đề; cả Naive và Agentic.
- [ ] Mỗi thành viên tạo/chấm ít nhất 50 mẫu phù hợp, lưu provenance và rà chéo; bổ sung hội thoại có ground truth mới.
- [ ] Đánh giá nâng cấp detail bằng ablation, có correctness và citation correctness; giữ nguyên artifact cũ để so sánh.
- [ ] Củng cố prompt/router theo lỗi đo được; chỉ sau đó cân nhắc dense text/parent–child/RAPTOR.
- [ ] Đồng bộ báo cáo A3 với code sau khi có số liệu mới; ghi rõ phần chưa đo, không thay kết quả bằng kỳ vọng.
- [ ] Làm slide, video, xuất báo cáo theo yêu cầu nộp; cả nhóm luyện trả lời về model, qrels, ablation, lỗi và giải pháp.

## 7. Bản đồ tài liệu và mã nguồn

| Nhóm | File chính |
|---|---|
| Cài đặt và chạy | `README.md`, `requirements.txt`, `.env.example` |
| A1 | `step1_build_catalog.py`, `step2_build_embeddings.py`, `step3_build_index.py`, `search_core.py`, `metrics.py`, `step8/9/10/12/13/14*`, `docs/BAO_CAO_TONG_HOP.md` |
| A2 | `docs/BAO_CAO_A2.md`, reranker DeBERTa trong `search_core.py`, `step_ablation_rerank.py` |
| A3 runtime | `rag_core.py`, `product_dialogue.py`, `app.py`, `frontend/chat.html` |
| A3 đánh giá | `rag_metrics.py`, `step15/16/17/18/19*`, `step_ablation_rag.py`, `data/*results.json`, `data/*details.jsonl`; regression offline trong `tests/test_product_dialogue.py` |
| Báo cáo A3 | `docs/BAO_CAO_A3_NEWEST.md` là bản bố cục mới; `BAO_CAO_A3.md` và `BAO_CAO_A3_IS6303.md` là các bản song song, cần chọn bản nộp và đồng bộ |
| Học/bảo vệ | `docs/GIAI_THICH_HE_THONG_RAG.md`, `docs/HOC_LY_THUYET.md`, `docs/CONG_THUC_DO_LUONG.md` |
| Kaggle | `kaggle/make_kaggle_rag_bundle.py`, `kaggle/kaggle_run_rag_ablation.py`, các script ablation/build |

Không chạy lại step15 một cách tùy tiện: script mở rộng ghi đè eval và không được thiết kế để chạy lặp trên bộ 250 câu hiện có.
