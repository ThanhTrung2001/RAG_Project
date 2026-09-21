"""
================================================================================
BƯỚC 5 — METRIC ĐÁNH GIÁ TỰ VIẾT TAY (Recall@k, nDCG@k)
================================================================================

TẠI SAO PHẢI TỰ VIẾT, KHÔNG DÙNG sklearn.metrics HAY THƯ VIỆN CÓ SẴN?
    Đây là YÊU CẦU BẮT BUỘC của rubric A1 (mục "cài đặt đúng cơ chế", 30%
    điểm): "vòng metric tự viết". Ngoài lý do điểm số, có lý do kỹ thuật
    thật: sklearn.ndcg_score được thiết kế cho bài toán ranking có điểm
    relevance LIÊN TỤC (ví dụ rating 1-5 sao), còn ở đây ta chỉ có ground
    truth NHỊ PHÂN (đúng/sai) -- công thức tự viết bên dưới khớp chính xác
    với bài toán retrieval nhị phân này.

RECALL@K LÀ GÌ?
    "Trong số các kết quả ĐÚNG đang tồn tại (theo ground truth), hệ thống
    tìm được BAO NHIÊU trong top-k trả về?" -- KHÔNG quan tâm thứ tự, chỉ
    quan tâm có nằm trong top-k hay không.

NDCG@K LÀ GÌ, KHÁC RECALL Ở ĐÂU?
    nDCG (normalized Discounted Cumulative Gain) khắt khe hơn: nó PHẠT NẶNG
    nếu kết quả đúng nằm ở HẠNG THẤP (ví dụ hạng 8 thay vì hạng 1) -- vì
    người dùng thực tế ít khi kéo xuống xem hết top-10. Công thức dùng
    log2(rank+1) làm mẫu số: rank càng lớn, mẫu số càng lớn, đóng góp của
    kết quả đó vào điểm càng nhỏ dần.

VÍ DỤ TÍNH TAY (để đối chiếu khi code chạy ra số):
    Câu query có 2 kết quả đúng: id={3, 7}. Hệ thống trả về top-5: [3,1,7,2,9]
        Recall@5 = 2/2 = 1.0   (tìm đủ cả 2, thứ tự không quan trọng)
        DCG   = 1/log2(2) + 0 + 1/log2(4) + 0 + 0 = 1.0 + 0.5 = 1.5
        IDCG  = 1/log2(2) + 1/log2(3) = 1.0 + 0.63 = 1.63   (nếu 2 kết quả
                đúng nằm ở hạng 1,2 -- kịch bản LÝ TƯỞNG NHẤT có thể)
        nDCG@5 = 1.5 / 1.63 ≈ 0.92
    -> nDCG thấp hơn 1.0 một chút vì id=7 (đúng) nằm ở hạng 3 thay vì hạng 2
       lý tưởng -- đây chính là thông tin mà Recall (=1.0, "hoàn hảo") KHÔNG
       thấy được.
"""
import math


def recall_at_k(retrieved_ids, relevant_ids, k):
    """
    retrieved_ids: list id trả về từ search(), ĐÃ SẮP theo thứ hạng (rank 1 trước)
    relevant_ids:  list/set id ĐÚNG theo ground truth (từ file eval_set.jsonl)
    """
    if not relevant_ids:
        # Câu query không có ground truth hợp lệ -- không tính được Recall,
        # trả None để evaluate_all() biết mà BỎ QUA câu này khi tính trung bình
        # (thay vì tính nhầm thành 0, làm sai lệch điểm trung bình).
        return None

    top_k = set(retrieved_ids[:k])            # chỉ xét k kết quả đầu tiên
    hit = len(top_k & set(relevant_ids))      # "&" = phép giao tập hợp -- đếm
                                                # số phần tử CÓ MẶT Ở CẢ HAI set
    return hit / len(relevant_ids)


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """
    Cài đặt nDCG@k với relevance NHỊ PHÂN: đúng=1.0, sai=0.0
    (khác với nDCG "đầy đủ" hỗ trợ relevance nhiều mức như sklearn, nhưng
    khớp đúng bản chất bài toán search ảnh/text ở đây -- 1 kết quả hoặc
    đúng hoặc sai, không có mức độ "hơi đúng").
    """
    relevant_ids = set(relevant_ids)   # set để tra cứu "in" nhanh (O(1)) thay vì
                                          # quét cả list (O(n)) mỗi lần kiểm tra

    # --- Tính DCG (Discounted Cumulative Gain) của kết quả THẬT ---
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        # i bắt đầu từ 0 (Python enumerate), nhưng công thức nDCG chuẩn đánh
        # số hạng bắt đầu từ 1 -- nên dùng log2(i + 2), tức là log2(rank + 1)
        # với rank = i + 1.
        dcg += rel / math.log2(i + 2)

    # --- Tính IDCG (Ideal DCG) -- DCG trong kịch bản LÝ TƯỞNG NHẤT có thể ---
    # tức là nếu TẤT CẢ kết quả đúng đều nằm ở top đầu (hạng 1, 2, 3...)
    ideal_hits = min(len(relevant_ids), k)   # không thể có nhiều hơn k kết quả
                                                # đúng trong top-k, dù ground truth
                                                # có nhiều hơn k kết quả đúng đi nữa
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        # Xảy ra khi relevant_ids rỗng hoặc k=0 -- tránh chia cho 0
        return None
    return dcg / idcg   # chuẩn hoá (normalize) DCG thật theo DCG lý tưởng,
                          # cho ra điểm trong khoảng [0, 1], dễ so sánh giữa các query


def evaluate_all(search_fn, eval_set, k=10, **search_kwargs):
    """
    Chạy search_fn (chính là search() trong search_core.py) trên TOÀN BỘ
    eval_set, tính Recall@k và nDCG@k trung bình.

    search_fn:     hàm search() -- truyền vào thay vì import trực tiếp, để
                   file này KHÔNG phụ thuộc cứng vào search_core.py (dễ test
                   độc lập, dễ tái sử dụng cho hệ thống khác).
    eval_set:      list dict {"query":..., "query_type":..., "relevant_ids":[...]}
    **search_kwargs: các tham số khác truyền thẳng vào search(), ví dụ
                   components=["bm25"] khi chạy ablation.
    """
    recalls, ndcgs = [], []

    for item in eval_set:
        results = search_fn(item["query"], item["query_type"], k=k, **search_kwargs)
        retrieved_ids = [r["id"] for r in results]   # chỉ cần id để tính metric,
                                                        # không cần ảnh/caption

        r = recall_at_k(retrieved_ids, item["relevant_ids"], k)
        n = ndcg_at_k(retrieved_ids, item["relevant_ids"], k)

        # Chỉ cộng vào danh sách nếu tính được (không phải None) -- xem giải
        # thích ở recall_at_k() phía trên.
        if r is not None:
            recalls.append(r)
        if n is not None:
            ndcgs.append(n)

    return {
        "recall@k": sum(recalls) / len(recalls) if recalls else None,
        "ndcg@k": sum(ndcgs) / len(ndcgs) if ndcgs else None,
        "n_queries": len(eval_set)
    }
