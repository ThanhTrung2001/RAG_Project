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

MRR LÀ GÌ, KHÁC nDCG Ở ĐÂU?
    MRR chỉ nhìn kết quả ĐÚNG ĐẦU TIÊN xuất hiện ở hạng nào -- 1/hạng đó.
    Không quan tâm các kết quả đúng còn lại (nếu có). Phù hợp mô phỏng
    hành vi người dùng thực tế: "tôi chỉ cần thấy 1 sản phẩm đúng là đủ,
    tôi sẽ dừng kéo xuống ngay khi thấy nó" -- khác nDCG (chấm điểm CẢ
    DANH SÁCH) và Recall (đếm % tìm được, không quan tâm thứ tự).

nDCG ĐA MỨC (0/1/2) LÀ GÌ, KHÁC BẢN NHỊ PHÂN Ở ĐÂU? (ndcg_at_k_graded())
    Bản nhị phân ở trên coi MỌI kết quả sai là như nhau (0 điểm), dù nó
    "sai hoàn toàn khác lĩnh vực" hay "sai nhưng cùng category, có thể vẫn
    hữu ích". ndcg_at_k_graded() phân biệt 3 mức: đúng tuyệt đối (2), cùng
    category nhưng không phải đúng id (1), khác hẳn (0) -- category được
    tra ĐỘNG từ catalog.jsonl lúc chấm điểm, KHÔNG cần đổi định dạng
    eval_set.jsonl đã viết ở Bước 6.
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


def mrr(retrieved_ids, relevant_ids):
    """
    MRR (Mean Reciprocal Rank) của 1 CÂU query -- "hạng của kết quả đúng đầu
    tiên xuất hiện trong danh sách trả về, đảo ngược lại".
        - Kết quả đúng ở hạng 1 -> 1/1 = 1.0 (điểm cao nhất có thể)
        - Kết quả đúng ở hạng 4 -> 1/4 = 0.25
        - Không có kết quả đúng nào trong toàn bộ retrieved_ids -> 0.0
    Khác Recall/nDCG ở chỗ: MRR CHỈ quan tâm kết quả ĐÚNG ĐẦU TIÊN, không
    quan tâm có tìm được HẾT các kết quả đúng hay không -- phù hợp để đo
    "người dùng phải kéo xuống bao xa mới thấy 1 kết quả dùng được", tách
    biệt với Recall (đo "tìm được bao nhiêu %") và nDCG (đo thứ hạng CỦA
    TẤT CẢ kết quả đúng, không chỉ cái đầu tiên).
    """
    if not relevant_ids:
        return None
    relevant_ids = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0   # không có kết quả đúng nào trong toàn bộ danh sách trả về


def ndcg_at_k_graded(retrieved_ids, relevant_ids, k, catalog):
    """
    nDCG@k với relevance ĐA MỨC (0/1/2), khác ndcg_at_k() (chỉ nhị phân 0/1):
        rel=2: đúng chính xác id (nằm trong relevant_ids)
        rel=1: SAI id nhưng CÙNG category với 1 trong các id đúng -- "gần
               đúng", vẫn có ích cho người dùng dù không phải sản phẩm họ hỏi
        rel=0: category khác hẳn -- không liên quan

    catalog: dict {id: row} (từ catalog.jsonl) -- cần để tra category của
             MỖI id trả về VÀ của các relevant_ids, không tính sẵn trong
             eval_set.jsonl (giữ nguyên định dạng eval_set.jsonl cũ, category
             được tra ĐỘNG lúc chấm điểm thay vì lưu tĩnh trong ground truth).

    So với ndcg_at_k() nhị phân: đây là bản CHẶT CHẼ HƠN, phạt nặng hơn nếu
    hệ thống trả về sản phẩm cùng category nhưng xếp hạng thấp -- nhưng
    cũng "thưởng" một phần cho kết quả cùng category dù không phải id đúng
    tuyệt đối (nhị phân coi đó là sai hoàn toàn, 0 điểm).
    """
    relevant_ids = set(relevant_ids)
    relevant_categories = {
        catalog[rid]["category"] for rid in relevant_ids if rid in catalog
    }

    def relevance(doc_id):
        if doc_id in relevant_ids:
            return 2.0
        row = catalog.get(doc_id)
        if row and row["category"] in relevant_categories:
            return 1.0
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        dcg += relevance(doc_id) / math.log2(i + 2)

    # IDCG lý tưởng: xếp hết các rel=2 lên trước (tối đa len(relevant_ids)),
    # rồi lấp đầy phần còn lại của top-k bằng rel=1 (giả định luôn có đủ sản
    # phẩm cùng category để lấp -- hợp lý với catalog vài nghìn sản phẩm/
    # category không quá hẹp).
    n_perfect = min(len(relevant_ids), k)
    idcg = sum(2.0 / math.log2(i + 2) for i in range(n_perfect))
    idcg += sum(1.0 / math.log2(i + 2) for i in range(n_perfect, k))

    if idcg == 0:
        return None
    return dcg / idcg


def evaluate_all(search_fn, eval_set, k=10, catalog=None, **search_kwargs):
    """
    Chạy search_fn (chính là search() trong search_core.py) trên TOÀN BỘ
    eval_set, tính Recall@k, nDCG@k, MRR và latency trung bình.

    search_fn:     hàm search() -- truyền vào thay vì import trực tiếp, để
                   file này KHÔNG phụ thuộc cứng vào search_core.py (dễ test
                   độc lập, dễ tái sử dụng cho hệ thống khác).
    eval_set:      list dict {"query":..., "query_type":..., "relevant_ids":[...]}
    catalog:       dict {id: row} -- CHỈ cần nếu muốn có thêm "ndcg@k_graded"
                   (relevance đa mức 0/1/2, xem ndcg_at_k_graded()). None ->
                   bỏ qua, không tính (không phá code cũ gọi evaluate_all()
                   mà không truyền catalog).
    **search_kwargs: các tham số khác truyền thẳng vào search(), ví dụ
                   components=["bm25"] khi chạy ablation.
    """
    import time

    recalls, ndcgs, mrrs, ndcgs_graded = [], [], [], []
    latencies = []   # thời gian mỗi lần gọi search_fn(), tính bằng mili-giây

    for item in eval_set:
        t0 = time.perf_counter()
        results = search_fn(item["query"], item["query_type"], k=k, **search_kwargs)
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved_ids = [r["id"] for r in results]   # chỉ cần id để tính metric,
                                                        # không cần ảnh/caption

        r = recall_at_k(retrieved_ids, item["relevant_ids"], k)
        n = ndcg_at_k(retrieved_ids, item["relevant_ids"], k)
        m = mrr(retrieved_ids, item["relevant_ids"])

        # Chỉ cộng vào danh sách nếu tính được (không phải None) -- xem giải
        # thích ở recall_at_k() phía trên.
        if r is not None:
            recalls.append(r)
        if n is not None:
            ndcgs.append(n)
        if m is not None:
            mrrs.append(m)
        if catalog is not None:
            ng = ndcg_at_k_graded(retrieved_ids, item["relevant_ids"], k, catalog)
            if ng is not None:
                ndcgs_graded.append(ng)

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else None
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else None

    return {
        "recall@k": sum(recalls) / len(recalls) if recalls else None,
        "ndcg@k": sum(ndcgs) / len(ndcgs) if ndcgs else None,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else None,
        "ndcg@k_graded": sum(ndcgs_graded) / len(ndcgs_graded) if ndcgs_graded else None,
        "latency_ms_avg": sum(latencies) / len(latencies) if latencies else None,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "n_queries": len(eval_set)
    }
