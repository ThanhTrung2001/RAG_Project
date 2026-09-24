"""
================================================================================
BƯỚC 8 — ABLATION STUDY (chạy nhiều cấu hình, so sánh số liệu)
================================================================================

TẠI SAO ABLATION PHẢI "MỖI LẦN CHỈ ĐỔI ĐÚNG 1 BIẾN"?
    Đây là tư duy thực nghiệm khoa học cốt lõi: nếu đổi CÙNG LÚC 2 thứ
    (ví dụ vừa tắt BM25 vừa đổi model embedding) rồi thấy điểm giảm, ta
    KHÔNG THỂ BIẾT cái nào gây ra sự giảm đó -- có thể do thiếu BM25, có
    thể do model mới tệ hơn, hoặc do cả 2 cộng lại. Ablation yêu cầu: mỗi
    lần chạy chỉ đổi ĐÚNG 1 biến, mọi thứ khác giữ nguyên -- khi đó CHÊNH
    LỆCH điểm số giữa 2 lần chạy CHÍNH LÀ đóng góp thuần của biến đó.

TẠI SAO SCRIPT NÀY "DÙNG CHUNG" VỚI API THẬT (app.py)?
    Cả 2 đều gọi CÙNG 1 hàm search() với tham số "components" khác nhau --
    không có logic ablation nào tách biệt, viết riêng. Điều này quan trọng
    vì: (1) đảm bảo số liệu ablation PHẢN ÁNH ĐÚNG hành vi thật của hệ
    thống khi demo (không có 2 phiên bản code có thể lệch nhau), và (2)
    khi demo trực tiếp, tick/bỏ tick checkbox trên UI CHÍNH LÀ đang chạy
    lại ablation theo thời gian thực.

CHẠY: python step8_run_ablation.py
YÊU CẦU TRƯỚC: đã có data/eval_set.jsonl (copy từ eval_set_template.jsonl
      rồi viết thêm cho đủ 20-40 câu -- xem README.md).
"""
import json

from search_core import search, available_components
from metrics import evaluate_all

EVAL_SET_PATH = "data/eval_set.jsonl"
K = 10   # đánh giá trên top-10 -- khớp với k mặc định của giao diện search thật

# Mỗi cấu hình dưới đây là 1 "thí nghiệm" -- so với cấu hình "hybrid" (đầy đủ),
# 2 cấu hình còn lại mỗi cái CHỈ THIẾU ĐÚNG 1 THÀNH PHẦN. Đây chính là thiết
# kế ablation "có kiểm soát" mà rubric yêu cầu (20% điểm).
CONFIGS = [
    {"name": "bm25 only",           "components": ["bm25"]},
    {"name": "dense only",          "components": ["dense"]},
    {"name": "hybrid (bm25+dense)", "components": ["bm25", "dense"]},
    {"name": "hybrid + rerank",     "components": ["bm25", "dense"], "use_rerank": True},
]


def load_eval_set():
    rows = []
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_catalog():
    """Đọc catalog.jsonl thành dict {id: row} -- dùng để tra category, phục vụ
    ndcg@k đa mức (xem metrics.ndcg_at_k_graded, mục 4.1 template A1: "ground
    truth đa mức: gốc=2, cùng category=1, khác=0")."""
    rows = {}
    with open("data/catalog.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def main():
    # In ra để chắc chắn registry đã đăng ký đúng những gì mong đợi --
    # nếu thiếu "dense" hoặc "bm25" ở đây, nghĩa là search_core.py có lỗi
    # import hoặc decorator không chạy đúng.
    print(f"Component đã đăng ký trong hệ thống: {available_components()}")

    eval_set = load_eval_set()
    catalog = load_catalog()
    print(f"Chạy ablation trên {len(eval_set)} câu query, k={K}\n")

    header = (f"{'Cấu hình':<24} {'Recall@'+str(K):<10} {'nDCG@'+str(K):<10} "
              f"{'nDCG_gr@'+str(K):<11} {'MRR':<8} {'Lat_avg(ms)':<12} {'Lat_p95(ms)':<12}")
    print(header)
    print("-" * len(header))

    for cfg in CONFIGS:
        # Gọi CHÍNH XÁC hàm search() mà app.py cũng gọi -- không có bản sao
        # logic riêng cho ablation, xem giải thích ở docstring đầu file.
        metrics = evaluate_all(
            search_fn=search,
            eval_set=eval_set,
            k=K,
            catalog=catalog,   # bật thêm ndcg@k_graded (đa mức), xem metrics.py
            components=cfg["components"],
            use_rerank=cfg.get("use_rerank", False),   # False cho 3 config cũ,
                                                          # True chỉ cho "hybrid + rerank"
        )

        def fmt(key, spec=".3f"):
            v = metrics[key]
            return format(v, spec) if v is not None else "N/A"

        print(f"{cfg['name']:<24} {fmt('recall@k'):<10} {fmt('ndcg@k'):<10} "
              f"{fmt('ndcg@k_graded'):<11} {fmt('mrr'):<8} "
              f"{fmt('latency_ms_avg', '.1f'):<12} {fmt('latency_ms_p95', '.1f'):<12}")

    print("\n-> Chênh lệch điểm số giữa 2 cấu hình = đóng góp của thành phần bị tắt.")
    print("-> nDCG_gr = nDCG đa mức (đúng id=2, cùng category=1, khác=0) -- xem metrics.ndcg_at_k_graded")
    print("-> MRR = hạng của kết quả đúng ĐẦU TIÊN, đảo ngược (1/hạng) -- xem metrics.mrr")
    print("-> Lat_avg/Lat_p95 = latency mỗi lần gọi search(), đo bằng time.perf_counter()")
    print("-> Copy bảng này vào báo cáo, kèm 2-3 ví dụ câu query sai cụ thể cho mỗi cấu hình")
    print("   (đọc lại từng câu trong eval_set.jsonl, xem search() trả về gì -- đây chính")
    print("   là bước 9 'error analysis', không code sẵn được vì cần bạn tự nhìn kết quả.)")


if __name__ == "__main__":
    main()
