"""
================================================================================
ABLATION — SWEEP THAM SỐ RRF: k × N (mục 5.2 template A1)
================================================================================

2 THAM SỐ CẦN SWEEP LÀ GÌ?
    k:  hằng số trong công thức RRF, score(doc) = sum(1/(k + rank + 1)).
        k CÀNG NHỎ -> chênh lệch điểm giữa hạng 1 và hạng 10 CÀNG LỚN (fusion
        "tin tưởng" mạnh vào thứ hạng gốc). k CÀNG LỚN -> các hạng được coi
        gần như ngang nhau (fusion "làm mờ" khác biệt thứ hạng).
    N:  số ứng viên lấy từ MỖI nhánh (BM25/Dense) TRƯỚC KHI fusion (tham số
        top_n của search()). N càng lớn -> fusion có nhiều "nguyên liệu" hơn
        để chọn, nhưng cũng có nguy cơ đưa vào nhiều ứng viên nhiễu ở hạng thấp.

CÁCH ĐỌC BẢNG KẾT QUẢ:
    Cormack et al. (2009) - paper gốc đề xuất RRF - nói rằng kết quả KHÔNG
    nhạy với k trong khoảng 20-100. Bảng dưới đây CHẠY THẬT trên dataset A1
    để tự kiểm chứng nhận định đó có đúng ở QUY MÔ NHỎ (2000 sản phẩm, 30
    câu eval) như ở đây hay không -- có thể khác paper gốc (chạy trên
    TREC, hàng triệu tài liệu).

CHẠY: python step10_sweep_rrf.py
YÊU CẦU TRƯỚC: đã có đủ data/dense.index, data/bm25.pkl, data/eval_set.jsonl
      (dùng lại search() thật của A1, không build lại gì mới).
"""
import json

from search_core import search
from metrics import evaluate_all

EVAL_SET_PATH = "data/eval_set.jsonl"
CATALOG_PATH = "data/catalog.jsonl"
K = 10

RRF_K_VALUES = [10, 60, 100]
TOP_N_VALUES = [20, 50, 100]


def load_eval_set():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_catalog():
    catalog = {}
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            catalog[row["id"]] = row
    return catalog


def main():
    eval_set = load_eval_set()
    catalog = load_catalog()

    print(f"Sweep RRF k x N -- {len(eval_set)} câu query, k={K}, components=[bm25, dense]\n")
    header = f"{'rrf_k':<8} {'N (top_n)':<12} {'Recall@'+str(K):<10} {'nDCG@'+str(K):<10} {'MRR':<8}"
    print(header)
    print("-" * len(header))

    for rrf_k in RRF_K_VALUES:
        for top_n in TOP_N_VALUES:
            metrics = evaluate_all(
                search_fn=search,
                eval_set=eval_set,
                k=K,
                catalog=catalog,
                components=["bm25", "dense"],
                top_n=top_n,
                rrf_k=rrf_k,
            )

            def fmt(key):
                v = metrics[key]
                return f"{v:.3f}" if v is not None else "N/A"

            print(f"{rrf_k:<8} {top_n:<12} {fmt('recall@k'):<10} {fmt('ndcg@k'):<10} {fmt('mrr'):<8}")
        print()

    print("-> Nếu số liệu gần như KHÔNG đổi khi k thay đổi (10 -> 60 -> 100) ở CÙNG 1 giá")
    print("   trị N -- đúng như Cormack et al. (2009) kết luận -- ghi rõ điều này vào báo")
    print("   cáo mục 5.2, kèm số liệu THẬT thay vì chỉ trích dẫn paper.")
    print("-> Nếu N (top_n) ảnh hưởng RÕ RỆT hơn k -- đây là 1 phát hiện đáng nêu: quy mô")
    print("   pool ứng viên trước fusion quan trọng hơn cách tính điểm fusion.")


if __name__ == "__main__":
    main()
