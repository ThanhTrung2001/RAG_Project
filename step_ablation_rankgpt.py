"""
================================================================================
ABLATION — SO SÁNH KHÔNG RERANK / CROSS-ENCODER / RANKGPT (model mạnh vs yếu)
================================================================================

MỤC ĐÍCH: thực nghiệm HOÁ đúng ví dụ giảng viên đưa ra trong lớp: "RankGPT
rerank không tốt vì dùng model mã nguồn mở kém hiệu quả". Thay vì chỉ trích
dẫn nhận định này, script này CHẠY THẬT với 2 model Ollama khác cỡ, đo bằng
Recall@k/nDCG@k, để có SỐ LIỆU chứng minh (hoặc bác bỏ) nhận định đó trên
chính dataset A1.

YÊU CẦU TRƯỚC KHI CHẠY:
    1. Đã có data/eval_set.jsonl (từ A1)
    2. Ollama đang chạy, đã pull ÍT NHẤT 2 model khác cỡ để so sánh, vd:
           ollama pull llama3.1        (mạnh hơn, ~4.7GB)
           ollama pull qwen2.5:0.5b    (rất nhỏ, để thấy rõ rủi ro model yếu)
       Nếu chỉ có 1 model, sửa RANKGPT_MODELS bên dưới còn 1 phần tử --
       vẫn chạy được, chỉ là không so sánh mạnh/yếu được.

CHẠY: python step_ablation_rankgpt.py
"""
import json

from search_core import search, rerank, rankgpt_rerank
from metrics import evaluate_all, recall_at_k, ndcg_at_k

EVAL_SET_PATH = "data/eval_set.jsonl"
K = 10
POOL_SIZE = 30   # lấy top-30 từ hybrid search trước khi rerank -- nhỏ hơn
                   # top_n=50 mặc định của A1 để prompt RankGPT không quá dài

# Đổi tên model cho khớp với model bạn đã "ollama pull" -- xem docstring trên.
RANKGPT_MODELS = ["llama3.1", "qwen2.5:0.5b"]


def load_eval_set():
    rows = []
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def run_config(name, rerank_fn, eval_set):
    """
    rerank_fn: hàm nhận (query, candidates) -> candidates đã sắp xếp lại,
               hoặc None nếu không rerank (dùng thẳng kết quả hybrid).
    """
    recalls, ndcgs = [], []

    for item in eval_set:
        query = item["query"]
        # Lấy pool RỘNG từ hybrid (bm25+dense), KHÔNG rerank ở bước search() --
        # để tự áp rerank_fn bên ngoài, so sánh công bằng cùng 1 pool đầu vào.
        candidates = search(query, query_type="text", k=POOL_SIZE,
                             components=["bm25", "dense"], top_n=POOL_SIZE)

        if rerank_fn is not None and candidates:
            candidates = rerank_fn(query, candidates)

        retrieved_ids = [c["id"] for c in candidates[:K]]
        r = recall_at_k(retrieved_ids, item["relevant_ids"], K)
        n = ndcg_at_k(retrieved_ids, item["relevant_ids"], K)
        if r is not None:
            recalls.append(r)
        if n is not None:
            ndcgs.append(n)

    return {
        "name": name,
        "recall@k": sum(recalls) / len(recalls) if recalls else None,
        "ndcg@k": sum(ndcgs) / len(ndcgs) if ndcgs else None,
    }


def main():
    eval_set = load_eval_set()
    print(f"Chạy ablation rerank trên {len(eval_set)} câu hỏi, pool={POOL_SIZE}, k={K}\n")

    configs = [
        ("Không rerank (hybrid thô)", None),
        ("Cross-encoder (Nogueira & Cho)", lambda q, c: rerank(q, c)),
    ]
    for model_name in RANKGPT_MODELS:
        configs.append((
            f"RankGPT ({model_name})",
            lambda q, c, m=model_name: rankgpt_rerank(q, c, llm_model=m)
        ))

    print(f"{'Cấu hình':<35} {'Recall@'+str(K):<12} {'nDCG@'+str(K):<12}")
    print("-" * 59)
    for name, fn in configs:
        print(f"Đang chạy: {name}...")
        result = run_config(name, fn, eval_set)
        r = f"{result['recall@k']:.3f}" if result["recall@k"] is not None else "N/A"
        n = f"{result['ndcg@k']:.3f}" if result["ndcg@k"] is not None else "N/A"
        print(f"{name:<35} {r:<12} {n:<12}")

    print("\nCách đọc bảng:")
    print("- So 'Không rerank' với 'Cross-encoder': đo giá trị của rerank NÓI CHUNG.")
    print("- So 2 dòng 'RankGPT (...)' với nhau: đo TRỰC TIẾP ảnh hưởng của việc")
    print("  chọn model mạnh/yếu -- đúng phát hiện paper Sun et al. (2023)/note")
    print("  giảng viên. Nếu RankGPT model yếu cho điểm THẤP HƠN cả 'không rerank',")
    print("  đó là bằng chứng cụ thể: rerank sai có thể làm KẾT QUẢ TỆ HƠN không")
    print("  làm gì cả -- một insight quan trọng cần nêu trong báo cáo.")


if __name__ == "__main__":
    main()
