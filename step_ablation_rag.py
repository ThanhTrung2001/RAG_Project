"""
================================================================================
ABLATION CHO A3 — SO SÁNH NAIVE RAG vs AGENTIC RAG BẰNG METRIC CỤ THỂ
================================================================================

TẠI SAO FILE NÀY BẮT BUỘC PHẢI CÓ (không phải tuỳ chọn)?
    Rubric đồ án ghi rõ: "Cải tiến có kiểm chứng: hệ thống phải có điểm cải
    tiến so với baseline VÀ CHỨNG MINH HIỆU QUẢ CẢI TIẾN BẰNG THÍ NGHIỆM
    ABLATION." Có code Naive RAG và Agentic RAG (2 hàm khác nhau trong
    rag_core.py) là ĐIỀU KIỆN CẦN, nhưng chưa ĐO được cái nào tốt hơn, tốt
    hơn ở đâu, đánh đổi gì -- đó là điều kiện ĐỦ mà rubric yêu cầu, và đó
    chính là việc file này làm.

    Đây là bản mở rộng của tư duy đã áp dụng ở A1 (step8_run_ablation.py:
    so BM25-only/Dense-only/Hybrid) -- giờ áp dụng lại cho tầng RAG: so
    Naive RAG / Agentic RAG trên CÙNG một bộ câu hỏi, đo bằng metric cụ thể,
    không chỉ mô tả bằng lời "agentic thông minh hơn".

3 METRIC ĐO ĐƯỢC MÀ KHÔNG CẦN NGƯỜI CHẤM TAY (quan trọng vì đánh giá RAG
thường cần con người đọc từng câu trả lời -- ở đây dùng 3 proxy tự động
được vì thiết kế prompt trong rag_core.py đã có cấu trúc rõ ràng):

    1. Recall@k của sources -- TÁI SỬ DỤNG eval_set.jsonl và recall_at_k()
       đã viết ở A1 (metrics.py). Vì rag_core.answer() gọi lại đúng
       search_core.search() bên trong, sources trả về CHÍNH LÀ kết quả
       search -- đo được y hệt cách đã đo ở A1.

    2. Tỷ lệ trích dẫn hợp lệ (citation compliance) -- prompt trong
       rag_core.build_prompt() YÊU CẦU LLM trích dẫn bằng [1], [2]... Đếm
       xem câu trả lời CÓ chứa ít nhất 1 trích dẫn hợp lệ hay không -- đây
       là proxy tự động cho "grounding" (LLM có bám vào nguồn không) mà
       KHÔNG cần con người đọc từng câu để đánh giá đúng/sai nội dung.

    3. Latency trung bình (giây/lần gọi) -- đo bằng time.time(), phản ánh
       trực tiếp đánh đổi mà BAO_CAO_BPM.md mục 7 đã phát hiện: Agentic có
       thể CHẬM HƠN vì chạy ngầm nhiều lượt search+đánh giá.

CHẠY: python step_ablation_rag.py
YÊU CẦU TRƯỚC: đã có data/eval_set.jsonl (từ A1) VÀ Ollama đang chạy
      (rag_core.py cần gọi được LLM).
"""
import json
import re
import time

from rag_core import answer as naive_answer
from rag_core import answer_agentic
from metrics import recall_at_k

EVAL_SET_PATH = "data/eval_set.jsonl"
K = 5   # dùng k nhỏ hơn A1 (k=10) vì RAG chỉ nên đưa vài nguồn vào prompt,
         # không phải toàn bộ top-10 -- context dài làm LLM dễ lạc hướng

CITATION_PATTERN = re.compile(r"\[\d+\]")   # khớp đúng định dạng [1], [2]... mà prompt yêu cầu


def load_eval_set():
    rows = []
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def has_valid_citation(answer_text: str) -> bool:
    """
    Kiểm tra câu trả lời có chứa ít nhất 1 trích dẫn dạng [n] không.
    Đây KHÔNG kiểm tra trích dẫn đó có ĐÚNG SẢN PHẨM hay không (việc đó cần
    người đọc) -- chỉ kiểm tra LLM có TUÂN THỦ ĐỊNH DẠNG grounding đã yêu
    cầu trong prompt hay không. Một proxy thô nhưng tự động hoá được.
    """
    return bool(CITATION_PATTERN.search(answer_text))


def run_ablation_config(name, answer_fn, eval_set):
    """
    Chạy 1 cấu hình (Naive hoặc Agentic) trên toàn bộ eval_set, đo 3 metric.
    answer_fn: rag_core.answer hoặc rag_core.answer_agentic -- cùng chữ ký
               gọi (query, k=...), khác nhau ở cơ chế BÊN TRONG.
    """
    recalls = []
    citation_hits = 0
    latencies = []

    for item in eval_set:
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        start = time.time()
        result = answer_fn(query, k=K)
        latency = time.time() - start
        latencies.append(latency)

        retrieved_ids = [s["id"] for s in result["sources"]]
        r = recall_at_k(retrieved_ids, relevant_ids, K)
        if r is not None:
            recalls.append(r)

        if has_valid_citation(result["answer"]):
            citation_hits += 1

    n = len(eval_set)
    return {
        "name": name,
        "recall@k": sum(recalls) / len(recalls) if recalls else None,
        "citation_rate": citation_hits / n if n else None,
        "avg_latency_sec": sum(latencies) / len(latencies) if latencies else None,
    }


def main():
    eval_set = load_eval_set()
    print(f"Chạy ablation RAG trên {len(eval_set)} câu hỏi, k={K}")
    print("Lưu ý: cần Ollama đang chạy -- mỗi câu hỏi sẽ gọi LLM thật.\n")

    configs = [
        ("Naive RAG", lambda q, k: naive_answer(q, k=k)),
        ("Agentic RAG", lambda q, k: answer_agentic(q, k=k)),
    ]

    results = []
    for name, fn in configs:
        print(f"Đang chạy: {name}...")
        results.append(run_ablation_config(name, fn, eval_set))

    print(f"\n{'Cấu hình':<15} {'Recall@'+str(K):<12} {'Tỷ lệ trích dẫn':<18} {'Latency TB (s)':<15}")
    print("-" * 60)
    for r in results:
        recall_str = f"{r['recall@k']:.3f}" if r["recall@k"] is not None else "N/A"
        citation_str = f"{r['citation_rate']*100:.1f}%" if r["citation_rate"] is not None else "N/A"
        latency_str = f"{r['avg_latency_sec']:.2f}" if r["avg_latency_sec"] is not None else "N/A"
        print(f"{r['name']:<15} {recall_str:<12} {citation_str:<18} {latency_str:<15}")

    print("\nCách đọc bảng:")
    print("- Recall@k giống hệt A1: 2 cấu hình này gọi cùng search_core.search()")
    print("  bên dưới nên Recall thường KHÔNG khác nhau nhiều -- điểm khác biệt")
    print("  thật sự nằm ở 2 cột còn lại.")
    print("- Tỷ lệ trích dẫn thấp -> LLM không tuân thủ grounding -> cần sửa lại")
    print("  prompt trong rag_core.build_prompt() cho rõ ràng hơn.")
    print("- Latency TB cao hơn ở Agentic là BÌNH THƯỜNG (chạy ngầm nhiều lượt) --")
    print("  đây chính là số liệu cần đối chiếu với phân tích đánh đổi trong")
    print("  BAO_CAO_BPM.md mục 7 (ngưỡng ~42% mới đáng bật Agentic).")


if __name__ == "__main__":
    main()
