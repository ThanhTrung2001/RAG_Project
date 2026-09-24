"""
================================================================================
ABLATION CHO A3 — SO SÁNH NAIVE RAG vs AGENTIC RAG BẰNG ĐỦ METRIC RAG CHUẨN
================================================================================

TẠI SAO FILE NÀY BẮT BUỘC PHẢI CÓ (không phải tuỳ chọn)?
    Rubric đồ án ghi rõ: "Cải tiến có kiểm chứng: hệ thống phải có điểm cải
    tiến so với baseline VÀ CHỨNG MINH HIỆU QUẢ CẢI TIẾN BẰNG THÍ NGHIỆM
    ABLATION." Có code Naive RAG và Agentic RAG (2 hàm khác nhau trong
    rag_core.py) là ĐIỀU KIỆN CẦN, chưa ĐO được cái nào tốt hơn -- đó là
    điều kiện ĐỦ mà file này làm.

DÙNG CHUNG 1 BỘ EVAL VỚI A1 (KHÔNG PHẢI FILE RIÊNG)
    Bản đầu tiên đọc nhầm `data/eval_set.jsonl` (KHÔNG có `answer_should_mention`
    lúc đó). Bản sau đó tách hẳn ra `data/rag_eval_set.jsonl` riêng -- nhưng
    A3 = "upgrade A1 bằng LLM", không phải hệ thống có qrels riêng biệt.
    Từ `step15_expand_eval_set.py`, `data/eval_set.jsonl` đã được MỞ RỘNG
    lên 250 câu VÀ có sẵn `answer_should_mention` cho TẤT CẢ 250 câu -- dùng
    ĐÚNG 1 file này cho cả A1 (step8/9/10/12) và A3 (file này).

6 METRIC ĐO ĐƯỢC MÀ KHÔNG CẦN NGƯỜI CHẤM TAY TỪNG CÂU:
    TẦNG RETRIEVAL (tái sử dụng nguyên vẹn metrics.py + rag_metrics.py):
      1. Context Recall@k  -- recall_at_k() của A1: tìm đúng sản phẩm không.
      2. Context Precision@k -- rag_metrics.py: có xếp đúng sản phẩm LÊN
         TRÊN sản phẩm sai không (khác Recall: quan tâm thứ hạng).
    TẦNG GENERATION (rag_metrics.py, CẦN gọi LLM/embedding thêm):
      3. Faithfulness -- câu trả lời có BỊA thông tin ngoài context không.
      4. Answer Relevancy -- câu trả lời có ĐÚNG CHỦ ĐỀ câu hỏi không.
      5. Answer-should-mention hit rate -- tỷ lệ từ khoá BẮT BUỘC (đã ghi
         tay ở ground truth) THỰC SỰ xuất hiện trong câu trả lời -- proxy
         RẺ (không cần LLM/embedding) cho "câu trả lời có đúng nội dung
         không", bổ sung cho Faithfulness (đo KHÔNG bịa) và Answer
         Relevancy (đo ĐÚNG CHỦ ĐỀ) -- 3 cái không thay thế nhau.
      6. Citation compliance -- có tuân thủ định dạng trích dẫn [1][2] đã
         yêu cầu trong prompt không (proxy grounding, không cần LLM chấm).
    7. Latency trung bình -- đánh đổi tốc độ, đối chiếu BAO_CAO_BPM.md mục 7.

CHẠY: python step_ablation_rag.py
YÊU CẦU TRƯỚC: đã có data/eval_set.jsonl (250 câu, Bước 15) VÀ Ollama đang chạy.
    Với SAMPLE_SIZE=250 x 2 cấu hình x (1 lượt trả lời + 1 lượt chấm
    faithfulness) = tới ~1000 lượt gọi LLM -- CÓ THỂ MẤT HÀNG GIỜ với model
    nhỏ chạy CPU. Xem SAMPLE_SIZE bên dưới để chạy thử nhanh trước.
"""
import json
import re
import time

from rag_core import answer as naive_answer
from rag_core import answer_agentic, call_llm
from rag_core import build_context
from search_core import search
from metrics import recall_at_k
from rag_metrics import context_precision_at_k, faithfulness, answer_relevancy

RAG_EVAL_SET_PATH = "data/eval_set.jsonl"   # dùng chung với A1 -- xem docstring đầu file
K = 5   # dùng k nhỏ hơn A1 (k=10) vì RAG chỉ nên đưa vài nguồn vào prompt,
         # không phải toàn bộ top-10 -- context dài làm LLM dễ lạc hướng

# Đổi giá trị này để chạy thử nhanh trước khi chạy full 250 câu (vd 20 để
# sanity-check trong ~5-10 phút, rồi mới chạy None = full cho số liệu nộp
# bài chính thức). None = dùng hết toàn bộ eval set.
SAMPLE_SIZE = None

CITATION_PATTERN = re.compile(r"\[\d+\]")   # khớp đúng định dạng [1], [2]... mà prompt yêu cầu


def load_eval_set():
    rows = []
    with open(RAG_EVAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if SAMPLE_SIZE is not None:
        rows = rows[:SAMPLE_SIZE]
    return rows


def has_valid_citation(answer_text: str) -> bool:
    """
    Kiểm tra câu trả lời có chứa ít nhất 1 trích dẫn dạng [n] không.
    Đây KHÔNG kiểm tra trích dẫn đó có ĐÚNG SẢN PHẨM hay không (việc đó cần
    người đọc) -- chỉ kiểm tra LLM có TUÂN THỦ ĐỊNH DẠNG grounding đã yêu
    cầu trong prompt hay không. Một proxy thô nhưng tự động hoá được.
    """
    return bool(CITATION_PATTERN.search(answer_text))


def mention_hit_rate(answer_text: str, must_mention: list) -> float:
    """Tỷ lệ từ khoá trong `answer_should_mention` THỰC SỰ xuất hiện (so
    khớp không phân biệt hoa/thường) trong câu trả lời. So khớp SUBSTRING
    đơn giản -- proxy thô nhưng không cần LLM, nhất quán cách A1 đã chọn
    "đơn giản hoá có chủ đích" (simple_tokenize() ở step3_build_index.py)."""
    if not must_mention:
        return None
    answer_lower = answer_text.lower()
    hits = sum(1 for kw in must_mention if kw.lower() in answer_lower)
    return hits / len(must_mention)


def run_ablation_config(name, answer_fn, eval_set):
    """
    Chạy 1 cấu hình (Naive hoặc Agentic) trên toàn bộ eval_set, đo đủ 6 metric.
    answer_fn: rag_core.answer hoặc rag_core.answer_agentic -- cùng chữ ký
               gọi (query, k=...), khác nhau ở cơ chế BÊN TRONG.
    """
    recalls, precisions, faiths, relevancies, mention_rates = [], [], [], [], []
    citation_hits = 0
    latencies = []

    for i, item in enumerate(eval_set, start=1):
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        start = time.time()
        result = answer_fn(query, k=K)
        latency = time.time() - start
        latencies.append(latency)

        retrieved_ids = [s["id"] for s in result["sources"]]
        r = recall_at_k(retrieved_ids, relevant_ids, K)
        p = context_precision_at_k(retrieved_ids, relevant_ids, K)
        if r is not None:
            recalls.append(r)
        if p is not None:
            precisions.append(p)

        if has_valid_citation(result["answer"]):
            citation_hits += 1

        m = mention_hit_rate(result["answer"], item.get("answer_should_mention", []))
        if m is not None:
            mention_rates.append(m)

        # 2 metric CẦN gọi thêm LLM/embedding -- nặng nhất trong toàn bộ vòng lặp
        context_text = build_context(result["sources"]) if result["sources"] else ""
        f = faithfulness(result["answer"], context_text, call_llm)
        if f is not None:
            faiths.append(f)

        rel = answer_relevancy(result["answer"], query)
        if rel is not None:
            relevancies.append(rel)

        if i % 10 == 0:
            print(f"    [{name}] đã chạy {i}/{len(eval_set)}")

    def avg(values):
        return sum(values) / len(values) if values else None

    n = len(eval_set)
    return {
        "name": name,
        "context_recall@k": avg(recalls),
        "context_precision@k": avg(precisions),
        "faithfulness": avg(faiths),
        "answer_relevancy": avg(relevancies),
        "mention_hit_rate": avg(mention_rates),
        "citation_rate": citation_hits / n if n else None,
        "avg_latency_sec": avg(latencies),
    }


def main():
    eval_set = load_eval_set()
    print(f"Chạy ablation RAG trên {len(eval_set)} câu hỏi (SAMPLE_SIZE={SAMPLE_SIZE}), k={K}")
    print("Lưu ý: cần Ollama đang chạy -- mỗi câu hỏi gọi LLM 2 lần (trả lời + chấm faithfulness).\n")

    configs = [
        ("Naive RAG", lambda q, k: naive_answer(q, k=k)),
        ("Agentic RAG", lambda q, k: answer_agentic(q, k=k)),
    ]

    results = []
    for name, fn in configs:
        print(f"Đang chạy: {name}...")
        results.append(run_ablation_config(name, fn, eval_set))

    header = (f"{'Cấu hình':<14} {'Ctx Recall':<11} {'Ctx Precision':<14} {'Faithfulness':<13} "
              f"{'Ans Relevancy':<14} {'Mention hit':<12} {'Citation':<10} {'Latency(s)':<10}")
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        def fmt(key, pct=False):
            v = r[key]
            if v is None:
                return "N/A"
            return f"{v*100:.1f}%" if pct else f"{v:.3f}"
        print(f"{r['name']:<14} {fmt('context_recall@k'):<11} {fmt('context_precision@k'):<14} "
              f"{fmt('faithfulness'):<13} {fmt('answer_relevancy'):<14} {fmt('mention_hit_rate'):<12} "
              f"{fmt('citation_rate', pct=True):<10} {r['avg_latency_sec']:.2f}")

    print("\nCách đọc bảng:")
    print("- Context Recall/Precision giống tầng retrieval -- 2 cấu hình gọi cùng search_core.search()")
    print("  bên dưới nên thường KHÔNG khác nhau nhiều ở Naive vs Agentic vòng đầu, trừ khi Agentic")
    print("  đã tự viết lại query (search KHÁC câu gốc) -- xem 'final_query_used' nếu cần debug.")
    print("- Faithfulness thấp -> LLM đang BỊA thông tin ngoài context -- lỗi tầng Generation.")
    print("- Answer Relevancy thấp nhưng Faithfulness cao -> câu trả lời ĐÚNG (không bịa) nhưng")
    print("  LẠC ĐỀ -- có thể do prompt chưa ép rõ 'phải trả lời thẳng vào câu hỏi'.")
    print("- Mention hit rate thấp -> đối chiếu qrels (answer_should_mention) cho thấy LLM bỏ sót")
    print("  chi tiết QUAN TRỌNG dù không bịa gì sai -- khác lỗi Faithfulness.")
    print("- Latency cao hơn ở Agentic là BÌNH THƯỜNG (chạy ngầm nhiều lượt) -- đối chiếu với")
    print("  phân tích đánh đổi trong BAO_CAO_BPM.md mục 7 (ngưỡng ~42% mới đáng bật Agentic).")


if __name__ == "__main__":
    main()
