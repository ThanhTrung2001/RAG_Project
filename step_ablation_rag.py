"""
Ablation A3 trên data/eval_set.jsonl (dùng chung với A1). Các cấu hình:
  Naive RAG | Naive RAG + cross-encoder MiniLM | Naive RAG + DeBERTa distill | Agentic RAG
Cấu hình DeBERTa bị bỏ qua nếu chưa có models/deberta-10k-rank_net.

Metric: Context Recall@k, Context Precision@k, Faithfulness, Answer Relevancy,
mention hit rate (answer_should_mention), citation rate, latency trung bình.
Mỗi câu: Naive 1 lượt LLM, Agentic 2 hoặc 4 lượt (max_iters=2), cộng 1 lượt chấm faithfulness.
CHẠY: python step_ablation_rag.py (cần Ollama đang chạy; đặt SAMPLE_SIZE để chạy thử nhanh).
"""
import json
import re
import time

from rag_core import answer as naive_answer
from rag_core import answer_agentic, call_llm
from rag_core import build_context
from search_core import search, available_rerankers
from metrics import recall_at_k
from rag_metrics import context_precision_at_k, faithfulness, answer_relevancy

RAG_EVAL_SET_PATH = "data/eval_set.jsonl"
K = 5   # nhỏ hơn A1 (k=10): context ngắn giúp LLM ít lạc hướng

SAMPLE_SIZE = None   # None = toàn bộ eval set; đặt số nhỏ (vd 20) để chạy thử

CITATION_PATTERN = re.compile(r"\[\d+\]")


def load_eval_set():
    rows = []
    with open(RAG_EVAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if SAMPLE_SIZE is not None:
        rows = rows[:SAMPLE_SIZE]
    return rows


def has_valid_citation(answer_text: str) -> bool:
    """Có ít nhất 1 trích dẫn [n] không (chỉ kiểm định dạng, không kiểm đúng sản phẩm)."""
    return bool(CITATION_PATTERN.search(answer_text))


def mention_hit_rate(answer_text: str, must_mention: list) -> float:
    """Tỷ lệ từ khoá bắt buộc xuất hiện trong câu trả lời (substring, không phân biệt hoa thường)."""
    if not must_mention:
        return None
    answer_lower = answer_text.lower()
    hits = sum(1 for kw in must_mention if kw.lower() in answer_lower)
    return hits / len(must_mention)


def run_ablation_config(name, answer_fn, eval_set):
    """Chạy 1 cấu hình trên eval_set. answer_fn(query, k) trả dict có "answer", "sources"."""
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

        # Faithfulness chấm trên đúng context mà câu trả lời đã dùng.
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
    print("Lưu ý: cần Ollama đang chạy -- mỗi câu: Naive 2 lượt LLM, Agentic 3 hoặc 5 lượt "
          "(gồm 1 lượt chấm faithfulness).\n")

    configs = [("Naive RAG", lambda q, k: naive_answer(q, k=k))]
    for name, label in [("cross_encoder", "Naive + MiniLM"), ("deberta", "Naive + DeBERTa")]:
        if name in available_rerankers():
            configs.append((label, lambda q, k, r=name: naive_answer(q, k=k, reranker=r)))
        else:
            print(f"Bỏ qua {label}: chưa có model")
    configs.append(("Agentic RAG", lambda q, k: answer_agentic(q, k=k)))

    results = []
    for name, fn in configs:
        print(f"Đang chạy: {name}...")
        results.append(run_ablation_config(name, fn, eval_set))

    header = (f"{'Cấu hình':<16} {'Ctx Recall':<11} {'Ctx Precision':<14} {'Faithfulness':<13} "
              f"{'Ans Relevancy':<14} {'Mention hit':<12} {'Citation':<10} {'Latency(s)':<10}")
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        def fmt(key, pct=False):
            v = r[key]
            if v is None:
                return "N/A"
            return f"{v*100:.1f}%" if pct else f"{v:.3f}"
        print(f"{r['name']:<16} {fmt('context_recall@k'):<11} {fmt('context_precision@k'):<14} "
              f"{fmt('faithfulness'):<13} {fmt('answer_relevancy'):<14} {fmt('mention_hit_rate'):<12} "
              f"{fmt('citation_rate', pct=True):<10} {r['avg_latency_sec']:.2f}")

    print("\nCách đọc bảng:")
    print("- Context Recall/Precision đo tầng retrieval: 'Naive' và 'Naive + reranker' chỉ khác ở")
    print("  bước rerank, nên chênh lệch 2 cột này là tác dụng của reranker lên context đưa cho LLM.")
    print("- Naive vs Agentic cùng gọi search() ở vòng đầu; khác nhau khi Agentic tự viết lại query.")
    print("- Faithfulness thấp -> LLM đang BỊA thông tin ngoài context -- lỗi tầng Generation.")
    print("- Answer Relevancy thấp nhưng Faithfulness cao -> câu trả lời ĐÚNG (không bịa) nhưng")
    print("  LẠC ĐỀ -- có thể do prompt chưa ép rõ 'phải trả lời thẳng vào câu hỏi'.")
    print("- Mention hit rate thấp -> đối chiếu qrels (answer_should_mention) cho thấy LLM bỏ sót")
    print("  chi tiết QUAN TRỌNG dù không bịa gì sai -- khác lỗi Faithfulness.")
    print("- Latency của Agentic cao hơn vì gọi LLM nhiều lượt; latency của reranker cộng thêm")
    print("  thời gian chấm 30 ứng viên (DeBERTa chậm hơn MiniLM trên CPU).")


if __name__ == "__main__":
    main()
