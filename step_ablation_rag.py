"""
Ablation A3 trên data/eval_set.jsonl (dùng chung với A1). Các cấu hình:
  naive          Naive RAG (baseline)
  naive_minilm   Naive RAG + rerank cross-encoder MiniLM (A1)
  naive_deberta  Naive RAG + rerank DeBERTa distill (A2); bỏ qua nếu chưa có models/deberta-10k-rank_net
  agentic        Agentic RAG

Metric: Context Recall@k, Context Precision@k, Faithfulness, Answer Relevancy,
mention hit rate (answer_should_mention), citation rate, latency; Agentic thêm tỷ lệ viết lại query.
Mỗi câu: Naive 1 lượt LLM, Agentic 2 hoặc 4 lượt (max_iters=2), cộng 1 lượt chấm faithfulness.
Model LLM lấy từ .env (OLLAMA_MODEL, xem rag_core.py).

CHẠY:
  python step_ablation_rag.py                      # 250 câu, cả 4 cấu hình
  python step_ablation_rag.py --sample 20          # chạy thử 20 câu đầu
  python step_ablation_rag.py --configs naive agentic
  python step_ablation_rag.py --resume             # chạy tiếp từ data/rag_ablation_details.jsonl
Output:
  data/rag_ablation_results.json   bảng tổng hợp (toàn bộ và 50 câu viết tay)
  data/rag_ablation_details.jsonl  chi tiết từng câu, ghi ngay sau mỗi câu
"""
import argparse
import json
import os
import re
import time
from datetime import datetime

from rag_core import LLM_MODEL
from rag_core import answer as naive_answer
from rag_core import answer_agentic, call_llm
from rag_core import build_context
from search_core import available_rerankers
from metrics import recall_at_k
from rag_metrics import context_precision_at_k, faithfulness, answer_relevancy

RAG_EVAL_SET_PATH = "data/eval_set.jsonl"
RESULTS_PATH = "data/rag_ablation_results.json"
DETAILS_PATH = "data/rag_ablation_details.jsonl"
K = 5   # nhỏ hơn A1 (k=10): context ngắn giúp LLM ít lạc hướng
HANDWRITTEN = 50   # 50 dòng đầu eval set là câu viết tay (30 A1 + 20 A3)

CITATION_PATTERN = re.compile(r"\[\d+\]")

CONFIGS = {
    "naive":         ("Naive RAG",       lambda q, k: naive_answer(q, k=k)),
    "naive_minilm":  ("Naive + MiniLM",  lambda q, k: naive_answer(q, k=k, reranker="cross_encoder")),
    "naive_deberta": ("Naive + DeBERTa", lambda q, k: naive_answer(q, k=k, reranker="deberta")),
    "agentic":       ("Agentic RAG",     lambda q, k: answer_agentic(q, k=k)),
}
METRIC_KEYS = ["context_recall", "context_precision", "faithfulness", "answer_relevancy",
               "mention_hit", "citation", "latency_sec"]


def load_eval_set(sample_size=None):
    with open(RAG_EVAL_SET_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:sample_size] if sample_size else rows


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


def evaluate_one(config_key, answer_fn, index, item):
    """Chạy 1 câu hỏi với 1 cấu hình, trả dict chi tiết (gồm các metric)."""
    query, relevant_ids = item["query"], item["relevant_ids"]

    start = time.time()
    result = answer_fn(query, K)
    latency = time.time() - start

    retrieved_ids = [s["id"] for s in result["sources"]]
    context_text = build_context(result["sources"]) if result["sources"] else ""

    row = {
        "config": config_key,
        "index": index,
        "query": query,
        "relevant_ids": relevant_ids,
        "retrieved_ids": retrieved_ids,
        "answer": result["answer"],
        "context_recall": recall_at_k(retrieved_ids, relevant_ids, K),
        "context_precision": context_precision_at_k(retrieved_ids, relevant_ids, K),
        # Faithfulness chấm trên đúng context mà câu trả lời đã dùng.
        "faithfulness": faithfulness(result["answer"], context_text, call_llm),
        "answer_relevancy": answer_relevancy(result["answer"], query),
        "mention_hit": mention_hit_rate(result["answer"], item.get("answer_should_mention", [])),
        "citation": 1.0 if has_valid_citation(result["answer"]) else 0.0,
        "latency_sec": latency,
    }
    if "final_query_used" in result:
        row["final_query_used"] = result["final_query_used"]
        row["rewritten"] = result["final_query_used"] != query
        row["judge_verdicts"] = result.get("judge_verdicts", [])
    return row


def summarize(rows):
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]

    def avg(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    summary = {key: avg(key) for key in METRIC_KEYS}
    summary["n"] = len(rows)
    summary["n_errors"] = len(errors)
    rewritten = [r["rewritten"] for r in rows if "rewritten" in r]
    if rewritten:
        summary["rewrite_rate"] = sum(rewritten) / len(rewritten)
    return summary


def fmt(value, pct=False):
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%" if pct else f"{value:.3f}"


def print_table(title, results, subset):
    print(f"\n{title}")
    header = (f"{'Cấu hình':<16} {'n':>4} {'Lỗi':>4} {'CtxRecall':>10} {'CtxPrec':>9} {'Faithful':>9} "
              f"{'AnsRel':>8} {'Mention':>8} {'Citation':>9} {'Latency':>8} {'Rewrite':>8}")
    print(header)
    print("-" * len(header))
    for r in results:
        s = r[subset]
        print(f"{r['name']:<16} {s['n']:>4} {s['n_errors']:>4} {fmt(s['context_recall']):>10} {fmt(s['context_precision']):>9} "
              f"{fmt(s['faithfulness']):>9} {fmt(s['answer_relevancy']):>8} {fmt(s['mention_hit']):>8} "
              f"{fmt(s['citation'], pct=True):>9} {fmt(s['latency_sec']):>8} "
              f"{fmt(s.get('rewrite_rate'), pct=True) if 'rewrite_rate' in s else '-':>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="chỉ chạy N câu đầu")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--resume", action="store_true",
                        help="giữ các câu đã chạy xong trong file chi tiết, chỉ chạy phần còn thiếu")
    args = parser.parse_args()

    eval_set = load_eval_set(args.sample)
    config_keys = [c for c in args.configs
                   if c != "naive_deberta" or "deberta" in available_rerankers()]
    if len(config_keys) < len(args.configs):
        print("Bỏ qua naive_deberta: chưa có models/deberta-10k-rank_net")

    print(f"Ablation RAG: {len(eval_set)} câu, k={K}, LLM={LLM_MODEL}, cấu hình={config_keys}")
    print("Mỗi câu: Naive 2 lượt LLM, Agentic 3 hoặc 5 lượt (gồm 1 lượt chấm faithfulness).\n")

    done = {}   # (config, index) -> row đã chạy xong (không lỗi) từ lần chạy trước
    if args.resume and os.path.exists(DETAILS_PATH):
        with open(DETAILS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "error" not in row:
                    done[(row["config"], row["index"])] = row
        print(f"Resume: đã có {len(done)} kết quả từ lần chạy trước\n")
    with open(DETAILS_PATH, "w", encoding="utf-8") as f:
        for row in done.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    results = []
    for key in config_keys:
        name, fn = CONFIGS[key]
        print(f"Đang chạy: {name}...")
        rows = []
        for i, item in enumerate(eval_set):
            if (key, i) in done:
                rows.append(done[(key, i)])
                continue
            try:
                row = evaluate_one(key, fn, i, item)
            except Exception as e:   # call_llm đã tự thử lại; ghi lỗi rồi chạy tiếp câu sau
                row = {"config": key, "index": i, "query": item["query"],
                       "error": f"{type(e).__name__}: {e}"}
                print(f"    [{name}] câu {i}: LỖI {row['error'][:120]}")
            rows.append(row)
            with open(DETAILS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"    [{name}] {i + 1}/{len(eval_set)}")
        results.append({
            "name": name,
            "config": key,
            "all": summarize(rows),
            "handwritten": summarize([r for r in rows if r["index"] < HANDWRITTEN]),
        })

    print_table("Toàn bộ", results, "all")
    print_table(f"{HANDWRITTEN} câu viết tay (nếu có trong mẫu)", results, "handwritten")

    output = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "llm_model": LLM_MODEL,
        "eval_set": RAG_EVAL_SET_PATH,
        "num_queries": len(eval_set),
        "k": K,
        "results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {RESULTS_PATH} và {DETAILS_PATH}")


if __name__ == "__main__":
    main()
