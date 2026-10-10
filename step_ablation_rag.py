"""
Ablation A3 trên data/eval_set.jsonl (dùng chung với A1). Các cấu hình:
  naive          Naive RAG (baseline)
  naive_minilm   Naive RAG + rerank cross-encoder MiniLM (A1)
  naive_deberta  Naive RAG + rerank DeBERTa distill (A2); bỏ qua nếu chưa có models/deberta-10k-rank_net
  agentic        Agentic RAG

Ground truth: qrels 0-3 (step16_build_qrels.py); relevant_ids = id có grade >= 2.
Metric: Context Recall@k, Context Precision@k, Context nDCG@k đa mức (theo qrels), Faithfulness, Answer Relevancy,
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import rag_core
from rag_core import LLM_MODEL, OLLAMA_URLS
from rag_core import answer as naive_answer
from rag_core import answer_agentic, call_llm
from rag_core import build_context
from search_core import available_rerankers
from metrics import ndcg_at_k_graded, recall_at_k
from rag_metrics import context_precision_at_k, faithfulness
from rag_metrics import answer_relevancy as _answer_relevancy

# Khi chạy nhiều luồng (--workers), chỉ các lượt gọi LLM chạy song song. Tìm kiếm, rerank và
# embedding dùng chung model trong tiến trình này nên được khoá để chạy lần lượt.
_MODEL_LOCK = threading.Lock()


def _locked(fn):
    def wrapper(*args, **kwargs):
        with _MODEL_LOCK:
            return fn(*args, **kwargs)
    return wrapper


rag_core.retrieve = _locked(rag_core.retrieve)   # answer()/answer_agentic() tra retrieve lúc gọi
answer_relevancy = _locked(_answer_relevancy)

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
METRIC_KEYS = ["context_recall", "context_precision", "context_ndcg", "faithfulness", "answer_relevancy",
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


def evaluate_one(config_key, answer_fn, index, item, with_faithfulness=True):
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
        # Thứ hạng của context theo nhãn đa mức: sp grade 2 ở [1] vẫn được điểm, ít hơn grade 3.
        "context_ndcg": ndcg_at_k_graded(retrieved_ids, relevant_ids, K, qrels=item.get("qrels")),
        # Faithfulness chấm trên đúng context mà câu trả lời đã dùng.
        "faithfulness": (faithfulness(result["answer"], context_text, call_llm)
                         if with_faithfulness else None),
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
    header = (f"{'Cấu hình':<16} {'n':>4} {'Lỗi':>4} {'CtxRecall':>10} {'CtxPrec':>9} {'CtxNDCG':>8} {'Faithful':>9} "
              f"{'AnsRel':>8} {'Mention':>8} {'Citation':>9} {'Latency':>8} {'Rewrite':>8}")
    print(header)
    print("-" * len(header))
    for r in results:
        s = r[subset]
        print(f"{r['name']:<16} {s['n']:>4} {s['n_errors']:>4} {fmt(s['context_recall']):>10} {fmt(s['context_precision']):>9} {fmt(s.get('context_ndcg')):>8} "
              f"{fmt(s['faithfulness']):>9} {fmt(s['answer_relevancy']):>8} {fmt(s['mention_hit']):>8} "
              f"{fmt(s['citation'], pct=True):>9} {fmt(s['latency_sec']):>8} "
              f"{fmt(s.get('rewrite_rate'), pct=True) if 'rewrite_rate' in s else '-':>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="chỉ chạy N câu đầu")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--resume", action="store_true",
                        help="giữ các câu đã chạy xong trong file chi tiết, chỉ chạy phần còn thiếu")
    parser.add_argument("--workers", type=int, default=1,
                        help="số câu chạy song song; >1 làm cột Latency cao hơn thực tế (tranh GPU)")
    parser.add_argument("--no-faithfulness", action="store_true",
                        help="bỏ lượt chấm Faithfulness (giảm 1 lượt LLM mỗi câu)")
    args = parser.parse_args()

    eval_set = load_eval_set(args.sample)
    config_keys = [c for c in args.configs
                   if c != "naive_deberta" or "deberta" in available_rerankers()]
    if len(config_keys) < len(args.configs):
        print("Bỏ qua naive_deberta: chưa có models/deberta-10k-rank_net")

    with_faith = not args.no_faithfulness
    print(f"Ablation RAG: {len(eval_set)} câu, k={K}, LLM={LLM_MODEL}, cấu hình={config_keys}")
    print(f"Ollama: {OLLAMA_URLS}, workers={args.workers}, faithfulness={'có' if with_faith else 'không'}")
    if args.workers > 1:
        print("Lưu ý: chạy song song nên cột Latency gồm cả thời gian chờ GPU, cao hơn khi chạy 1 câu.")
    print()

    done = {}   # (config, index) -> row đã chạy xong (không lỗi) từ lần chạy trước
    if args.resume and os.path.exists(DETAILS_PATH):
        with open(DETAILS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "error" not in row:
                    done[(row["config"], row["index"])] = row
        # Bỏ kết quả chấm theo ground truth khác bản hiện tại (vd trước khi có qrels, hoặc câu đã sửa).
        stale = [key for key, row in done.items()
                 if row["index"] >= len(eval_set)
                 or row["query"] != eval_set[row["index"]]["query"]
                 or row["relevant_ids"] != eval_set[row["index"]]["relevant_ids"]
                 or "context_ndcg" not in row]
        for key in stale:
            del done[key]
        print(f"Resume: đã có {len(done)} kết quả từ lần chạy trước"
              f" (bỏ {len(stale)} kết quả theo ground truth cũ)\n")
    with open(DETAILS_PATH, "w", encoding="utf-8") as f:
        for row in done.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    results = []
    for key in config_keys:
        name, fn = CONFIGS[key]
        print(f"Đang chạy: {name}...")
        rows = [done[(key, i)] for i in range(len(eval_set)) if (key, i) in done]
        pending = [(i, item) for i, item in enumerate(eval_set) if (key, i) not in done]
        started = time.time()

        def run_one(i, item, key=key, fn=fn):
            try:
                return evaluate_one(key, fn, i, item, with_faithfulness=with_faith)
            except Exception as e:   # call_llm đã tự thử lại; ghi lỗi rồi chạy tiếp câu khác
                return {"config": key, "index": i, "query": item["query"],
                        "error": f"{type(e).__name__}: {e}"}

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(run_one, i, item) for i, item in pending]
            for count, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                if "error" in row:
                    print(f"    [{name}] câu {row['index']}: LỖI {row['error'][:120]}")
                rows.append(row)
                with open(DETAILS_PATH, "a", encoding="utf-8") as f:   # chỉ luồng chính ghi file
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if count % 10 == 0 or count == len(pending):
                    elapsed = time.time() - started
                    print(f"    [{name}] {count}/{len(pending)} câu mới "
                          f"({elapsed / count:.1f} s/câu, đã {elapsed / 60:.1f} phút)")

        results.append({
            "name": name,
            "config": key,
            "all": summarize(rows),
            "handwritten": summarize([r for r in rows if r["index"] < HANDWRITTEN]),
            "template": summarize([r for r in rows if r["index"] >= HANDWRITTEN]),
        })

    print_table("Toàn bộ", results, "all")
    print_table(f"{HANDWRITTEN} câu viết tay (nếu có trong mẫu)", results, "handwritten")
    print_table("Câu sinh từ mẫu (nếu có trong mẫu)", results, "template")

    output = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "llm_model": LLM_MODEL,
        "eval_set": RAG_EVAL_SET_PATH,
        "num_queries": len(eval_set),
        "k": K,
        "workers": args.workers,
        "faithfulness": with_faith,
        "ollama_urls": OLLAMA_URLS,
        "results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {RESULTS_PATH} và {DETAILS_PATH}")


if __name__ == "__main__":
    main()
