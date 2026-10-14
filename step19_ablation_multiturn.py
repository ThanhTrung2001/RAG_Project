"""
Bước 19 (A3): ablation Memory khi TÌM KIẾM -- viết lại câu hỏi nối tiếp (rag_core.contextualize_query).

Hội thoại 2 lượt tạo tự động từ data/eval_set.jsonl, dùng lại qrels của câu i làm ground truth của lượt 2:
  followup  lượt 1 = câu i, lượt 2 = câu mơ hồ ("show me more options like that"...): cần lịch sử mới hiểu.
  switch    lượt 1 = câu j (cách i nửa eval set, chủ đề khác), lượt 2 = câu i: lịch sử KHÔNG được kéo vào.
Lượt 1 chạy thật bằng rag_core.answer() (1 lượt LLM, không rerank) để có lịch sử giống chatbot.
Lượt 2 tìm bằng 4 cách, cùng retrieve(k=5), không rerank:
  raw      câu lượt 2 nguyên văn (Memory chỉ dùng khi trả lời, như trước khi có contextualize_query)
  concat   câu lượt 1 + câu lượt 2 (Memory không cần LLM)
  rewrite  contextualize_query(câu lượt 2, lịch sử)  (cách chatbot đang dùng, +1 lượt LLM)
  oracle   câu i viết đầy đủ (mức trần)
Metric: Context Recall@5, Precision@5, nDCG@5 đa mức, sản phẩm đúng ở [1]; rewrite thêm tỷ lệ truy vấn có tiếng Việt.

CHẠY (trên Kaggle qua kaggle/kaggle_run_rag_ablation.py --with-multiturn, hoặc local):
  python step19_ablation_multiturn.py --sample 10 --workers 4
  python step19_ablation_multiturn.py --resume
Output:
  data/multiturn_ablation_results.json   bảng tổng hợp theo loại hội thoại x cách tìm
  data/multiturn_ablation_details.jsonl  chi tiết từng hội thoại, ghi ngay sau mỗi hội thoại
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import rag_core
from rag_core import LLM_MODEL, OLLAMA_URLS, _has_vietnamese, contextualize_query
from metrics import ndcg_at_k_graded, recall_at_k
from rag_metrics import context_precision_at_k

# Như step_ablation_rag.py: chỉ lượt gọi LLM chạy song song, tìm kiếm dùng chung model nên khoá lại.
_MODEL_LOCK = threading.Lock()
_original_retrieve = rag_core.retrieve


def _locked_retrieve(*args, **kwargs):
    with _MODEL_LOCK:
        return _original_retrieve(*args, **kwargs)


rag_core.retrieve = _locked_retrieve   # answer() tra rag_core.retrieve lúc gọi

EVAL_SET_PATH = "data/eval_set.jsonl"
RESULTS_PATH = "data/multiturn_ablation_results.json"
DETAILS_PATH = "data/multiturn_ablation_details.jsonl"
K = 5   # cùng k với step_ablation_rag.py
HANDWRITTEN = 50

# Câu nối tiếp giữ nguyên nhu cầu của lượt 1, nên qrels của câu lượt 1 vẫn đúng cho lượt 2.
# Không dùng "other color"/"cheaper": các câu đó đổi tập sản phẩm đúng, qrels hiện có không chấm được.
FOLLOWUPS = ["show me more options like that",
             "do you have any others similar to that?",
             "what else do you have like this one?",
             "any alternatives to that?"]
KINDS = ["followup", "switch"]
METHODS = ["raw", "concat", "rewrite", "oracle"]


def load_eval_set(sample_size=None):
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows, (rows[:sample_size] if sample_size else rows)


def make_dialog(kind, i, eval_set):
    """Trả (câu lượt 1, câu lượt 2) của hội thoại thứ i."""
    item = eval_set[i]
    if kind == "followup":
        return item["query"], FOLLOWUPS[i % len(FOLLOWUPS)]
    j = (i + len(eval_set) // 2) % len(eval_set)
    return eval_set[j]["query"], item["query"]


def first_turn_history(query):
    """Chạy lượt 1 như chatbot (Naive, có Memory) và trả lịch sử sau lượt đó."""
    session_id = f"step19-{threading.get_ident()}-{time.time_ns()}"
    rag_core.answer(query, k=K, session_id=session_id)
    history = list(rag_core.get_history(session_id))
    rag_core.clear_history(session_id)
    return history


def score(retrieved_ids, item):
    relevant_ids = item["relevant_ids"]
    return {
        "retrieved_ids": retrieved_ids,
        "context_recall": recall_at_k(retrieved_ids, relevant_ids, K),
        "context_precision": context_precision_at_k(retrieved_ids, relevant_ids, K),
        "context_ndcg": ndcg_at_k_graded(retrieved_ids, relevant_ids, K, qrels=item.get("qrels")),
        "top1_hit": 1.0 if retrieved_ids[:1] and retrieved_ids[0] in relevant_ids else 0.0,
    }


def evaluate_dialog(kind, i, eval_set):
    item = eval_set[i]
    turn1, turn2 = make_dialog(kind, i, eval_set)
    history = first_turn_history(turn1)

    start = time.time()
    rewritten = contextualize_query(turn2, history)
    rewrite_latency = time.time() - start

    queries = {"raw": turn2, "concat": f"{turn1} {turn2}", "rewrite": rewritten, "oracle": item["query"]}
    row = {"kind": kind, "index": i, "turn1": turn1, "turn2": turn2,
           "turn1_answer": history[-1]["answer"] if history else "",
           "relevant_ids": item["relevant_ids"], "queries": queries,
           "rewrite_has_vietnamese": _has_vietnamese(rewritten),
           "rewrite_latency_sec": rewrite_latency}
    for method, query in queries.items():
        results = rag_core.retrieve(query, k=K)
        row[method] = score([r["id"] for r in results], item)
    return row


def summarize(rows):
    rows = [r for r in rows if "error" not in r]
    summary = {"n": len(rows)}
    for method in METHODS:
        summary[method] = {}
        for key in ["context_recall", "context_precision", "context_ndcg", "top1_hit"]:
            values = [r[method][key] for r in rows if r[method][key] is not None]
            summary[method][key] = sum(values) / len(values) if values else None
    if rows:
        summary["rewrite_vietnamese_rate"] = sum(r["rewrite_has_vietnamese"] for r in rows) / len(rows)
        summary["rewrite_latency_sec"] = sum(r["rewrite_latency_sec"] for r in rows) / len(rows)
    return summary


def print_table(title, summary):
    def fmt(v):
        return f"{v:.3f}" if v is not None else "N/A"
    print(f"\n{title} (n={summary['n']})")
    print(f"{'Cách tìm':<10} {'CtxRecall@5':>12} {'CtxPrec@5':>10} {'CtxNDCG@5':>10} {'Đúng ở [1]':>11}")
    for method in METHODS:
        s = summary[method]
        print(f"{method:<10} {fmt(s['context_recall']):>12} {fmt(s['context_precision']):>10} "
              f"{fmt(s['context_ndcg']):>10} {fmt(s['top1_hit']):>11}")
    if summary["n"]:
        print(f"Truy vấn viết lại có tiếng Việt: {summary['rewrite_vietnamese_rate'] * 100:.1f}%, "
              f"thời gian viết lại: {summary['rewrite_latency_sec']:.1f} s/câu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="chỉ chạy N câu đầu (x 2 loại hội thoại)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    # Câu j của hội thoại "switch" lấy từ toàn bộ eval set, để --sample không đổi nội dung hội thoại.
    full_set, eval_set = load_eval_set(args.sample)
    print(f"Ablation Memory khi tìm kiếm: {len(eval_set)} câu x {len(KINDS)} loại hội thoại, k={K}, "
          f"LLM={LLM_MODEL}, Ollama={OLLAMA_URLS}, workers={args.workers}\n")

    done = {}
    if args.resume and os.path.exists(DETAILS_PATH):
        with open(DETAILS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "error" not in row:
                    done[(row["kind"], row["index"])] = row
        print(f"Resume: đã có {len(done)} hội thoại\n")
    with open(DETAILS_PATH, "w", encoding="utf-8") as f:
        for row in done.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = list(done.values())
    pending = [(kind, i) for kind in KINDS for i in range(len(eval_set)) if (kind, i) not in done]

    def run_one(kind, i):
        try:
            return evaluate_dialog(kind, i, full_set)
        except Exception as e:   # call_llm đã tự thử lại; ghi lỗi rồi chạy tiếp
            return {"kind": kind, "index": i, "error": f"{type(e).__name__}: {e}"}

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_one, kind, i) for kind, i in pending]
        for count, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            if "error" in row:
                print(f"    {row['kind']} câu {row['index']}: LỖI {row['error'][:120]}")
            rows.append(row)
            with open(DETAILS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if count % 20 == 0 or count == len(pending):
                elapsed = time.time() - started
                print(f"    {count}/{len(pending)} hội thoại ({elapsed / count:.1f} s/hội thoại, "
                      f"đã {elapsed / 60:.1f} phút)")

    errors = [r for r in rows if "error" in r]
    summary = {}
    for kind in KINDS:
        kind_rows = [r for r in rows if r["kind"] == kind]
        summary[kind] = {"all": summarize(kind_rows),
                         "handwritten": summarize([r for r in kind_rows if r["index"] < HANDWRITTEN])}
        print_table(f"{kind}, toàn bộ", summary[kind]["all"])
        print_table(f"{kind}, {HANDWRITTEN} câu viết tay", summary[kind]["handwritten"])

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"date": datetime.now().isoformat(timespec="seconds"), "llm_model": LLM_MODEL,
                   "eval_set": EVAL_SET_PATH, "num_queries": len(eval_set), "k": K,
                   "followups": FOLLOWUPS, "n_errors": len(errors), "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {RESULTS_PATH} và {DETAILS_PATH} ({len(errors)} lỗi)")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
