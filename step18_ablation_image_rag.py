"""
Bước 18 (A3): ablation RAG khi hỏi bằng ảnh, đo cả tầng Retrieval lẫn Generation (cần Ollama).

Dữ liệu: 100 ảnh của data/eval_set_image.jsonl (Bước 13). Mỗi ảnh thuộc một sản phẩm đúng trong
data/eval_set.jsonl, nên ghép được với câu hỏi chữ, qrels và answer_should_mention của câu đó.
Ba cấu hình, cùng 100 sản phẩm, cùng Naive RAG (rag_core.answer):
  text        chỉ câu chữ (mốc so sánh trên đúng 100 sản phẩm này)
  image       chỉ ảnh, không gõ chữ
  image_text  ảnh + câu chữ (gộp kết quả tìm theo ảnh và theo chữ bằng RRF)
Metric giống step_ablation_rag.py. Answer Relevancy so câu trả lời với câu chữ đã ghép (cả ở cấu hình
chỉ ảnh), coi đó là "nhu cầu" người dùng muốn diễn đạt.

CHẠY:
  python step18_ablation_image_rag.py --sample 5      # chạy thử
  python step18_ablation_image_rag.py --workers 4     # trên Kaggle (xem kaggle_run_rag_ablation.py --with-image)
  python step18_ablation_image_rag.py --resume
Output: data/image_rag_ablation_results.json, data/image_rag_ablation_details.jsonl
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from PIL import Image

import step_ablation_rag as sar   # khoá model dùng chung, metric và cách tổng hợp giống A3
from metrics import ndcg_at_k_graded, recall_at_k
from rag_core import LLM_MODEL, OLLAMA_URLS, answer, build_context, call_llm
from rag_metrics import context_precision_at_k, faithfulness

EVAL_SET_PATH = "data/eval_set.jsonl"
EVAL_SET_IMAGE_PATH = "data/eval_set_image.jsonl"
RESULTS_PATH = "data/image_rag_ablation_results.json"
DETAILS_PATH = "data/image_rag_ablation_details.jsonl"
K = sar.K

CONFIGS = {
    "text":       ("Chỉ chữ",    True,  False),
    "image":      ("Chỉ ảnh",    False, True),
    "image_text": ("Ảnh + chữ",  True,  True),
}


def load_items(sample_size=None):
    """Ghép mỗi ảnh với câu chữ cùng sản phẩm đúng trong eval_set.jsonl."""
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        text_items = [json.loads(line) for line in f if line.strip()]
    by_target = {}
    for item in text_items:
        for tid in item.get("target_ids", item["relevant_ids"]):
            by_target.setdefault(tid, item)

    items = []
    with open(EVAL_SET_IMAGE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            img = json.loads(line)
            text = by_target[img["relevant_ids"][0]]
            items.append({
                # file sinh trên Windows có dấu "\" trong đường dẫn -> đổi để chạy được trên Linux/Kaggle
                "query_image": img["query_image"].replace("\\", "/"),
                "transform": img["transform"],
                "query": text["query"],
                "relevant_ids": text["relevant_ids"],
                "qrels": text.get("qrels"),
                "answer_should_mention": text.get("answer_should_mention", []),
            })
    return items[:sample_size] if sample_size else items


def evaluate_one(config_key, index, item, with_faithfulness=True):
    _, use_text, use_image = CONFIGS[config_key]
    query = item["query"] if use_text else ""
    image = Image.open(item["query_image"]).convert("RGB") if use_image else None

    start = time.time()
    result = answer(query, k=K, image=image)
    latency = time.time() - start

    retrieved_ids = [s["id"] for s in result["sources"]]
    relevant_ids = item["relevant_ids"]
    context_text = build_context(result["sources"]) if result["sources"] else ""
    return {
        "config": config_key,
        "index": index,
        "query_image": item["query_image"],
        "transform": item["transform"],
        "query": query,
        "relevant_ids": relevant_ids,
        "retrieved_ids": retrieved_ids,
        "answer": result["answer"],
        "context_recall": recall_at_k(retrieved_ids, relevant_ids, K),
        "context_precision": context_precision_at_k(retrieved_ids, relevant_ids, K),
        "context_ndcg": ndcg_at_k_graded(retrieved_ids, relevant_ids, K, qrels=item.get("qrels")),
        "faithfulness": (faithfulness(result["answer"], context_text, call_llm)
                         if with_faithfulness else None),
        "answer_relevancy": sar.answer_relevancy(result["answer"], item["query"]),
        "mention_hit": sar.mention_hit_rate(result["answer"], item["answer_should_mention"]),
        "citation": 1.0 if sar.has_valid_citation(result["answer"]) else 0.0,
        "latency_sec": latency,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="chỉ chạy N ảnh đầu")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-faithfulness", action="store_true")
    args = parser.parse_args()

    items = load_items(args.sample)
    with_faith = not args.no_faithfulness
    print(f"Ablation RAG hỏi bằng ảnh: {len(items)} ảnh, k={K}, LLM={LLM_MODEL}, cấu hình={args.configs}")
    print(f"Ollama: {OLLAMA_URLS}, workers={args.workers}, faithfulness={'có' if with_faith else 'không'}\n")

    done = {}
    if args.resume and os.path.exists(DETAILS_PATH):
        with open(DETAILS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "error" not in row and row["index"] < len(items) \
                        and row["relevant_ids"] == items[row["index"]]["relevant_ids"]:
                    done[(row["config"], row["index"])] = row
        print(f"Resume: đã có {len(done)} kết quả\n")
    with open(DETAILS_PATH, "w", encoding="utf-8") as f:
        for row in done.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    results = []
    for key in args.configs:
        name = CONFIGS[key][0]
        print(f"Đang chạy: {name}...")
        rows = [done[(key, i)] for i in range(len(items)) if (key, i) in done]
        pending = [(i, item) for i, item in enumerate(items) if (key, i) not in done]

        def run_one(i, item, key=key):
            try:
                return evaluate_one(key, i, item, with_faithfulness=with_faith)
            except Exception as e:
                return {"config": key, "index": i, "error": f"{type(e).__name__}: {e}"}

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(run_one, i, item) for i, item in pending]
            for count, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                if "error" in row:
                    print(f"    [{name}] ảnh {row['index']}: LỖI {row['error'][:120]}")
                rows.append(row)
                with open(DETAILS_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if count % 10 == 0 or count == len(pending):
                    print(f"    [{name}] {count}/{len(pending)} ảnh mới")

        results.append({"name": name, "config": key, "all": sar.summarize(rows)})

    sar.print_table("100 ảnh (cùng sản phẩm, cùng Naive RAG)", results, "all")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"date": datetime.now().isoformat(timespec="seconds"), "llm_model": LLM_MODEL,
                   "eval_set_image": EVAL_SET_IMAGE_PATH, "num_images": len(items), "k": K,
                   "workers": args.workers, "faithfulness": with_faith, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {RESULTS_PATH} và {DETAILS_PATH}")


if __name__ == "__main__":
    main()
