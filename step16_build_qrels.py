"""
Bước 16: nâng ground truth từ "1 id đúng / câu" lên qrels đa mức 0-3 (kiểu TREC).

Thang nhãn:
  3 = đúng hoàn toàn (khớp mọi ràng buộc của truy vấn: loại sản phẩm + thuộc tính/brand/model)
  2 = liên quan tốt (đúng loại sản phẩm, lệch nhẹ hoặc thiếu một thuộc tính phụ)
  1 = một phần (cùng nhóm / phụ kiện / chỉ khớp một khía cạnh)
  0 = không liên quan
Nhãn được chấm từ caption/brand/category/mô tả trong catalog, lưu ở data/qrels_judgments/.

Hai chế độ:
  python step16_build_qrels.py pool   -> data/qrels_pool.jsonl
      Pooling (như TREC): gộp top-POOL_DEPTH của BM25, dense và hybrid (RRF) cho mỗi câu,
      luôn thêm id đúng gốc. Mỗi ứng viên kèm caption/brand/category/mô tả rút gọn để chấm.
  python step16_build_qrels.py merge  -> ghi đè data/eval_set.jsonl
      Đọc nhãn đã chấm ở data/qrels_judgments/*.jsonl, ghi trường "qrels" {id: grade} và
      đặt lại relevant_ids = các id có grade >= RELEVANT_THRESHOLD.
      id đúng gốc (known item) luôn giữ grade 3.
Ứng viên ngoài pool coi như grade 0 (giả định chuẩn của TREC pooling).
"""
import glob
import json
import os
import sys

CATALOG_PATH = "data/catalog.jsonl"
EVAL_SET_PATH = "data/eval_set.jsonl"
POOL_PATH = "data/qrels_pool.jsonl"
JUDGMENTS_GLOB = "data/qrels_judgments/*.jsonl"
POOL_DEPTH = 10
RELEVANT_THRESHOLD = 2
DESC_CHARS = 300

# Truy vấn gốc mâu thuẫn với chính sản phẩm đúng (phát hiện lúc chấm qrels, đối chiếu
# toàn văn catalog) -> sửa truy vấn cho khớp dữ liệu thay vì ép nhãn 3 cho sản phẩm sai.
# {target_id: (truy vấn cũ, truy vấn mới)}
QUERY_FIXES = {
    192: ("cornhole board set with eagle flag design",       # sp là bộ decal dán bảng
          "cornhole board wraps with eagle flag design"),
    1758: ("fast fold portable projection screen",           # sp là tấm màn thay thế
           "replacement screen surface for a fast fold projection screen"),
    1397: ("slide stopper accessory for a trombone",         # mô tả: "trumpet slide stopper"
           "slide stopper accessory for a trumpet"),
}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def apply_query_fixes(eval_set):
    for item in eval_set:
        target_ids = item.get("target_ids", item["relevant_ids"])
        for tid in target_ids:
            if tid in QUERY_FIXES and item["query"] == QUERY_FIXES[tid][0]:
                item["query"] = QUERY_FIXES[tid][1]
    return eval_set


def build_pool():
    import search_core

    catalog = {row["id"]: row for row in load_jsonl(CATALOG_PATH)}
    eval_set = apply_query_fixes(load_jsonl(EVAL_SET_PATH))

    with open(POOL_PATH, "w", encoding="utf-8") as f:
        for qi, item in enumerate(eval_set):
            # id đúng gốc lấy từ "target_id" nếu đã merge trước đó, không thì từ relevant_ids.
            target_ids = item.get("target_ids", item["relevant_ids"])
            candidate_ids = list(target_ids)
            for components in (["bm25"], ["dense"], ["bm25", "dense"]):
                results = search_core.search(item["query"], "text", k=POOL_DEPTH,
                                             components=components)
                candidate_ids += [r["id"] for r in results]
            candidate_ids = list(dict.fromkeys(candidate_ids))   # bỏ trùng, giữ thứ tự

            candidates = []
            for doc_id in candidate_ids:
                row = catalog[doc_id]
                desc = row["search_text"][len(row["caption"]):].strip()
                candidates.append({
                    "id": doc_id,
                    "caption": row["caption"],
                    "brand": row.get("brand"),
                    "category": row["category"],
                    "desc": desc[:DESC_CHARS],
                })
            f.write(json.dumps({"qid": qi, "query": item["query"], "target_ids": target_ids,
                                "candidates": candidates}, ensure_ascii=False) + "\n")
            print(f"[{qi + 1}/{len(eval_set)}] {len(candidates)} ứng viên")


def merge():
    eval_set = apply_query_fixes(load_jsonl(EVAL_SET_PATH))

    judgments = {}   # qid -> {id: grade}
    for path in sorted(glob.glob(JUDGMENTS_GLOB)):
        for row in load_jsonl(path):
            judgments[row["qid"]] = {int(k): int(v) for k, v in row["grades"].items()}

    # Nhãn phải phủ đúng pool hiện tại (pool đổi khi truy vấn được sửa).
    pool = {row["qid"]: row for row in load_jsonl(POOL_PATH)}
    for qi, row in pool.items():
        ungraded = [c["id"] for c in row["candidates"] if c["id"] not in judgments.get(qi, {})]
        if ungraded:
            sys.exit(f"qid {qi} còn ứng viên chưa chấm: {ungraded}")

    missing = [qi for qi in range(len(eval_set)) if qi not in judgments]
    if missing:
        sys.exit(f"Thiếu nhãn cho {len(missing)} câu, vd qid {missing[:10]}")

    for qi, item in enumerate(eval_set):
        target_ids = item.get("target_ids", item["relevant_ids"])
        grades = judgments[qi]
        for tid in target_ids:
            grades[tid] = 3
        # Sắp theo grade giảm dần cho dễ đọc; key là str vì JSON.
        qrels = {str(k): v for k, v in sorted(grades.items(), key=lambda x: (-x[1], x[0]))}
        item["target_ids"] = target_ids
        item["qrels"] = qrels
        item["relevant_ids"] = [int(k) for k, v in qrels.items() if v >= RELEVANT_THRESHOLD]

    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        for item in eval_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    n_rel = [len(item["relevant_ids"]) for item in eval_set]
    dist = {}
    for item in eval_set:
        for v in item["qrels"].values():
            dist[v] = dist.get(v, 0) + 1
    print(f"Đã ghi {len(eval_set)} câu vào {EVAL_SET_PATH}")
    print(f"relevant_ids/câu: trung bình {sum(n_rel) / len(n_rel):.2f}, tối đa {max(n_rel)}")
    print(f"Phân bố nhãn: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "pool":
        build_pool()
    elif mode == "merge":
        merge()
    else:
        sys.exit("Dùng: python step16_build_qrels.py pool|merge")
