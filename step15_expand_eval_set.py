"""
================================================================================
BƯỚC 15 — GỘP GROUND TRUTH A1+A3 THÀNH 1 BỘ 250 CÂU DUY NHẤT (tiếng Anh)
================================================================================

TẠI SAO GỘP LÀM 1 FILE, KHÔNG TÁCH RIÊNG eval_set.jsonl (A1) VÀ
rag_eval_set.jsonl (A3) NHƯ TRƯỚC?
    A3 = "upgrade hệ search A1 lên bằng LLM" -- KHÔNG PHẢI một hệ thống
    tách biệt với qrels riêng. Ground truth cho retrieval (relevant_ids)
    dùng CHUNG cho cả 2 tầng: A1 đo retrieval thuần, A3 đo thêm tầng
    generation TRÊN CÙNG retrieval đó. Tách 2 file trước đây tạo cảm giác
    sai là A3 có 1 bộ dữ liệu riêng, trong khi thực chất A3 chỉ cần THÊM
    1 field (`answer_should_mention`) vào ĐÚNG bộ eval đã có của A1.

TẠI SAO TOÀN BỘ QUERY PHẢI LÀ TIẾNG ANH?
    30 câu gốc của A1 (`eval_set.jsonl`) viết bằng tiếng Anh (khớp
    caption/search_text tiếng Anh trong catalog). Để BM25 (so khớp từ vựng
    thô) hoạt động nhất quán trên toàn bộ 250 câu, KHÔNG được trộn ngôn ngữ
    -- 230 câu Vietnamese không dấu đã sinh trước đó (rag_eval_set.jsonl,
    đã XOÁ) là SAI QUY ƯỚC, phải viết lại bằng tiếng Anh.

QUY MÔ 250 CÂU (nhóm 5 người x 50 câu/người), gồm 3 nhóm:
    1. 30 câu GỐC của A1 (viết tay, giữ đúng thứ tự đầu file -- các script
       khác như step13_build_image_eval_set.py lấy N dòng ĐẦU nên KHÔNG
       được đổi thứ tự 30 dòng này).
    2. 20 câu GỐC của A3 cũ (rag_eval_set.jsonl trước đây, viết TAY nhưng
       bằng tiếng Việt) -- viết lại bằng tiếng Anh, GIỮ NGUYÊN id sản phẩm.
    3. 200 câu MỚI, sinh bằng template tiếng Anh (giống tinh thần
       step15_expand_rag_eval_set.py cũ, nhưng đổi ngôn ngữ) -- lấy 200 sản
       phẩm KHÁC (không trùng 50 câu trên) để tối đa hoá độ đa dạng.

CHẠY: python step15_expand_eval_set.py
OUTPUT: ghi ĐÈ data/eval_set.jsonl (30 câu gốc giữ nguyên vị trí đầu, thêm
        answer_should_mention cho TẤT CẢ 250 dòng). data/rag_eval_set.jsonl
        không còn cần thiết -- step_ablation_rag.py (A3) giờ đọc thẳng
        data/eval_set.jsonl.
"""
import json
import random

CATALOG_PATH = "data/catalog.jsonl"
EVAL_SET_PATH = "data/eval_set.jsonl"
TARGET_TOTAL = 250   # 5 người x 50 câu/người
SEED = 42

# 20 câu A3 gốc -- viết TAY bằng tiếng Anh (dịch lại đúng ý 20 câu tiếng Việt
# cũ, GIỮ NGUYÊN 20 id sản phẩm cũ -- không đổi ground truth, chỉ đổi ngôn ngữ).
HANDWRITTEN_A3_QUERIES = [
    {"query": "10-in-1 air fryer toaster oven combo", "relevant_ids": [272],
     "answer_should_mention": ["NuWave", "Air Fryer"]},
    {"query": "smart thermostat with air quality sensor", "relevant_ids": [443],
     "answer_should_mention": ["ecobee", "Thermostat"]},
    {"query": "16 inch chainsaw guide bar", "relevant_ids": [508],
     "answer_should_mention": ["STIHL", "Guide Bar"]},
    {"query": "steel side tool cabinet with keyed lock", "relevant_ids": [514],
     "answer_should_mention": ["Proto", "Cabinet"]},
    {"query": "brake drum for a Jeep", "relevant_ids": [779],
     "answer_should_mention": ["Crown Automotive", "Brake Drum"]},
    {"query": "bathroom scale that can hold 440 pounds", "relevant_ids": [1227],
     "answer_should_mention": ["Etekcity", "Scale"]},
    {"query": "toilet seat with a built-in bidet spray", "relevant_ids": [1401],
     "answer_should_mention": ["Duravit", "SensoWash"]},
    {"query": "inflatable water roller for the pool", "relevant_ids": [1434],
     "answer_should_mention": ["Inflatable", "Water Roller"]},
    {"query": "one-person camping tent with a stove jack", "relevant_ids": [1623],
     "answer_should_mention": ["ONETIGRIS", "Tent"]},
    {"query": "SFI rated junior racing suit", "relevant_ids": [1764],
     "answer_should_mention": ["RaceQuip", "Racing Suit"]},
    {"query": "memory foam cervical pillow for neck pain", "relevant_ids": [1847],
     "answer_should_mention": ["Memory Foam", "Pillow"]},
    {"query": "rotating makeup organizer with multiple tiers", "relevant_ids": [1921],
     "answer_should_mention": ["SONGMICS", "Organizer"]},
    {"query": "asphalt lute with a long handle", "relevant_ids": [366],
     "answer_should_mention": ["Kraft Tool", "Lute"]},
    {"query": "microfibre applicator pad for car wax", "relevant_ids": [794],
     "answer_should_mention": ["Gtechniq", "Applicator"]},
    {"query": "tool lanyard with a carabiner for working at height", "relevant_ids": [827],
     "answer_should_mention": ["Ergodyne", "Lanyard"]},
    {"query": "toggle style guitar capo", "relevant_ids": [1002],
     "answer_should_mention": ["Dunlop", "Capo"]},
    {"query": "slide stopper accessory for a trombone", "relevant_ids": [1397],
     "answer_should_mention": ["Yamaha", "Slide"]},
    {"query": "long track speed skates", "relevant_ids": [1493],
     "answer_should_mention": ["Zandstra", "Skate"]},
    {"query": "electric combi oven for a commercial kitchen", "relevant_ids": [1614],
     "answer_should_mention": ["Convotherm", "Combi Oven"]},
    {"query": "wall plate with an HDMI and ethernet port", "relevant_ids": [1479],
     "answer_should_mention": ["RiteAV", "HDMI"]},
]

TEMPLATES = [
    "{caption}",
    "looking for {caption}",
    "need to buy {caption}",
    "do you have {caption}",
    "price of {caption}",
    "{brand} {caption}",
    "is there a cheaper {caption}",
    "where can I find {caption}",
    "recommend a good {caption}",
    "in stock: {caption}",
]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def short_caption(caption: str, max_words: int = 8) -> str:
    words = caption.strip().split()
    return " ".join(words[:max_words]).lower()


def build_answer_should_mention(row: dict) -> list:
    mentions = []
    if row.get("brand"):
        mentions.append(row["brand"])
    first_words = " ".join(row["caption"].strip().split()[:3])
    if first_words:
        mentions.append(first_words)
    return mentions or [row["caption"][:30]]


def main():
    random.seed(SEED)
    catalog_rows = load_jsonl(CATALOG_PATH)
    catalog = {r["id"]: r for r in catalog_rows}

    # --- Nhóm 1: 30 câu gốc A1, giữ NGUYÊN thứ tự + nội dung, chỉ thêm
    # answer_should_mention (field mới, các câu này trước đây chưa có). ---
    original_a1 = load_jsonl(EVAL_SET_PATH)
    for item in original_a1:
        target_id = item["relevant_ids"][0]
        item["answer_should_mention"] = build_answer_should_mention(catalog[target_id])
    print(f"Nhóm 1 (A1 gốc, giữ nguyên): {len(original_a1)} câu")

    # --- Nhóm 2: 20 câu gốc A3, viết lại tiếng Anh (đã soạn tay ở trên). ---
    print(f"Nhóm 2 (A3 gốc, viết lại tiếng Anh): {len(HANDWRITTEN_A3_QUERIES)} câu")

    # --- Nhóm 3: sinh thêm bằng template tiếng Anh cho đủ 250. ---
    used_ids = {item["relevant_ids"][0] for item in original_a1}
    used_ids |= {item["relevant_ids"][0] for item in HANDWRITTEN_A3_QUERIES}

    n_needed = TARGET_TOTAL - len(original_a1) - len(HANDWRITTEN_A3_QUERIES)
    candidates = [row for row in catalog_rows if row["id"] not in used_ids]
    sampled = random.sample(candidates, n_needed)

    templated = []
    for row in sampled:
        template = random.choice(TEMPLATES)
        query = template.format(brand=row.get("brand") or "this shop", caption=short_caption(row["caption"]))
        templated.append({
            "query": query,
            "relevant_ids": [row["id"]],
            "answer_should_mention": build_answer_should_mention(row),
        })
    print(f"Nhóm 3 (sinh bằng template tiếng Anh): {len(templated)} câu")

    all_rows = original_a1 + HANDWRITTEN_A3_QUERIES + templated
    # query_type: giữ "text" cho tất cả -- rag_core.answer() luôn gọi search()
    # với query_type="text" cố định, còn step8/step9/step10 (A1) đọc field
    # này trực tiếp từ eval_set.jsonl.
    for item in all_rows:
        item.setdefault("query_type", "text")

    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nXong: {len(all_rows)} câu -> {EVAL_SET_PATH} (dùng chung cho A1 và A3)")


if __name__ == "__main__":
    main()
