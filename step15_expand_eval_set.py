"""
Bước 15: gộp ground truth A1 + A3 thành một bộ eval tiếng Anh TARGET_TOTAL = 250 câu.

Gồm: các câu gốc A1 (giữ nguyên thứ tự ở đầu file), 20 câu A3 viết tay, phần còn lại sinh
bằng template trên sản phẩm chưa dùng. Thêm answer_should_mention cho mọi dòng.
Output: ghi đè data/eval_set.jsonl (dùng chung cho A1 và A3).
Chạy: python step15_expand_eval_set.py (chỉ chạy trên eval_set.jsonl gốc, xem ghi chú trong main).
"""
import json
import random

CATALOG_PATH = "data/catalog.jsonl"
EVAL_SET_PATH = "data/eval_set.jsonl"
TARGET_TOTAL = 250
SEED = 42

# 20 câu A3 viết tay (tiếng Anh), mỗi câu 1 id sản phẩm đúng.
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
    """Lấy max_words từ đầu của caption, viết thường."""
    words = caption.strip().split()
    return " ".join(words[:max_words]).lower()


def build_answer_should_mention(row: dict) -> list:
    """Từ khoá câu trả lời nên nhắc: brand + 3 từ đầu caption (fallback: 30 ký tự đầu caption)."""
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

    # Nhóm 1: câu gốc A1, giữ nguyên thứ tự (step13 lấy N dòng đầu), chỉ thêm answer_should_mention.
    # Đọc và ghi đè cùng file: chạy lần 2 sẽ lỗi (n_needed âm) vì file đã có 250 dòng.
    original_a1 = load_jsonl(EVAL_SET_PATH)
    for item in original_a1:
        target_id = item["relevant_ids"][0]
        item["answer_should_mention"] = build_answer_should_mention(catalog[target_id])
    print(f"Nhóm 1 (A1 gốc, giữ nguyên): {len(original_a1)} câu")

    # Nhóm 2: 20 câu A3 viết tay ở trên.
    print(f"Nhóm 2 (A3 gốc, viết lại tiếng Anh): {len(HANDWRITTEN_A3_QUERIES)} câu")

    # Nhóm 3: sinh bằng template trên sản phẩm chưa dùng cho đủ TARGET_TOTAL.
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
    # Các script đánh giá A1 đọc query_type từ file; mọi câu ở đây đều là text.
    for item in all_rows:
        item.setdefault("query_type", "text")

    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nXong: {len(all_rows)} câu -> {EVAL_SET_PATH} (dùng chung cho A1 và A3)")


if __name__ == "__main__":
    main()
