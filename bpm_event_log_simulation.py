"""
BPM: mô phỏng event log cho quy trình "Sales tư vấn sản phẩm" có Loop "khách chưa hài lòng -> tìm lại".

3 kịch bản: A không hỗ trợ, B chỉ có search A1, C có A1 + Agentic RAG (A3).
Thời gian mỗi lần tìm và xác suất lặp p là giả định trong SCENARIOS; so với công thức Loop CT = T / (1 - p).
Caption lấy từ vài dòng đầu data/catalog.jsonl, chỉ làm nhãn cho case.
CHẠY: python bpm_event_log_simulation.py -> ghi bpm_event_log.csv và in bảng so sánh CT.
"""
import csv
import json
import random
from datetime import datetime, timedelta

CATALOG_PATH = "data/catalog.jsonl"
OUTPUT_CSV = "bpm_event_log.csv"
N_CASES_PER_SCENARIO = 8
MAX_ATTEMPTS_SAFETY = 10   # chặn vòng lặp vô hạn khi mô phỏng

# duration_range: (T_min, T_max) phút cho 1 lần tìm; loop_probability: xác suất phải tìm lại
SCENARIOS = {
    "A_khong_ho_tro": {
        "duration_range": (4.0, 6.0),
        "loop_probability": 0.40,
    },
    "B_chi_co_A1": {
        "duration_range": (0.5, 1.0),
        "loop_probability": 0.15,
    },
    "C_co_A1_va_A3_agentic": {
        "duration_range": (1.0, 1.5),   # lâu hơn B vì Agentic có thể search ngầm thêm lượt
        "loop_probability": 0.03,       # phần lớn việc lặp đã nằm trong Agentic RAG
    },
}


def load_sample_products():
    products = []
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            products.append(json.loads(line))
            if len(products) >= N_CASES_PER_SCENARIO:
                break
    return products


def simulate_case(scenario_cfg, case_id, product, start_time):
    """Lặp "tìm kiếm" tới khi khách hài lòng hoặc chạm MAX_ATTEMPTS_SAFETY; mỗi lần là 1 dòng log."""
    rows = []
    lo, hi = scenario_cfg["duration_range"]
    p_loop = scenario_cfg["loop_probability"]
    current_time = start_time
    attempt = 0

    while True:
        attempt += 1
        duration = round(random.uniform(lo, hi), 2)
        task_end = current_time + timedelta(minutes=duration)

        rows.append({
            "case_id": case_id,
            "product_context": product["caption"],
            "task": "Sales tim kiem san pham",
            "attempt_number": attempt,
            "start_time": current_time.isoformat(),
            "end_time": task_end.isoformat(),
            "duration_minutes": duration,
        })
        current_time = task_end

        must_loop = random.random() < p_loop   # khách chưa hài lòng -> tìm lại
        if not must_loop or attempt >= MAX_ATTEMPTS_SAFETY:
            break

    return rows, current_time


def compute_ct_by_case(rows):
    """CT mỗi case = tổng thời gian mọi lần tìm."""
    case_ids = sorted(set(r["case_id"] for r in rows))
    cycle_times = []
    for cid in case_ids:
        case_rows = [r for r in rows if r["case_id"] == cid]
        cycle_times.append(sum(r["duration_minutes"] for r in case_rows))
    return cycle_times


def theoretical_ct_loop_formula(scenario_cfg):
    """CT = T / (1 - p), với T là trung bình của duration_range."""
    lo, hi = scenario_cfg["duration_range"]
    t_avg = (lo + hi) / 2
    p = scenario_cfg["loop_probability"]
    return t_avg / (1 - p)


def main():
    products = load_sample_products()
    print(f"Dùng {len(products)} sản phẩm thật từ catalog.jsonl làm ngữ cảnh case\n")

    start_time = datetime(2026, 9, 7, 9, 0)
    all_rows_by_scenario = {}

    for scenario_name, cfg in SCENARIOS.items():
        rows = []
        t = start_time
        for i, product in enumerate(products):
            case_id = f"CONSULT-{scenario_name[:1]}-{i+1:03d}"
            case_rows, t = simulate_case(cfg, case_id, product, t)
            rows.extend(case_rows)
            t += timedelta(minutes=1)   # nghỉ giữa các case
        all_rows_by_scenario[scenario_name] = rows

    all_rows_flat = [r for rows in all_rows_by_scenario.values() for r in rows]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case_id", "product_context", "task", "attempt_number",
            "start_time", "end_time", "duration_minutes"
        ])
        writer.writeheader()
        writer.writerows(all_rows_flat)
    print(f"Đã ghi {len(all_rows_flat)} dòng event log vào {OUTPUT_CSV}\n")

    print(f"{'Kịch bản':<24} {'CT mô phỏng (phút)':<20} {'CT công thức Loop':<20} {'Số lần lặp TB':<15}")
    print("-" * 79)

    ct_a_sim = None
    for scenario_name, cfg in SCENARIOS.items():
        rows = all_rows_by_scenario[scenario_name]
        cycle_times = compute_ct_by_case(rows)
        ct_sim = sum(cycle_times) / len(cycle_times)
        ct_theory = theoretical_ct_loop_formula(cfg)

        case_ids = set(r["case_id"] for r in rows)
        avg_attempts = len(rows) / len(case_ids)

        if ct_a_sim is None:
            ct_a_sim = ct_sim

        print(f"{scenario_name:<24} {ct_sim:<20.2f} {ct_theory:<20.2f} {avg_attempts:<15.2f}")

    print("\n-> Cột 'CT mô phỏng' và 'CT công thức Loop' nên gần nhau -- nếu lệch nhiều,")
    print("   tăng N_CASES_PER_SCENARIO để mô phỏng ổn định hơn (giảm nhiễu do random).")
    print("-> 'Số lần lặp TB' cho thấy trực quan: kịch bản A Sales phải thử ~1.6-1.7 lần/case,")
    print("   kịch bản C gần như luôn xong ở lần đầu (Loop đã bị 'nuốt' vào bên trong Agentic RAG).")


if __name__ == "__main__":
    main()
