"""
================================================================================
BPM — MÔ PHỎNG EVENT LOG CHO QUY TRÌNH "SALES TƯ VẤN SẢN PHẨM" (3 KỊCH BẢN)
================================================================================

QUY TRÌNH ĐÚNG (không phải Order Processing -- đó là quy trình khác, sau khi
khách đã CHỐT mua):

    Khách nêu yêu cầu -> Sales tìm kiếm sản phẩm -> Trình bày kết quả
         -> Gateway "Khách hài lòng?" --Không--> LOOP quay lại tìm kiếm
                                     --Có--> Chuyển sang lập đơn hàng (kết thúc)

TẠI SAO ĐÂY LÀ LOOP (Buổi 2), KHÔNG PHẢI XOR LỒNG NHAU?
    Khâu nghẽn là "Sales tìm kiếm sản phẩm": giao diện search hiện tại yêu
    cầu nhập riêng từng ô (tên, category, khoảng giá), KHÔNG có filter màu
    sắc/thuộc tính tự do -- khách nói "áo đen khoảng 200 nghìn" nhưng Sales
    không nhập được đúng ý, khách thường CHƯA HÀI LÒNG lần đầu, quay lại
    tìm lần nữa. Đây đúng cấu trúc Loop:
        CT = T(tìm kiếm) / (1 - p)
    với p = xác suất phải lặp lại (khách chưa hài lòng).

TẠI SAO AGENTIC RAG (đã build trong rag_core.py) LÀ LỜI GIẢI ĐÚNG CHO LOOP NÀY?
    answer_agentic() tự đánh giá kết quả, tự viết lại query, tự search lại
    tối đa 2 lần -- TRƯỚC KHI trả kết quả cuối cùng. Nói cách khác, nó đưa
    cái Loop "khách chưa hài lòng -> tìm lại" vào BÊN TRONG 1 lần gọi API,
    thay vì để Sales phải lặp lại thao tác nhiều lần với khách đứng chờ.
    Đây là lý do kịch bản C dưới đây có T/lần cao hơn kịch bản B (vì mỗi
    lần gọi có thể chạy ngầm 1-2 lượt search+đánh giá) nhưng xác suất phải
    lặp lại (nhìn từ phía Sales/khách) THẤP HƠN NHIỀU.

3 KỊCH BẢN:
    A. Không có gì hỗ trợ -- giao diện search nhiều ô, thiếu filter
    B. Chỉ có A1 (search)  -- 1 câu tự nhiên, hiểu được "đen", "200 nghìn"
    C. Có A1 + A3 Agentic  -- hệ thống tự lặp lại BÊN TRONG, Sales chỉ gọi 1 lần

CHẠY: python bpm_event_log_simulation.py
OUTPUT: bpm_event_log.csv + bảng so sánh CT (công thức Loop + mô phỏng thật)
"""
import csv
import json
import random
from datetime import datetime, timedelta

CATALOG_PATH = "data/catalog.jsonl"
OUTPUT_CSV = "bpm_event_log.csv"
N_CASES_PER_SCENARIO = 8
MAX_ATTEMPTS_SAFETY = 10   # chặn vòng lặp vô hạn khi mô phỏng (thực tế Loop có thể kéo dài)

# (T_min, T_max) mỗi lần thử tìm kiếm (phút), và xác suất PHẢI LẶP LẠI (p)
SCENARIOS = {
    "A_khong_ho_tro": {
        "duration_range": (4.0, 6.0),
        "loop_probability": 0.40,   # 40% Sales phải tìm lại vì thiếu filter đúng ý khách
    },
    "B_chi_co_A1": {
        "duration_range": (0.5, 1.0),
        "loop_probability": 0.15,   # search ngữ nghĩa hiểu tốt hơn, nhưng vẫn có thể trật
    },
    "C_co_A1_va_A3_agentic": {
        "duration_range": (1.0, 1.5),   # cao hơn B vì hệ thống có thể tự search ngầm 1-2 lần
        "loop_probability": 0.03,       # Loop đã được xử lý NGẦM bên trong, hiếm khi lộ ra ngoài
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
    """
    Mô phỏng 1 case: lặp lại "tìm kiếm" cho tới khi thành công hoặc chạm
    ngưỡng an toàn. Mỗi lần lặp là 1 dòng event log riêng -- đúng cấu trúc
    event log thật (1 case có thể có NHIỀU dòng cùng activity, khác timestamp).
    """
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

        # random.random() < p_loop -> khách CHƯA hài lòng, lặp lại
        must_loop = random.random() < p_loop
        if not must_loop or attempt >= MAX_ATTEMPTS_SAFETY:
            break

    return rows, current_time


def compute_ct_by_case(rows):
    """CT thật của mỗi case = tổng thời gian TẤT CẢ các lần lặp cộng lại."""
    case_ids = sorted(set(r["case_id"] for r in rows))
    cycle_times = []
    for cid in case_ids:
        case_rows = [r for r in rows if r["case_id"] == cid]
        cycle_times.append(sum(r["duration_minutes"] for r in case_rows))
    return cycle_times


def theoretical_ct_loop_formula(scenario_cfg):
    """
    So sánh với công thức Loop lý thuyết đã học: CT = T(activity) / (1 - p)
    Dùng T trung bình của khoảng (lo, hi) làm T(activity).
    """
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
            t += timedelta(minutes=1)   # nghỉ ngắn giữa các case
        all_rows_by_scenario[scenario_name] = rows

    # Ghi toàn bộ ra 1 file CSV chung
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
