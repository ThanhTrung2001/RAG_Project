"""
Đóng gói những file cần để chạy step_ablation_rag.py (A3) và các ablation retrieval A1
(step8/9/10, step_ablation_rerank) trên Kaggle thành 1 file zip.
Chạy lại script này mỗi khi đổi data/eval_set.jsonl hoặc code, rồi upload bản mới lên Kaggle.

Chạy ở thư mục gốc project:
    python kaggle/make_kaggle_rag_bundle.py               # gồm DeBERTa (~670 MB)
    python kaggle/make_kaggle_rag_bundle.py --no-deberta  # bỏ DeBERTa (~10 MB), Kaggle sẽ bỏ qua cấu hình naive_deberta
Output: dist/kaggle_rag_bundle.zip -> upload lên Kaggle làm Dataset, rồi chạy kaggle/kaggle_run_rag_ablation.py.
Không cần ảnh sản phẩm: tầng tìm kiếm chỉ đọc catalog và 2 index.
"""
import argparse
import os
import zipfile

CODE_FILES = ["search_core.py", "rag_core.py", "rag_metrics.py", "metrics.py", "step_ablation_rag.py",
              # ablation retrieval (A1), chạy bằng kaggle/kaggle_run_retrieval_ablation.py
              "step8_run_ablation.py", "step9_ablation_index_fields.py", "step10_sweep_rrf.py",
              "step_ablation_rerank.py",
              # hỏi bằng ảnh (A3), chạy bằng kaggle_run_rag_ablation.py --with-image
              "step18_ablation_image_rag.py",
              # Memory khi tìm kiếm (A3), chạy bằng kaggle_run_rag_ablation.py --with-multiturn
              "step19_ablation_multiturn.py"]
DATA_FILES = ["data/catalog.jsonl", "data/dense.index", "data/bm25.pkl", "data/eval_set.jsonl",
              "data/eval_set_image.jsonl"]
IMAGE_DIR = "data/query_images"   # 100 ảnh truy vấn của step18 (~vài MB)
# Kết quả đã chạy: đóng gói để Kaggle chạy --resume, chỉ tính phần còn thiếu (vd cấu hình mới).
RESUME_FILES = ["data/rag_ablation_details.jsonl", "data/image_rag_ablation_details.jsonl",
                "data/multiturn_ablation_details.jsonl"]
DEBERTA_DIR = "models/deberta-10k-rank_net"
OUT_PATH = "dist/kaggle_rag_bundle.zip"
ROOT_IN_ZIP = "a1_rag"   # mọi file nằm trong thư mục này để script Kaggle dễ tìm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-deberta", action="store_true", help="không đóng gói model DeBERTa")
    args = parser.parse_args()

    files = CODE_FILES + DATA_FILES
    files += sorted(f"{IMAGE_DIR}/{n}" for n in os.listdir(IMAGE_DIR))
    files += [f for f in RESUME_FILES if os.path.isfile(f)]
    if not args.no_deberta:
        if not os.path.isdir(DEBERTA_DIR):
            raise SystemExit(f"Không thấy {DEBERTA_DIR}; dùng --no-deberta hoặc tải checkpoint trước")
        for root, _, names in os.walk(DEBERTA_DIR):
            files += [os.path.join(root, n).replace("\\", "/") for n in names]

    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        raise SystemExit(f"Thiếu file: {missing}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with zipfile.ZipFile(OUT_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f"{ROOT_IN_ZIP}/{f}")
            print(f"  + {f}")

    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"\nXong: {OUT_PATH} ({size_mb:.1f} MB, {len(files)} file)")
    print("Upload file này lên Kaggle (Datasets -> New Dataset), rồi gắn vào notebook.")


if __name__ == "__main__":
    main()
