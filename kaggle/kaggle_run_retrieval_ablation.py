"""
Chạy các ablation retrieval A1 trên Kaggle (GPU) với ground truth qrels mới (step16_build_qrels.py):
  step8_run_ablation.py          -- bật/tắt BM25, dense, hybrid, rerank
  step9_ablation_index_fields.py -- trường dữ liệu đưa vào index
  step10_sweep_rrf.py            -- sweep k x N của RRF
  step_ablation_rerank.py        -- cross-encoder vs DeBERTa distilled
Dùng chung bundle với A3:
  1. Ở máy local: python kaggle/make_kaggle_rag_bundle.py -> upload dist/kaggle_rag_bundle.zip làm Kaggle Dataset
     (bản mới, vì bundle cũ chứa eval_set.jsonl/metrics.py trước khi có qrels).
  2. Kaggle notebook: Accelerator = GPU, Internet = On, Add Input = dataset vừa tạo.
  3. Upload file này rồi chạy trong cell:
       !python kaggle_run_retrieval_ablation.py                 # chạy cả 4 bước
       !python kaggle_run_retrieval_ablation.py --steps 8 rerank  # chỉ chạy vài bước
     hoặc dán cả file vào 1 cell và sửa NOTEBOOK_ARGS.
Output (/kaggle/working): retrieval_ablation_output.zip gồm log từng bước (bảng in ra màn hình, step8/9/10
không ghi file json) và data/rerank_ablation_results.json.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile

WORK_DIR = "/kaggle/working/a1_retrieval"
OUT_ZIP = "/kaggle/working/retrieval_ablation_output.zip"
# Tham số dùng khi dán file này vào 1 cell notebook (cell không nhận tham số dòng lệnh).
NOTEBOOK_ARGS = []

PIP_PACKAGES = ["transformers==4.57.1", "sentence-transformers==4.1.0", "faiss-cpu",
                "rank_bm25", "sentencepiece"]
STEPS = {
    "8": "step8_run_ablation.py",
    "9": "step9_ablation_index_fields.py",
    "10": "step10_sweep_rrf.py",
    "rerank": "step_ablation_rerank.py",
}


def find_bundle_dir(explicit=None):
    """Thư mục chứa step8_run_ablation.py trong dataset đã gắn (/kaggle/input/...)."""
    if explicit:
        return explicit
    hits = glob.glob("/kaggle/input/**/step8_run_ablation.py", recursive=True)
    if not hits:
        raise SystemExit("Không tìm thấy step8_run_ablation.py trong /kaggle/input -- "
                         "đã upload bundle MỚI (make_kaggle_rag_bundle.py) và Add Input chưa?")
    return os.path.dirname(hits[0])


def prepare_workdir(bundle_dir, work_dir):
    """Copy code + data sang thư mục ghi được (luôn ghi đè để chắc chắn dùng eval_set mới nhất)."""
    os.makedirs(os.path.join(work_dir, "data"), exist_ok=True)
    for path in glob.glob(os.path.join(bundle_dir, "*.py")):
        shutil.copy(path, work_dir)
    for path in glob.glob(os.path.join(bundle_dir, "data", "*")):
        shutil.copy(path, os.path.join(work_dir, "data", os.path.basename(path)))

    deberta_src = os.path.join(bundle_dir, "models", "deberta-10k-rank_net")
    deberta_dst = os.path.join(work_dir, "models", "deberta-10k-rank_net")
    if os.path.isdir(deberta_src) and not os.path.exists(deberta_dst):
        os.makedirs(os.path.dirname(deberta_dst), exist_ok=True)
        os.symlink(deberta_src, deberta_dst)

    with open(os.path.join(work_dir, "data", "eval_set.jsonl"), encoding="utf-8") as f:
        first = f.readline()
    if '"qrels"' not in first:
        raise SystemExit("eval_set.jsonl trong bundle chưa có qrels -- đóng gói lại bundle sau step16.")
    print(f"Thư mục làm việc: {work_dir} (DeBERTa: {'có' if os.path.exists(deberta_dst) else 'không'})")


def run_step(script, work_dir, log_dir):
    log_path = os.path.join(log_dir, script.replace(".py", ".log"))
    print(f"\n========== {script} ==========", flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, script], cwd=work_dir, env=env, text=True,
                                encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:   # vừa in ra cell vừa ghi log
            print(line, end="", flush=True)
            log.write(line)
        proc.wait()
    if proc.returncode != 0:
        print(f"!! {script} lỗi (exit {proc.returncode}), xem {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", default=list(STEPS), choices=list(STEPS))
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--work-dir", default=WORK_DIR)
    parser.add_argument("--skip-setup", action="store_true", help="bỏ qua pip install")
    in_notebook = "ipykernel" in sys.modules
    args = parser.parse_args(NOTEBOOK_ARGS if in_notebook else None)

    if not args.skip_setup:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PIP_PACKAGES], check=True)

    prepare_workdir(find_bundle_dir(args.bundle_dir), args.work_dir)
    log_dir = os.path.join(args.work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    outputs = [run_step(STEPS[s], args.work_dir, log_dir) for s in args.steps]
    rerank_json = os.path.join(args.work_dir, "data", "rerank_ablation_results.json")
    if "rerank" in args.steps and os.path.exists(rerank_json):
        outputs.append(rerank_json)

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in outputs:
            zf.write(path, os.path.relpath(path, args.work_dir))
    print(f"\nXong: {OUT_ZIP} -- tải ở tab Output; giải nén data/rerank_ablation_results.json vào data/ ở máy local.")


if __name__ == "__main__":
    main()
