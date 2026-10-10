"""
Chạy ablation A3 (step_ablation_rag.py) trên Kaggle với GPU: cài Ollama, tải LLM, chạy, nén kết quả.

Tăng tốc: mỗi GPU chạy 1 `ollama serve` riêng (T4 x2 -> 2 server), mỗi server xử lý --parallel yêu cầu
cùng lúc; step_ablation_rag.py chạy --workers câu song song và chia đều lượt gọi LLM cho các server.

Chuẩn bị:
  1. Ở máy local: python kaggle/make_kaggle_rag_bundle.py  -> upload dist/kaggle_rag_bundle.zip làm Kaggle Dataset.
  2. Kaggle notebook: Settings -> Accelerator = GPU T4 x2 (hoặc P100), Internet = On, Add Input = dataset vừa tạo.
  3. Chạy theo 1 trong 2 cách:
       - Upload file này vào notebook rồi chạy trong cell:
           !python kaggle_run_rag_ablation.py --sample 10    # chạy thử, xem tốc độ
           !python kaggle_run_rag_ablation.py                # chạy đủ 250 câu
       - Hoặc dán toàn bộ file vào 1 cell: sửa NOTEBOOK_ARGS bên dưới rồi chạy cell.
Output (tab Output của Kaggle): rag_ablation_output.zip gồm rag_ablation_results.json,
rag_ablation_details.jsonl và log của Ollama. Giải nén 2 file json vào data/ ở máy local.

Tham số riêng: --model, --parallel (số yêu cầu song song mỗi GPU, mặc định 2), --bundle-dir, --work-dir,
--skip-setup. Các tham số khác (--sample, --configs, --resume, --workers, --no-faithfulness) được chuyển
thẳng cho step_ablation_rag.py; nếu không ghi --workers thì tự đặt = số GPU x --parallel.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile

import requests

WORK_DIR = "/kaggle/working/a1_rag"   # zip kết quả và log Ollama nằm ở thư mục cha (/kaggle/working)
BASE_PORT = 11434
# Tham số dùng khi dán file này vào 1 cell notebook (cell không nhận tham số dòng lệnh).
# Sửa ở đây rồi chạy lại cell, vd [] (đủ 250 câu), ["--sample", "100"], ["--no-faithfulness"].
NOTEBOOK_ARGS = ["--sample", "10"]

PIP_PACKAGES = ["transformers==4.57.1", "sentence-transformers==4.1.0", "faiss-cpu",
                "rank_bm25", "python-dotenv", "sentencepiece"]


def run(cmd, **kwargs):
    print(f"\n$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def find_bundle_dir(explicit=None):
    """Thư mục chứa rag_core.py trong dataset đã gắn (/kaggle/input/<tên dataset>/...)."""
    if explicit:
        return explicit
    hits = glob.glob("/kaggle/input/**/rag_core.py", recursive=True)
    if not hits:
        raise SystemExit("Không tìm thấy rag_core.py trong /kaggle/input -- đã Add Input dataset chưa?")
    return os.path.dirname(hits[0])


def prepare_workdir(bundle_dir, work_dir):
    """Copy code + data sang thư mục ghi được; model DeBERTa chỉ tạo symlink (không copy 700 MB)."""
    os.makedirs(os.path.join(work_dir, "data"), exist_ok=True)
    for path in glob.glob(os.path.join(bundle_dir, "*.py")):
        shutil.copy(path, work_dir)
    for path in glob.glob(os.path.join(bundle_dir, "data", "*")):
        target = os.path.join(work_dir, "data", os.path.basename(path))
        if not os.path.exists(target):   # giữ file chi tiết của lần chạy trước để --resume
            shutil.copy(path, target)

    deberta_src = os.path.join(bundle_dir, "models", "deberta-10k-rank_net")
    deberta_dst = os.path.join(work_dir, "models", "deberta-10k-rank_net")
    if os.path.isdir(deberta_src) and not os.path.exists(deberta_dst):
        os.makedirs(os.path.dirname(deberta_dst), exist_ok=True)
        os.symlink(deberta_src, deberta_dst)
    print(f"Thư mục làm việc: {work_dir} (DeBERTa: {'có' if os.path.exists(deberta_dst) else 'không'})")


def count_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return max(1, sum(1 for line in out.splitlines() if line.startswith("GPU ")))
    except (OSError, subprocess.CalledProcessError):
        return 1   # không có GPU: 1 server chạy CPU


def install_ollama():
    if shutil.which("ollama"):
        print("Ollama đã có sẵn")
        return
    # Bản cài mới của Ollama dùng file .tar.zst nên cần zstd.
    run("apt-get install -y -qq zstd > /dev/null 2>&1 || true", shell=True)
    run("curl -fsSL https://ollama.com/install.sh | sh", shell=True)


def output_dir(work_dir):
    return os.path.dirname(os.path.abspath(work_dir))


def server_url(i):
    return f"http://127.0.0.1:{BASE_PORT + i}"


def wait_ready(url, seconds=60):
    for _ in range(seconds):
        try:
            requests.get(f"{url}/api/tags", timeout=2)
            return True
        except requests.RequestException:
            time.sleep(1)
    return False


def start_servers(n_gpus, parallel, log_dir):
    """Mỗi GPU một `ollama serve` (Kaggle không có systemd nên phải tự chạy nền)."""
    urls = []
    for i in range(n_gpus):
        url = server_url(i)
        if not wait_ready(url, seconds=1):
            env = dict(os.environ,
                       CUDA_VISIBLE_DEVICES=str(i),
                       OLLAMA_HOST=f"127.0.0.1:{BASE_PORT + i}",
                       OLLAMA_NUM_PARALLEL=str(parallel),
                       OLLAMA_KEEP_ALIVE="-1")   # giữ model trong GPU suốt lượt chạy
            log = open(os.path.join(log_dir, f"ollama_serve_{i}.log"), "w")
            subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=subprocess.STDOUT)
            if not wait_ready(url):
                raise SystemExit(f"Ollama trên GPU {i} không khởi động được, xem ollama_serve_{i}.log")
        print(f"Ollama GPU {i}: {url} (parallel={parallel})")
        urls.append(url)
    return urls


def warm_up(url, model):
    """Gọi 1 lần để nạp model vào GPU, tránh tính thời gian nạp vào câu hỏi đầu tiên."""
    start = time.time()
    response = requests.post(f"{url}/api/generate",
                             json={"model": model, "prompt": "Say OK", "stream": False}, timeout=600)
    response.raise_for_status()
    ps = requests.get(f"{url}/api/ps", timeout=5).json()
    for m in ps.get("models", []):
        print(f"  {url}: {m['name']} nạp {time.time() - start:.1f} s, "
              f"{m.get('size_vram', 0) / 1e9:.1f} GB trên GPU / {m.get('size', 0) / 1e9:.1f} GB")


def zip_outputs(work_dir):
    out_dir = output_dir(work_dir)
    zip_path = os.path.join(out_dir, "rag_ablation_output.zip")
    files = [os.path.join(work_dir, "data", "rag_ablation_results.json"),
             os.path.join(work_dir, "data", "rag_ablation_details.jsonl"),
             *glob.glob(os.path.join(out_dir, "ollama_serve_*.log"))]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if os.path.exists(path):
                zf.write(path, os.path.basename(path))
    print(f"\nĐã nén kết quả: {zip_path} (tải ở tab Output)")


def in_notebook():
    # Trong cell Jupyter/Kaggle, sys.argv là tham số của kernel ("-f .../kernel-xxx.json"), không phải của ta.
    return "ipykernel" in sys.modules


def main(argv=None):
    if argv is None:
        argv = NOTEBOOK_ARGS if in_notebook() else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--parallel", type=int, default=2, help="số yêu cầu song song mỗi GPU")
    parser.add_argument("--bundle-dir", default=None, help="mặc định tự tìm trong /kaggle/input")
    parser.add_argument("--work-dir", default=WORK_DIR)
    parser.add_argument("--skip-setup", action="store_true",
                        help="bỏ qua pip/cài Ollama (dùng khi đã cài, hoặc thử ở máy local)")
    args, ablation_args = parser.parse_known_args(argv)
    print(f"Tham số: {argv}")

    started = time.time()
    prepare_workdir(find_bundle_dir(args.bundle_dir), args.work_dir)

    n_gpus = count_gpus()
    print(f"Số GPU: {n_gpus}")
    if not args.skip_setup:
        run([sys.executable, "-m", "pip", "install", "-q", *PIP_PACKAGES])
        install_ollama()
    urls = start_servers(n_gpus, args.parallel, output_dir(args.work_dir))
    if not args.skip_setup:
        # Các server dùng chung thư mục model nên chỉ cần tải 1 lần qua server đầu tiên.
        run(["ollama", "pull", args.model], env=dict(os.environ, OLLAMA_HOST=urls[0].split("//")[1]))
    for url in urls:
        warm_up(url, args.model)

    if "--workers" not in ablation_args:
        ablation_args += ["--workers", str(len(urls) * args.parallel)]
    env = dict(os.environ, OLLAMA_MODEL=args.model, PYTHONUNBUFFERED="1",
               OLLAMA_URL=",".join(f"{u}/api/generate" for u in urls))
    cmd = [sys.executable, "step_ablation_rag.py", *ablation_args]
    print(f"\n$ {' '.join(cmd)}  (cwd={args.work_dir})", flush=True)
    result = subprocess.run(cmd, cwd=args.work_dir, env=env)

    zip_outputs(args.work_dir)
    print(f"Tổng thời gian: {(time.time() - started) / 60:.1f} phút")
    if result.returncode != 0:
        message = (f"step_ablation_rag.py kết thúc với mã lỗi {result.returncode}; "
                   "chạy lại kèm --resume để tiếp tục từ chỗ dừng")
        if in_notebook():
            print(f"\nLỖI: {message}")   # không raise trong cell để khỏi văng traceback dài
        else:
            raise SystemExit(message)


if __name__ == "__main__":
    main()
