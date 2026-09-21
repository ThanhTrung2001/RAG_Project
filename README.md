# A1 — Multimodal Search System (text + image)

**Dataset:** Shopify/product-catalogue (Hugging Face) — first 2000 products, images + text + metadata.
**Architecture:** 100% local, no external server/account needed — FAISS (dense) + BM25 (sparse) + RRF fusion.
**Read before starting:** `BAO_CAO_TONG_HOP.md` — full theory + code + rubric, used to write the submitted report.

## Setup
```bash
pip install -r requirements.txt --break-system-packages
```

## Run in the correct order (required — each step depends on the previous one)

```bash
# Step 1: download the Shopify/product-catalogue dataset, normalize into catalog.jsonl
python step1_build_catalog.py

# Step 2: generate CLIP embeddings for images + captions (most time-consuming step)
python step2_build_embeddings.py

# Step 3: build FAISS (dense) + BM25 (sparse)
python step3_build_index.py

# Step 6 (MANUAL — cannot be scripted): open data/catalog.jsonl to view the
# actual products, write 20-40 queries with the correct id for each one
cp data/eval_set_template.jsonl data/eval_set.jsonl
# open the copied eval_set.jsonl file and add enough queries

# Step 8: run ablation, print Recall@k / nDCG@k table for each configuration
python step8_run_ablation.py

# Step 7 + 10: run the API + demo frontend
uvicorn app:app --reload --port 8000
# open in browser: http://localhost:8000
```

## File structure — mapped to the steps in BAO_CAO_TONG_HOP.md

| File | Step | Role |
|---|---|---|
| `step1_build_catalog.py` | 1 | Download dataset, create `catalog.jsonl` + `images/` folder |
| `step2_build_embeddings.py` | 2 | Generate CLIP embeddings for images + captions, save as `.npy` |
| `step3_build_index.py` | 3 | Build FAISS (`dense.index`) + BM25 (`bm25.pkl`) |
| `search_core.py` | 4 | Registry pattern: `@component("bm25")`, `@component("dense")`, `search()`, `rrf_fusion()` |
| `metrics.py` | 5 | Hand-written `recall_at_k()`, `ndcg_at_k()` (no sklearn) |
| `data/eval_set_template.jsonl` | 6 | 8 real sample queries from the dataset — needs to be extended to 20-40 |
| `app.py` | 7 | FastAPI: `/api/v1/search`, `/api/v1/search_by_image`, `/api/v1/components` |
| `step8_run_ablation.py` | 8 | Runs 3 configurations (bm25-only / dense-only / hybrid), prints comparison table |
| *(do it yourself, cannot be pre-coded)* | 9 | Error analysis — read ablation results, group errors by cause |
| `frontend/index.html` | 10 | Minimal UI, checkboxes auto-generated from `/api/v1/components` |
| `BAO_CAO_TONG_HOP.md` | 11 | Full theory + code + rubric — condense into a ≤8-page report for submission |

## Every `.py` file is fully commented following a consistent structure

Each file starts with a docstring block explaining:
- **What problem this step solves** (why it's needed, what happens if skipped)
- **How it works** (algorithm, formulas, worked examples where relevant)
- **How to run it + expected output**

Inside function bodies, lines whose meaning isn't obvious (e.g. `feat.norm()`,
`np.argsort()[::-1]`, the `@component` decorator) have inline comments
explaining WHY they're written that way — read the `.py` files directly for
more depth than this README summary provides.

## What you still need to do yourself (cannot be pre-coded since it depends on your dataset/domain)

1. **Write the real `data/eval_set.jsonl`** — 20-40 queries, look at the actual
   images in `data/images/` to know which id is correct for each query. The
   most time-consuming part but required (graded separately, 20% of the score).
2. **Error analysis (step 9)** — after running `step8_run_ablation.py`, filter
   queries with low Recall, open the images, cross-reference `category`/`brand`
   in the catalog, group by root cause. No code can replace manually looking
   and analyzing (20% of the rubric).
3. **(Optional, for extra engineering credit) Add a rerank component** — write
   a function using a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`),
   register it as `@component("rerank")` in `search_core.py`. Since rerank
   must run AFTER the fusion results are available (not as a parallel branch
   alongside BM25/Dense), it needs its own post-processing logic — it isn't
   registered in the same registry as the other two components.
