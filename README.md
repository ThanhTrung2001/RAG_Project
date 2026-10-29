# A1 — Multimodal Search System (text + image) + A3 RAG Chatbot

**Dataset:** Shopify/product-catalogue (Hugging Face) — first 2000 products, images + text + metadata.
**Architecture:** 100% local — SigLIP + FAISS (dense) + BM25 (sparse) + RRF fusion for search; a local
LLM served by [Ollama](https://ollama.com) generates the chatbot answers.
**Theory + report:** `docs/BAO_CAO_TONG_HOP.md` — full theory + code + rubric.

## Quick start — run only the chatbot

### Requirements
- Python 3.10+
- [Ollama](https://ollama.com/download) installed (the chatbot calls it to generate answers)
- Internet on the first run (downloads the dataset + the `google/siglip-base-patch16-224` model)

### 1. Install

```bash
git clone <repo-url>
cd a1_multimodal_search

python -m venv venv
# Windows:      venv\Scripts\activate
# Linux/macOS:  source venv/bin/activate
pip install -r requirements.txt

# Config: OLLAMA_MODEL, OLLAMA_URL (defaults work as-is)
cp .env.example .env            # Windows cmd: copy .env.example .env

# Pull the LLM — must match OLLAMA_MODEL in .env
ollama pull llama3.1
```

### 2. Build the data (once)

The images and indexes are **not on GitHub** (too large, ignored in `.gitignore`), so they
must be rebuilt locally. Only steps 1–3 are needed for the chatbot:

```bash
python step1_build_catalog.py     # download the dataset -> data/catalog.jsonl + data/images/
python step2_build_embeddings.py  # SigLIP image embeddings (slowest step, much faster with a GPU)
python step3_build_index.py       # build data/dense.index (FAISS) + data/bm25.pkl (BM25)
```

> **Shortcut:** if you were given `a1_data_output_siglip.zip` (prebuilt by
> `kaggle/kaggle_build_A1_data_siglip.py`), extract it into the project root — the files land in
> `data/...` — and skip the 3 steps above.

### 3. Run

```bash
ollama serve                      # skip if Ollama is already running (e.g. as a background app)
uvicorn app:app --port 8000
```

Open in a browser:
- Chatbot: http://localhost:8000/chat
- Search page: http://localhost:8000/search

### Ask follow-up questions about a product

After searching, click a product card, then ask e.g. “Cái này chất liệu gì?” or
“Is it waterproof?”. You can also say “Sản phẩm số 2 nặng bao nhiêu?” using the
latest search's source list. That list and its numbering stay available while you
ask about product 1, then product 2; short follow-ups use the last product discussed.
A new search replaces the list and resets focus to its sole result, or to the one
distinct product cited in the answer. Multiple cited products leave no focus.
The UI automatically selects and opens that product; no click is required.
Search responses expose this selection as `focused_product_id` (null if ambiguous).
Detail answers load that product's catalog record by ID;
missing attributes must be reported as unavailable. Several possible products
without a selection trigger a clarification question.

Requests for recommendations, alternatives or a new topic still use search.
Clear rules handle explicit references; ambiguous messages use an LLM intent
classifier (`search`, `detail`, `clarify`). This classifier still needs evaluation
on real conversations. Numbered citations refer to the current response's `sources`;
the separate `product_options` field on follow-up responses keeps the search list
for selecting products and resolving spoken numbers.

The text API accepts `selected_product_id` and `product_dialogue=false` to disable
the new routing for baseline comparisons. State is session-local RAM and is lost
on restart. New image uploads always use search.

Offline regressions (stub catalog/LLM, including API integration):

```bash
pip install httpx  # optional test dependency for FastAPI TestClient
python -m unittest discover -s tests -v
```

Conversation scenarios live in `tests/product_dialogue_scenarios.json`. They include
the air-fryer and thermostat examples, switching products in Vietnamese, comparison
then choosing one product, changing topic, one-result follow-ups, clarification
then choosing by number, the original no-click examples, image-only follow-ups,
and image+text search followed by switching products and uploading a different image.
The offline suite runs all eleven in both pipelines, with and without the UI's
selection ID (44 variants, 276 turns). Retrieval and LLM output
are fixtures here; these checks verify routing/state rather than answer quality.

```bash
node tests/check_chat_product_flow.cjs  # actual ask() wiring with DOM/API fixtures
python tests/run_product_dialogue_live.py --output .claude/tasks/happy-live.json
python tests/run_product_dialogue_live.py --scenario image_text_switch_and_replace --scenario image_only_followup --modes naive agentic --output .claude/tasks/happy-image-live.json
```

The live runner is opt-in: it uses the cached search model/indexes and real Ollama,
records answers and intent/identity/state checks after every turn, and runs the
original air-fryer example in Naive and thermostat example in Agentic, now without
clicking or selecting a product by number before the first follow-up. `--modes`
runs each selected scenario in both pipelines. Image steps load actual local photos.
It does not score factual answer quality. The longer switching scenarios identify
products by number when moving between them;
“this product” alone is ambiguous when several products are proposed and no focus
exists. A single proposed product is focused automatically.

Image+text uses both an image search and a hybrid text search, merged with RRF;
text-only reranking is disabled for image input. Every new upload is a search,
even if its text looks like a detail question. A following text-only message can
ask about the focused product or select a numbered item from the retained list.
The upload is cleared after sending: "more options" on a later text-only turn uses
textual history, not the previous raw image. Upload it again for another visual search.
For slow local models the runner uses the app's 600-second timeout per LLM call.
To resume after a failure, use `--resume <previous-report.json>` and a different
`--output` path; passing prefix turns are explicitly marked as reused, then the
remaining conversation runs again. Keep the original report as failure evidence.

### Optional: DeBERTa reranker

The chatbot's reranker defaults to `none`; `cross_encoder` (MiniLM) downloads automatically on first
use. The `deberta` option only appears when the distilled checkpoint has been placed in
`models/deberta-10k-rank_net/` (the folder containing `config.json`) — it is not in the repo.

### Troubleshooting
- **`RuntimeError: Error in faiss::FileIOReader ... data/dense.index`** / missing `bm25.pkl` /
  `Directory 'data/images' does not exist` → section "2. Build the data" has not been run.
- **Chatbot replies that it cannot reach Ollama / model not found** → check `ollama serve` is
  running and `ollama list` contains the model named in `OLLAMA_MODEL`.

## Full pipeline (evaluation + ablations)

After steps 1–3 above:

```bash
# Step 8: ablation on data/eval_set.jsonl, prints Recall@k / nDCG@k per configuration
python step8_run_ablation.py

# Further ablations / analyses (each script's docstring explains its input + output)
python step9_ablation_index_fields.py
python step10_sweep_rrf.py
python step_ablation_rerank.py
python step_ablation_rag.py          # needs Ollama
```

Heavy runs (embedding-model ablation, RAG ablation on GPU) have Kaggle versions in `kaggle/`.

## File structure — mapped to the steps in `docs/BAO_CAO_TONG_HOP.md`

| File | Step | Role |
|---|---|---|
| `step1_build_catalog.py` | 1 | Download dataset, create `catalog.jsonl` + `images/` folder |
| `step2_build_embeddings.py` | 2 | Generate SigLIP embeddings for images + captions, save as `.npy` |
| `step3_build_index.py` | 3 | Build FAISS (`dense.index`) + BM25 (`bm25.pkl`) |
| `search_core.py` | 4 | Registry pattern: `@component("bm25")`, `@component("dense")`, `search()`, `rrf_fusion()`, rerankers |
| `metrics.py` | 5 | Hand-written `recall_at_k()`, `ndcg_at_k()` (no sklearn) |
| `data/eval_set.jsonl` | 6 | Evaluation queries with the correct product ids |
| `app.py` | 7 | FastAPI: search APIs (`/api/v1/search`, `/api/v1/search_by_image`, …) + chatbot APIs (`/api/v1/ask`, `/api/v1/ask_image`, `/api/v1/reset_session`) |
| `step8_run_ablation.py` | 8 | Runs bm25-only / dense-only / hybrid, prints comparison table |
| `rag_core.py`, `rag_metrics.py` | A3 | RAG pipeline (retrieve → prompt → Ollama), multi-turn sessions, agentic mode; RAG metrics |
| `frontend/search.html`, `frontend/chat.html` | 10 | Search UI and chatbot UI |
| `docs/BAO_CAO_TONG_HOP.md` | 11 | Full theory + code + rubric |

## Every `.py` file is fully commented following a consistent structure

Each file starts with a docstring block explaining:
- **What problem this step solves** (why it's needed, what happens if skipped)
- **How it works** (algorithm, formulas, worked examples where relevant)
- **How to run it + expected output**

Inside function bodies, lines whose meaning isn't obvious (e.g. `feat.norm()`,
`np.argsort()[::-1]`, the `@component` decorator) have inline comments
explaining WHY they're written that way — read the `.py` files directly for
more depth than this README summary provides.
