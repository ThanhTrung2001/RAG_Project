# Task 7
- Status: IN PROGRESS
- Estimated time: Not estimated
- Goal: Vietnamese classroom explanation of A3, experiment counts, model choices and source-grounded success/failure traces.
## Detailed description
User asks to read docs/results/logs, explain entire A3 in accessible presentation form, including ablation counts, models, cross-encoder, reranking, memory, Naive/Agentic, routing, and concrete top5 success/failure attribution through BM25/dense/RRF/rerank. User supplies numerical case analysis to verify, not treat as authoritative. Deliver in chat. Preserve current unrelated uncommitted drift, including new docs/KICH_BAN_DEMO.md and changed search/rag/router/frontend.
## Acceptance criteria
- [ ] AC1: Explain requested core flows and verified experiment counts/results.
- [ ] AC2: Verify supplied failure cases and add successful case with per-stage evidence and top5 interpretation.
- [ ] AC3: Distinguish historical ablation results/current runtime, ranking scores/thresholds, retrieval/generation/grounding errors.
## File paths
- Read: docs, data logs/results, search_core.py, rag_core.py, product_dialogue.py, metrics/scripts.
- New: this journal and read-only trace evidence under .claude/tasks/.
## Research / flow / impact
Current git status differs from prior turn; inspect diff before describing behavior. Recompute branch full ranks but fuse serving top50, rerank serving top30 to avoid step12's top50 rerank mismatch. No runtime edits intended.
## Implementation steps
1. [ ] Read current flow/diff and relevant docs/logs.
2. [ ] Recompute bounded trace cases using cached local model/index and save evidence.
3. [ ] Compose presentation with caveats and concrete error analysis.
## Verification
| ID/AC | Status | Exact check | Actual evidence |
|---|---|---|---|
| AC1-3 | NOT RUN | Pending | Pending. |
## Result / next
Proceed read-only; no skills invoked, no commits or experiments beyond selected diagnostic queries.

## Verified result
- Status: COMPLETE. AC1[x],AC2[x],AC3[x]; steps1–3[x].
- Recomputed6queries with cached real SigLIP/index and both rerankers, no LLM calls/downloads. Exit0; evidence a3-case-trace-2026-10-29.json. Serving top50 fusion; reranking uses top30 per branch and30fused candidates.
- Basket1396: BM251219(score0),dense1(.1238346),RRF9(.0163934),rerank MiniLM17/DeBERTa6. Skates1493: BM25555(score0),dense2(.0612726),RRF6(.016129),rerank15/15. SP1801: BM251(60.61569),dense137(.0167576),RRF7(.0163934),rerank1/1. Ace748: BM2514(8.11287),dense1(.1198992),RRF2(.029907),rerank24/23; still2inpool30, proves reranker demotion rather than pool truncation. NuWave272: BM251(35.54366),dense1(.1325207),RRF1(.0327869),rerank1/1. Hotwaterbag888: BM256,dense1,RRF1,rerank1/1.
- BM25 rank among zero-score ties not meaningful; outside top50 contributes zero to serving RRF, not1/(60+full_rank). Scores not calibrated probabilities or universal decision thresholds; no absolute relevance cutoff found.
- Main logs1500=250x6; image300; memory500;0error records. Naive misses3;Agentic7;eachrerankNaive3. FaithfulnessNone240total. Historical scores date12/10; no multi-seed runs established.
- Current drift includes alternatives exclusion, keep old state if no alternatives cited, discussed_ids for comparison followups, component score metadata. Preserve all; explain current features separately from historical ablation. Sweep old single-target ground truth, not rerun qrels; k10 improvement conditional.
- No runtime files modified. Deliver comprehensive explanation in chat with evidence links; diagnostics/journal are only new files.

**Update on 29/10/2026 10:22 (Asia/Saigon)**
- User narrows scope explicitly to A3 and authorizes explanation of flows using existing evidence, accepting variance; requests no additional measurement. Previous broader trace research remains historical evidence. No further experiments, benchmarks or tests will run for this explanation.
- Revised scope: A1 retrieval/model internals only as necessary to explain A3 context/top5 and error attribution; A2 only checkpoint provenance for reranker. Cover A3 ablation design/counts, historical results, Naive/Agentic routing/memory/image flows, and provided failure/success examples.
- All AC remain satisfied by prior code/log/artifact review; final explanation in chat with existing local evidence links. No standalone report/slide artifact requested. No runtime edits.
