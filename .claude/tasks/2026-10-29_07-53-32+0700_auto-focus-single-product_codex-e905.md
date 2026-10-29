# Task 5
- Status: IN PROGRESS
- Estimated time: Not estimated
- Goal: Automatically focus the sole proposed product after search.
## Detailed description
User requests auto focus if only one product is recommended. Interpret recommendation as the one distinct valid product citation rendered as a product card, even if search has multiple sources; also focus a sole search source without requiring citation. Several valid recommended products or no identifiable proposal among multiple sources leaves focus unset. Preserve ordered search choices and existing verbal switching.
## Acceptance criteria
- [ ] AC1: Single search source automatically focused in both pipelines.
- [ ] AC2: One distinct cited recommendation among several sources automatically focused; multiple/invalid/absent citations do not guess.
- [ ] AC3: UI reflects focus and next question sends correct ID; search list remains stable.
## File paths
- Planned: rag_core.py, frontend/chat.html, tests/test_product_dialogue.py, tests/product_dialogue_scenarios.json, tests/check_chat_product_flow.cjs, tests/run_product_dialogue_live.py, README.md.
## Research / flow / impact
Current append_history search always clears focus; UI also always clears selection. Change shared focus inference -> state and additive search response focused_product_id -> UI selection/detail display. Both Naive/Agentic and image search use shared state writer. Preserve response-local citation numbering and original source order. Valid citations are a deterministic proxy for proposed cards; cannot infer uncited prose recommendations or positive/negative sentiment from citations.
## Implementation steps
1. [ ] Add shared single-recommendation focus inference and API/UI synchronization.
2. [ ] Update affected existing scenario/regression expectations and optional live checker.
3. [ ] Run full offline/API suite, UI harness and patch checks.
## Verification
| ID/AC | Status | Exact check | Actual evidence |
|---|---|---|---|
| AC1-3 | NOT RUN | Pending | Pending. |
## Result / next
No skill invoked, no unrelated changes or commits planned. Preserve previous uncommitted work.

**Update on 29/10/2026 07:56 (Asia/Saigon)**
- Status: COMPLETE.
- AC1 [x]: Shared search focus inference selects sole source, including ID0/no citation. Naive/Agentic state and additive focused_product_id response synchronized; empty results remain None. Existing image path also uses shared search history writer.
- AC2 [x]: One distinct valid cited ID across multiple sources auto-focuses; repeated citations deduplicated; multiple/no valid citations leave focus None. Original choice order retained. Existing search-focus regression expanded for both pipelines; sole recommendation points to second source to detect accidental first-product default.
- AC3 [x]: UI consumes focused_product_id, falls back to sole source for older responses, opens selected detail panel using original list position, sends selected ID on follow-up. API regression proves no-selection follow-up uses auto focus; UI harness covers ID20 at position2, source-only auto-focus, ambiguous and empty new searches.
- Updated existing single-result JSON scenario with one recommendation among multiple sources -> short capacity question -> switch to original item1. Total seven scenarios x4 variants=28 variants/204 turns. Updated README and opt-in live checker; old-policy reports do not reuse passing prefixes under new focus semantics.
- Steps1–3 [x]. Changed this task: rag_core.py, frontend/chat.html, tests/test_product_dialogue.py, tests/product_dialogue_scenarios.json, tests/check_chat_product_flow.cjs, tests/run_product_dialogue_live.py, README.md. Preserved prior uncommitted product_dialogue.py changes. No commits or new dependencies.

| ID/AC | Status | Exact check | Actual evidence |
|---|---|---|---|
| AC1-3 / backend/API | PASS | .\venv\Scripts\python.exe -m unittest discover -s tests -v (outside sandbox for asyncio socketpair) |18 methods PASS, exit0,1.316s; includes28scenario variants, both modes, auto-focus API/follow-up. |
| AC3 / UI wiring | PASS | node tests/check_chat_product_flow.cjs |24 turns PASS, exit0, both modes. |
| First UI verification | FAIL (corrected fixture) | node tests/check_chat_product_flow.cjs |Assertion at openedDetail.id: actual20 expected10. Fixture assumed panel always opens sources[0], invalid for sole cited second product. Corrected assertion to explicit focus; no runtime fix required. Initial batched shell final exit0 due subsequent git command; actual Node failure retained here. |
| Syntax | PASS | ast.parse for rag_core.py, tests/test_product_dialogue.py, tests/run_product_dialogue_live.py |All parsed successfully, exit0. |
| Patch integrity | PASS | git diff --check |Exit0, CRLF warnings only. |
| Live LLM/browser this task | NOT RUN |No live dialogue or rendered browser calls |Deterministic focus logic validated with mocked LLM/API and actual ask() wiring; prior live reports are historical. |

## Result / next
User can search -> receive one recommendation -> ask Capacity?/Chất liệu gì? directly. Multiple recommendations still require identity. Proposals are determined by valid numeric citation cards (or one source), not sentiment analysis of uncited prose.
