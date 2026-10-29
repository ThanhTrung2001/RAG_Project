# Task 1
- Status: IN PROGRESS
- Estimated time: Not estimated
- Goal: Explain current product-detail and conversational-memory behavior.
## Detailed description
User requests reading documentation and code, especially latest changes believed uncommitted, and explaining ambiguity with two returned products, no click, and switching detail questions from product 1 to product 2. Read-only investigation; no implementation or new tests requested.
## Acceptance criteria
- [ ] AC1: Establish current working-tree changes and memory/detail flow from code.
- [ ] AC2: Explain ambiguous no-click detail request and switching between returned products with evidence.
## File paths
- New: this journal.
- Investigate: repository documents, chat backend and UI (to locate).
## Research / flow / impact
- Initial git status --short returned empty (clean tracked working tree).
## Implementation steps
1. [ ] Read documentation and locate memory/detail code.
2. [ ] Trace UI selection, conversation resolution, ambiguity and switching behavior.
3. [ ] Report evidence and limitations.
## Verification
| ID/AC | Status | Exact check | Actual evidence |
|---|---|---|---|
| AC1 | IN PROGRESS | git status --short | Exit 0, no changes listed before journal creation. |
| AC2 | NOT RUN | Static flow tracing | Pending. |
## Result / next
Read documents and code. No app behavior changes authorized or planned.

**Update on 29/10/2026 07:19 (Asia/Saigon)**
- Status: COMPLETE (investigation); no source changes.
- AC1 [x]: Initial tracked working tree clean. HEAD 3db7616 already includes memory/detail feature. Traced README, plan section 4, product_dialogue.py, rag_core.py, app.py, frontend/chat.html, search_core.get_product and existing tests. Other documents inspected for relevant memory/flow references; no claim of exhaustive line-by-line reading.
- AC2 [x]: Executed real answer() with existing DialogueTests mocked catalog/LLM and two seeded sources. Ambiguous detail -> clarify with both IDs; product 1 -> detail with ID 0; subsequent product 2 -> clarify/unknown_position with only ID 0. Selecting old second card ID 1 -> detail ID 1.
- Flow: UI selection/position -> router -> catalog by ID -> detail LLM -> append_history replaces latest sources and sets focus -> UI redraws latest list and auto-selects single detail source. This replaces original candidate list, so ordinal switching back to original item 2 fails. Old cited cards retain click closures with actual IDs.
- Evidence: rag_core.py:66-70,88-100,110-126; product_dialogue.py:38-51; frontend/chat.html:151,203,316,326,344.
- Remaining limits: Bare ambiguous wording uses real LLM classifier; exact intent cannot be guaranteed without Ollama. Numbering is source-list numbering, not order of mentions in generated text. UI requests k=5 even if answer mentions only two.

| ID/AC | Status | Exact check | Actual evidence |
|---|---|---|---|
| AC1 | PASS | git status --short; git log -5 --oneline; git show --stat --oneline HEAD | Initially clean; HEAD 3db7616 feature commit. Only this journal creates untracked .claude/. |
| AC2 | PASS | Inline Python using runpy on tests/test_product_dialogue.py and existing fixture, two sources | clarify [0,1] -> detail [0] -> clarify unknown_position [0]; explicit selected_product_id=1 -> detail [1]. |
| Regression | PASS | venv Python inline unittest suite selecting existing RoutingTests/DialogueTests except API test | 14 existing tests passed, exit 0. No new test files/cases. |
| Full suite default Python | BLOCKED | python -m unittest discover -s tests -v | Exit 1: missing dotenv; five routing tests passed, DialogueTests setup failed. |
| Full suite venv Python | BLOCKED | .\venv\Scripts\python.exe -m unittest discover -s tests -v | Stalled in existing API test after deprecation warnings; no completed result. |
| Diagnostic | BLOCKED | Get-CimInstance Win32_Process filtered to unittest command | Exit 1, CIM access unavailable. |
| Real LLM/browser | NOT RUN | No Ollama or browser calls | Static UI trace plus mocked runtime only. |

## Result / next
Answer user in Vietnamese: clicking optional for initial ordinal detail; ambiguous no-selection asks clarification; original-list ordinal switching after detail currently fails due to source replacement; clicking original second product works. No automatic fix or commit.
