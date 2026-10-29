# Task 2
- Status: IN PROGRESS
- Estimated time: Not estimated
- Goal: Preserve original search choices while switching product details verbally and retaining follow-up focus.
## Detailed description
User requests keeping the original product list so a customer can ask about item 1, switch to item 2 without clicking, and continue asking about the focused product. Latest search defines original choices; a new search/reset still replaces/clears them. Prior investigation: .claude/tasks/2026-10-29_07-12-50+0700_product-memory-question_codex-92af.md.
## Acceptance criteria
- [ ] AC1: Detail turns retain ordered choices from latest search, supporting verbal ordinal switching.
- [ ] AC2: Follow-up attribute questions use the most recently identified product.
- [ ] AC3: UI retains original numbered choices while detail citations remain correct for their own response.
## File paths
- Planned: rag_core.py, frontend/chat.html, tests/test_product_dialogue.py (existing assertions only), README.md.
## Research / flow / impact
- Current: ask -> router(latest sources, focus) -> catalog -> append_history overwrites sources with detail subset -> UI replaces list.
- Changed: search sets ordered choices; detail changes focus only; response keeps local detail sources and separately exposes product_options for list rendering. Both answer and answer_agentic share this path. Existing API consumers retain sources contract; added options are optional/additive.
- Edges: clarify keeps choices/focus, compare clears single focus, new/empty search replaces choices and clears focus, reset/session isolation unchanged, clicked older card retains explicit ID without renumbering search options, no-session detail falls back to identified sources.
- No live browser/LLM proof yet. Existing API test hung in prior investigation. No skills invoked.
## Implementation steps
1. [ ] Separate stable choice state from detail focus and response sources.
2. [ ] Update UI choice rendering and concise usage docs.
3. [ ] Update relevant existing tests and run bounded offline verification.
## Verification
| ID/AC | Status | Exact check | Actual evidence |
|---|---|---|---|
| AC1-3 | NOT RUN | Existing offline tests and code review | Pending. |
## Result / next
Implement requested behavior only; no commits or new test cases.

**Update on 29/10/2026 07:17 (Asia/Saigon)**
- Status: COMPLETE.
- AC1 [x]: Search alone replaces ordered state.sources. Detail retains original choices and changes focused_id. Existing regression now checks two choices, verbal switching 1 -> 2 -> 1, with previous UI-selected ID passed to demonstrate explicit ordinal wins.
- AC2 [x]: Existing regression checks four short follow-ups after each switch, across Naive and Agentic; history remains bounded at three without losing choices/focus.
- AC3 [x]: UI code uses additive product_options for sidebar, keeps data.sources for citation cards, derives detail panel numbering from original list. Verified by diff review and JavaScript syntax check; interactive browser behavior NOT RUN.
- Steps 1–3 [x]. Changed: rag_core.py, frontend/chat.html, README.md, tests/test_product_dialogue.py. No new test methods/files, commits or dependency changes.
- CodeGraph cache dated 13/10 precedes feature; used current rg/code instead. All direct state consumers are rag_core and existing tests. API passes dictionaries unchanged; sources contract stays response-local.

| ID/AC | Status | Exact check | Actual evidence |
|---|---|---|---|
| AC1, AC2 / regressions | PASS | .\venv\Scripts\python.exe - via runpy + unittest selecting existing RoutingTests and DialogueTests except test_api_selection_baseline_and_reset | 14 existing tests pass, exit 0; final run 0.282s. |
| AC3 syntax | PASS | node via stdin: read frontend/chat.html script blocks and compile with vm.Script | JavaScript syntax OK (1 script blocks), exit 0. |
| Patch integrity | PASS | git diff --check | Exit 0; only CRLF conversion warnings. |
| Full existing suite | BLOCKED | venv Python subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], timeout=20) | Timed out in existing API test, subprocess terminated; wrapper shell exit 1 (script exit 124). Same stall observed before these changes. |
| Live browser/Ollama | NOT RUN | No browser or real LLM execution | UI validated statically; LLM output generation mocked. |

## Result / next
Requested behavior implemented and relevant offline regressions verified. New searches (including image search) replace choices and clear focus; clarify preserves state and comparison clears singular focus. Session/reset semantics unchanged. User can review uncommitted changes; full API/live verification remains limited as above.
