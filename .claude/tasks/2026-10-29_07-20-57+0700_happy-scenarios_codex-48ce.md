# Task 3
- Status: IN PROGRESS
- Estimated time: Not estimated
- Goal: Update and expand happy-path conversation scenarios and verify them thoroughly.
## Detailed description
User supplies Naive air-fryer and Agentic thermostat sequences and requests updated/additional scenarios after original-list/focus preservation changes. Explicitly authorizes new scenarios/tests. Keep prior uncommitted implementation. Distinguish mocked routing/state checks from real LLM correctness. Previous implementation journal: .claude/tasks/2026-10-29_07-16-26+0700_preserve-product-list_codex-31bd.md.
## Acceptance criteria
- [ ] AC1: Update air-fryer and thermostat scenarios with unambiguous selection and verbal product switching, follow-up focus and alternatives.
- [ ] AC2: Add happy cases for both pipelines and run them with explicit expected intent/identity/state evidence.
- [ ] AC3: Report actual results and real-service/API limitations.
## File paths
- Planned: tests/product_dialogue_scenarios.json, tests/test_product_dialogue.py; runtime only if demonstrated defect requires scoped fix.
## Research / flow / impact
Search -> numbered choices -> explicit detail -> catalog record -> retained choices + focused ID -> short follow-up -> switch by position -> alternatives search replaces choices/focus. Original examples ambiguously say this product immediately after multi-result search; updated scenarios select product 1 verbally first. Shared router applies to both pipelines. API and frontend consume response-local sources and stable product_options separately.
## Implementation steps
1. [ ] Define reusable scenario fixtures with expected intents and product identities.
2. [ ] Add parameterized offline execution and state assertions; diagnose API stall.
3. [ ] Probe local Ollama availability, run bounded verification, report results.
## Verification
| ID/AC | Status | Exact check | Actual evidence |
|---|---|---|---|
| AC1-3 | NOT RUN | Pending | Pending. |
## Result / next
Implement scenarios and verify without new dependencies or commits.

**Update on 29/10/2026 07:26 (Asia/Saigon)**
- Defined seven reusable JSON scenarios: updated air fryer, updated thermostat, Vietnamese verbal switching, compare then select, new topic, single result, ambiguity recovery.
- Added one parameterized unittest method: seven scenarios x Naive/Agentic x direct API/no selected ID versus simulated UI selected ID = 28 variants, 192 turns. Classifier/retrieval/generation are fixtures; expectations assert state and grounded detail context, not model accuracy.
- Extended existing API test with verbal 2 -> follow-up -> 1 switching and original product_options assertions in both modes.
- API stall diagnosed with faulthandler: stuck in Windows socket._fallback_socketpair accept while creating asyncio loop, before TestClient request. Sandbox caused it. Same full suite outside sandbox passed all 15 original test methods; after scenario/API updates all 16 pass (1.384s).
- Added node DOM/API harness exercising actual ask() wiring: 14 turns, both modes, checks original sidebar options, outgoing selected IDs, detail panel numbering, local citation identity, comparison, and new-search reset. PASS, exit 0. This is not a rendered browser test.
- Local Ollama unavailable in sandbox but outside sandbox /api/tags returned HTTP 200, llama3.1:latest. Started opt-in real retrieval/Ollama runner with cached HF model only, saving each completed turn to .claude/tasks/happy-live-2026-10-29.json. First actual search returns NuWave ID272 + Breville ID1336; detail target 1 passed. Live run in progress.
- Added README commands and exact limits. No runtime changes in this task so far.

**Update on 29/10/2026 07:29 (Asia/Saigon)**
- Live evidence invalidated broad happy-path confidence: Naive turn 7, How much capacity? with focus Breville ID1336, classifier returned both [272,1336], causing focus=None. First failure preserved in happy-live-2026-10-29.json. Other completed turns passed identity/state.
- Scoped runtime fix: product_dialogue.py treats How much <known attribute>? as elliptical, resolving the focused product before classifier. Previously this phrase fell through to LLM. No change to explicit positions, alternatives/new-search priority, no-focus ambiguity or source-list retention.
- Added precise regression with a classifier fixture deliberately returning both products; assert classifier is never called for focused How much capacity?. Existing routing table also now includes this phrase with focus ID1.
- Full suite after fix: .\venv\Scripts\python.exe -m unittest discover -s tests -v outside sandbox: 17 tests PASS, exit 0, 2.372s. UI harness rerun: 14 turns PASS. git diff --check PASS.
- Live process was started before fix; it still runs old imported router. Its baseline evidence is preserved; need fresh-process real regression recheck after it completes. Additional modified path: product_dialogue.py.

**Update on 29/10/2026 07:36 (Asia/Saigon)**
- Baseline live run finished: Naive 10 PASS / 1 FAIL (capacity classifier bug); Agentic search PASS then detailed answer exceeded smoke runner timeout 90 seconds. Baseline retained unchanged.
- Updated opt-in runner to app-default 600s timeout, optional --llm-timeout, and explicit --resume passing-prefix replay. Reused turns marked reused_from in new report; failed turn and all successors rerun. Saves new evidence in happy-live-recheck-2026-10-29.json without deleting first failure.
- Capacity fix real recheck PASS: focus and only source ID1336 retained, router reason=rules, answer reports Breville 0.8-cubic-foot interior, matching catalog. Subsequent Naive turns all PASS; final Naive status PASS (6 reused prefix turns + 5 fresh turns).
- Real NuWave material stainless steel/capacity21quarts match catalog. Agentic still running. Search sources IDs443(ecobee thermostat),1229(Kelon air conditioner): second is not equally relevant. This is retrieval quality limitation, separate from target memory.
- Scope: real-run report tests identity/state only; manual spot checks do not prove overall factual/citation correctness. Breville baseline material response once used literal [n], so do not claim universally correct generated citations.

**Update on 29/10/2026 07:45 (Asia/Saigon)**
- First Agentic recheck last turn FAIL: How does it monitor air quality? was classified as ecobee ID443 even with focused Kelon1229. Scoped fix: functional How does/can it/this product/that product follow-ups resolve focused product (or sole source; ambiguous multiple no focus clarifies) before LLM classifier. Expanded adversarial classifier regression.
- Manual inspection found response-local number mismatch not detected by identity-only checker: after selecting product2, generated answer said no product2 because local context is [1]. Initial prompt-only clarification failed in real recheck. Preserved happy-live-agentic-final-2026-10-29.json as evidence even though automatic ID/state status was PASS.
- Fixed actual input: resolve_product_references(query, sources) replaces positional references with resolved catalog titles for detail prompt only; original user history and response-local sources unchanged. Covers English numeric/ordinal, citations and Vietnamese accented/unaccented references. Added routing helper unit coverage and assertion that detail prompt receives title rather than original Product5.
- Detail prompt also explicitly requests actual citation numbers and forbids literal placeholder [n]. This is guidance, not a guarantee; earlier generated responses sometimes used [n].
- Full suite latest: 18 tests PASS, exit0,1.160s. API happy chain and 28 scenario variants included. Final real two-turn recheck now running in new process, reusing prior eight-turn Agentic prefix, output happy-live-agentic-resolved-2026-10-29.json.
- Latest user requested A3 core presentation explanation after demo is finished; keep this task active until demo recheck completes, then start separate explanation task.

**Update on 29/10/2026 07:46 (Asia/Saigon)**
- Status: COMPLETE.
- AC1 [x]: Seven reusable scenarios include both original examples updated with explicit product selection, switching and alternatives.
- AC2 [x]: 28 offline variants/192 turns PASS; API chain PASS; 14 actual ask() wiring turns PASS. New regression tests cover both live-discovered router defects and resolved-reference prompt input.
- AC3 [x]: Final real recheck last Agentic two turns PASS with manual content inspection: product2 now receives Kelon title and produces catalog-grounded details with [1]; next question says catalog lacks air-quality monitoring. Naive overall 11 identity/state turns PASS in recheck; Agentic 10 identity/state turns PASS using passing prefixes plus fresh reruns. These were not full uninterrupted post-fix reruns. Historical failing reports retained.
- Final suite: 18 tests PASS outside sandbox; no remaining API stall outside sandbox. Syntax/diff check and UI harness PASS. No interactive browser test or exhaustive answer-quality scoring.
- Changed in this task: tests/product_dialogue_scenarios.json, tests/check_chat_product_flow.cjs, tests/run_product_dialogue_live.py (new); tests/test_product_dialogue.py, README.md, product_dialogue.py, rag_core.py (modified). Prior frontend changes preserved. No commits.
- Final live evidence: .claude/tasks/happy-live-recheck-2026-10-29.json (Naive), .claude/tasks/happy-live-agentic-resolved-2026-10-29.json (Agentic). Original evidence and failed intermediate reports retained.
## Result / next
Demo scenario task complete; proceed to user's requested A3 core explanation. Remaining limits: relevance of second thermostat result, model-generated citation/fact quality not universally guaranteed, session state still RAM-only.
