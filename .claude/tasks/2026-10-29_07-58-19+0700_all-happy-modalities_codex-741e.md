# Task 6
- Status: IN PROGRESS
- Estimated time: Not estimated
- Goal: Expand and execute happy conversation coverage across text, image and image+text in Naive and Agentic.
## Detailed description
User requests scenario examples like original air-fryer/thermostat demos, actually testing and updating them, explicitly including image+text and all meaningful happy paths for both pipelines. Preserve latest auto-focus behavior. Explain image+text flow and follow-up memory. Cover bounded meaningful combinations rather than claim exhaustive coverage of arbitrary natural language.
## Acceptance criteria
- [ ] AC1: Updated reusable scenarios cover original no-click auto-focus examples plus text/image/image+text, switching, follow-ups, alternatives and new input replacing old choices in both modes.
- [ ] AC2: Execute offline/core/API/UI checks and real retrieval/Ollama scenarios where available; record actual outcomes and limitations.
- [ ] AC3: Explain how image+text and subsequent memory work; provide usable scenario sequences and artifact pointers.
## File paths
- Planned: tests/product_dialogue_scenarios.json, tests/test_product_dialogue.py, tests/check_chat_product_flow.cjs, tests/run_product_dialogue_live.py, README.md. Runtime fixes only for demonstrated defects.
## Research / flow / impact
Image attachments use /api/v1/ask_image multipart, bypass detail router and always search. Image+text -> image search plus hybrid text search -> RRF -> source text to LLM; image reranker disabled. Shared append_history selects unique recommendation, preserving IDs for next text-only detail. New image replaces choices; prior raw image is not retained across turns. Agentic repeats image retrieval with current rewritten query. Need actual fusion branch/API multipart/UI upload checks in addition to mocked high-level scenarios.
## Implementation steps
1. [ ] Add modality and auto-focus scenario fixtures and enable image inputs in runners.
2. [ ] Add focused fusion/API/UI verification without external downloads.
3. [ ] Run full tests and real representative scenarios; update results/documentation.
## Verification
| ID/AC | Status | Exact check | Actual evidence |
|---|---|---|---|
| AC1-3 | NOT RUN | Pending | Pending. |
## Result / next
No skills invoked, commits or external integration services planned. Prior uncommitted work preserved.

**Update on 29/10/2026 08:03 (Asia/Saigon)**
- Added four compact scenarios: original air-fryer no-click auto-focus, original thermostat no-click auto-focus, image+text -> follow-up -> verbal switch -> new image+text/new topic -> follow-up -> alternatives, and image-only -> text follow-up -> alternatives -> numbered detail. Existing seven cases retained. Total11groups x2pipelines xUI/noUI selection=44variants,276turns.
- Scenario runner now supplies image inputs and checks image history marker and retrieve image argument. Offline fixture search responses deliberately distinguish single versus multiple proposed citations.
- Added fusion orchestration check (both branches, pool50, overlap/fused order, no rerank), Agentic forced NO->rewrite->YES check proving same image retained in both retrievals/reranker disabled, and multipart PNG upload API test for image-only/image+text in both modes with following text detail, new image replacement and item2 switching.
- Full unittest suite:21methods PASS,exit0,1.647s outside sandbox. UI harness expanded to actual image FormData branch and attachment clearing plus follow-up selection;32turns PASS,exit0. No runtime changes needed so far.
- Live runner now loads actual images and --modes can run both pipelines; resumed prefixes match image path too. Started8real conversations across4newscenarios and2modes,36planned turns, real SigLIP/FAISS/BM25/Ollama.
- First live attempt failed while printing Vietnamese diagnostic on Windows(cp1252 UnicodeEncodeError); no usable report saved before print. Updated stdout/stderr UTF8/backslashreplace and restarted. Ollama /api/tags outside sandbox HTTP200, llama3.1:latest.
- New live run .claude/tasks/happy-multimodal-live-2026-10-29.json ongoing; first real Naive airfryer search, material and capacity turns PASS with auto-focus. Later outputs pending. Syntax/diff checks PASS.

**Update on 29/10/2026 08:13 (Asia/Saigon)**
- Status: COMPLETE. AC1[x],AC2[x],AC3[x]; steps1–3[x].
- Real uninterrupted run completed exit0: eight conversations,36turns,all PASS. Four compact scenarios each run Naive and Agentic: original air-fryer auto-focus4turns, original thermostat auto-focus4turns, image+text switch/new-image replacement6turns, image-only followups4turns. Actual local catalog photos000272.jpg and000443.jpg loaded, no mocks/reused prefixes in this run. Evidence: .claude/tasks/happy-multimodal-live-2026-10-29.json.
- Full offline suite21methods PASS;44parameterized variants/276turns included. API image-only/image+text PNG multipart and followup/new-upload flows PASS in both modes. Agentic forced rewrite retaining original image PASS; fusion orchestration PASS.
- Actual ask() JavaScript with DOM/API fixtures32turns PASS, including image FormData and attachment clearing. Syntax/diff checks PASS. No real browser visual verification; live report scores intent/identity/list/focus, not exhaustive generated-fact/citation correctness.
- Updated files this task: tests/product_dialogue_scenarios.json, tests/test_product_dialogue.py, tests/check_chat_product_flow.cjs, tests/run_product_dialogue_live.py, README.md. Runtime app/rag/router unchanged in this task. Prior edits retained; no commits/dependencies.
- Image+text behavior confirmed: new image always search; image and text result lists fused with RRF; rerank disabled; next text question uses focused catalog product and stable options. Raw image not retained for later text-only alternatives; reattach for another image-based search.
## Result / next
Deliver usable scenarios and actual verification counts, explain image+text and limitations. Coverage is a bounded happy-path matrix, not proof for all possible natural-language messages.
