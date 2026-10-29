"""Opt-in real retrieval/Ollama check, never run by unittest discovery.

Run from the repo root:
  python tests/run_product_dialogue_live.py --output .claude/tasks/happy-live.json
Uses cached Hugging Face models only; does not install/download dependencies.
The report checks intent/identity/state, not factual correctness of generated text.
"""
import argparse
import json
import os
import re
from pathlib import Path
import sys
import time


def main():
    # Windows console defaults may not encode Vietnamese diagnostics from Ollama.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", action="append", help="Default: the two user examples")
    parser.add_argument("--modes", nargs="+", choices=["naive", "agentic"],
                        help="Run selected scenarios in these modes; default uses preferred mode")
    parser.add_argument("--resume", type=Path, help="Reuse a previous report's passing prefix; rerun from first failure")
    parser.add_argument("--llm-timeout", type=int, default=600, help="Per LLM call, same default as app")
    parser.add_argument("--k", type=int, help="Sources per search (UI uses 5); default 2, or 1 for 'single' steps")
    args = parser.parse_args()
    previous = json.loads(args.resume.read_text(encoding="utf-8")) if args.resume else {}
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("Loading local search model/indexes (offline cache only)...", flush=True)
    import rag_core as rag

    rag.LLM_TIMEOUT_SEC = args.llm_timeout
    rag.LLM_RETRIES = 1
    scenarios = json.loads(Path(__file__).with_name("product_dialogue_scenarios.json")
                           .read_text(encoding="utf-8"))["scenarios"]
    wanted = args.scenario or ["naive_airfryer_auto_focus", "agentic_thermostat_auto_focus"]
    unknown = set(wanted) - {s["id"] for s in scenarios}
    if unknown:
        parser.error(f"Unknown scenarios: {sorted(unknown)}")
    report = {"kind": "real retrieval and Ollama; no UI clicks; no mocks",
              "focus_policy": "single_source_or_single_cited_product; uncited_alternatives_keep_focus",
              "checks": "intent, target IDs, retained choices, focus; NOT answer fact scoring",
              "model": rag.LLM_MODEL, "scenarios": []}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    runs = [(scenario, mode) for scenario in scenarios if scenario["id"] in wanted
            for mode in (args.modes or [scenario.get("preferred_mode", "naive")])]
    for scenario, mode in runs:
        if scenario["id"] not in wanted:
            continue
        fn = rag.answer_agentic if mode == "agentic" else rag.answer
        session = "live-check-" + scenario["id"]
        rag.clear_history(session)
        record = {"id": scenario["id"], "mode": mode, "turns": []}
        report["scenarios"].append(record)
        options = []
        old_turns = next((s["turns"] for s in previous.get("scenarios", [])
                          if s["id"] == scenario["id"] and s["mode"] == mode), [])
        reuse_prefix = previous.get("focus_policy") == report["focus_policy"]
        for n, step in enumerate(scenario["steps"], 1):
            old = old_turns[n - 1] if n <= len(old_turns) else {}
            if (reuse_prefix and old.get("status") == "PASS"
                    and old.get("query") == step["query"]
                    and old.get("image") == step.get("image")):
                result = old["response"]
                kept = "product_options" in result
                rag.append_history(session, step["query"], result["answer"],
                                   result["sources"], intent=result["intent"], keep_products=kept)
                if result["intent"] == "search" and not kept:
                    options = result["sources"]
                record["turns"].append({**old, "reused_from": str(args.resume)})
                print(f'{scenario["id"]} {n}: reusing previous PASS', flush=True)
                save()
                continue
            reuse_prefix = False
            print(f'{scenario["id"]} {n}/{len(scenario["steps"])}: {step["query"]}', flush=True)
            turn = {"query": step["query"], "image": step.get("image"),
                    "expected_intent": step["intent"]}
            record["turns"].append(turn)
            started = time.monotonic()
            try:
                expected = [options[p - 1]["id"] for p in step.get("targets", [])]
                image = None
                if step.get("image"):
                    from PIL import Image
                    with Image.open(root / step["image"]) as opened:
                        image = opened.convert("RGB")
                previous_focus = rag._PRODUCT_STATE.get(session, {}).get("focused_id")
                result = fn(step["query"], k=args.k or (1 if step.get("result_set") == "single" else 2),
                            session_id=session, image=image)
                state = rag._PRODUCT_STATE.get(session, {})
                errors = []
                if result["intent"] != step["intent"]:
                    errors.append("intent mismatch")
                if step["intent"] == "search":
                    sources = result["sources"]
                    if not sources:
                        errors.append("empty search")
                    excluded = set(result.get("excluded_ids", []))
                    if excluded & {p["id"] for p in sources}:
                        errors.append("discussed product offered again")
                    cited = {sources[int(n) - 1]["id"]
                             for n in re.findall(r"\[(\d+)\]", result["answer"])
                             if 1 <= int(n) <= len(sources)}
                    if "product_options" in result:
                        # Alternatives request with no cited alternative: old choices and focus stay.
                        if cited or not excluded:
                            errors.append("kept old choices although an alternative was cited")
                        expected_focus = previous_focus
                    else:
                        options = sources
                        expected_focus = (options[0]["id"] if len(options) == 1 else
                                          next(iter(cited)) if len(cited) == 1 else None)
                    if result.get("focused_product_id") != expected_focus:
                        errors.append("response focus mismatch")
                elif step["intent"] == "detail":
                    if [p["id"] for p in result["sources"]] != expected:
                        errors.append("target mismatch")
                    expected_focus = expected[0] if len(expected) == 1 else None
                else:
                    expected_focus = state.get("focused_id")
                if state.get("sources") != options:
                    errors.append("choice list changed")
                if state.get("focused_id") != expected_focus:
                    errors.append("focus mismatch")
                turn.update({"status": "FAIL" if errors else "PASS", "errors": errors,
                             "expected_ids": expected, "response": result,
                             "option_ids": [p["id"] for p in options],
                             "focus": state.get("focused_id")})
            except Exception as exc:
                turn.update({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
            turn["seconds"] = round(time.monotonic() - started, 2)
            print(f'  {turn["status"]} ({turn["seconds"]}s) {turn.get("errors", turn.get("error", ""))}', flush=True)
            save()
            if turn["status"] == "ERROR":
                break
        record["status"] = ("PASS" if len(record["turns"]) == len(scenario["steps"])
                            and all(t["status"] == "PASS" for t in record["turns"]) else "FAIL")
        rag.clear_history(session)
        save()
    return 0 if all(s["status"] == "PASS" for s in report["scenarios"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
