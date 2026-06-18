"""
pipeline.py — the multi-agent handoff with a bounded cross-model review loop.

run_snippet(snippet) orchestrates:
  detect (RAG-grounded) -> author a failing test -> [refactor -> run tests ->
  behavior check -> review by a DIFFERENT model]* (<= K iterations) -> trust gate.

Returns a structured trace (per-iteration history + the deterministic verdict)
for the UI and the evaluation harness. Every stage is wrapped so one failure
becomes an error row, never a crash. Runs fully offline via MockClient.
"""
from __future__ import annotations

import difflib
import json
import time

import config
from agents import (DETECTOR_SYSTEM_PROMPT, ESCALATE,
                    REFACTORER_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT,
                    TEST_AUTHOR_SYSTEM_PROMPT, build_detector_input,
                    build_refactor_input, build_review_input,
                    build_test_author_input, trust_decision)
from executor import behavior_preserved, evaluate_cases
from llm import get_client
from retriever import Retriever


def count_diff_lines(a: str, b: str) -> int:
    d = difflib.unified_diff((a or "").splitlines(), (b or "").splitlines(), lineterm="", n=0)
    return sum(1 for ln in d if ln[:1] in ("+", "-") and not ln.startswith(("+++", "---")))


def _build_feedback(fix_run: dict, review: dict) -> "str | None":
    parts = []
    if not fix_run.get("all_passed", False):
        fails = [c for c in fix_run.get("cases", []) if not c.get("passed")]
        if fails:
            parts.append("Failing tests: " + "; ".join(
                f"{c['args']} -> {c['result'].get('exc') or c['result'].get('value')}" for c in fails[:3]))
        elif fix_run.get("load_error"):
            parts.append(f"Code did not run: {fix_run['load_error']}")
    if not review.get("approved", False) and review.get("comments"):
        parts.append("Reviewer: " + review["comments"])
    return "\n".join(parts) if parts else None


def _load_practices(path: str = "practices.json") -> list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run_snippet(snippet: dict, *, retriever: "Retriever | None" = None,
                practices: "list | None" = None, overrides: "dict | None" = None,
                max_iters: "int | None" = None) -> dict:
    """Run one snippet end-to-end and return its trace.

    `snippet` needs at least {code, func_name}; optional {id, behavior_inputs}.
    Pass a shared `retriever` to avoid re-embedding the KB per snippet.
    """
    t0 = time.perf_counter()
    code = snippet.get("code", "")
    func_name = snippet.get("func_name", "")
    behavior_inputs = snippet.get("behavior_inputs")
    max_iters = max(1, max_iters if max_iters is not None else config.REVIEW_LOOP_MAX_ITERS)

    trace = {"id": snippet.get("id"), "code": code, "func_name": func_name,
             "detection": None, "retrieved": [], "test": None, "repro": False,
             "iterations": [], "converged": False, "final_refactor": None,
             "trust": None, "error": None, "latency_ms": 0.0}

    try:
        if retriever is None:
            retriever = Retriever(practices=practices or _load_practices(), overrides=overrides)
        retrieved = retriever.retrieve(code)
        trace["retrieved"] = retrieved

        detection = get_client("detector", overrides).complete_json(
            DETECTOR_SYSTEM_PROMPT, build_detector_input(code, retrieved))
        trace["detection"] = detection
        confidence = float(detection.get("confidence", 0.0) or 0.0)
        severity = detection.get("severity", "low")
        ambiguous = bool(detection.get("ambiguous", False))

        if not detection.get("has_flaw", False):
            dec, _ = trust_decision(
                repro=False, fix_passes=False, behavior_preserved=True,
                reviewer_approved=False, converged=True, flaw_confidence=confidence,
                diff_lines=0, severity=severity, ambiguous=ambiguous)
            trace["converged"] = True
            trace["trust"] = {"decision": dec, "reason": "no flaw detected — no change proposed"}
        else:
            test = get_client("test_author", overrides).complete_json(
                TEST_AUTHOR_SYSTEM_PROMPT, build_test_author_input(code, func_name, detection))
            trace["test"] = test
            cases = test.get("cases", []) or []
            orig_run = evaluate_cases(code, func_name, cases) if cases else {"any_failed": False, "cases": []}
            repro = bool(orig_run.get("any_failed", False))
            trace["repro"] = repro
            if behavior_inputs is None:
                known_good = [c["args"] for c, per in zip(cases, orig_run.get("cases", []))
                              if per.get("result", {}).get("ok")]
            else:
                known_good = behavior_inputs

            refactorer = get_client("refactorer", overrides)
            reviewer = get_client("reviewer", overrides)
            feedback = None
            fix_passes = preserved = approved = False
            final_code = None
            for _ in range(max_iters):
                refac = refactorer.complete_json(
                    REFACTORER_SYSTEM_PROMPT, build_refactor_input(code, detection, feedback))
                final_code = refac.get("code", "") or ""
                fix_run = evaluate_cases(final_code, func_name, cases) if cases else {"all_passed": False, "cases": []}
                fix_passes = bool(fix_run.get("all_passed", False))
                preserved, _ = behavior_preserved(code, final_code, func_name, known_good)
                review = reviewer.complete_json(
                    REVIEWER_SYSTEM_PROMPT, build_review_input(code, final_code, detection, retrieved))
                approved = bool(review.get("approved", False))
                trace["iterations"].append({
                    "refactor": final_code, "explanation": refac.get("explanation", ""),
                    "diff_lines": count_diff_lines(code, final_code),
                    "fix_passes": fix_passes, "behavior_preserved": preserved,
                    "review": {"approved": approved, "comments": review.get("comments", "")},
                })
                if fix_passes and approved:
                    break
                feedback = _build_feedback(fix_run, review)

            converged = bool(fix_passes and approved)
            trace["converged"] = converged
            trace["final_refactor"] = final_code
            dec, reason = trust_decision(
                repro=repro, fix_passes=fix_passes, behavior_preserved=preserved,
                reviewer_approved=approved, converged=converged, flaw_confidence=confidence,
                diff_lines=count_diff_lines(code, final_code) if final_code else 0,
                severity=severity, ambiguous=ambiguous)
            trace["trust"] = {"decision": dec, "reason": reason}
    except Exception as exc:  # one snippet must never crash a whole eval run
        trace["error"] = f"{type(exc).__name__}: {exc}"
        if trace["trust"] is None:
            trace["trust"] = {"decision": ESCALATE, "reason": "pipeline error — routed to human"}

    trace["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return trace
