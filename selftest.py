#!/usr/bin/env python3
"""
selftest.py — offline, dependency-free verification of the deterministic spine.

Runs with plain `python3 selftest.py` — no third-party packages, no API key.
Covers the trust gate (incl. its safety invariant), the safe executor
(reproduce → fix → behavior-preservation → timeout), and the metric math.
Exits non-zero if any check fails. Data/consistency checks are added later.
"""
from __future__ import annotations

import difflib
import itertools
import json
import os
import sys

import agents
import config
import evals
from agents import (AUTO_APPLY, DECISIONS, ESCALATE, SEVERITIES, SUGGEST,
                    trust_decision)
from executor import behavior_preserved, evaluate_cases, run_inputs
from llm import (LLMError, MockClient, _is_recoverable, _pick_model, get_client,
                 parse_json)
from pipeline import run_snippet

_PASS = 0
_FAIL = 0
_FAILURES: list = []


def check(name: str, cond: bool) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
    else:
        _FAIL += 1
        _FAILURES.append(name)


# --------------------------------------------------------------------------- #
# 1. Trust gate — explicit branches
# --------------------------------------------------------------------------- #
def base(**kw):
    args = dict(repro=True, fix_passes=True, behavior_preserved=True,
                reviewer_approved=True, converged=True, flaw_confidence=0.9,
                diff_lines=5, severity="low", ambiguous=False)
    args.update(kw)
    return trust_decision(**args)

check("trust: clean fix -> auto_apply", base()[0] == AUTO_APPLY)
check("trust: no repro -> escalate", base(repro=False)[0] == ESCALATE)
check("trust: behavior broken -> escalate", base(behavior_preserved=False)[0] == ESCALATE)
check("trust: fix fails test -> escalate", base(fix_passes=False)[0] == ESCALATE)
check("trust: not converged -> suggest", base(converged=False)[0] == SUGGEST)
check("trust: reviewer rejects -> suggest", base(reviewer_approved=False)[0] == SUGGEST)
check("trust: ambiguous -> suggest", base(ambiguous=True)[0] == SUGGEST)
check("trust: low confidence -> suggest", base(flaw_confidence=0.3)[0] == SUGGEST)
check("trust: big diff -> suggest", base(diff_lines=25)[0] == SUGGEST)
check("trust: critical severity -> suggest", base(severity="critical")[0] == SUGGEST)

# --------------------------------------------------------------------------- #
# 2. Trust gate — exhaustive safety invariant across an input grid
# --------------------------------------------------------------------------- #
_valid_ok = True
_invariant_ok = True
for repro, fix_passes, beh, rev, conv, amb in itertools.product([True, False], repeat=6):
    for conf in (0.3, 0.6, 0.9):
        for diff in (5, 25):
            for sev in ("low", "high", "critical"):
                dec, _ = trust_decision(
                    repro=repro, fix_passes=fix_passes, behavior_preserved=beh,
                    reviewer_approved=rev, converged=conv, flaw_confidence=conf,
                    diff_lines=diff, severity=sev, ambiguous=amb)
                if dec not in DECISIONS:
                    _valid_ok = False
                if dec == AUTO_APPLY and not (repro and fix_passes and beh):
                    _invariant_ok = False
check("trust: every decision is valid across grid", _valid_ok)
check("trust: NEVER auto_apply an unsafe fix (safety invariant)", _invariant_ok)

# --------------------------------------------------------------------------- #
# 3. Executor — reproduce, fix, behavior-preservation, timeout
# --------------------------------------------------------------------------- #
BUGGY = "def avg(nums):\n    return sum(nums) / len(nums)\n"
FIXED = "def avg(nums):\n    return sum(nums) / len(nums) if nums else 0\n"
CHANGED = "def avg(nums):\n    return 0\n"
CASES = [
    {"args": [[2, 4]], "expect": {"kind": "returns", "value": 3.0}},
    {"args": [[]], "expect": {"kind": "not_raises"}},
]

buggy_run = evaluate_cases(BUGGY, "avg", CASES)
fixed_run = evaluate_cases(FIXED, "avg", CASES)
check("exec: test reproduces flaw on original (any_failed)", buggy_run["any_failed"] is True)
check("exec: original not all-passing", buggy_run["all_passed"] is False)
check("exec: refactor passes all cases", fixed_run["all_passed"] is True)

pres_ok, _ = behavior_preserved(BUGGY, FIXED, "avg", [[[2, 4]], [[1, 2, 3]]])
pres_bad, diffs = behavior_preserved(BUGGY, CHANGED, "avg", [[[2, 4]], [[1, 2, 3]]])
check("exec: safe refactor preserves behavior", pres_ok is True)
check("exec: behavior-changing refactor flagged", pres_bad is False and len(diffs) > 0)

LOOP = "def loop(x):\n    while True:\n        pass\n"
to = run_inputs(LOOP, "loop", [[1]], timeout=0.5)
check("exec: infinite loop hits timeout (no host hang)", to["timeout"] is True and to["results"][0]["ok"] is False)

SYNTAX = "def broken(:\n    return 1\n"
syn = run_inputs(SYNTAX, "broken", [[1]])
check("exec: malformed code -> load_error, not a crash", syn["load_error"] is not None and syn["results"][0]["ok"] is False)

# --------------------------------------------------------------------------- #
# 4. Metric math — known answers
# --------------------------------------------------------------------------- #
check("eval: accuracy", abs(evals.accuracy([(1, 1), (2, 3), (4, 4)]) - 2 / 3) < 1e-9)
check("eval: rate", evals.rate([1, 2, 3, 4], lambda x: x % 2 == 0) == 0.5)
check("eval: safe_div by zero", evals.safe_div(1, 0) == 0.0)
check("eval: mean", evals.mean([1, 2, 3]) == 2.0)
check("eval: median", evals.median([1, 2, 3, 4]) == 2.5)
check("eval: percentile p50 == median", evals.percentile([1, 2, 3, 4], 50) == 2.5)
check("eval: percentile single value", evals.percentile([10], 95) == 10.0)

prf_items = [{"p": 1, "g": 1}, {"p": 1, "g": 0}, {"p": 0, "g": 1}, {"p": 0, "g": 0}]
prf = evals.precision_recall_f1(prf_items, lambda it: it["p"] == 1, lambda it: it["g"] == 1)
check("eval: precision", prf["precision"] == 0.5)
check("eval: recall", prf["recall"] == 0.5)
check("eval: f1", prf["f1"] == 0.5)
check("eval: confusion counts", evals.confusion_matrix([("a", "a"), ("a", "b"), ("b", "b")])["a"]["b"] == 1)

# --------------------------------------------------------------------------- #
# 5. Dataset + KB — schema, validity, and label/policy consistency
# --------------------------------------------------------------------------- #
with open("snippets.json", encoding="utf-8") as _fh:
    SNIPPETS = json.load(_fh)
with open("practices.json", encoding="utf-8") as _fh:
    PRACTICES = json.load(_fh)

_REQ = {"id", "func_name", "flaw_type", "severity", "edge_case",
        "should_auto_apply", "code", "reference_fix", "behavior_inputs",
        "test_cases", "notes"}
_ids = [s["id"] for s in SNIPPETS]
check("data: >= 10 snippets", len(SNIPPETS) >= 10)
check("data: snippet ids unique", len(set(_ids)) == len(_ids))
check("data: snippet schema complete", all(_REQ.issubset(s) for s in SNIPPETS))
check("data: severities valid", all(s["severity"] in SEVERITIES for s in SNIPPETS))
check("data: practices schema + size",
      len(PRACTICES) >= 5 and all({"id", "title", "content"}.issubset(p) for p in PRACTICES))


def _diff_lines(a: str, b: str) -> int:
    d = difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
    return sum(1 for ln in d if ln[:1] in ("+", "-") and not ln.startswith(("+++", "---")))


# For each snippet, derive the trust signals by ACTUALLY running the reference
# fix through the executor, then confirm (a) the dataset is valid and (b) the
# should_auto_apply label matches what the trust policy decides. Fully offline.
_NO_REPRO = {"no_flaw", "perf_only"}
_validity_ok = True
_consistency_ok = True
for s in SNIPPETS:
    repro = evaluate_cases(s["code"], s["func_name"], s["test_cases"])["any_failed"]
    fix_ok = evaluate_cases(s["reference_fix"], s["func_name"], s["test_cases"])["all_passed"]
    preserved, _ = behavior_preserved(s["code"], s["reference_fix"], s["func_name"], s["behavior_inputs"])
    if s["edge_case"] in _NO_REPRO:
        if repro:
            _validity_ok = False          # a "clean" snippet must have no failing test
    elif not (repro and fix_ok):
        _validity_ok = False              # a flawed snippet's test must fail on it and pass on the fix
    dec, _ = trust_decision(
        repro=repro, fix_passes=fix_ok, behavior_preserved=preserved,
        reviewer_approved=True, converged=True, flaw_confidence=0.9,
        diff_lines=_diff_lines(s["code"], s["reference_fix"]),
        severity=s["severity"], ambiguous=s["edge_case"] in ("subtle", "multi"))
    if (dec == AUTO_APPLY) != bool(s["should_auto_apply"]):
        _consistency_ok = False
check("data: reference fixes valid (flaw reproduces & fix passes; clean snippets don't)", _validity_ok)
check("data: labels match the trust policy on idealized signals", _consistency_ok)

# --------------------------------------------------------------------------- #
# 6. LLM client — parse-ladder, offline mock shapes, routing, retry
# --------------------------------------------------------------------------- #
check("llm: parses clean json", parse_json('{"a": 1}') == {"a": 1})
check("llm: parses fenced json", parse_json('```json\n{"a": 1}\n```') == {"a": 1})
check("llm: parses json wrapped in prose", parse_json('Sure, here: {"a": 1} done') == {"a": 1})
_raised = False
try:
    parse_json("not json at all")
except LLMError:
    _raised = True
check("llm: raises on unparseable output", _raised)

_role_keys = {
    "detector": {"has_flaw", "flaw_type", "severity", "confidence", "rationale"},
    "test_author": {"cases", "rationale"},
    "refactorer": {"code", "explanation"},
    "reviewer": {"approved", "comments"},
    "judge": {"grounded", "unsupported_claims"},
}
check("llm: mock client returns role-shaped JSON",
      all(keys.issubset(MockClient(role=r).complete_json("s", "u")) for r, keys in _role_keys.items()))

_vecs = MockClient(role="embedder").embed(["abc", "defgh"])
check("llm: mock embed -> equal-length vectors",
      len(_vecs) == 2 and len(_vecs[0]) == len(_vecs[1]) > 0)

os.environ["TRIAGE_OFFLINE"] = "1"
check("llm: offline get_client -> MockClient", isinstance(get_client("detector"), MockClient))
del os.environ["TRIAGE_OFFLINE"]


class _FlakyMock(MockClient):
    """Bad JSON on the first call, valid on the second — exercises the retry."""
    def __init__(self):
        super().__init__(role="detector")
        self.calls = 0

    def _raw(self, system, user, temperature=0.0):
        self.calls += 1
        return "garbage" if self.calls == 1 else '{"ok": true}'


_flaky = _FlakyMock()
check("llm: complete_json retries once then parses",
      _flaky.complete_json("s", "u") == {"ok": True} and _flaky.calls == 2)

check("agents: all role prompts present", all(hasattr(agents, n) for n in (
    "DETECTOR_SYSTEM_PROMPT", "TEST_AUTHOR_SYSTEM_PROMPT", "REFACTORER_SYSTEM_PROMPT",
    "REVIEWER_SYSTEM_PROMPT", "GROUNDEDNESS_JUDGE_PROMPT")))

# --------------------------------------------------------------------------- #
# 7. Pipeline — end-to-end smoke test, fully offline via the mock client
# --------------------------------------------------------------------------- #
os.environ["TRIAGE_OFFLINE"] = "1"
try:
    _trace = run_snippet({"id": "smoke", "func_name": "average",
                          "code": "def average(nums):\n    return sum(nums) / len(nums)\n"})
finally:
    os.environ.pop("TRIAGE_OFFLINE", None)
check("pipeline: produces a detection", _trace.get("detection") is not None)
check("pipeline: trust decision is valid", _trace.get("trust", {}).get("decision") in DECISIONS)
check("pipeline: review loop is bounded", len(_trace.get("iterations", [])) <= config.REVIEW_LOOP_MAX_ITERS)
check("pipeline: one bad refactor doesn't crash the run", _trace.get("error") is None)

# --------------------------------------------------------------------------- #
# 8. Eval harness — offline run over a subset returns a well-formed report
# --------------------------------------------------------------------------- #
os.environ["TRIAGE_OFFLINE"] = "1"
try:
    _res = evals.run_eval(snippets=SNIPPETS[:3])
finally:
    os.environ.pop("TRIAGE_OFFLINE", None)
_summ = _res.get("summary", {})
check("eval: report has summary + per-snippet rows",
      "summary" in _res and len(_res.get("per_snippet", [])) == 3)
check("eval: decisions sum to n", sum(_summ.get("decisions", {}).values()) == 3)
check("eval: unsafe_auto_applies is a non-negative int",
      isinstance(_summ.get("unsafe_auto_applies"), int) and _summ["unsafe_auto_applies"] >= 0)
check("eval: no pipeline errors offline", _summ.get("pipeline_errors") == 0)
check("eval: config records models + offline flag",
      "models" in _res.get("config", {}) and _res["config"].get("offline") is True)

# --------------------------------------------------------------------------- #
# 9. Model fallback (pure helpers behind the self-heal for missing/busy models)
# --------------------------------------------------------------------------- #
check("llm: 404 is recoverable", _is_recoverable(404, "not found"))
check("llm: overloaded/503 is recoverable", _is_recoverable(503, "model is overloaded, try again"))
check("llm: timeout message is recoverable", _is_recoverable(None, "deadline exceeded"))
check("llm: 400 invalid-argument is NOT recoverable", not _is_recoverable(400, "invalid argument"))
check("llm: keep requested model when it's available",
      _pick_model("gemini-2.5-flash", ["gemini-2.5-flash", "x"], ["y"]) == "gemini-2.5-flash")
check("llm: fall back to a listed model when requested is missing",
      _pick_model("gemini-3.5-flash-lite", ["gemini-3.1-flash-lite", "gemini-2.5-flash"],
                  ["gemini-2.5-flash", "gemini-3.1-flash-lite"]) == "gemini-2.5-flash")
check("llm: never reuse the model that just failed",
      _pick_model("a", ["a", "b"], ["a", "b"], exclude="a") == "b")
check("llm: None when nothing is available", _pick_model("a", [], ["x"]) is None)

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
print(f"selftest: {_PASS} passed, {_FAIL} failed")
if _FAILURES:
    print("FAILURES:")
    for name in _FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("OK")
