#!/usr/bin/env python3
"""
selftest.py — offline, dependency-free verification of the deterministic spine.

Runs with plain `python3 selftest.py` — no third-party packages, no API key.
Covers the trust gate (incl. its safety invariant), the safe executor
(reproduce → fix → behavior-preservation → timeout), and the metric math.
Exits non-zero if any check fails. Data/consistency checks are added later.
"""
from __future__ import annotations

import itertools
import sys

import evals
from agents import (AUTO_APPLY, DECISIONS, ESCALATE, SUGGEST, trust_decision)
from executor import behavior_preserved, evaluate_cases, run_inputs

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
# Summary
# --------------------------------------------------------------------------- #
print(f"selftest: {_PASS} passed, {_FAIL} failed")
if _FAILURES:
    print("FAILURES:")
    for name in _FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("OK")
