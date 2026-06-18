"""
executor.py — safe, timeboxed execution of candidate functions.

Used to turn an LLM's refactor into *runnable proof*: run the original and the
refactored function on structured inputs and compare. Only a single function is
ever executed; "tests" are structured cases (data), not executed LLM code, so
the only untrusted code run is the refactor itself.

Safety model: each run happens in a separate Python subprocess with a wall-clock
timeout; the child's stdout is isolated so a snippet that prints can't corrupt
results. This is NOT a hardened sandbox (no syscall/network filtering) — it is
sized for trusted, self-contained synthetic snippets. Production would use a
container / gVisor / Firecracker.

Pure stdlib — importable by the offline selftest with no third-party deps.
"""
from __future__ import annotations

import json
import subprocess
import sys

DEFAULT_TIMEOUT = 5.0

# Runner executed via `python -c`. Reads a job {code, func, inputs} from stdin,
# defines the function, calls it on each input arg-list, and prints a JSON
# {load_error, results:[{ok, exc, value}]} to the REAL stdout (snippet prints
# are redirected to a throwaway buffer so they can't corrupt the JSON).
_RUNNER = r'''
import sys, json, io
real_out = sys.stdout
job = json.load(sys.stdin)
buf = io.StringIO()
sys.stdout = buf
ns = {}
load_error = None
try:
    exec(job["code"], ns)
    fn = ns[job["func"]]
except Exception as e:
    load_error = type(e).__name__
results = []
if load_error is None:
    for args in job["inputs"]:
        try:
            val = fn(*args)
            try:
                json.dumps(val)
                v = val
            except (TypeError, ValueError):
                v = repr(val)
            results.append({"ok": True, "exc": None, "value": v})
        except Exception as e:
            results.append({"ok": False, "exc": type(e).__name__, "value": None})
else:
    results = [{"ok": False, "exc": load_error, "value": None} for _ in job["inputs"]]
sys.stdout = real_out
print(json.dumps({"load_error": load_error, "results": results}))
'''


def run_inputs(code: str, func_name: str, inputs: list, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Run ``func_name`` (defined in ``code``) on each arg-list in ``inputs``.

    ``inputs`` is a list where each element is the positional-args list for one
    call (e.g. ``[[[2, 4]], [[]]]`` calls ``f([2, 4])`` then ``f([])``).

    Returns ``{"timeout": bool, "load_error": str|None, "results": [...]}`` with
    one result ``{ok, exc, value}`` per input.
    """
    payload = json.dumps({"code": code, "func": func_name, "inputs": [list(a) for a in inputs]})
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER],
            input=payload, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"timeout": True, "load_error": "Timeout",
                "results": [{"ok": False, "exc": "Timeout", "value": None} for _ in inputs]}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"timeout": False, "load_error": "RunnerError",
                "results": [{"ok": False, "exc": "RunnerError", "value": None} for _ in inputs],
                "stderr": proc.stderr[-500:]}
    data = json.loads(proc.stdout)
    data["timeout"] = False
    return data


def case_passes(result: dict, expect: dict) -> bool:
    """Does one execution result satisfy an expected outcome?

    expect kinds: {"kind": "returns", "value": X} | {"kind": "raises", "exc": "Name"}
    | {"kind": "not_raises"}.
    """
    kind = expect.get("kind")
    if kind == "returns":
        return bool(result.get("ok")) and result.get("value") == expect.get("value")
    if kind == "raises":
        return (not result.get("ok")) and result.get("exc") == expect.get("exc")
    if kind == "not_raises":
        return bool(result.get("ok"))
    return False


def evaluate_cases(code: str, func_name: str, cases: list, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Run ``cases`` (each {"args": [...], "expect": {...}}) against ``code``.

    Returns ``{all_passed, any_failed, n_pass, n_total, cases:[...], timeout, load_error}``.
    ``any_failed`` is how we detect the flaw was *reproduced* on the original code;
    ``all_passed`` is how we detect the refactor *fixed* it.
    """
    inputs = [c["args"] for c in cases]
    run = run_inputs(code, func_name, inputs, timeout)
    per = []
    for c, r in zip(cases, run["results"]):
        per.append({"args": c["args"], "expect": c["expect"], "result": r,
                    "passed": case_passes(r, c["expect"])})
    n_pass = sum(1 for p in per if p["passed"])
    return {
        "all_passed": len(per) > 0 and n_pass == len(per),
        "any_failed": any(not p["passed"] for p in per),
        "n_pass": n_pass, "n_total": len(per), "cases": per,
        "timeout": run.get("timeout", False), "load_error": run.get("load_error"),
    }


def behavior_preserved(original_code: str, refactor_code: str, func_name: str,
                       inputs: list, timeout: float = DEFAULT_TIMEOUT) -> tuple:
    """Does the refactor match the original on inputs the original handled?

    ``inputs`` are known-good arg-lists. Returns ``(preserved: bool, diffs: list)``.
    Inputs where the original itself errors are skipped (the fix is allowed to
    change error behavior — that is often the whole point).
    """
    if not inputs:
        return True, []
    orig = run_inputs(original_code, func_name, inputs, timeout)["results"]
    refac = run_inputs(refactor_code, func_name, inputs, timeout)["results"]
    diffs = []
    for args, ro, rr in zip(inputs, orig, refac):
        if not ro.get("ok"):
            continue
        if not rr.get("ok") or rr.get("value") != ro.get("value"):
            diffs.append({"args": args, "original": ro, "refactor": rr})
    return (len(diffs) == 0), diffs
