"""
evals.py — metric functions for the evaluation harness.

Everything here is PURE and stdlib-only: functions take already-computed values
and return numbers. No google SDK, numpy, or pipeline imports at module level —
so the offline selftest can import and exercise the metric math with zero deps
and no API key. The orchestration that actually runs the pipeline (``run_eval``)
is added in a later phase and imports the pipeline lazily, inside the function.
"""
from __future__ import annotations

import statistics


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def accuracy(pairs: list) -> float:
    """pairs: list of (predicted, gold). Fraction where predicted == gold."""
    if not pairs:
        return 0.0
    return sum(1 for pred, gold in pairs if pred == gold) / len(pairs)


def rate(items: list, predicate) -> float:
    """Fraction of items for which predicate(item) is truthy."""
    if not items:
        return 0.0
    return sum(1 for it in items if predicate(it)) / len(items)


def confusion_matrix(pairs: list, labels: "list | None" = None) -> dict:
    """pairs: list of (gold, predicted). Returns matrix[gold][pred] = count."""
    if labels is None:
        labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for gold, pred in pairs:
        matrix.setdefault(gold, {p: 0 for p in labels})
        row = matrix[gold]
        row[pred] = row.get(pred, 0) + 1
    return matrix


def precision_recall_f1(items: list, is_pred_pos, is_gold_pos) -> dict:
    """Binary precision/recall/F1; positive class defined by the two predicates."""
    tp = fp = fn = tn = 0
    for it in items:
        p, g = bool(is_pred_pos(it)), bool(is_gold_pos(it))
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif not p and g:
            fn += 1
        else:
            tn += 1
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def mean(xs: list) -> float:
    return statistics.fmean(xs) if xs else 0.0


def median(xs: list) -> float:
    return statistics.median(xs) if xs else 0.0


def percentile(xs: list, p: float) -> float:
    """Linear-interpolation percentile (p in [0, 100]). Stdlib only."""
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    k = (len(ys) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


# --------------------------------------------------------------------------- #
# Orchestration: run the whole pipeline over the dataset and assemble metrics.
# Lazy imports keep this module importable (for the pure metrics above) without
# the pipeline / LLM stack — so the offline selftest can use those metrics with
# zero third-party deps.
# --------------------------------------------------------------------------- #
def run_eval(snippets=None, *, overrides=None, max_iters=None, judge=True):
    """Run every snippet through the pipeline and assemble a metrics report.

    The deterministic spine (trust gate + executor) makes the headline numbers
    trustworthy. Runs fully offline when TRIAGE_OFFLINE=1 (mock client).
    Returns a dict: {n, summary, per_snippet, edge_cases, config}.
    """
    import json as _json

    import config
    from agents import (AUTO_APPLY, GROUNDEDNESS_JUDGE_PROMPT,
                        build_groundedness_input)
    from llm import get_client
    from pipeline import run_snippet
    from retriever import Retriever

    if snippets is None:
        with open("snippets.json", encoding="utf-8") as fh:
            snippets = _json.load(fh)
    with open("practices.json", encoding="utf-8") as fh:
        practices = _json.load(fh)

    retriever = Retriever(practices=practices, overrides=overrides)
    judge_client = get_client("judge", overrides) if judge else None

    per = []
    iter_pairs = []          # (reviewer_approved, fix_passes) across all iterations
    reviewer_rejections = 0

    for snip in snippets:
        trace = run_snippet(snip, retriever=retriever, overrides=overrides, max_iters=max_iters)
        det = trace.get("detection") or {}
        iters = trace.get("iterations") or []
        decision = (trace.get("trust") or {}).get("decision")
        final = iters[-1] if iters else None

        for it in iters:
            approved = bool(it.get("review", {}).get("approved"))
            iter_pairs.append((approved, bool(it.get("fix_passes"))))
            if not approved:
                reviewer_rejections += 1

        rec = {
            "id": snip.get("id"),
            "edge_case": snip.get("edge_case"),
            "severity_label": snip.get("severity"),
            "label_auto_apply": bool(snip.get("should_auto_apply")),
            "true_has_flaw": snip.get("edge_case") != "no_flaw",
            "decision": decision,
            "reason": (trace.get("trust") or {}).get("reason"),
            "predicted_auto": decision == AUTO_APPLY,
            "detected_has_flaw": bool(det.get("has_flaw", False)),
            "detected_severity": det.get("severity"),
            "repro": bool(trace.get("repro")),
            "iter0_fix_passes": bool(iters[0].get("fix_passes")) if iters else False,
            "final_fix_passes": bool(final.get("fix_passes")) if final else False,
            "final_behavior_preserved": bool(final.get("behavior_preserved")) if final else True,
            "n_iterations": len(iters),
            "converged": bool(trace.get("converged")),
            "latency_ms": trace.get("latency_ms", 0.0),
            "error": trace.get("error"),
            "grounded": None,
        }
        if judge_client and rec["detected_has_flaw"] and det.get("rationale"):
            try:
                verdict = judge_client.complete_json(
                    GROUNDEDNESS_JUDGE_PROMPT,
                    build_groundedness_input(det.get("rationale", ""), trace.get("retrieved") or []))
                rec["grounded"] = bool(verdict.get("grounded"))
            except Exception:
                rec["grounded"] = None
        rec["unsafe_auto_apply"] = rec["predicted_auto"] and not (
            rec["final_fix_passes"] and rec["final_behavior_preserved"])
        per.append(rec)

    flawed = [r for r in per if r["n_iterations"] > 0]      # a fix was attempted
    judged = [r for r in per if r["grounded"] is not None]

    summary = {
        # Headline safety guarantee — must be 0.
        "unsafe_auto_applies": sum(1 for r in per if r["unsafe_auto_apply"]),
        # Trust-gate decision quality vs the should_auto_apply labels.
        "auto_apply": precision_recall_f1(per, lambda r: r["predicted_auto"], lambda r: r["label_auto_apply"]),
        "decisions": {d: sum(1 for r in per if r["decision"] == d)
                      for d in ("auto_apply", "suggest", "escalate")},
        # Does the cross-model review loop earn its cost?
        "fix_rate_iter0": rate(flawed, lambda r: r["iter0_fix_passes"]),
        "fix_rate_final": rate(flawed, lambda r: r["final_fix_passes"]),
        "fix_rate_lift": rate(flawed, lambda r: r["final_fix_passes"]) - rate(flawed, lambda r: r["iter0_fix_passes"]),
        "median_iterations": median([r["n_iterations"] for r in flawed]),
        "reviewer_rejections": reviewer_rejections,
        "reviewer_test_agreement": rate(iter_pairs, lambda p: p[0] == p[1]),
        # Detection + safety quality.
        "has_flaw_accuracy": accuracy([(r["detected_has_flaw"], r["true_has_flaw"]) for r in per]),
        "severity_accuracy": accuracy([(r["detected_severity"], r["severity_label"])
                                       for r in per if r["true_has_flaw"] and r["detected_has_flaw"]]),
        "behavior_preservation_rate": rate(flawed, lambda r: r["final_behavior_preserved"]),
        "groundedness_rate": rate(judged, lambda r: r["grounded"]),   # proxy metric
        "groundedness_n": len(judged),
        "latency_ms": {"median": median([r["latency_ms"] for r in per]),
                       "p95": percentile([r["latency_ms"] for r in per], 95)},
        "pipeline_errors": sum(1 for r in per if r["error"]),
    }
    edge_cases = [{"edge_case": r["edge_case"], "id": r["id"], "decision": r["decision"],
                   "label_auto_apply": r["label_auto_apply"],
                   "ok": r["predicted_auto"] == r["label_auto_apply"]}
                  for r in per if r["edge_case"]]
    return {
        "n": len(per),
        "summary": summary,
        "per_snippet": per,
        "edge_cases": edge_cases,
        "config": {
            "max_iters": max_iters if max_iters is not None else config.REVIEW_LOOP_MAX_ITERS,
            "offline": config.offline_mode(),
            "models": {role: config.model_for(role, overrides) for role in config.ROLES},
        },
    }
