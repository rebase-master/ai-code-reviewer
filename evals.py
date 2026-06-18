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
