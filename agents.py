"""
agents.py — agent prompts + the deterministic trust policy.

Pure stdlib, no third-party imports, so it is safe to import from the offline
selftest and the metric layer without pulling in the LLM SDK.

The agent *prompt* strings are added in a later phase; the trust policy lives
here now because it is the deterministic spine the evaluation harness relies on.
"""
from __future__ import annotations

# Flaw severities the detector may assign.
SEVERITIES = ("low", "medium", "high", "critical")

# Trust-gate decisions.
AUTO_APPLY = "auto_apply"
SUGGEST = "suggest"
ESCALATE = "escalate"
DECISIONS = (AUTO_APPLY, SUGGEST, ESCALATE)

# Tunable thresholds for the trust gate.
MAX_AUTO_APPLY_DIFF_LINES = 20      # larger diffs always go to a human
MIN_AUTO_APPLY_CONFIDENCE = 0.6     # below this, suggest rather than auto-apply


def trust_decision(
    *,
    repro: bool,
    fix_passes: bool,
    behavior_preserved: bool,
    reviewer_approved: bool,
    converged: bool,
    flaw_confidence: float,
    diff_lines: int,
    severity: str,
    ambiguous: bool = False,
) -> "tuple[str, str]":
    """Decide whether the agent may act on a proposed fix.

    Returns ``(decision, reason)`` with decision in AUTO_APPLY / SUGGEST / ESCALATE.

    Deterministic and auditable on purpose: this is the spine the eval trusts,
    and it is independent of any LLM. Keyword-only so every call site is explicit.

    Safety invariant (asserted in the eval): never AUTO_APPLY unless the fix both
    passes its test and preserves existing behavior.
    """
    # --- Hard safety gates: any failure means we cannot trust the fix. ------- #
    if not repro:
        return ESCALATE, "could not produce a test that reproduces the flaw"
    if not behavior_preserved:
        return ESCALATE, "refactor changed existing behavior — unsafe to apply"
    if not fix_passes:
        return ESCALATE, "fix does not pass its own test"

    # --- Soft gates: safe, but warrants a human's eyes before acting. -------- #
    if not converged or not reviewer_approved:
        return SUGGEST, "review loop did not fully converge — human review"
    if ambiguous:
        return SUGGEST, "ambiguous or multiple issues — human review"
    if flaw_confidence < MIN_AUTO_APPLY_CONFIDENCE:
        return SUGGEST, "low detection confidence — human review"
    if diff_lines > MAX_AUTO_APPLY_DIFF_LINES or severity == "critical":
        return SUGGEST, "large or high-risk change — human review"

    return AUTO_APPLY, "test-backed, behavior-preserving, peer-reviewed, small, high-confidence"
