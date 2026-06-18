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


# --------------------------------------------------------------------------- #
# Agent prompts (the LLM stages). All return strict JSON; the client parses it.
# Pure strings — agents.py stays import-light.
# --------------------------------------------------------------------------- #
DETECTOR_SYSTEM_PROMPT = """You are a senior code reviewer. Given a single Python function, identify the most important design flaw or bug, if any.

You receive the CODE and a set of BEST_PRACTICES (each with an id) to ground your judgment.

Return ONLY a JSON object, no prose, no markdown fences:
{
  "has_flaw": boolean,
  "flaw_type": string,                  // short snake_case label, e.g. "empty_input_crash"; "none" if no flaw
  "severity": one of ["low","medium","high","critical"],
  "confidence": number,                 // 0.0-1.0
  "cited_practice_id": string or null,  // the BEST_PRACTICES id that supports your finding
  "rationale": string                   // one or two sentences
}

Rules:
- Report at most one primary flaw — the highest-impact issue.
- Ground your rationale in the BEST_PRACTICES and cite the most relevant id.
- If the code is correct and idiomatic, set has_flaw false, flaw_type "none", severity "low", and a high confidence.
- Use "critical" only for safety, security, or financial-integrity issues.
"""

TEST_AUTHOR_SYSTEM_PROMPT = """You write a single focused test that FAILS on the buggy code and will PASS once the flaw is fixed.

You receive the CODE, its FUNCTION name, and the detected FLAW.

Express the test as STRUCTURED CASES (data, not code) so it can be run deterministically. Return ONLY JSON:
{
  "cases": [
    {"args": [ ... ], "expect": {"kind": "returns", "value": <any>}}
    // or {"kind": "raises", "exc": "ExceptionName"} or {"kind": "not_raises"}
  ],
  "rationale": string
}

Rules:
- Each case's "args" is the list of positional arguments for ONE call of the function.
- Include at least one case that exercises the flaw (the buggy code should fail it).
- Use simple, JSON-serializable inputs and expected values.
"""

REFACTORER_SYSTEM_PROMPT = """You fix the identified flaw in a single Python function with the smallest change that is correct and preserves existing behavior.

You receive the CODE, the detected FLAW, and optionally FEEDBACK from a previous attempt (failing tests or reviewer comments).

Return ONLY JSON:
{
  "code": string,          // the complete fixed function, ready to run
  "explanation": string    // one or two sentences on what you changed
}

Rules:
- Keep the same function name and signature.
- Make the minimal change that fixes the flaw; no unrelated edits.
- If FEEDBACK is present, address it specifically.
- Return the entire function, not a diff.
"""

REVIEWER_SYSTEM_PROMPT = """You are a second, independent reviewer. Decide whether a proposed fix is correct, minimal, and safe to apply.

You receive the ORIGINAL code, the PROPOSED fix, the detected FLAW, and BEST_PRACTICES for reference.

Return ONLY JSON:
{
  "approved": boolean,
  "comments": string,                  // concise; if not approved, say what to change
  "cited_practice_id": string or null
}

Rules:
- Approve only if the fix resolves the flaw, keeps the signature, and introduces no new issues or behavior changes.
- If unsure, or the change is broad or risky, do NOT approve and explain why.
- Be specific and brief.
"""

GROUNDEDNESS_JUDGE_PROMPT = """You evaluate whether a reviewer's rationale is grounded in the provided BEST_PRACTICES. This is an automated eval check.

You receive the RATIONALE and the BEST_PRACTICES available.

Return ONLY JSON:
{
  "grounded": boolean,
  "unsupported_claims": [string]
}

Mark grounded false if the rationale asserts a rule or best practice NOT supported by the snippets. Reasoning about the code itself is allowed. This is a proxy metric.
"""


def _practices_block(practices: list) -> str:
    if not practices:
        return "(none)"
    return "\n".join(f"[{p['id']}] {p['title']}: {p['content']}" for p in practices)


def build_detector_input(code: str, practices: list) -> str:
    return f"CODE:\n{code}\nBEST_PRACTICES:\n{_practices_block(practices)}"


def build_test_author_input(code: str, func_name: str, detection: dict) -> str:
    return (f"CODE:\n{code}\nFUNCTION: {func_name}\n"
            f"FLAW: {detection.get('flaw_type')} (severity {detection.get('severity')}) — "
            f"{detection.get('rationale', '')}")


def build_refactor_input(code: str, detection: dict, feedback: "str | None" = None) -> str:
    fb = f"\nFEEDBACK:\n{feedback}" if feedback else ""
    return f"CODE:\n{code}\nFLAW: {detection.get('flaw_type')} — {detection.get('rationale', '')}{fb}"


def build_review_input(code: str, refactor_code: str, detection: dict, practices: list) -> str:
    return (f"ORIGINAL:\n{code}\nPROPOSED FIX:\n{refactor_code}\n"
            f"FLAW: {detection.get('flaw_type')} — {detection.get('rationale', '')}\n"
            f"BEST_PRACTICES:\n{_practices_block(practices)}")


def build_groundedness_input(rationale: str, practices: list) -> str:
    return f"RATIONALE: {rationale}\nBEST_PRACTICES:\n{_practices_block(practices)}"
