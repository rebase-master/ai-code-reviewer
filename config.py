"""
config.py — central configuration for the AI Pair Engineer.

Every agent role maps to a provider + model, so the pipeline can run on a single
model or *mix* models — notably a different reviewer model for independent,
cross-model review (a model is poor at catching its own mistakes). Override any
value via Streamlit secrets ([models] table) or env vars; see secrets.toml.example.

Import-light on purpose (stdlib only) so it is safe to import anywhere,
including the offline, dependency-free selftest.
"""
from __future__ import annotations

import os

# --- Review loop ----------------------------------------------------------- #
# Max refine iterations in the code -> review -> feedback loop.
# 1 disables the loop (single-shot refactor). Bounded to cap latency/cost.
REVIEW_LOOP_MAX_ITERS: int = int(os.environ.get("REVIEW_LOOP_MAX_ITERS", "2"))

# --- Per-role model routing ------------------------------------------------ #
# provider is one of: "gemini" (shipped), "mock" (offline, deterministic),
# and — documented — "anthropic" / "openai" (add an adapter + key to enable).
# The reviewer intentionally defaults to a DIFFERENT model than the generator.
# Unknown / busy / timed-out models self-heal via llm.py (it lists the available
# models and retries on a valid one); rolling aliases like "gemini-flash-lite-latest"
# also sidestep version-churn 404s.
DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "detector":    {"provider": "gemini", "model": "gemini-2.5-flash-lite"},
    "test_author": {"provider": "gemini", "model": "gemini-2.5-flash-lite"},
    "refactorer":  {"provider": "gemini", "model": "gemini-2.5-flash-lite"},
    "reviewer":    {"provider": "gemini", "model": "gemini-3.1-flash-lite"},  # different lite model
    "judge":       {"provider": "gemini", "model": "gemini-2.5-flash-lite"},
    "embedder":    {"provider": "gemini", "model": "gemini-embedding-001"},
}

ROLES = tuple(DEFAULT_MODELS.keys())


def offline_mode() -> bool:
    """True when every role should resolve to the deterministic MockClient.

    Lets the whole app + eval dashboard run with zero credentials. Toggled by
    the TRIAGE_OFFLINE env var (or the Streamlit sidebar, which sets it).
    """
    return os.environ.get("TRIAGE_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def model_for(role: str, overrides: "dict | None" = None) -> dict[str, str]:
    """Resolve {provider, model} for an agent role.

    Precedence: offline mode > explicit overrides (e.g. st.secrets["models"]) > defaults.
    """
    if role not in DEFAULT_MODELS:
        raise KeyError(f"unknown role {role!r}; expected one of {ROLES}")
    if offline_mode():
        return {"provider": "mock", "model": f"mock-{role}"}
    resolved = dict(DEFAULT_MODELS[role])
    if overrides and isinstance(overrides.get(role), dict):
        resolved.update(overrides[role])
    return resolved
