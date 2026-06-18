"""Shared UI helpers: secrets access, model overrides, key bootstrap, data loading."""
import json
import os

import streamlit as st


def secret(name, default=None):
    """Read a Streamlit secret, tolerating a missing secrets.toml."""
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def model_overrides():
    """Per-role model overrides from secrets ([models] table), as plain dicts."""
    m = secret("models")
    if not m:
        return None
    try:
        return {role: dict(spec) for role, spec in m.items()}
    except Exception:
        return None


def bootstrap_keys():
    """Copy any provider keys from secrets into the environment for the clients."""
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        v = secret(k)
        if v:
            os.environ[k] = v


@st.cache_data(show_spinner=False)
def load_snippets():
    with open("snippets.json", encoding="utf-8") as fh:
        return json.load(fh)
