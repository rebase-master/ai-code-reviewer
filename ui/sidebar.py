"""Sidebar: API-key bootstrap, offline toggle, per-role model display."""
import os

import streamlit as st

import config
from ui.common import bootstrap_keys, model_overrides, secret


def render():
    """Render the sidebar, set the offline env flag, and return model overrides."""
    bootstrap_keys()
    st.sidebar.title("AI Pair Engineer")
    offline = st.sidebar.toggle("Offline mode (mock, no API key)",
                                value=not bool(secret("GEMINI_API_KEY")))
    if offline:
        os.environ["TRIAGE_OFFLINE"] = "1"
    else:
        os.environ.pop("TRIAGE_OFFLINE", None)

    overrides = model_overrides()
    st.sidebar.subheader("Models per agent")
    for role in config.ROLES:
        spec = config.model_for(role, overrides)
        tag = " ←  different" if role == "reviewer" else ""
        st.sidebar.write(f"`{role}` → `{spec['model']}`{tag}")
    st.sidebar.caption(f"Review loop: up to {config.REVIEW_LOOP_MAX_ITERS} iteration(s)")
    if not offline and not secret("GEMINI_API_KEY"):
        st.sidebar.warning("No GEMINI_API_KEY found — switch on offline mode, or add it to "
                           ".streamlit/secrets.toml.")
    return overrides
