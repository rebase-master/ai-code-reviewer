"""
app.py — Streamlit entry point for the AI Pair Engineer.

Thin orchestration only; the sidebar and the three tabs live in the ui/ package
(ui.sidebar, ui.live_demo, ui.dashboard, ui.ship).

Run:
  uv run streamlit run app.py                       (needs GEMINI_API_KEY in .streamlit/secrets.toml)
  TRIAGE_OFFLINE=1 uv run streamlit run app.py       (offline replay — no key)
"""
import streamlit as st

from ui import dashboard, live_demo, ship, sidebar

st.set_page_config(page_title="AI Pair Engineer", layout="wide")

overrides = sidebar.render()

st.title("AI Pair Engineer — cross-model review loop & trust gate")
st.caption("Detect a flaw → write a failing test → refactor in a bounded, cross-model review "
           "loop → a deterministic trust gate decides: auto-apply / suggest / escalate.")

tab1, tab2, tab3 = st.tabs(["🔎 Live demo", "📊 Eval dashboard", "🚢 How I'd ship it"])
with tab1:
    live_demo.render(overrides)
with tab2:
    dashboard.render(overrides)
with tab3:
    ship.render()
