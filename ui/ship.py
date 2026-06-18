"""Tab 3 — productionization notes + the trust-gate diagram."""
import os

import streamlit as st
import streamlit.components.v1 as components

SHIP_NOTES = """
### How I'd ship this

**Integration — as a PR bot / CI check**
- On each pull request, run the pipeline per changed function.
- Post the flaw + grounded rationale as a review comment; attach the generated test.
- Open the refactor as a *suggested change*.
- **Auto-merge only when the trust gate returns `auto-apply`**; otherwise request human review. The gate's `reason` string is the audit trail.

**Hardening the sandbox**
- The executor here is a timeboxed subprocess — fine for trusted snippets, not a security boundary. In production, run candidate code in a container / gVisor / Firecracker with no network and a strict CPU/memory/time budget.

**Risks & mitigations**
- *Prompt injection via code comments* — strip/ignore untrusted instructions; the trust gate stays deterministic so a manipulated model can't grant itself auto-apply.
- *Single-model blind spots* — a different reviewer model + the behavior-preservation check catch what one model misses.
- *Knowledge drift* — version the best-practices KB and re-run the eval on every change.
- *Cost / latency* — the review loop is bounded (K iterations); cache embeddings; batch where possible.
"""


def render():
    st.markdown(SHIP_NOTES)
    if os.path.exists("docs/trust-gate-decision-flow.svg"):
        with open("docs/trust-gate-decision-flow.svg", encoding="utf-8") as fh:
            components.html(fh.read(), height=640)
