"""Tab 1 — live demo: run one snippet and render the full per-stage trace."""
import streamlit as st

import config
from agents import AUTO_APPLY, SUGGEST
from pipeline import run_snippet
from ui.common import load_snippets


def _decision_box(trust):
    dec, reason = trust.get("decision"), trust.get("reason", "")
    if dec == AUTO_APPLY:
        st.success(f"**AUTO-APPLY** — {reason}")
    elif dec == SUGGEST:
        st.warning(f"**SUGGEST** (human review) — {reason}")
    else:
        st.error(f"**ESCALATE** (route to human) — {reason}")


def _render_trace(trace, reviewer_model):
    if trace.get("error"):
        st.error(f"Pipeline error: {trace['error']}")
    det = trace.get("detection") or {}

    st.markdown("##### 1 · Flaw detection")
    if det.get("has_flaw"):
        st.markdown(
            f"**{det.get('flaw_type')}** · severity **{det.get('severity')}** · "
            f"confidence {det.get('confidence')}")
        st.write(det.get("rationale", ""))
        if det.get("cited_practice_id"):
            st.caption(f"Grounded in best-practice `{det['cited_practice_id']}`")
    else:
        st.info("No flaw detected — no change proposed.")

    with st.expander("Retrieved best-practices (RAG grounding)"):
        for r in trace.get("retrieved", []):
            st.write(f"`{r['id']}` {r['title']} — score {r['score']:.3f}")

    if trace.get("test"):
        st.markdown("##### 2 · Generated test (structured cases)")
        st.json(trace["test"].get("cases", []), expanded=False)

    iters = trace.get("iterations") or []
    if iters:
        st.markdown(f"##### 3 · Cross-model review loop — {len(iters)} iteration(s)")
        for i, it in enumerate(iters, 1):
            head = (f"Iter {i} — tests {'✓' if it['fix_passes'] else '✗'} · "
                    f"behavior {'✓' if it['behavior_preserved'] else '✗'} · "
                    f"reviewer {'approved' if it['review']['approved'] else 'changes requested'}")
            with st.expander(head, expanded=(i == len(iters))):
                st.code(it["refactor"], language="python")
                st.caption(f"diff: {it['diff_lines']} line(s) · {it.get('explanation', '')}")
                rc = it["review"]
                st.markdown(f"**Reviewer** (`{reviewer_model}`): "
                            f"{'✅ approved' if rc['approved'] else '📝 changes requested'}")
                if rc.get("comments"):
                    st.write(rc["comments"])

    st.markdown("##### 4 · Trust-gate verdict")
    _decision_box(trace.get("trust") or {})


def render(overrides):
    reviewer_model = config.model_for("reviewer", overrides)["model"]
    if config.offline_mode():
        st.caption("⚠️ Offline replay — for the labeled snippets, agents replay the dataset's "
                   "*reference* solutions (a ground-truth ceiling, not a live model); custom code "
                   "falls back to a flat stub. Add a GEMINI_API_KEY (offline off) for real analysis.")
    snippets = load_snippets()
    labels = {f"{s['id']} — {s['flaw_type']}": s for s in snippets}
    choice = st.selectbox("Pick a labeled snippet, or choose Custom to paste your own",
                          ["Custom…"] + list(labels))
    if choice == "Custom…":
        code = st.text_area("Function code", height=170,
                            value="def average(nums):\n    return sum(nums) / len(nums)\n")
        func_name = st.text_input("Function name", value="average")
        snippet = {"id": "custom", "func_name": func_name, "code": code}
        sig = f"custom::{func_name}::{code}"
    else:
        snippet = labels[choice]
        st.code(snippet["code"], language="python")
        sig = choice

    if st.button("Run pipeline", type="primary"):
        with st.spinner("detect → test → review loop → trust gate…"):
            st.session_state["last_trace"] = run_snippet(snippet, overrides=overrides)
            st.session_state["last_sig"] = sig
    # Only show the trace if it belongs to the current selection — otherwise the
    # panel below would be a stale result from a previous run (confusing).
    if st.session_state.get("last_trace") and st.session_state.get("last_sig") == sig:
        st.divider()
        _render_trace(st.session_state["last_trace"], reviewer_model)
    elif st.session_state.get("last_trace"):
        st.info("Selection changed — click **Run pipeline** to analyze this snippet.")
