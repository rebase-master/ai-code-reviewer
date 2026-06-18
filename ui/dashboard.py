"""Tab 2 — eval dashboard (the submission): run the harness and render metrics."""
import streamlit as st

from evals import run_eval


def _render_dashboard(res):
    import pandas as pd
    s = res["summary"]
    if res["config"]["offline"]:
        st.warning("Offline replay — agents return the dataset's *reference* solutions, so these are "
                   "ideal/ceiling numbers, not a live model's. Run with a key (offline off) for real results.")

    unsafe = s["unsafe_auto_applies"]
    c = st.columns(4)
    c[0].metric("Unsafe auto-applies", unsafe)
    c[1].metric("Fix-rate lift (loop)", f"{s['fix_rate_iter0']:.0%} → {s['fix_rate_final']:.0%}")
    c[2].metric("Auto-apply F1", f"{s['auto_apply']['f1']:.2f}")
    c[3].metric("Latency p95 (ms)", f"{s['latency_ms']['p95']:.0f}")
    if unsafe == 0:
        st.success("Safety guarantee held: **0 unsafe auto-applies** across the set.")
    else:
        st.error(f"{unsafe} unsafe auto-apply(s) — the gate let through a bad fix. Investigate.")

    st.markdown("##### Trust-gate quality (predicted auto-apply vs `should_auto_apply` labels)")
    aa = s["auto_apply"]
    q = st.columns(4)
    q[0].metric("Precision", f"{aa['precision']:.2f}")
    q[1].metric("Recall", f"{aa['recall']:.2f}")
    q[2].metric("F1", f"{aa['f1']:.2f}")
    q[3].metric("Decisions", " / ".join(f"{k.split('_')[0]}:{v}" for k, v in s["decisions"].items()))

    st.markdown("##### Detection, safety & loop")
    d = st.columns(4)
    d[0].metric("Has-flaw accuracy", f"{s['has_flaw_accuracy']:.0%}")
    d[1].metric("Severity accuracy", f"{s['severity_accuracy']:.0%}")
    d[2].metric("Behavior preserved", f"{s['behavior_preservation_rate']:.0%}")
    d[3].metric(f"Groundedness (n={s['groundedness_n']})", f"{s['groundedness_rate']:.0%}")
    st.caption(f"n={res['n']} snippets (small — numbers are directional). "
               f"Groundedness is an LLM-judge proxy. Reviewer/test agreement: "
               f"{s['reviewer_test_agreement']:.0%}; reviewer rejections: {s['reviewer_rejections']}.")

    st.markdown("##### Per-snippet")
    st.dataframe(pd.DataFrame([{
        "id": r["id"], "edge": r["edge_case"], "label_auto": r["label_auto_apply"],
        "decision": r["decision"], "repro": r["repro"], "fix": r["final_fix_passes"],
        "behavior": r["final_behavior_preserved"], "iters": r["n_iterations"],
        "grounded": r["grounded"], "ms": r["latency_ms"], "error": r["error"],
    } for r in res["per_snippet"]]), use_container_width=True, hide_index=True)

    st.markdown("##### Edge cases")
    st.dataframe(pd.DataFrame(res["edge_cases"]), use_container_width=True, hide_index=True)
    with st.expander("Run configuration"):
        st.json(res["config"])


def render(overrides):
    st.write("Runs every snippet through the pipeline and measures the system. "
             "**This dashboard is the submission.**")
    if st.button("Run eval", type="primary"):
        with st.spinner("Evaluating… (real models ~1–2 min; offline is instant)"):
            st.session_state["eval"] = run_eval(overrides=overrides)
    res = st.session_state.get("eval")
    if not res:
        st.info("Click **Run eval** to populate the dashboard.")
    else:
        _render_dashboard(res)
