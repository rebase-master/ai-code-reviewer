"""
app.py — Streamlit UI for the AI Pair Engineer.

Tab 1  Live demo        — run one snippet and watch detect -> test -> review loop -> trust gate.
Tab 2  Eval dashboard   — run the harness over the dataset; lead with the safety guarantee. (The submission.)
Tab 3  How I'd ship it  — productionization notes + the trust-gate diagram.

Run:  streamlit run app.py            (needs GEMINI_API_KEY in .streamlit/secrets.toml)
      TRIAGE_OFFLINE=1 streamlit run app.py   (mock client, no key — illustrative numbers)
"""
import json
import os

import streamlit as st
import streamlit.components.v1 as components

import config
from agents import AUTO_APPLY, ESCALATE, SUGGEST
from evals import run_eval
from pipeline import run_snippet

st.set_page_config(page_title="AI Pair Engineer", layout="wide")


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def _secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def _model_overrides():
    m = _secret("models")
    if not m:
        return None
    try:
        return {role: dict(spec) for role, spec in m.items()}
    except Exception:
        return None


def _bootstrap_keys():
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        v = _secret(k)
        if v:
            os.environ[k] = v


@st.cache_data(show_spinner=False)
def _load_snippets():
    with open("snippets.json", encoding="utf-8") as fh:
        return json.load(fh)


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


def _render_dashboard(res):
    import pandas as pd
    s = res["summary"]
    if res["config"]["offline"]:
        st.warning("Offline mock run — numbers are illustrative only. Add a key and turn off "
                   "offline mode for real results.")

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


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
_bootstrap_keys()

st.sidebar.title("AI Pair Engineer")
offline = st.sidebar.toggle("Offline mode (mock, no API key)",
                            value=not bool(_secret("GEMINI_API_KEY")))
if offline:
    os.environ["TRIAGE_OFFLINE"] = "1"
else:
    os.environ.pop("TRIAGE_OFFLINE", None)

overrides = _model_overrides()
st.sidebar.subheader("Models per agent")
for role in config.ROLES:
    spec = config.model_for(role, overrides)
    tag = " ←  different" if role == "reviewer" else ""
    st.sidebar.write(f"`{role}` → `{spec['model']}`{tag}")
st.sidebar.caption(f"Review loop: up to {config.REVIEW_LOOP_MAX_ITERS} iteration(s)")
if not offline and not _secret("GEMINI_API_KEY"):
    st.sidebar.warning("No GEMINI_API_KEY found — switch on offline mode, or add it to "
                       ".streamlit/secrets.toml.")

st.title("AI Pair Engineer — cross-model review loop & trust gate")
st.caption("Detect a flaw → write a failing test → refactor in a bounded, cross-model review "
           "loop → a deterministic trust gate decides: auto-apply / suggest / escalate.")

tab1, tab2, tab3 = st.tabs(["🔎 Live demo", "📊 Eval dashboard", "🚢 How I'd ship it"])
reviewer_model = config.model_for("reviewer", overrides)["model"]

with tab1:
    snippets = _load_snippets()
    labels = {f"{s['id']} — {s['flaw_type']}": s for s in snippets}
    choice = st.selectbox("Pick a labeled snippet, or choose Custom to paste your own",
                          ["Custom…"] + list(labels))
    if choice == "Custom…":
        code = st.text_area("Function code", height=170,
                            value="def average(nums):\n    return sum(nums) / len(nums)\n")
        func_name = st.text_input("Function name", value="average")
        snippet = {"id": "custom", "func_name": func_name, "code": code}
    else:
        snippet = labels[choice]
        st.code(snippet["code"], language="python")

    if st.button("Run pipeline", type="primary"):
        with st.spinner("detect → test → review loop → trust gate…"):
            st.session_state["last_trace"] = run_snippet(snippet, overrides=overrides)
    if st.session_state.get("last_trace"):
        st.divider()
        _render_trace(st.session_state["last_trace"], reviewer_model)

with tab2:
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

with tab3:
    st.markdown(SHIP_NOTES)
    if os.path.exists("docs/trust-gate-decision-flow.svg"):
        with open("docs/trust-gate-decision-flow.svg", encoding="utf-8") as fh:
            components.html(fh.read(), height=640)
