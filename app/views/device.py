import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import DECISION_COLORS, decision_badge, fmt, models, provenance_banner, require_run, settings

from drishti import data, reasons
from drishti.decision import STAGE_LABEL
from drishti.explain import describe_trajectory, parameter_evidence, parameter_sentences, shap_note
from drishti.module_a import MAD_K
from drishti.provenance import DATASETS

run = require_run()
st.title("Device analysis")
provenance_banner(run.provenance_class)

devs = run.devices.device_id.tolist()
qp = st.query_params.get("device")
default = devs.index(qp) if qp in devs else (devs.index("SYN_L159_D009") if "SYN_L159_D009" in devs else 0)
dev = st.selectbox("Device", devs, index=default, help="Type to search")
st.query_params["device"] = dev
drow = run.devices.set_index("device_id").loc[dev]
p = run.params[run.params.device_id == dev]

st.markdown(f"### {dev} &nbsp; {decision_badge(drow.decision, drow.basis)}", unsafe_allow_html=True)
st.caption(f"Lot **{drow.lot_id}** · part **{drow.part_number}** · {STAGE_LABEL}")
if drow.missing_parameters:
    st.warning(f"Registered parameter(s) not supplied: {drow.missing_parameters} → device cannot PASS.")
meaning = {"PASS": "All parameters: calibrated 168 h interval inside the limits and no peer, lot or data-quality "
                   "warning. This is an early-screening PASS, not qualification.",
           "ESCALATE": "Evidence is insufficient or suspicious. Recommended action: continue burn-in / engineering "
                       "review; do not accept or reject on this evidence alone.",
           "FAIL": "A limit is violated: observed now (OBSERVED) or the whole calibrated interval is beyond a "
                   "limit (PREDICTED)."}
st.info(meaning[drow.decision])

fc = models().module_b
k = float(settings().module_a["robust_z_threshold"])
tabs = st.tabs([f"{r.parameter} · {r.decision}" for r in p.itertuples()])
for tab, r in zip(tabs, p.itertuples()):
    with tab:
        left, right = st.columns([3, 2])
        peers = run.params[(run.params.lot_id == r.lot_id) & (run.params.parameter == r.parameter)
                           & (run.params.test_condition == r.test_condition) & (run.params.device_id != dev)]
        with left:
            fig = go.Figure()
            for q in peers.itertuples():
                fig.add_trace(go.Scatter(x=[0, 24], y=[q.value_0h, q.value_24h], mode="lines", showlegend=False,
                                         line=dict(color="rgba(150,155,160,0.45)", width=1), hoverinfo="skip"))
            if np.isfinite(r.peer_median_level) and r.peer_mad_level > 0:
                band = k * MAD_K * r.peer_mad_level
                fig.add_trace(go.Scatter(x=[24, 24], y=[r.peer_median_level - band, r.peer_median_level + band],
                                         mode="lines", line=dict(color="rgba(26,86,219,0.35)", width=14),
                                         name=f"peer band (median ± {k}·robust σ)"))
            fig.add_trace(go.Scatter(x=[0, 24], y=[r.value_0h, r.value_24h], mode="lines+markers",
                                     line=dict(color="#1a56db", width=3), marker=dict(size=9), name=f"{dev} (measured)"))
            if np.isfinite(r.prediction):
                fig.add_trace(go.Scatter(x=[24, 168], y=[r.value_24h, r.prediction], mode="lines",
                                         line=dict(color="#1a56db", dash="dot"), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=[168], y=[r.prediction], mode="markers", name="168 h forecast (q50)",
                                         marker=dict(size=11, symbol="square", color="#1a56db"),
                                         error_y=dict(type="data", symmetric=False, array=[r.upper - r.prediction],
                                                      arrayminus=[r.prediction - r.lower], thickness=2, width=10)))
            show_future = False
            if run.provenance_class == "SYNTHETIC" and r.lot_id.startswith("SYN_"):
                show_future = st.toggle("Show hidden synthetic readings at 48/96/168 h (evaluation only)",
                                        key=f"fut_{r.parameter}")
            if show_future:
                meas = data.synthetic_measurements()
                fut = meas[(meas.device_id == dev) & (meas.parameter == r.parameter)].sort_values("burn_in_hours")
                fig.add_trace(go.Scatter(x=fut.burn_in_hours, y=fut.value, mode="lines+markers", name="actual (hidden at 24 h)",
                                         line=dict(color="#7e3ff2", dash="dash")))
            fig.add_hline(y=r.spec_upper, line=dict(color="#b3261e", dash="dash"),
                          annotation_text=f"upper limit {r.spec_upper:g}", annotation_position="top left")
            fig.add_hline(y=r.spec_lower, line=dict(color="#b3261e", dash="dash"),
                          annotation_text=f"lower limit {r.spec_lower:g}", annotation_position="bottom left")
            fig.update_layout(height=430, xaxis=dict(title="burn-in hours", tickvals=[0, 24, 48, 96, 168]),
                              yaxis_title=f"{r.parameter} [{r.unit}]", margin=dict(t=20, l=10, r=10, b=10),
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
            if show_future:
                shape = describe_trajectory(fut.burn_in_hours, fut.value)
                st.caption(f"Retrospective shape of the full synthetic trajectory: **{shape}** (descriptive only; "
                           "not a failure-mechanism diagnosis; readings after 24 h never enter the decision).")
            st.caption(f"Grey: {len(peers)} lot peers under the same test condition. Error bar: calibrated 90% "
                       "lot-level interval.")
        with right:
            st.markdown(f"**Outcome:** {decision_badge(r.decision, r.basis)} by rule **{r.rule}**",
                        unsafe_allow_html=True)
            ev = pd.DataFrame(parameter_evidence(r), columns=["Evidence", "Value"])
            st.table(ev.set_index("Evidence"))  # static table wraps long values
        st.markdown("**Why**")
        for s in parameter_sentences(r):
            st.markdown(f"- {s}")
        rc = pd.DataFrame([{"code": c, "category": reasons.category(c), "role": reasons.role(c),
                            "decisive": c in r.decisive_codes} for c in r.reason_codes])
        with st.expander("Reason codes and roles"):
            st.dataframe(rc, hide_index=True, width="stretch")
        if np.isfinite(r.prediction) and fc is not None:
            with st.expander("Forecast model contributions (TreeSHAP)"):
                sh = pd.DataFrame({"input": ["value_0h", "value_24h"], "contribution": [r.shap_value_0h, r.shap_value_24h]})
                bar = go.Figure(go.Bar(x=sh.contribution, y=sh.input, orientation="h", marker_color="#1a56db"))
                bar.update_layout(height=180, margin=dict(t=10, l=10, r=10, b=10),
                                  xaxis_title=f"contribution to q50 forecast [{r.unit}] (base {fmt(r.shap_base)})")
                st.plotly_chart(bar, width="stretch")
                st.caption(shap_note())

st.divider()
st.subheader("Provenance")
src = DATASETS["synthetic"] if run.provenance_class == "SYNTHETIC" else None
st.markdown(f"- Evidence class: **{run.provenance_class}**; input file `{run.source_name}` "
            f"(SHA-256 `{run.input_sha256[:16]}…`); run `{run.run_id}`")
st.markdown(f"- Specification: `{p.spec_revision.iloc[0]}` · origin **{p.spec_origin.iloc[0]}**")
if src:
    st.markdown(f"- Source: {src['title']} — {src['note']}")
if drow.lot_id.startswith("ILLUSTRATIVE_"):
    st.warning("This lot is an **illustrative fixture** (a synthetic lot rounded to a 0.1 V instrument "
               "resolution to demonstrate zero-MAD handling), not generator output.")
st.markdown(f"- Model: `{run.model_version}`")
