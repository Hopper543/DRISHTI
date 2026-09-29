import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import settings

from drishti import data
from drishti.provenance import DATASETS
from drishti.real import REQUIRED, analyze_real, trajectory_shapes, validate_real

st.title("Real-data explorer")
st.info("**REAL component measurements — exploratory electrical-drift analysis.** These cohorts come from NASA "
        "radiation (TID) reports and a capacitor aging experiment. The x-axis is **radiation dose or aging hours, "
        "never burn-in hours**, there is no 168 h endpoint and no latent-defect label, so DRISHTI gives peer "
        "statistics only — no PASS/FAIL and no forecast.", icon="📐")

src = st.radio("Data", ["Bundled prepared real data", "Upload real long-format CSV"], horizontal=True)
if src.startswith("Bundled"):
    frame, scope = data.real_drift_view()
    st.caption("Scope: **" + ("full local table (incl. local-only sources)" if scope == "local_full" else
                              "public subset committed to the repository (AD620, ReRAM, OP484, U309)") + "**")
    name = "real_drift_view"
else:
    st.caption(f"Required columns: `{', '.join(REQUIRED)}` (the `data/public/real_measurements` layout).")
    up = st.file_uploader("Real-data CSV", type=["csv"], key="real_up")
    if up is None:
        st.stop()
    frame = pd.read_csv(io.BytesIO(up.getvalue()), dtype={"source_device_id": str, "lot_id": str})
    name = up.name

errors, warnings, f = validate_real(frame)
for e in errors:
    st.error(e)
for w in warnings:
    st.caption("⚠️ " + w)
if errors:
    st.stop()


@st.cache_data(show_spinner="Scoring real cohorts ...")
def _analyze(key: str, _f: pd.DataFrame, min_peers: int):
    cfg = dict(settings().module_a, min_peers=min_peers)
    return analyze_real(_f, cfg), trajectory_shapes(_f)


mp = st.sidebar.number_input("Minimum peers (exploratory)", 3, 20, int(settings().module_a["min_peers"]),
                             help="The default screening policy uses 8. Lower values are exploratory only.")
res, shapes = _analyze(f"{name}-{len(f)}", f, int(mp))

c1, c2, c3 = st.columns(3)
ds = c1.selectbox("Dataset", sorted(f.dataset_id.unique()),
                  format_func=lambda d: f"{d} — {DATASETS.get(d, {}).get('title', d)}")
g = f[f.dataset_id == ds]
par = c2.selectbox("Parameter", sorted(g.parameter.unique()))
g = g[g.parameter == par]
ch = c3.selectbox("Channel", sorted(g.channel.astype(str).unique()))
g = g[g.channel.astype(str) == ch]
meta = DATASETS.get(ds, {})
if meta:
    st.markdown(f"**Source:** [{meta['title']}]({meta['url']}) · evidence **{meta['evidence']}** · reuse: "
                f"{meta['redistribution']}  \n{meta['note']}")
unit = g.unit.iloc[0]
axis = g.stress_axis.iloc[0]
xlab = "nominal dose checkpoint [krad(Si)]" if "krad" in axis else "aging hours (capacitor stress, not burn-in)"

fig = go.Figure()
palette = {"biased": "#1a56db", "Biased": "#1a56db", "unbiased": "#0e9f6e", "control": "#6b7280",
           "Control": "#6b7280", "read_only": "#1a56db", "read_write": "#d61f69", "UNRESOLVED": "#b86e00"}
for dev, d in g.sort_values("measurement_stage").groupby("device_id"):
    pg = str(d.peer_group.iloc[0])
    fig.add_trace(go.Scatter(x=d.stress_value, y=d.value, mode="lines+markers", name=f"{dev} ({pg})",
                             line=dict(color=palette.get(pg, "#7e3ff2"), width=1.5,
                                       dash="dot" if "control" in pg.lower() else "solid"),
                             hovertemplate=f"{dev}<br>%{{x}}: %{{y:.4g}} {unit}<br>phase %{{customdata}}<extra>{pg}</extra>",
                             customdata=d.phase))
fig.update_layout(height=430, xaxis_title=xlab, yaxis_title=f"{par} [{unit}]", margin=dict(t=20, l=10, r=10, b=10))
st.plotly_chart(fig, width="stretch")
st.caption("Anneal / post-storage points are plotted at their recorded stress value where one exists; anneal rows "
           "with unknown final dose are omitted from the x-axis. Controls are dotted and never used as peers.")

r = res[(res.dataset_id == ds) & (res.parameter == par) & (res.channel.astype(str) == ch)]
if r.empty:
    st.info("No scorable stressed rows for this selection.")
    st.stop()
stage = st.select_slider("Checkpoint for peer comparison", options=sorted(r.measurement_stage.unique()),
                         format_func=lambda s: f"stage {s} · {r[r.measurement_stage == s].stress_value.iloc[0]}")
rs = r[r.measurement_stage == stage]
counts = rs.exploratory_status.value_counts()
st.markdown(" · ".join(f"**{k}** {v}" for k, v in counts.items()))
if "UNTRUSTED_NO_VERDICT" in counts:
    st.warning("Rows with unresolved bias assignment or suspect source cells (e.g. AD620) receive no automated "
               "peer verdict.")
if "INSUFFICIENT_PEERS" in counts:
    st.caption(f"Peer groups smaller than {mp} devices get no verdict; values are still listed.")
cols = ["device_id", "peer_group", "stress_value", "value", "baseline_value", "change", "n_peers", "z_level",
        "z_change", "exploratory_status", "source_file", "source_page", "quality_flags"]
st.dataframe(rs[[c for c in cols if c in rs.columns]], hide_index=True, width="stretch")
if ds == "reram":
    st.markdown("**Reported functional-failure intervals (interval-censored, separate evidence type)**")
    st.dataframe(data.reram_outcomes(), hide_index=True, width="stretch")
with st.expander("Descriptive trajectory shapes (retrospective; not mechanism diagnoses)"):
    st.dataframe(shapes[(shapes.dataset_id == ds) & (shapes.parameter == par) & (shapes.channel.astype(str) == ch)],
                 hide_index=True, width="stretch")
