import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui import DECISION_COLORS, fmt, provenance_banner, require_run, settings

from drishti.module_a import MAD_K

run = require_run()
st.title("Lot analysis")
provenance_banner(run.provenance_class)

p, lots = run.params, run.lots
c1, c2, c3 = st.columns(3)
lot_ids = sorted(p.lot_id.unique())
default_lot = lot_ids.index("SYN_L159") if "SYN_L159" in lot_ids else 0
lot = c1.selectbox("Lot", lot_ids, index=default_lot)
params = sorted(p[p.lot_id == lot].parameter.unique())
param = c2.selectbox("Parameter", params)
conds = sorted(p[(p.lot_id == lot) & (p.parameter == param)].test_condition.unique())
cond = c3.selectbox("Test-condition group", conds)

g = p[(p.lot_id == lot) & (p.parameter == param) & (p.test_condition == cond)].copy()
lg = lots[(lots.lot_id == lot) & (lots.parameter == param) & (lots.test_condition == cond)].iloc[0]
unit = g.unit.iloc[0]
cfg = settings().module_a
k = float(cfg["robust_z_threshold"])

m = st.columns(6)
m[0].metric("Devices", len(g))
m[1].metric("Valid peers", int(lg.n_peers))
m[2].metric("Peer status", lg.status)
m[3].metric(f"Median 24 h [{unit}]", fmt(lg.median_level))
m[4].metric(f"Median change [{unit}]", fmt(lg.median_change))
m[5].metric("Flagged (Module A)", int(g.a_unusual.fillna(False).sum()))

if lg.status == "INSUFFICIENT_PEERS":
    st.warning(f"**Small lot:** only {int(lg.n_peers)} valid peers (minimum {cfg['min_peers']}). A robust median "
               "and MAD from so few devices are unstable, so DRISHTI gives **no peer verdict** and escalates. "
               "The device is still compared with the historical reference population (shown as evidence only).")
elif lg.status == "ZERO_MAD":
    st.warning(f"**Zero spread (MAD = 0) in {lg.zero_mad_features}:** at least half the devices report the same "
               "value, typically from instrument quantisation. Any z-score would divide by zero, and replacing the "
               "MAD with a tiny number would create huge artificial scores, so DRISHTI escalates instead.")
codes = list(lg.lot_codes)
if "A_LOT_SHIFT" in codes:
    st.error(f"**Lot shift:** this lot's median differs from historical lots (robust z level "
             f"{fmt(lg.lot_z_level, 3)}, change {fmt(lg.lot_z_change, 3)}; threshold ±{cfg['lot_shift_z_threshold']}). "
             "A uniformly shifted lot looks normal to peer comparison, so every device is escalated for lot review.")
if "A_LOT_SPREAD" in codes:
    st.error(f"**Lot spread:** spread ratio vs historical within-lot MAD: level {fmt(lg.spread_ratio_level, 3)}, "
             f"change {fmt(lg.spread_ratio_change, 3)}.")
if not codes and lg.status == "VALID":
    st.caption(f"Historical comparison: lot median robust z = {fmt(lg.lot_z_level, 3)} (level), "
               f"{fmt(lg.lot_z_change, 3)} (change) — within ±{cfg['lot_shift_z_threshold']}.")

tab1, tab2, tab3 = st.tabs(["Level vs change", "0 h → 24 h trajectories", "Device table"])
g["color"] = g.decision.map(DECISION_COLORS)
hover = ("<b>%{customdata[0]}</b><br>24 h: %{x:.4g} " + unit + "<br>change: %{y:.4g} " + unit +
         "<br>z level %{customdata[1]:.2f} · z change %{customdata[2]:.2f}<br>%{customdata[3]}<extra></extra>")
with tab1:
    fig = go.Figure()
    if np.isfinite(lg.mad_level) and np.isfinite(lg.mad_change) and lg.mad_level > 0 and lg.mad_change > 0:
        fig.add_shape(type="rect", x0=lg.median_level - k * MAD_K * lg.mad_level,
                      x1=lg.median_level + k * MAD_K * lg.mad_level,
                      y0=lg.median_change - k * MAD_K * lg.mad_change, y1=lg.median_change + k * MAD_K * lg.mad_change,
                      fillcolor="rgba(26,86,219,0.08)", line=dict(color="rgba(26,86,219,0.5)", dash="dot"))
    for d, sub in g.groupby("decision"):
        fig.add_trace(go.Scatter(
            x=sub.value_24h, y=sub.value_24h - sub.value_0h, mode="markers", name=d,
            marker=dict(size=11, color=DECISION_COLORS[d], line=dict(width=1, color="white"),
                        symbol=np.where(sub.a_unusual.fillna(False), "diamond", "circle")),
            customdata=np.column_stack([sub.device_id, sub.z_level.fillna(np.nan), sub.z_change.fillna(np.nan),
                                        sub.reason_codes.map(lambda c: ", ".join(x for x in c if x != "DEMO_SPEC_NOT_APPROVED"))]),
            hovertemplate=hover))
    fig.update_layout(height=460, xaxis_title=f"24 h value [{unit}]", yaxis_title=f"change 0→24 h [{unit}]",
                      legend_title="Device decision (this parameter)", margin=dict(t=30, l=10, r=10, b=10))
    if lg.status == "VALID" and st.toggle("Zoom to the peer reference band (outliers may fall outside the view)"):
        bx, by = 2 * k * MAD_K * lg.mad_level, 2 * k * MAD_K * lg.mad_change
        fig.update_xaxes(range=[lg.median_level - bx, lg.median_level + bx])
        fig.update_yaxes(range=[lg.median_change - by, lg.median_change + by])
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Shaded box: robust reference band, peer median ± {k} × 1.4826 × MAD on each axis (only when "
               "the peer group is valid). Diamonds: Module A unusual level/change. Scores are statistical "
               "unusualness, not failure probabilities.")
with tab2:
    fig2 = go.Figure()
    for r in g.itertuples():
        flagged = bool(r.a_unusual) if r.a_unusual == r.a_unusual else False
        fig2.add_trace(go.Scatter(x=[0, 24], y=[r.value_0h, r.value_24h], mode="lines+markers",
                                  line=dict(color=DECISION_COLORS[r.decision] if r.decision != "PASS" else "#9aa0a6",
                                            width=3 if flagged else 1.2), name=r.device_id, showlegend=False,
                                  hovertemplate=f"{r.device_id}<br>%{{y:.4g}} {unit}<extra>{r.decision}</extra>"))
    if np.isfinite(lg.median_level) and lg.mad_level > 0:
        band = k * MAD_K * lg.mad_level
        fig2.add_trace(go.Scatter(x=[24, 24], y=[lg.median_level - band, lg.median_level + band], mode="lines",
                                  line=dict(color="rgba(26,86,219,0.5)", width=12), name="24 h peer band",
                                  hoverinfo="skip"))
    lo, hi = g.spec_lower.iloc[0], g.spec_upper.iloc[0]
    fig2.add_hline(y=hi, line=dict(color="#b3261e", dash="dash"), annotation_text=f"upper limit {hi:g}")
    if lo != 0 or (g.value_24h.min() - lo) < 0.3 * (hi - lo):
        fig2.add_hline(y=lo, line=dict(color="#b3261e", dash="dash"), annotation_text=f"lower limit {lo:g}")
    fig2.update_layout(height=460, xaxis=dict(title="burn-in hours", tickvals=[0, 24]), yaxis_title=f"{param} [{unit}]",
                       margin=dict(t=30, l=10, r=10, b=10))
    st.plotly_chart(fig2, width="stretch")
    st.caption("Grey: devices passing this parameter; coloured: ESCALATE/FAIL; thick: Module A flagged. "
               "Only readings available by 24 h are used.")
with tab3:
    t = g[["device_id", "value_0h", "value_24h", "z_level", "z_change", "ensemble_score", "prediction", "lower",
           "upper", "decision", "rule"]].copy()
    t["change"] = g.value_24h - g.value_0h
    t["reasons"] = g.reason_codes.map(lambda c: ", ".join(x for x in c if x not in ("DEMO_SPEC_NOT_APPROVED",)))
    st.dataframe(t.sort_values("decision"), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4g") for c in
                                ["value_0h", "value_24h", "change", "z_level", "z_change", "ensemble_score",
                                 "prediction", "lower", "upper"]})
