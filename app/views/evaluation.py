import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from drishti.config import EVAL_DIR

st.title("Evaluation")
if not (EVAL_DIR / "module_a_synthetic_summary.csv").exists():
    st.error("No evaluation results found. Run `python scripts/evaluate.py`.")
    st.stop()
meta = json.loads((EVAL_DIR / "evaluation_meta.json").read_text())
st.caption(f"Results from `scripts/evaluate.py` · model `{meta['model_version']}`. Evidence classes are shown "
           "separately and never averaged together; errors are never averaged across physical units.")

SPLIT_LABEL = {"validation": "Validation (thresholds set here)", "test": "Ordinary test", "ood_test": "Shifted test",
               "edge_test": "Edge / small-lot test"}
t_a, t_b, t_d, t_s, t_r = st.tabs(["Module A · SYNTHETIC", "Module B · SYNTHETIC", "Decisions · SYNTHETIC",
                                   "SECOM · REAL process data", "Real component cohorts · exploratory"])


def cm_fig(tn, fp, fn, tp, labels=("normal", "anomalous"), title=""):
    z = np.array([[tn, fp], [fn, tp]])
    fig = go.Figure(go.Heatmap(z=z, x=[f"flag: no", f"flag: yes"], y=[f"truth: {labels[0]}", f"truth: {labels[1]}"],
                               text=z, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(height=260, title=title, margin=dict(t=40, l=10, r=10, b=10), yaxis=dict(autorange="reversed"))
    return fig


with t_a:
    st.warning("**SYNTHETIC evidence only.** Planted anomalies follow generator assumptions; this is not real "
               "defect-detection performance.", icon="🧪")
    a = pd.read_csv(EVAL_DIR / "module_a_synthetic_summary.csv")
    det = a[a.method != "lot_shift_check"].copy()
    det["split"] = det.split.map(SPLIT_LABEL)
    st.markdown("Unit of analysis: one device × parameter at 24 h. Positive = any planted abnormality on that "
                "parameter. **Delayed and step defects start after 30–130 h, so they are invisible at 24 h by "
                "construction** — `recall_detectable_24h` counts gradual drift, stable outliers and measurement "
                "faults only. Thresholds: robust |z| ≥ 3.5 (fixed a priori); IF/ECOD/ensemble at a 2% "
                "healthy-parameter flag rate on validation lots.")
    fig = px.bar(det, x="method", y="recall_detectable_24h", color="split", barmode="group",
                 hover_data=["n_scored", "healthy_flag_rate", "roc_auc", "average_precision"],
                 labels={"recall_detectable_24h": "recall (24 h-detectable planted anomalies)"}, height=360)
    st.plotly_chart(fig, width="stretch")
    st.caption("Edge / small-lot bars rest on very few scorable rows (most small-lot rows cannot be peer-scored); "
               "treat them as anecdotal.")
    show = det[["split", "method", "threshold", "n_scored", "n_not_scored", "tp", "fp", "fn", "tn", "healthy_flag_rate",
                "recall_detectable_24h", "recall_all_planted", "precision", "roc_auc", "average_precision"]]
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                ["threshold", "healthy_flag_rate", "recall_detectable_24h", "recall_all_planted",
                                 "precision", "roc_auc", "average_precision"]})
    c1, c2 = st.columns(2)
    sp = c1.selectbox("Confusion matrix · split", list(SPLIT_LABEL), format_func=SPLIT_LABEL.get, index=1)
    me = c2.selectbox("method", ["robust_rule", "isolation_forest", "ecod", "ensemble"])
    r = a[(a.split == sp) & (a.method == me)].iloc[0]
    st.plotly_chart(cm_fig(r.tn, r.fp, r.fn, r.tp, title=f"{me} · {SPLIT_LABEL[sp]}"), width="stretch")
    sc = pd.read_csv(EVAL_DIR / "module_a_synthetic_by_scenario.csv")
    piv = sc[sc.method == me].pivot(index="scenario", columns="split", values="flag_rate").reindex(columns=list(SPLIT_LABEL))
    st.markdown("**Flag rate by planted scenario** (healthy rows = false-flag rate)")
    st.dataframe(piv.rename(columns=SPLIT_LABEL), width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in SPLIT_LABEL.values()})
    ls = a[a.method == "lot_shift_check"][["split", "planted_lot_degradation", "lot_parameter_groups", "flag_rate"]]
    st.markdown("**Historical lot-shift check** (lot × parameter groups flagged `A_LOT_SHIFT`)")
    st.dataframe(ls.assign(split=ls.split.map(SPLIT_LABEL)), hide_index=True)
    st.info("Finding: on validation lots the simple robust rule ranked planted anomalies as well as Isolation "
            "Forest and better than ECOD or the percentile ensemble at equal flag rates, so it is the default "
            "decision rule. The ML detectors remain as baselines and corroborating evidence.")

with t_b:
    st.warning("**SYNTHETIC evidence only.**", icon="🧪")
    b = pd.read_csv(EVAL_DIR / "module_b_synthetic_metrics.csv")
    par = st.selectbox("Parameter", sorted(b.parameter.unique()), index=sorted(b.parameter.unique()).index("leakage_current"))
    g = b[b.parameter == par]
    unit = g.unit.iloc[0]
    long = g.melt(id_vars=["split"], value_vars=["MAE", "persistence_MAE", "linear_extrapolation_MAE"],
                  var_name="model", value_name="mae")
    long["split"] = long.split.map(SPLIT_LABEL)
    st.plotly_chart(px.bar(long, x="split", y="mae", color="model", barmode="group", height=340,
                           labels={"mae": f"MAE [{unit}]"}, title=f"{par}: 168 h forecast error [{unit}]"),
                    width="stretch")
    st.dataframe(g.assign(split=g.split.map(SPLIT_LABEL))[
        ["split", "unit", "n", "n_scored", "n_escalated_missing", "n_out_of_domain", "MAE", "median_AE",
         "persistence_MAE", "linear_extrapolation_MAE", "interval_coverage", "coverage_in_domain",
         "whole_lot_coverage", "n_lots", "mean_interval_width", "coverage_target"]],
        hide_index=True, width="stretch")
    st.markdown("**All parameters — coverage (target 0.90 whole-lot on exchangeable lots)**")
    cov = b.pivot(index="parameter", columns="split", values="whole_lot_coverage").reindex(columns=list(SPLIT_LABEL))
    st.dataframe(cov.rename(columns=SPLIT_LABEL), width="stretch")
    st.caption("Shifted lots violate the exchangeability assumption: coverage there drops to about 0.5 and MAE rises "
               "sharply (the shifted lots' inputs are often outside the training range and are flagged "
               "B_OUT_OF_DOMAIN, so they cannot PASS). Edge lots have only 3–12 devices. Intervals are wide by "
               "design: the lot-level conformal margin is the largest per-lot miss over 15 calibration lots.")

with t_d:
    st.warning("**SYNTHETIC evidence only.** Device-level decisions of the full pipeline vs generator truth.", icon="🧪")
    ct = pd.read_csv(EVAL_DIR / "decision_synthetic_confusion.csv")
    for split in ["test", "ood_test", "edge_test"]:
        g = ct[ct.split == split].set_index("truth")[["PASS", "ESCALATE", "FAIL"]]
        fig = go.Figure(go.Heatmap(z=g.values, x=g.columns, y=g.index, text=g.values, texttemplate="%{text}",
                                   colorscale="Oranges", showscale=False))
        fig.update_layout(height=250, title=SPLIT_LABEL[split], margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig, width="stretch")
    st.markdown("""
* **Escapes** = `latent_defect_within_spec` devices that PASS. Most are delayed/step defects whose change begins
  after 24 h; no 24 h method can see them.
* No device whose true 168 h value is out of specification received PASS in these splits.
* Healthy devices escalated on ordinary lots are dominated by **RDS_on interval crossings**: the calibrated
  interval (≈ ±0.12 Ω) is wide relative to the gap between typical values (~0.35 Ω) and the 0.5 Ω demo limit.
  Under the default policy an interval crossing a limit must ESCALATE; narrowing it needs more calibration lots
  or a less conservative calibration target, not tuning on test lots.
""")

with t_s:
    st.info("**REAL process data (UCI SECOM, CC BY 4.0).** A separate manufacturing anomaly benchmark with 590 "
            "anonymous features; it is not burn-in data and calendar dates are not lots.", icon="📐")
    s = json.loads((EVAL_DIR / "secom_results.json").read_text())
    st.markdown(f"Protocol: {s['protocol']}  \n**Caveat:** {s['caveat']}")
    rows = []
    for split, res in s["results"].items():
        for m, v in res.items():
            rows.append(dict(split=split, method=m + (" (preserved baseline)" if v["baseline"] else ""), **{
                k: v[k] for k in ("n", "n_fail", "tp", "fp", "fn", "tn", "recall", "false_positive_rate", "roc_auc",
                                  "average_precision")}))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    t = s["results"]["test"]
    cols = st.columns(3)
    for c, (m, v) in zip(cols, t.items()):
        c.plotly_chart(cm_fig(v["tn"], v["fp"], v["fn"], v["tp"], labels=("pass", "fail"), title=f"test · {m}"),
                       width="stretch")
    st.error("The preserved Isolation Forest baseline detects **0 of 23** failed test examples (30 false alarms), "
             "reproduced exactly. ECOD under the same pre-declared protocol also detects 0 of 23. We did not tune "
             "against the test labels to manufacture a better number.")

with t_r:
    st.info("**REAL component cohorts** (NASA radiation reports, capacitor aging). Exploratory electrical-drift "
            "analysis: there is no 168 h burn-in endpoint and there are no latent-defect labels, so these are "
            "descriptive counts, not accuracy.", icon="📐")
    r = json.loads((EVAL_DIR / "real_exploratory.json").read_text())
    st.caption(f"Data scope: {r['scope']} · {r['note']}")
    st.markdown("**Default (minimum 8 peers)**")
    st.dataframe(pd.DataFrame(r["default_min_peers_8"]).fillna(0), hide_index=True, width="stretch")
    st.markdown("**Sensitivity: minimum 5 peers** (exploratory only)")
    st.dataframe(pd.DataFrame(r["sensitivity_min_peers_5"]).fillna(0), hide_index=True, width="stretch")
    rr = pd.DataFrame(r["reram_standby_current_vs_reported_functional_failure"])
    if len(rr):
        st.markdown("**ReRAM: standby-current peer status vs reported functional-failure interval** (dose in "
                    "krad(Si); functional outcomes are interval-censored and are not latent-defect truth)")
        st.dataframe(rr, hide_index=True, width="stretch")
