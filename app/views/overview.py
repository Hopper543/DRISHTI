import pandas as pd
import streamlit as st

from ui import current_run, load_demo, models, provenance_banner, registry, settings

from drishti.decision import STAGE_LABEL
from drishti.provenance import DATASETS

st.title("DRISHTI — early burn-in screening")
st.markdown("**Smart India Hackathon 2026 · Problem Statement 26170 (ISRO) · Team Agamemnon** — "
            "*AI-Driven Anomaly Detection in Component Burn-In & Screening*")

st.warning("**Screening-round prototype.** The shipped models and specification limits are built on "
           "**SYNTHETIC** demonstration data. Nothing here is a qualified acceptance decision, and 24 h "
           "recommendations are early screening decisions, not qualification completion.", icon="⚠️")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Module A · peers")
    st.markdown("Compares each device with **comparable devices from its own lot** (same part, parameter, unit, "
                "test condition): the 24 h value and the change since 0 h, as robust median/MAD scores. "
                "Isolation Forest, ECOD and an ensemble are kept as baselines. Whole-lot shifts are checked "
                "against historical lots.")
with c2:
    st.subheader("Module B · forecast")
    st.markdown("Predicts the **168 h value** from exactly two inputs, `value_0h` and `value_24h`, using per "
                "part/parameter LightGBM quantile models. The **interval is calibrated on held-out lots** "
                "(lot-level conformal). Unsupported, invalid or out-of-range inputs are escalated.")
with c3:
    st.subheader("Decision engine")
    st.markdown("One documented table combines observed limit checks, peer evidence, forecast uncertainty and "
                "data quality into **PASS / FAIL / ESCALATE** with reason codes. An interval crossing a limit "
                "escalates; a forecast PASS never hides a peer warning.")

st.divider()
run = current_run()
if run is None or not run.ok:
    st.subheader("Try it")
    st.markdown("The demo loads five whole held-out synthetic lots (a normal lot, early abnormal drift, an interval "
                "crossing a limit, a shifted lot, a 3-device lot with a missing 24 h reading) and one clearly "
                "labelled illustrative quantised lot (zero MAD). No data preparation is needed.")
    if st.button("▶ Load demo", type="primary"):
        load_demo()
        st.rerun()
else:
    provenance_banner(run.provenance_class)
    s = run.summary()
    st.subheader(f"Current run `{run.run_id}`")
    cols = st.columns(4)
    cols[0].metric("Devices", s["n_devices"])
    for col, d in zip(cols[1:], ["PASS", "ESCALATE", "FAIL"]):
        col.metric(d, s["device_decisions"].get(d, 0))
    st.caption(f"{STAGE_LABEL} · source {run.source_name} · input SHA-256 {run.input_sha256[:16]}…")
    st.markdown("Guided demo cases: open **Device analysis** and pick "
                + ", ".join(f"`{d}`" for d in ["SYN_L150_D000", "SYN_L159_D009", "SYN_L152_D000", "SYN_L192_D000",
                                               "SYN_L180_D000", "ILLUSTRATIVE_Q155_D000"])
                + ".")

st.divider()
st.subheader("Models and specification registry")
m = models()
mb, ma = m.module_b, m.module_a
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Model version** `{m.version}`")
    if mb:
        st.markdown(f"- Module B: {mb.manifest['model_family']}; features `{mb.manifest['features']}`; target "
                    f"`{mb.manifest['target']}`; evidence **{mb.manifest['evidence']}**; trained "
                    f"{mb.manifest.get('created_utc', '?')}")
    if ma:
        ca = settings().module_a
        st.markdown(f"- Module A: flag rule **{ca['flag_rule']}** (robust |z| ≥ {ca['robust_z_threshold']}, "
                    f"min {ca['min_peers']} peers); "
                    f"detectors fitted on {ma.meta['detector_fit_population'].split(':')[0]}; "
                    f"normalised on {ma.meta['normalisation_population'].split('(')[0].strip()}")
        st.caption(ma.meta["caveat"])
with c2:
    reg = registry()
    st.markdown(f"**Specification registry** `{reg.revision}` · production approved: "
                f"**{reg.production_approved}**")
    st.dataframe(reg.table[["part_number", "parameter", "unit", "spec_lower", "spec_upper", "spec_origin"]],
                 hide_index=True, width="stretch", height=280)

st.subheader("Data provenance")
st.markdown("Every table and result carries one evidence class: **REAL_MEASURED**, **REAL_DERIVED** or "
            "**SYNTHETIC**. They are never merged into one training matrix or one accuracy number.")
rows = [{"dataset": k, "evidence": v["evidence"], "source": v["title"], "in repository": "yes" if v["in_repo"]
         else "local only", "reuse basis": v["redistribution"], "note": v["note"], "url": v["url"] or ""}
        for k, v in DATASETS.items()]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
             column_config={"url": st.column_config.LinkColumn("url", display_text="source")})
