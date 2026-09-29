"""Shared dashboard helpers. All analysis logic lives in the drishti package."""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from drishti import EVIDENCE_NOTICE, __version__, audit  # noqa: E402
from drishti.config import DATA_FIXTURES, load_settings  # noqa: E402
from drishti.pipeline import load_models, run_screening  # noqa: E402
from drishti.provenance import PROVENANCE_BADGE  # noqa: E402
from drishti.specs import load_registry  # noqa: E402

DECISION_COLORS = {"PASS": "#1b7f3b", "ESCALATE": "#b86e00", "FAIL": "#b3261e"}
DEMO_FILE = DATA_FIXTURES / "demo_batch_SYNTHETIC.csv"


@st.cache_resource(show_spinner="Loading models ...")
def models():
    return load_models()


@st.cache_resource
def registry():
    return load_registry()


def settings():
    return load_settings()


def current_run():
    return st.session_state.get("run")


def execute(raw: pd.DataFrame, raw_bytes: bytes | None, source_name: str) -> None:
    """Run the pipeline, store in session, append the audit record."""
    with st.spinner("Validating, scoring peers, forecasting and deciding ..."):
        run = run_screening(raw, registry=registry(), models=models(), raw_bytes=raw_bytes, source_name=source_name)
    st.session_state["run"] = run
    st.session_state["raw"] = raw
    if run.ok:
        entry = audit.record_run(run)
        st.session_state["last_audit"] = entry


def load_demo() -> None:
    b = DEMO_FILE.read_bytes()
    execute(pd.read_csv(io.BytesIO(b)), b, DEMO_FILE.name)


def decision_badge(decision: str, basis: str | None = None) -> str:
    c = DECISION_COLORS.get(decision, "#555")
    extra = f" <span style='font-weight:400;opacity:.85'>({basis})</span>" if basis else ""
    return (f"<span style='background:{c};color:white;padding:2px 10px;border-radius:12px;font-weight:600;"
            f"font-size:0.9em'>{decision}{extra}</span>")


def provenance_banner(prov: str | None) -> None:
    if not prov:
        return
    label = PROVENANCE_BADGE.get(prov, prov)
    if prov == "SYNTHETIC":
        st.warning(f"**Evidence class: {label}.** Results demonstrate the workflow only; they are not real "
                   "defect-detection performance. Specification limits are DEMO assumptions.", icon="🧪")
    else:
        st.info(f"**Evidence class: {label}.**", icon="📐")


def sidebar_status() -> None:
    with st.sidebar:
        st.markdown(f"**DRISHTI** v{__version__} · SIH 2026 · PS 26170")
        run = current_run()
        if run is not None and run.ok:
            s = run.summary()
            st.success(f"Run `{run.run_id}` · {s['n_devices']} devices\n\n"
                       + " · ".join(f"{k} {v}" for k, v in sorted(s["device_decisions"].items())))
            st.caption(f"Source: {run.source_name or 'upload'} · {run.provenance_class}")
        elif run is not None:
            st.error("Last upload failed validation.")
        else:
            st.info("No data loaded yet.")
        if st.button("▶ Load demo", width="stretch", key="sidebar_demo"):
            load_demo()
            st.rerun()
        st.caption("Mode: **" + settings().mode + "** · spec revision **" + registry().revision + "**")
        st.caption(EVIDENCE_NOTICE)


def require_run():
    run = current_run()
    if run is None or not run.ok:
        st.info("No screening results yet. Use **▶ Load demo** in the sidebar, or upload a CSV on the "
                "*Upload & validate* page.")
        if st.button("▶ Load demo now", type="primary"):
            load_demo()
            st.rerun()
        st.stop()
    return run


def fmt(v, nd=4):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}g}"
