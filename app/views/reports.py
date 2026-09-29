import pandas as pd
import streamlit as st

from ui import current_run, provenance_banner, require_run

from drishti import audit
from drishti.pipeline import export_frames, results_json
from drishti.report import build_pdf

run = require_run()
st.title("Reports & audit")
provenance_banner(run.provenance_class)

st.subheader(f"Export run `{run.run_id}`")
devs = run.devices.device_id.tolist()
flagged = run.devices[run.devices.decision != "PASS"].device_id.tolist()
focus = st.selectbox("Device to feature first in the PDF (optional)", ["(none)"] + devs,
                     index=(devs.index("SYN_L159_D009") + 1) if "SYN_L159_D009" in devs else 0)
max_detail = st.slider("Flagged devices with full evidence pages", 1, max(1, min(60, len(flagged) or 1)),
                       min(20, max(1, len(flagged))))


@st.cache_data(show_spinner="Building PDF ...", max_entries=4)
def _pdf(run_id: str, focus: str, n: int) -> bytes:
    return build_pdf(current_run(), focus_device=None if focus == "(none)" else focus, max_detail=n)


fr = export_frames(run)
c1, c2, c3, c4 = st.columns(4)
if c1.button("Build PDF report", type="primary", width="stretch"):
    st.session_state["pdf"] = (run.run_id, _pdf(run.run_id, focus, max_detail))
    audit.append("REPORT_EXPORTED", {"run_id": run.run_id, "format": "pdf", "focus_device": focus,
                                     "input_sha256": run.input_sha256})
if st.session_state.get("pdf", (None,))[0] == run.run_id:
    c1.download_button("⬇ Download PDF", st.session_state["pdf"][1], f"drishti_report_{run.run_id}.pdf",
                       "application/pdf", width="stretch")
c2.download_button("⬇ Devices CSV", fr["devices"].to_csv(index=False).encode(), f"drishti_devices_{run.run_id}.csv",
                   "text/csv", width="stretch")
c3.download_button("⬇ Parameters CSV", fr["parameters"].to_csv(index=False).encode(),
                   f"drishti_parameters_{run.run_id}.csv", "text/csv", width="stretch")
c4.download_button("⬇ Full results JSON", results_json(run).encode(), f"drishti_results_{run.run_id}.json",
                   "application/json", width="stretch")

st.divider()
st.subheader("Audit history")
st.markdown("Each screening run is appended to `runtime/audit/audit_log.jsonl` with run ID, input hash, model "
            "version, specification revision, timestamp and reason codes. Each line stores the SHA-256 of the "
            "previous line, so accidental edits or truncation are **detectable**. This is ordinary file logging, "
            "**not tamper-proof**: anyone with write access could rewrite the file and its chain.")
ok, msg = audit.verify_chain()
(st.success if ok else st.error)(msg)
log = audit.read()
if log:
    t = pd.DataFrame([{"logged_utc": e["logged_utc"], "event": e["event"], "run_id": e.get("run_id"),
                       "source": e.get("source_name", ""), "provenance": e.get("provenance_class", ""),
                       "input_sha256": (e.get("input_sha256") or "")[:16],
                       "model_version": e.get("model_version", ""), "spec_revision": e.get("spec_revision", ""),
                       "decisions": e.get("device_decisions", ""), "format": e.get("format", "")} for e in log])
    st.dataframe(t.iloc[::-1], hide_index=True, width="stretch")
    run_ids = [e.get("run_id") for e in log if e["event"] == "SCREENING_RUN"][::-1]
    sel = st.selectbox("Inspect entry (run ID)", run_ids) if run_ids else None
    entry = next((e for e in reversed(log) if e.get("run_id") == sel and e["event"] == "SCREENING_RUN"), None)
    if entry is not None:
        with st.expander("Entry JSON", expanded=False):
            st.json({k: v for k, v in entry.items() if k != "devices"})
            st.caption(f"{len(entry.get('devices', []))} per-device decisions stored in the entry.")
    st.download_button("⬇ Audit log (JSONL)", audit.audit_path().read_bytes(), "drishti_audit_log.jsonl",
                       "application/json")
else:
    st.info("Audit log is empty.")
