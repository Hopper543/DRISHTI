import io

import pandas as pd
import streamlit as st

from ui import DEMO_FILE, current_run, execute, load_demo, provenance_banner

from drishti.config import DATA_FIXTURES
from drishti.schema import FEATURE_COLUMNS, FORBIDDEN, OPTIONAL, REQUIRED

st.title("Upload & validate")
st.markdown("Upload **early-input** readings (one row per device and parameter, 0 h and 24 h only). The "
            "specification registry supplies limits; files may repeat them only if they agree. Real component "
            "files in the prepared long format use the separate **Real-data explorer** route.")

c1, c2, c3 = st.columns(3)
c1.download_button("⬇ Blank template (CSV)", (DATA_FIXTURES / "early_input_blank.csv").read_bytes(),
                   "drishti_early_input_blank.csv", "text/csv", width="stretch")
c2.download_button("⬇ Example upload — SYNTHETIC lot", (DATA_FIXTURES / "example_upload_SYNTHETIC.csv").read_bytes(),
                   "drishti_example_upload_SYNTHETIC.csv", "text/csv", width="stretch")
c3.download_button("⬇ Full demo batch — SYNTHETIC", DEMO_FILE.read_bytes(), DEMO_FILE.name, "text/csv",
                   width="stretch")

with st.expander("Input contract", expanded=False):
    st.markdown(f"""
* **Required columns:** `{', '.join(REQUIRED)}`
* **Optional:** `{', '.join(OPTIONAL)}` (`test_condition` separates peer groups measured under different
  conditions; `quality_flags` marks rows that must not receive an automated peer verdict)
* **Model inputs:** exactly `{FEATURE_COLUMNS}`. IDs, lots, limits and split names are never features.
* **Rejected:** endpoint/label columns ({', '.join(sorted(FORBIDDEN))}) and any `value_<t>h` with t > 24.
* **Units:** SI-prefix conversions of the registry unit are applied and recorded (e.g. nA → uA); anything else
  must match exactly.
* **Missing readings:** leave blank. They are escalated, never imputed. Text in a numeric column is an error.
* **Identity:** unique `sample_id`; one lot per device; one part number per lot; one provenance class per file.
""")

up = st.file_uploader("CSV file", type=["csv"])
cc1, cc2 = st.columns([1, 3])
if cc1.button("▶ Load demo instead", type="secondary"):
    load_demo()
    st.rerun()

if up is not None:
    raw_bytes = up.getvalue()
    try:
        raw = pd.read_csv(io.BytesIO(raw_bytes), dtype={"device_id": str, "lot_id": str, "sample_id": str})
    except Exception as exc:  # noqa: BLE001 - shown to the user
        st.error(f"Could not parse the file as CSV: {exc}. Save it as UTF-8 comma-separated values.")
        st.stop()
    st.markdown(f"**{up.name}** — {len(raw)} rows × {len(raw.columns)} columns")
    st.dataframe(raw.head(20), width="stretch", hide_index=True)
    if st.button("Validate and analyse", type="primary"):
        execute(raw, raw_bytes, up.name)

run = current_run()
if run is not None:
    st.divider()
    st.subheader(f"Validation result · {run.source_name}")
    v = run.validation
    if not v.ok:
        st.error("**Upload refused — nothing was scored.** Fix the items below and upload again.")
        for e in v.errors:
            st.markdown(f"- ❌ {e}")
    else:
        provenance_banner(run.provenance_class)
        st.success(f"Validated and analysed. Run `{run.run_id}` recorded in the audit log.")
    for w in v.warnings:
        st.markdown(f"- ⚠️ {w}")
    for i in v.info:
        st.markdown(f"- ℹ️ {i}")
    if v.ok:
        f = v.frame
        dq = pd.DataFrame({
            "check": ["rows", "devices", "lots", "parameters", "missing 0 h/24 h readings", "identical 0 h/24 h",
                      "unit conversions", "unknown part/parameter", "source quality flags"],
            "value": [len(f), f.device_id.nunique(), f.lot_id.nunique(), f.parameter.nunique(),
                      int(f.row_flags.map(lambda x: "DQ_MISSING_INPUT" in x).sum()),
                      int(f.row_flags.map(lambda x: "DQ_IDENTICAL_READINGS" in x).sum()),
                      int(f.row_flags.map(lambda x: "DQ_UNIT_CONVERTED" in x).sum()),
                      int(f.row_flags.map(lambda x: "SPEC_UNKNOWN" in x).sum()),
                      int(f.row_flags.map(lambda x: "DQ_SOURCE_QUALITY_FLAG" in x).sum())]})
        st.dataframe(dq, hide_index=True)
        st.page_link("views/lot.py", label="Continue to Lot analysis →", icon=":material/arrow_forward:")
