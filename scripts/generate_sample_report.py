"""Generate the committed sample outputs from the demo batch.

    python scripts/generate_sample_report.py

Writes docs/sample_report/: PDF report, device/parameter CSV, results JSON and
the audit-log lines produced by the run (written to a temporary runtime folder
so the real audit log is not touched). Contents are SYNTHETIC demo evidence.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "sample_report"


def main() -> None:
    (ROOT / "runtime").mkdir(exist_ok=True)  # git-ignored scratch on the repo drive
    with tempfile.TemporaryDirectory(dir=ROOT / "runtime") as tmp:
        os.environ["DRISHTI_RUNTIME_DIR"] = tmp
        import pandas as pd

        from drishti import audit
        from drishti.config import DATA_FIXTURES
        from drishti.pipeline import export_frames, results_json, run_screening
        from drishti.report import build_pdf

        src = DATA_FIXTURES / "demo_batch_SYNTHETIC.csv"
        raw_bytes = src.read_bytes()
        run = run_screening(pd.read_csv(src), raw_bytes=raw_bytes, source_name=src.name)
        audit.record_run(run)
        pdf = build_pdf(run, focus_device="SYN_L159_D009", max_detail=8)
        audit.append("REPORT_EXPORTED", {"run_id": run.run_id, "format": "pdf", "bytes": len(pdf)})
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "drishti_sample_report_SYNTHETIC.pdf").write_bytes(pdf)
        fr = export_frames(run)
        fr["devices"].to_csv(OUT / "drishti_sample_devices_SYNTHETIC.csv", index=False)
        fr["parameters"].to_csv(OUT / "drishti_sample_parameters_SYNTHETIC.csv", index=False)
        (OUT / "drishti_sample_results_SYNTHETIC.json").write_text(results_json(run), encoding="utf-8")
        (OUT / "drishti_sample_audit_log.jsonl").write_bytes(audit.audit_path().read_bytes())
        print(f"run {run.run_id}: {run.summary()['device_decisions']}; PDF {len(pdf) / 1024:.0f} KiB -> {OUT}")


if __name__ == "__main__":
    main()
