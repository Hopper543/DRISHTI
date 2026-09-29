"""End-to-end: load demo -> validate -> analyze -> predict -> explain -> export
report -> inspect audit entry. Also production gate and audit chain."""
import json

import numpy as np
import pytest

from drishti import audit
from drishti.explain import parameter_evidence, parameter_sentences
from drishti.pipeline import ProductionGateError, export_frames, results_json, run_screening
from drishti.report import build_pdf


@pytest.fixture(scope="module")
def run(demo_raw):
    return run_screening(demo_raw, source_name="demo_batch_SYNTHETIC.csv")


def test_end_to_end_demo(run, tmp_path):
    assert run.ok and run.provenance_class == "SYNTHETIC"
    assert set(run.devices.decision) <= {"PASS", "FAIL", "ESCALATE"}
    p = run.params.set_index("sample_id")
    # named demo cases (values verified against the dataset files)
    d9 = p.loc["SYN_L159_D009:leakage_current"]
    assert d9.value_0h == pytest.approx(10.1304, abs=1e-3) and d9.value_24h == pytest.approx(11.2391, abs=1e-3)
    assert "A_UNUSUAL_CHANGE" in d9.reason_codes and d9.decision == "ESCALATE"
    assert d9.lower >= d9.spec_lower and d9.upper <= d9.spec_upper  # forecast alone would have passed
    r = p.loc["SYN_L152_D000:RDS_on"]
    assert r.lower < r.spec_upper < r.upper and "B_INTERVAL_CROSSES_LIMIT" in r.reason_codes
    assert r.decision == "ESCALATE"
    assert "DQ_MISSING_INPUT" in p.loc["SYN_L192_D000:leakage_current"].reason_codes
    assert all("A_INSUFFICIENT_PEERS" in c for c in p[p.lot_id == "SYN_L192"].reason_codes if "DQ_MISSING_INPUT" not in c)
    assert all("A_ZERO_MAD" in c for c in p[(p.lot_id == "ILLUSTRATIVE_Q155") & (p.parameter == "threshold_voltage")].reason_codes)
    assert p[p.lot_id == "SYN_L180"].reason_codes.map(lambda c: "A_LOT_SHIFT" in c).all()
    assert (run.devices[run.devices.lot_id == "SYN_L150"].decision == "PASS").any()

    # explain
    ev = dict(parameter_evidence(d9))
    assert "Peer median change" in ev and "Calibrated 90% interval" in ev
    assert any("A_UNUSUAL_CHANGE" in s for s in parameter_sentences(d9))

    # export
    fr = export_frames(run)
    assert len(fr["devices"]) == run.devices.device_id.nunique()
    doc = json.loads(results_json(run))
    assert doc["summary"]["run_id"] == run.run_id and doc["summary"]["spec_revision"] == "DEMO_ONLY_v1"
    pdf = build_pdf(run, focus_device="SYN_L159_D009", max_detail=3)
    assert pdf[:5] == b"%PDF-" and len(pdf) > 10_000

    # audit
    entry = audit.record_run(run)
    audit.append("REPORT_EXPORTED", {"run_id": run.run_id, "format": "pdf", "bytes": len(pdf)})
    log = audit.read()
    assert [e["event"] for e in log] == ["SCREENING_RUN", "REPORT_EXPORTED"]
    first = log[0]
    for k in ("run_id", "input_sha256", "model_version", "spec_revision", "timestamp_utc", "reason_code_counts"):
        assert first[k] not in (None, "", {})
    assert first["input_sha256"] == run.input_sha256 and entry["run_id"] == run.run_id
    assert audit.verify_chain()[0]


def test_audit_chain_detects_edit():
    audit.append("A", {"x": 1})
    audit.append("B", {"x": 2})
    audit.append("C", {"x": 3})
    path = audit.audit_path()
    lines = path.read_text().splitlines()
    lines[1] = lines[1].replace('"x": 2', '"x": 9')
    path.write_text("\n".join(lines) + "\n")
    ok, msg = audit.verify_chain()
    assert not ok and "line 3" in msg


def test_invalid_upload_produces_no_decisions(demo_raw):
    bad = demo_raw.assign(value_168h=1.0)
    run = run_screening(bad)
    assert not run.ok and run.devices.empty


def test_production_gate_refuses_synthetic(demo_raw):
    with pytest.raises(ProductionGateError):
        run_screening(demo_raw, mode="production")
