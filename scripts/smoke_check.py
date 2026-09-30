"""Verify shipped models, demo decisions and real-data loading without starting a server."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from drishti import data  # noqa: E402
from drishti.pipeline import run_screening  # noqa: E402


def main():
    source = ROOT / "data/fixtures/demo_batch_SYNTHETIC.csv"
    run = run_screening(pd.read_csv(source), raw_bytes=source.read_bytes(), source_name=source.name)
    assert run.ok, run.validation.as_dict()
    summary = run.summary()
    assert summary["n_devices"] == 147, summary
    assert summary["device_decisions"] == {"PASS": 50, "FAIL": 1, "ESCALATE": 96}, summary
    decisions = run.devices.set_index("device_id").decision
    for device, expected in {"SYN_L150_D000": "PASS", "SYN_L159_D009": "ESCALATE",
                             "SYN_L152_D000": "ESCALATE", "SYN_L180_D020": "FAIL",
                             "SYN_L192_D000": "ESCALATE"}.items():
        assert decisions[device] == expected, (device, decisions[device])
    real, scope = data.real_drift_view()
    assert not real.empty
    print(json.dumps({"demo": "OK", "devices": 147, "decisions": summary["device_decisions"],
                      "real_data_scope": scope, "real_rows": len(real),
                      "spec_revision": run.spec_revision}, indent=2))


if __name__ == "__main__":
    main()
