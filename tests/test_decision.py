"""Decision table precedence and device aggregation."""
import numpy as np
import pandas as pd
import pytest

from drishti.decision import decide, decide_parameter

CFG = {"predicted_violation_action": "FAIL"}


@pytest.mark.parametrize("codes,expected,rule", [
    (["B_INTERVAL_INSIDE"], "PASS", "R5"),
    (["B_INTERVAL_CROSSES_LIMIT"], "ESCALATE", "R4"),                         # interval crossing -> ESCALATE
    (["A_UNUSUAL_CHANGE", "B_INTERVAL_INSIDE"], "ESCALATE", "R4"),            # forecast PASS can't erase A warning
    (["A_LOT_SHIFT", "B_INTERVAL_INSIDE"], "ESCALATE", "R4"),
    (["SPEC_OBSERVED_FAIL", "B_INTERVAL_INSIDE"], "FAIL", "R1"),              # observed failure first
    (["B_PREDICTED_VIOLATION"], "FAIL", "R2"),
    (["B_PREDICTED_VIOLATION", "B_OUT_OF_DOMAIN"], "ESCALATE", "R3"),         # no predicted FAIL by extrapolation
    (["B_PREDICTED_VIOLATION", "DQ_IDENTICAL_READINGS"], "ESCALATE", "R3"),
    (["DQ_MISSING_INPUT", "B_NOT_RUN_MISSING_INPUT"], "ESCALATE", "R3"),
    (["A_INSUFFICIENT_PEERS", "B_INTERVAL_INSIDE"], "ESCALATE", "R3"),
    (["A_ZERO_MAD", "B_INTERVAL_INSIDE"], "ESCALATE", "R3"),
    (["B_UNSUPPORTED_PART"], "ESCALATE", "R3"),
    (["DEMO_SPEC_NOT_APPROVED"], "ESCALATE", "R6"),                           # no evidence -> never PASS
])
def test_parameter_rules(codes, expected, rule):
    d, basis, r, _ = decide_parameter(codes, CFG)
    assert (d, r) == (expected, rule)


def test_observed_and_predicted_bases_differ():
    assert decide_parameter(["SPEC_OBSERVED_FAIL"], CFG)[1] == "OBSERVED"
    assert decide_parameter(["B_PREDICTED_VIOLATION"], CFG)[1] == "PREDICTED"


def _row(dev, param, v0, v24, lo, hi, a=(), b=(), ic="B_INTERVAL_INSIDE"):
    return dict(device_id=dev, lot_id="L", part_number="DEMO_MOSFET", parameter=param, value_0h=v0, value_24h=v24,
                spec_lower=lo, spec_upper=hi, row_flags=[], a_codes=list(a), b_codes=list(b), interval_code=ic,
                provenance_class="SYNTHETIC")


def test_device_aggregation(registry):
    p = pd.DataFrame([
        _row("D1", "RDS_on", .3, .31, 0, .5), _row("D1", "threshold_voltage", 1.8, 1.8, 1, 2.5),       # PASS
        _row("D2", "RDS_on", .3, .31, 0, .5), _row("D2", "threshold_voltage", 1.8, 2.6, 1, 2.5),       # observed FAIL
        _row("D3", "RDS_on", .3, .31, 0, .5, a=["A_UNUSUAL_CHANGE"]),
        _row("D3", "threshold_voltage", 1.8, 1.8, 1, 2.5),                                            # ESCALATE
        _row("D4", "RDS_on", .3, .31, 0, .5),                                                         # missing param
    ])
    params, dev = decide(p, registry, CFG)
    d = dev.set_index("device_id")
    assert d.loc["D1", "decision"] == "PASS"
    assert d.loc["D2", "decision"] == "FAIL" and d.loc["D2", "basis"] == "OBSERVED"
    assert d.loc["D3", "decision"] == "ESCALATE"
    assert d.loc["D4", "decision"] == "ESCALATE"
    assert "DEVICE_INCOMPLETE_PARAMETER_SET" in d.loc["D4", "reason_codes"]
    assert params.loc[3, "reason_codes"][0] == "SPEC_OBSERVED_FAIL"
    assert (dev.stage.str.contains("not qualification completion")).all()


def test_missing_reading_still_checks_observed_value(registry):
    p = pd.DataFrame([_row("D1", "RDS_on", 0.9, np.nan, 0, .5, ic="")])
    params, _ = decide(p, registry, CFG)
    assert params.decision.iloc[0] == "FAIL" and params.basis.iloc[0] == "OBSERVED"
