"""Decision engine: one documented table combining both modules.

Parameter level (first matching rule wins; ALL codes are still reported)
------------------------------------------------------------------------
R1  SPEC_OBSERVED_FAIL                        -> FAIL      basis OBSERVED
R2  B_PREDICTED_VIOLATION, forecast usable    -> FAIL      basis PREDICTED
    (usable = no B_OUT_OF_DOMAIN, DQ_IDENTICAL_READINGS,
     A_UNTRUSTED_SOURCE_FLAG or B_UNIT_MISMATCH)
    (action configurable: decision.predicted_violation_action)
R3  any BLOCK code (evidence gap)             -> ESCALATE  basis INSUFFICIENT_EVIDENCE
R4  any WARN code (interval crossing, Module A
    unusual/lot shift/spread)                 -> ESCALATE  basis REVIEW
R5  B_INTERVAL_INSIDE and nothing above       -> PASS      basis EARLY_SCREEN
R6  otherwise                                 -> ESCALATE  basis INSUFFICIENT_EVIDENCE

Consequences: an interval crossing a limit is ESCALATE; a forecast-only PASS
cannot erase a Module A warning (R4 precedes R5); predicted and observed
failures are reported with different bases.

Device level
------------
FAIL      if any parameter is FAIL (basis OBSERVED if any observed failure)
ESCALATE  otherwise if any parameter is ESCALATE or a registered parameter is
          missing (DEVICE_INCOMPLETE_PARAMETER_SET)
PASS      only if every registered parameter is PASS

FAIL vs ESCALATE: FAIL = evidence that a limit is (or, with the calibrated
interval entirely beyond it, will be) violated. ESCALATE = evidence is
insufficient or suspicious; the device goes to extended burn-in / engineering
review. At 24 h every outcome is an EARLY SCREENING recommendation, not
qualification completion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import reasons

PASS, FAIL, ESCALATE = "PASS", "FAIL", "ESCALATE"
# A predicted FAIL may not rest on extrapolation or suspect measurements.
UNRELIABLE_FORECAST = {"B_OUT_OF_DOMAIN", "DQ_IDENTICAL_READINGS", "A_UNTRUSTED_SOURCE_FLAG", "B_UNIT_MISMATCH"}
STAGE_LABEL = "EARLY SCREENING (24 h) - not qualification completion"


def observed_spec_fail(row) -> bool:
    lo, hi = row.spec_lower, row.spec_upper
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False
    vals = [v for v in (row.value_0h, row.value_24h) if np.isfinite(v)]
    return any(v < lo or v > hi for v in vals)


def decide_parameter(codes: list[str], cfg: dict) -> tuple[str, str, str, list[str]]:
    """Return (decision, basis, rule_id, decisive_codes)."""
    roles = {c: reasons.role(c) for c in codes}
    if "SPEC_OBSERVED_FAIL" in codes:
        return FAIL, "OBSERVED", "R1", ["SPEC_OBSERVED_FAIL"]
    blocks = [c for c, r in roles.items() if r == "BLOCK"]
    if "B_PREDICTED_VIOLATION" in codes and not UNRELIABLE_FORECAST.intersection(codes):
        action = cfg.get("predicted_violation_action", FAIL)
        return action, "PREDICTED", "R2", ["B_PREDICTED_VIOLATION"]
    if blocks:
        return ESCALATE, "INSUFFICIENT_EVIDENCE", "R3", blocks
    warns = [c for c, r in roles.items() if r == "WARN"]
    if warns:
        return ESCALATE, "REVIEW", "R4", warns
    if "B_INTERVAL_INSIDE" in codes:
        return PASS, "EARLY_SCREEN", "R5", ["B_INTERVAL_INSIDE"]
    return ESCALATE, "INSUFFICIENT_EVIDENCE", "R6", []


def decide(params: pd.DataFrame, registry, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """params must contain value_0h/24h, spec limits, row_flags, a_codes, b_codes,
    interval_code. Adds decision columns; returns (params, devices)."""
    p = params.copy()
    all_codes, dec, basis, rule, decisive = [], [], [], [], []
    for _, row in p.iterrows():
        codes = list(row.row_flags) + list(row.a_codes) + list(row.b_codes)
        if isinstance(row.interval_code, str) and row.interval_code:
            codes.append(row.interval_code)
        if observed_spec_fail(row):
            codes.insert(0, "SPEC_OBSERVED_FAIL")
        codes = list(dict.fromkeys(codes))
        d, b, r, dc = decide_parameter(codes, cfg)
        all_codes.append(codes); dec.append(d); basis.append(b); rule.append(r); decisive.append(dc)
    p["reason_codes"], p["decision"], p["basis"], p["rule"], p["decisive_codes"] = all_codes, dec, basis, rule, decisive

    devices = []
    for dev, g in p.groupby("device_id", sort=False):
        part = g.part_number.iloc[0]
        expected = set(registry.parameters_for(part)) if registry is not None else set()
        missing = sorted(expected - set(g.parameter)) if expected else []
        codes = []
        if (g.decision == FAIL).any():
            d = FAIL
            b = "OBSERVED" if ((g.decision == FAIL) & (g.basis == "OBSERVED")).any() else "PREDICTED"
        elif (g.decision == ESCALATE).any() or missing:
            d, b = ESCALATE, "REVIEW" if (g.basis == "REVIEW").any() else "INSUFFICIENT_EVIDENCE"
            if missing:
                codes.append("DEVICE_INCOMPLETE_PARAMETER_SET")
        else:
            d, b = PASS, "EARLY_SCREEN"
        for c in g.reason_codes:
            codes.extend(c)
        codes = list(dict.fromkeys(codes))
        drivers = g[g.decision == d].parameter.tolist()
        devices.append(dict(device_id=dev, lot_id=g.lot_id.iloc[0], part_number=part, decision=d, basis=b,
                            n_parameters=len(g), missing_parameters=",".join(missing),
                            driving_parameters=",".join(drivers), reason_codes=codes,
                            provenance_class=g.provenance_class.iloc[0], stage=STAGE_LABEL))
    return p, pd.DataFrame(devices)
