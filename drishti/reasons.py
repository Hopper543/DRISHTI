"""Reason-code catalogue: category, decision role and inspector-facing text.

Categories keep different evidence types apart:
  OBSERVED_FAILURE      a measured reading is outside the specification
  PREDICTED_VIOLATION   the whole calibrated 168 h interval is beyond a limit
  FORECAST_UNCERTAIN    the interval straddles a limit
  STATISTICAL           unusual relative to peers (not a failure probability)
  SUSPECTED_DEGRADATION unusual change from baseline
  LOT_LEVEL             whole-lot shift/spread vs historical lots
  MEASUREMENT           possible measurement / data problem
  INSUFFICIENT_EVIDENCE the check could not be performed reliably
  INFO                  context only
"""
from __future__ import annotations

# role: FAIL (decisive fail), BLOCK (evidence gap -> ESCALATE), WARN (-> ESCALATE),
#       OK (supports PASS), INFO (never changes the outcome)
CODES: dict[str, tuple[str, str, str]] = {
    # code: (category, role, text)
    "SPEC_OBSERVED_FAIL": ("OBSERVED_FAILURE", "FAIL",
                           "A measured reading (0 h or 24 h) is outside the specification limits."),
    "B_PREDICTED_VIOLATION": ("PREDICTED_VIOLATION", "FAIL",
                              "The entire calibrated 168 h forecast interval lies beyond a specification limit."),
    "B_INTERVAL_CROSSES_LIMIT": ("FORECAST_UNCERTAIN", "WARN",
                                 "The calibrated 168 h forecast interval crosses a specification limit."),
    "B_INTERVAL_INSIDE": ("INFO", "OK", "The calibrated 168 h forecast interval lies inside the limits."),
    "A_UNUSUAL_CHANGE": ("SUSPECTED_DEGRADATION", "WARN",
                         "Change from 0 h to 24 h is unusual relative to lot peers (re-measure to rule out a "
                         "measurement error)."),
    "A_UNUSUAL_LEVEL": ("STATISTICAL", "WARN", "The 24 h value is unusual relative to lot peers."),
    "A_HISTORICAL_UNUSUAL": ("STATISTICAL", "WARN",
                             "Unusual relative to the historical reference population (small-lot fallback)."),
    "A_LOT_SHIFT": ("LOT_LEVEL", "WARN",
                    "The lot median differs strongly from historical lots; the whole lot needs review because "
                    "peer comparison cannot detect a uniformly shifted lot."),
    "A_LOT_SPREAD": ("LOT_LEVEL", "WARN", "Lot spread is much wider than in historical lots."),
    "A_INSUFFICIENT_PEERS": ("INSUFFICIENT_EVIDENCE", "BLOCK",
                             "Fewer valid peers than the minimum; no trusted peer verdict."),
    "A_ZERO_MAD": ("INSUFFICIENT_EVIDENCE", "BLOCK",
                   "Peer spread (MAD) is zero, e.g. quantised readings; a robust score would be meaningless."),
    "A_UNTRUSTED_SOURCE_FLAG": ("MEASUREMENT", "BLOCK",
                                "Source quality flag present; no trusted automated peer verdict."),
    "A_NOT_SCORED_MISSING": ("INSUFFICIENT_EVIDENCE", "INFO", "Not peer-scored because a reading is missing."),
    "A_NOT_SCORED_UNSUPPORTED": ("INSUFFICIENT_EVIDENCE", "INFO", "Not peer-scored: parameter not in registry."),
    "A_NO_HISTORICAL_REFERENCE": ("INSUFFICIENT_EVIDENCE", "BLOCK",
                                  "No historical reference for this part/parameter; lot shift cannot be checked."),
    "DQ_MISSING_INPUT": ("MEASUREMENT", "BLOCK", "The 0 h or 24 h reading is missing; nothing is imputed."),
    "DQ_IDENTICAL_READINGS": ("MEASUREMENT", "BLOCK",
                              "0 h and 24 h readings are bit-identical: possible stuck or quantised measurement."),
    "DQ_SOURCE_QUALITY_FLAG": ("MEASUREMENT", "INFO", "The source row carries a quality flag."),
    "DQ_UNIT_CONVERTED": ("INFO", "INFO", "Values were converted between SI prefixes to the registry unit."),
    "SPEC_UNKNOWN": ("INSUFFICIENT_EVIDENCE", "BLOCK", "Part/parameter not in the specification registry."),
    "B_UNSUPPORTED_PART": ("INSUFFICIENT_EVIDENCE", "BLOCK", "No trained forecast model for this part/parameter."),
    "B_NO_CALIBRATION": ("INSUFFICIENT_EVIDENCE", "BLOCK", "Forecast model has no valid calibration."),
    "B_UNIT_MISMATCH": ("MEASUREMENT", "BLOCK", "Unit differs from the unit the model was trained on."),
    "B_NOT_RUN_MISSING_INPUT": ("INSUFFICIENT_EVIDENCE", "INFO", "Forecast not run because a reading is missing."),
    "B_OUT_OF_DOMAIN": ("INSUFFICIENT_EVIDENCE", "BLOCK",
                        "Inputs are outside the training range; the forecast is extrapolation and cannot support "
                        "a PASS."),
    "B_INVALID_INTERVAL": ("INSUFFICIENT_EVIDENCE", "BLOCK", "Forecast interval is not finite."),
    "DEVICE_INCOMPLETE_PARAMETER_SET": ("INSUFFICIENT_EVIDENCE", "BLOCK",
                                        "Not every registered parameter of this part was supplied."),
    "DEMO_SPEC_NOT_APPROVED": ("INFO", "INFO",
                               "Specification limits are DEMO assumptions (approved_for_production = False)."),
}


def category(code: str) -> str:
    return CODES.get(code, ("INFO", "INFO", code))[0]


def role(code: str) -> str:
    return CODES.get(code, ("INFO", "INFO", code))[1]


def text(code: str) -> str:
    return CODES.get(code, ("INFO", "INFO", code))[2]
