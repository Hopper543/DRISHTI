# Decision table (default demo policy)

Implemented in `drishti/decision.py`; reason texts in `drishti/reasons.py`; tested in `tests/test_decision.py`.
Configuration: `config/drishti.yaml → decision`.

## Parameter level — first matching rule wins; all codes are still reported

| Rule | Condition | Outcome | Basis |
|---|---|---|---|
| R1 | A measured 0 h or 24 h reading is outside the registry limits (`SPEC_OBSERVED_FAIL`) | **FAIL** | OBSERVED |
| R2 | The whole calibrated 168 h interval is beyond a limit (`B_PREDICTED_VIOLATION`) **and** the forecast is usable (no `B_OUT_OF_DOMAIN`, `DQ_IDENTICAL_READINGS`, `A_UNTRUSTED_SOURCE_FLAG`, `B_UNIT_MISMATCH`) | **FAIL** (configurable: `predicted_violation_action`) | PREDICTED |
| R3 | Any evidence gap (role BLOCK): missing reading, unknown part/parameter, no model, no calibration, unit mismatch, out-of-domain inputs, invalid interval, insufficient peers, zero MAD, identical readings, source quality flag, no historical reference | **ESCALATE** | INSUFFICIENT_EVIDENCE |
| R4 | Any warning (role WARN): interval crosses a limit, Module A unusual change / level, historical-unusual (small lot), lot shift, lot spread | **ESCALATE** | REVIEW |
| R5 | Interval inside the limits and nothing above | **PASS** | EARLY_SCREEN |
| R6 | Anything else (no usable evidence) | **ESCALATE** | INSUFFICIENT_EVIDENCE |

Required properties and where they hold:

* *Interval crossing a limit ⇒ ESCALATE* — R4 (`B_INTERVAL_CROSSES_LIMIT` is a WARN).
* *A forecast-only PASS cannot erase a significant Module A warning* — R4 precedes R5.
* *Observed failures vs predicted violations* — different rules and bases (R1 OBSERVED, R2 PREDICTED); the PDF,
  CSV and dashboard show the basis.
* *Data-quality failures, insufficient evidence and unsupported inputs are visible* — R3 with explicit codes.
* *No PASS without evidence* — R6.

## Device level

| Outcome | Condition |
|---|---|
| **FAIL** | any parameter FAIL (basis OBSERVED if any observed failure, else PREDICTED) |
| **ESCALATE** | otherwise, any parameter ESCALATE, or a parameter registered for the part is missing (`DEVICE_INCOMPLETE_PARAMETER_SET`) |
| **PASS** | every registered parameter PASS |

## FAIL vs ESCALATE

* **FAIL** — there is evidence that a limit *is* violated (observed) or that the *whole* calibrated interval is
  beyond it (predicted). Recommended action: reject or send to failure analysis per the approved procedure.
* **ESCALATE** — evidence is insufficient, uncertain or suspicious. Recommended action: continue burn-in
  (extended screening) and/or engineering review. ESCALATE is not a soft FAIL; it is "do not decide yet".
* **PASS at 24 h** — an *early screening* outcome only. It is not qualification completion and does not replace
  the approved 168 h burn-in unless an organisation validates and approves such a reduction on real lots.

## What is deliberately not implemented

* **Safety-slope rule.** No rule compares slopes to thresholds, because no approved threshold source (with units
  and a reference checkpoint) exists for these parts. A predicted value is never compared with a slope.
* **Mechanism diagnosis.** Drift-shape descriptions ("accelerating drift", "step change") are descriptive and
  retrospective; DRISHTI never infers TDDB, NBTI or electromigration.

## Production gate

`mode: production` in `config/drishti.yaml` makes `run_screening` refuse to produce decisions unless the input is
`REAL_MEASURED`, every registry row has `approved_for_production = True`, and the models are not marked
`SYNTHETIC_ONLY`. The shipped registry and models fail all three checks by design (`tests/test_e2e.py`).
