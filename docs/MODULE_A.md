# Module A — lot-relative anomaly detection

Code: `drishti/module_a.py` · Config: `config/drishti.yaml → module_a` · Tests: `tests/test_module_a.py`

## Peer groups

Devices are compared only with devices that share **part number, lot, parameter, unit and test condition**
(`test_condition` is an optional upload column; the synthetic fixture uses `stress125C_meas25C`). Rows with a
missing reading, an unknown part/parameter or a source quality flag are **not peers** and receive no verdict.
At the 24 h decision only `value_0h` and `value_24h` are used.

## Features and scores

| Quantity | Definition |
|---|---|
| level | `value_24h` |
| change | `value_24h − value_0h` (original unit, signed) |
| robust z | `(x − median) / (1.4826 · MAD)` within the peer group, for level and change separately |
| robust score | `max(|z_level|, |z_change|)` |
| Isolation Forest score | `−score_samples([z_level, z_change])` from a forest fitted **once per part/parameter** on the lot-standardised features of the 90 training lots (200 trees, seed 26170) |
| ECOD score | PyOD ECOD `decision_function` on the same features, fitted on the same training population |
| percentile | empirical CDF position of each score among the same scores on the 45 **calibration** lots |
| ensemble | weighted mean of the three percentiles (weights 1 : 1 : 1, configurable) |

Why the detectors are fitted on historical lots, not inside each lot: a lot has ~30 devices. ECOD's empirical-CDF
tail probabilities and Isolation Forest path lengths computed within 30 points are essentially ranks; they can
say which device is most isolated but not *how* extreme it is. In an early experiment on validation lots, per-lot
IF/ECOD flagged 0% of planted stable outliers while the robust rule caught 100%. Fitting on lot-standardised
features from many historical lots keeps the comparison lot-relative while letting the detectors express
magnitude.

## Flag rule (default: robust)

| `flag_rule` | Unusual when |
|---|---|
| `robust` (default) | robust score ≥ 3.5 — the Iglewicz & Hoaglin (1993) modified z-score cut-off; not tuned on labels |
| `ensemble` | ensemble ≥ τ, τ = 98th percentile of the ensemble over **known-healthy parameters of the 15 synthetic validation lots** (2% healthy flag rate) |
| `either` | robust OR ensemble |

The code of an unusual row is `A_UNUSUAL_CHANGE` if `|z_change| ≥ |z_level|`, else `A_UNUSUAL_LEVEL`.
"Unusual change" is reported as *suspected degradation* with a re-measure recommendation: at 24 h a single
jump cannot be distinguished from a measurement error.

**Why robust is the default.** On validation lots (thresholds set there, test untouched) the robust rule and
Isolation Forest ranked planted anomalies about equally (gradual-drift ROC AUC 0.79 vs 0.80), ECOD was weaker
(0.74) and the percentile ensemble did not improve on the robust rule at equal healthy flag rates. The simpler,
fully inspectable rule is therefore the decision rule; the ML detectors remain as documented baselines and as
corroborating evidence on the device page. Test-lot results are in [EVALUATION.md](EVALUATION.md).

## Guards

| Situation | Code | Behaviour |
|---|---|---|
| fewer than 8 valid peers | `A_INSUFFICIENT_PEERS` | no peer verdict; device compared with the pooled historical distribution (`A_HISTORICAL_UNUSUAL` if |z| ≥ 3.5) as evidence only; ESCALATE |
| MAD ≤ 1e−9 · max(1, |median|) for level or change | `A_ZERO_MAD` | no score; the MAD is **never** replaced by a small number (that would create enormous artificial scores); ESCALATE |
| missing 0 h / 24 h | `A_NOT_SCORED_MISSING` + `DQ_MISSING_INPUT` | not a peer; ESCALATE |
| source quality flag (e.g. AD620 unresolved bias) | `A_UNTRUSTED_SOURCE_FLAG` | not a peer, no verdict; ESCALATE |
| bit-identical 0 h and 24 h | `DQ_IDENTICAL_READINGS` | possible stuck/quantised measurement; ESCALATE |

## Historical lot comparison (distribution shift)

The historical reference (detectors, pooled distributions, lot medians) is keyed by part/parameter only and was
built from synthetic lots measured under one fixed context (125 °C stress, 25 °C measurement). Uploads under a
different `test_condition` are peer-compared correctly within their own condition group, but their historical
comparison is against that single reference context; a real deployment needs a reference per condition.

Peer comparison cannot see a lot that is shifted as a whole. For each valid group DRISHTI compares the lot
median level and median change with the medians of the 90 historical training lots (robust z, threshold ±4) →
`A_LOT_SHIFT`, and the lot MAD with the median within-lot MAD (ratio ≥ 3) → `A_LOT_SPREAD`. Every device of a
flagged lot × parameter is escalated for lot review.

## Separation of evidence types

| Evidence | Where it comes from |
|---|---|
| Statistical unusualness | `A_UNUSUAL_LEVEL`, ensemble percentiles |
| Confirmed specification failure | `SPEC_OBSERVED_FAIL` (decision engine; registry limits) |
| Suspected degradation | `A_UNUSUAL_CHANGE` |
| Possible measurement problem | `DQ_MISSING_INPUT`, `DQ_IDENTICAL_READINGS`, `A_UNTRUSTED_SOURCE_FLAG`, `B_UNIT_MISMATCH` |

**An anomaly score is not a probability of failure.** Calibration statements refer only to the stated reference
populations (synthetic train/calibration/validation lots, which themselves contain planted anomalies); no
unconditional false-alarm control is claimed, and unseen defect types may not be detected.

## Real-data route

`drishti/real.py` applies the same robust peer scoring to the prepared real long-format data at each dose/aging
checkpoint, with peers = same dataset, part, lot, parameter, unit, **channel**, bias group and checkpoint.
Controls are never peers. There is no historical reference for these single-lot cohorts, so only the robust rule
is used and results are labelled exploratory.
