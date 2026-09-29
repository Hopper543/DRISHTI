# Model card and limitations

| Field | Value |
|---|---|
| System | DRISHTI 0.1.0 — screening-round prototype (SIH 2026, PS 26170, Team Agamemnon) |
| Intended use | Demonstrate an auditable 24 h early-screening workflow for burn-in data: lot-relative anomaly evidence, 168 h forecast with calibrated uncertainty, and a documented PASS / FAIL / ESCALATE recommendation for an inspector. |
| Not intended for | Accepting or rejecting flight or production hardware; replacing MIL-STD-883 / organisation burn-in procedures; qualification; diagnosing failure mechanisms. |
| Status | `evidence = SYNTHETIC_ONLY`; specification registry `DEMO_ONLY_v1`, `approved_for_production = False`; `mode: production` refuses to run with these artifacts. |

## Models

| Component | Model | Fitted on | Calibrated / thresholded on |
|---|---|---|---|
| Module A decision rule | Robust modified z (median/MAD), |z| ≥ 3.5, min 8 peers | — (no fitting) | — (literature cut-off) |
| Module A baselines | Isolation Forest (200 trees), PyOD ECOD on lot-standardised [z_level, z_change] | 90 synthetic training lots | percentiles: 45 calibration lots; ensemble τ: 15 validation lots (2% healthy flag rate) |
| Module A lot check | lot median vs historical lot medians (robust z ≥ 4); spread ratio ≥ 3 | 90 training lots | — |
| Module B | LightGBM quantile q05/q50/q95 per part/parameter, inputs [value_0h, value_24h] | 90 training lots (30 per part) | lot-level CQR, 15 calibration lots per part, 90% target |

## Training data

The DRISHTI synthetic fixture (seed 26170): 5,772 devices, 204 lots, three demo part families (DEMO_MEMORY,
DEMO_OPAMP, DEMO_MOSFET), seven parameters, fixed 125 °C stress / 25 °C measurement context. Scenarios (healthy,
gradual, delayed, step, stable outlier, measurement fault, uniformly degrading lots) are **mathematical shapes,
not fitted device physics**, and their names are not failure mechanisms. No real burn-in data was used to fit
anything.

## Performance

See [EVALUATION.md](EVALUATION.md). All Module A/B numbers there are **synthetic** and depend on generator
assumptions. On the only real anomaly benchmark (SECOM) the unsupervised detectors do not work (0 of 23 test
failures). No real 0 h / 24 h → 168 h multi-lot burn-in cohort with adjudicated labels exists in this project,
so **real screening performance is unknown**.

## Known limitations

1. **Synthetic-only fitting.** Models, thresholds and intervals would have to be re-fitted and re-validated on
   real, multi-lot burn-in data of the target parts before any real use.
2. **Demo limits.** All limits are demonstration assumptions, not datasheet or ISRO values.
3. **24 h blind spot.** Defects whose drift starts after 24 h are invisible at 24 h; on synthetic ordinary lots
   84 of 130 latent defects passed the early screen (mostly delayed/step onsets).
4. **Conservative intervals ⇒ many ESCALATEs.** Lot-level conformal intervals are wide; with demo limits ~36% of
   healthy synthetic devices were escalated (mainly RDS_on). The escalation rate is a property of the policy
   and the limits, and must be reviewed with real data and approved limits.
5. **Exchangeability.** Interval coverage and ensemble thresholds hold (approximately) only for lots like the
   calibration lots. Shifted, tiny and new-part lots are not covered; shifted-lot coverage was ≈ 0.5.
   Coverage is per parameter, not joint across a device.
6. **Small lots.** Lots with fewer than 8 valid devices get no peer verdict; small-lot ESCALATE is by design.
7. **Zero MAD / quantisation.** Low-resolution instruments can make peer spread zero; DRISHTI escalates instead
   of scoring.
8. **Measurement vs degradation.** From two readings, an unusual change cannot be separated from a measurement
   error; the recommendation is to re-measure.
9. **Anomaly ≠ failure.** Scores express statistical unusualness; they are not failure probabilities.
10. **Unseen defect types** may not be detected; the system never auto-PASSes unsupported inputs but can PASS a
    device whose defect is invisible in the two early readings.
11. **Real cohorts are proxies.** NASA radiation (TID) and capacitor aging data are used only for exploratory
    peer-drift analysis. Radiation dose is never converted into burn-in hours. AD620 rows get no verdict
    (unresolved bias assignment). The IGBT data is quarantined (unverified scaling) and unused.
12. **Audit log** is ordinary append-only file logging with a hash chain: accidental edits are detectable, but
    it is **not tamper-proof**.
13. **Explainability.** TreeSHAP explains the q50 model's use of its two inputs, not physics. Drift-shape labels
    are descriptive and retrospective; no TDDB/NBTI/electromigration diagnosis is made.
14. **Not implemented** (presentation items): Explainable Boosting Machine; mechanism suggestion; a safety-slope
    rule (no approved threshold source exists); STDF ingestion.

## Path to production (gated)

Actual target-part devices from several independent lots with trustworthy lot/date codes; controlled bias and
temperature; calibrated instruments and replicates; approved, versioned specification registry; 168 h
endpoints, functional tests and reviewed failure analysis; re-fit and re-calibrate on real lots; freeze
thresholds; estimate escape and false-reject rates with confidence intervals on untouched lots and compare with
conventional screening; independent review; then set `approved_for_production` and `mode: production`.
