# Evaluation

All numbers come from `python scripts/evaluate.py` (files in `evaluation_results/`) with the committed models.
Evidence classes are reported separately and never pooled; forecast errors are never averaged across units.

**Protocol.** Thresholds were fixed before test lots were scored: robust |z| ≥ 3.5 (literature value); Isolation
Forest, ECOD and ensemble thresholds at a 2% healthy-parameter flag rate on the 15 synthetic **validation**
lots. Test (30 lots), shifted (12) and edge (12) lots were each scored once.

## 1. Module A on held-out SYNTHETIC lots

Unit: device × parameter at 24 h. Positive = any planted abnormality on that parameter. Delayed and step defects
begin between 30 and 130 h, so no 24 h method can see them; `recall_detectable_24h` covers gradual drift, stable
outliers and measurement faults.

| Split | Method | TP | FP | FN | TN | Healthy flag rate | Recall (24 h-detectable) | ROC AUC |
|---|---|---|---|---|---|---|---|---|
| Ordinary test | **robust rule** | 100 | 9 | 119 | 1872 | 0.005 | 0.697 | 0.787 |
| Ordinary test | Isolation Forest | 98 | 24 | 121 | 1857 | 0.013 | 0.676 | 0.772 |
| Ordinary test | ECOD | 48 | 27 | 171 | 1854 | 0.014 | 0.331 | 0.759 |
| Ordinary test | ensemble | 80 | 20 | 139 | 1861 | 0.011 | 0.556 | 0.778 |
| Shifted test | **robust rule** | 35 | 2 | 35 | 600 | 0.003 | 0.745 | 0.796 |
| Shifted test | Isolation Forest | 31 | 3 | 39 | 599 | 0.005 | 0.660 | 0.797 |
| Shifted test | ECOD | 12 | 5 | 58 | 597 | 0.008 | 0.255 | 0.763 |
| Shifted test | ensemble | 25 | 4 | 45 | 598 | 0.007 | 0.532 | 0.790 |

Robust rule, ordinary test, flag rate by planted scenario: stable outlier 1.00 (49), measurement fault 1.00 (36),
gradual 0.25 (57), step 0.03 (33), delayed 0.00 (44), healthy 0.004 (681), healthy parameter of an affected
device 0.005 (1200). Edge lots: 112 of 196 rows could not be peer-scored (lots of 3 and 5 devices) and are
escalated as `A_INSUFFICIENT_PEERS`; rates on the remaining 77 rows are anecdotal.

**Lot-shift check:** all 14 shifted-lot × parameter groups with planted uniform degradation were flagged
`A_LOT_SHIFT`; 0 of 14 shifted-but-not-degrading groups, 1 of 70 ordinary-test groups and 0 of 35 validation
groups were flagged.

## 2. Module B on held-out SYNTHETIC lots

MAE in each parameter's own unit. Coverage target 0.90 (whole-lot, exchangeable lots).

| Parameter | Unit | Split | MAE | Persistence | Linear extr. | Coverage | Whole-lot coverage (lots) | Mean width |
|---|---|---|---|---|---|---|---|---|
| leakage_current | µA | ordinary | 1.176 | 2.675 | 2.222 | 0.997 | 0.9 (10) | 25.4 |
| supply_current | mA | ordinary | 0.082 | 0.159 | 0.174 | 1.000 | 1.0 (10) | 1.59 |
| RDS_on | Ω | ordinary | 0.015 | 0.033 | 0.027 | 0.983 | 0.6 (10) | 0.237 |
| threshold_voltage | V | ordinary | 0.032 | 0.077 | 0.057 | 1.000 | 1.0 (10) | 0.837 |
| input_bias_current | nA | ordinary | 0.046 | 0.117 | 0.089 | 1.000 | 1.0 (10) | 1.34 |
| input_offset_voltage | µV | ordinary | 3.685 | 5.465 | 4.001 | 1.000 | 1.0 (10) | 76.9 |
| quiescent_current | mA | ordinary | 0.015 | 0.042 | 0.038 | 0.997 | 0.9 (10) | 0.471 |
| leakage_current | µA | shifted | 17.12 | 16.30 | 10.45 | 0.500 | 0.5 (4) | 31.9 |
| input_offset_voltage | µV | shifted | 60.69 | 39.35 | 23.27 | 0.500 | 0.5 (4) | 97.6 |
| RDS_on | Ω | shifted | 0.081 | 0.088 | 0.054 | 0.542 | 0.5 (4) | 0.246 |

(Full table, including edge lots and all shifted parameters: `evaluation_results/module_b_synthetic_metrics.csv`.)

Reading: on exchangeable lots the model beats both baselines and intervals over-cover per device (the lot-level
margin is conservative) while whole-lot coverage for RDS_on is 0.6, below target — 10 test lots is a small
sample and the guarantee is marginal over lots, not per parameter set. On shifted lots the model fails
(tree models cannot extrapolate) and linear extrapolation is better; those inputs are flagged `B_OUT_OF_DOMAIN`
and cannot PASS. These results reproduce the dataset package's benchmark exactly (same data, parameters, seed).

## 3. Full-pipeline device decisions vs generator truth (SYNTHETIC)

| Truth \ decision | Ordinary: PASS | ESCALATE | FAIL | Shifted: PASS | ESCALATE | FAIL | Edge: PASS | ESCALATE | FAIL |
|---|---|---|---|---|---|---|---|---|---|
| healthy | 470 | 260 | 0 | 64 | 49 | 0 | 1 | 66 | 0 |
| latent defect, within spec at 168 h | 84 | 46 | 0 | 14 | 9 | 0 | 2 | 7 | 0 |
| measurement fault | 0 | 36 | 0 | 0 | 8 | 0 | 0 | 8 | 0 |
| out of spec at 168 h | 0 | 4 | 0 | 0 | 137 | 7 | — | — | — |

* No device that is truly out of specification at 168 h received PASS.
* 84 latent defects PASSed on ordinary lots, mostly delayed/step onsets after 24 h (see §1).
* 36% of healthy ordinary-lot devices were escalated. The dominant cause is `B_INTERVAL_CROSSES_LIMIT` on
  RDS_on: 214 of the 267 escalated healthy parameters were decided by an interval crossing, 136 of them RDS_on. Its calibrated interval (≈ ±0.12 Ω) is wide relative to the
  gap between typical values (~0.35 Ω) and the 0.5 Ω demo limit. Under the required policy an interval crossing a
  limit must escalate; reducing this needs more calibration lots or a different, approved calibration target,
  not tuning on test lots.

## 4. SECOM — REAL process data (separate benchmark)

1,567 examples, 590 anonymous features, chronological date-group splits (dates are not lots). Protocol fixed in
advance: training-only feature selection (468 features) and median imputation; detectors fitted on training
rows; threshold = finite-sample 95% quantile on known-pass calibration rows.

| Split | Method | TP | FP | FN | TN | Recall | FPR | ROC AUC | AP |
|---|---|---|---|---|---|---|---|---|---|
| test (380, 23 fails) | Isolation Forest **(preserved baseline)** | 0 | 30 | 23 | 327 | 0.00 | 0.084 | 0.443 | 0.053 |
| test | ECOD | 0 | 15 | 23 | 342 | 0.00 | 0.042 | 0.498 | 0.068 |
| test | IF+ECOD rank-mean ensemble | 1 | 20 | 22 | 337 | 0.04 | 0.056 | 0.483 | 0.062 |
| validation (180, 7 fails) | Isolation Forest | 1 | 9 | 6 | 164 | 0.14 | 0.052 | 0.605 | 0.075 |

The preserved baseline is reproduced exactly (TP 0, FP 30, FN 23, TN 327). None of the unsupervised detectors
is useful on this benchmark; we report that rather than tuning against test labels.

## 5. REAL component cohorts — exploratory only

No 168 h burn-in endpoint and no latent-defect labels exist, so these are descriptive counts, not accuracy.
With the default minimum of 8 peers, only AD648 bias groups (10 devices) receive verdicts (40 rows, all typical);
AD620 rows receive none (`UNTRUSTED_NO_VERDICT`, unresolved bias assignment); every other cohort is below the
peer minimum. In a labelled **sensitivity** run with a minimum of 5 peers, the ReRAM standby-current groups at
50 krad(Si) flag DUT 4 (0.0215 mA — below the 0.045 mA limit) and DUT 6 as unusual; both are reported as
functionally failed by 50 krad, while DUT 7 and DUT 9 also failed by then but look typical. With five peers per
group this is an anecdote, not validation. Counts were produced from the full local table; a fresh clone
contains only the public subset (AD620, ReRAM, OP484, U309).
