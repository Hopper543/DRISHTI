# Module B — early forecasting (0 h, 24 h → 168 h)

Code: `drishti/module_b.py` · Training: `scripts/train.py` · Config: `config/drishti.yaml → module_b` ·
Tests: `tests/test_module_b.py`

## Contract

* **X = [value_0h, value_24h]** — nothing else. Every booster is checked to have exactly two features.
* **y = value_168h**, joined by `sample_id` with a one-to-one validated merge. The target never appears in
  inference inputs; uploads containing it (or any `value_<t>h` with t > 24) are refused.
* Separate models per **(part_number, parameter)**. IDs, lots, scenarios, limits and split names are not
  features (test: changing them leaves predictions bit-identical).

## Models

LightGBM quantile regression at q = 0.05, 0.50, 0.95 (`n_estimators=120, max_depth=4, num_leaves=12,
min_child_samples=30, learning_rate=0.05, seed 26170`), identical to the dataset package's benchmark so results
are comparable. Predictions are sorted to prevent quantile crossing. The q50 model is the central forecast.

## Calibration (lot-level conformalized quantile regression)

For each of the 15 calibration lots per part: nonconformity = max over the lot's devices of
`max(q05 − y, y − q95)`. The margin is the ⌈(n_lots + 1) · 0.90⌉-th smallest lot score (with 15 lots: the
largest). Interval = [q05 − margin, q95 + margin]. Saved in `models/module_b/calibration.json` with the number
of calibration lots and method.

Assumption: the screened lot is exchangeable with the calibration lots. Shifted lots, tiny lots and new parts
violate it; coverage is reported separately for them and is **not** guaranteed. Coverage is per parameter, not
joint across a device's parameters.

## Inference checks

| Check | Code | Effect |
|---|---|---|
| no model for part/parameter | `B_UNSUPPORTED_PART` | ESCALATE |
| calibration missing or infinite | `B_NO_CALIBRATION` | ESCALATE |
| unit differs from training unit | `B_UNIT_MISMATCH` | no forecast; ESCALATE |
| missing / non-finite input | `B_NOT_RUN_MISSING_INPUT` | no forecast; ESCALATE |
| input outside the 0.5–99.5 % training range of value_0h, value_24h or change | `B_OUT_OF_DOMAIN` | forecast shown but cannot support PASS or a predicted FAIL |
| artifact hash differs from manifest | load error | models refused |

## Baselines

* Persistence: ŷ = value_24h
* Linear extrapolation: ŷ = value_0h + 7 · (value_24h − value_0h)

## Explanations

TreeSHAP contributions of the q50 model (LightGBM `pred_contrib`) are shown for value_0h and value_24h. They
explain how the fitted model uses its two inputs; they do not explain device physics or the interval width.

## Results summary (SYNTHETIC only)

See [EVALUATION.md](EVALUATION.md). On ordinary test lots the model beats both baselines for every parameter;
on shifted lots it is worse than linear extrapolation and its coverage collapses to ≈ 0.5, which is why
out-of-domain inputs are escalated. Errors are always reported per parameter in the parameter's own unit.

## Not implemented

* Explainable Boosting Machine (mentioned in the presentation) — not implemented; LightGBM + TreeSHAP and the
  inspectable evidence card are used instead.
* Monotonic constraints, hyperparameter search — not done; validation lots are used only for Module A's ensemble
  threshold.
