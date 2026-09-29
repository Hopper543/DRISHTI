"""Reproducible evaluation. Writes evaluation_results/*.{json,csv}.

    python scripts/evaluate.py

Evidence classes are kept apart; nothing here is averaged across units or
across datasets.
  module_a_synthetic_*   planted-label detection on held-out SYNTHETIC lots
  module_b_synthetic_*   forecast error / coverage per part, parameter, split
  decision_synthetic_*   device decisions vs generator truth (escapes)
  secom_*                REAL process data, separate anomaly benchmark;
                         preserved Isolation Forest baseline + same-protocol ECOD
  real_exploratory_*     REAL component cohorts: descriptive only, no accuracy

Operating points were fixed BEFORE scoring test lots: robust |z| >= 3.5
(literature cut-off); IF/ECOD/ensemble thresholds at a 2% healthy-parameter
flag rate on VALIDATION lots. Test / shifted / edge lots are scored once.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyod.models.ecod import ECOD  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score  # noqa: E402

from drishti import data  # noqa: E402
from drishti.config import EVAL_DIR, load_settings  # noqa: E402
from drishti.module_a import run_module_a  # noqa: E402
from drishti.module_b import baselines, interval_status  # noqa: E402
from drishti.pipeline import load_models, run_screening  # noqa: E402
from drishti.real import analyze_real, validate_real  # noqa: E402

SPLITS = {"validation": "VALIDATION (threshold setting)", "test": "ORDINARY TEST", "ood_test": "SHIFTED TEST",
          "edge_test": "EDGE / SMALL-LOT TEST"}
ABNORMAL = ["gradual", "delayed", "step", "stable_outlier", "measurement_fault"]
DETECTABLE_AT_24H = ["gradual", "stable_outlier", "measurement_fault"]  # delayed/step onset is 30-130 h


def _safe(fn, *a):
    try:
        return float(fn(*a))
    except ValueError:
        return None


def module_a_synthetic(models, cfg) -> dict:
    e, gt = data.synthetic_early_inputs(), data.synthetic_ground_truth()
    ref = models.module_a
    scored = {}
    for split in SPLITS:
        rows, lots = run_module_a(e[e.split == split], ref, cfg)
        rows = rows.merge(gt[["sample_id", "scenario", "lot_wide_degradation_planted"]], on="sample_id",
                          validate="one_to_one")
        scored[split] = (rows, lots)
    v = scored["validation"][0]
    vh = v.scenario.str.startswith("healthy") & v.robust_score.notna()
    rate = float(cfg["target_healthy_flag_rate"])
    thr = {"robust_rule": float(cfg["robust_z_threshold"]),
           "isolation_forest": float(np.quantile(v.loc[vh, "if_score"], 1 - rate)),
           "ecod": float(np.quantile(v.loc[vh, "ecod_score"], 1 - rate)),
           "ensemble": float(ref.meta["ensemble_threshold_validation"])}
    score_col = {"robust_rule": "robust_score", "isolation_forest": "if_score", "ecod": "ecod_score",
                 "ensemble": "ensemble_score"}
    by_scen, summary = [], []
    for split, (rows, lots) in scored.items():
        scorable = rows[rows.robust_score.notna()].copy()
        scorable["positive"] = scorable.scenario.isin(ABNORMAL)
        healthy = scorable.scenario.str.startswith("healthy")
        for m, col in score_col.items():
            flag = scorable[col] >= thr[m]
            tn, fp, fn, tp = confusion_matrix(scorable.positive, flag, labels=[False, True]).ravel()
            summary.append(dict(split=split, split_label=SPLITS[split], method=m, threshold=thr[m],
                                n_scored=int(len(scorable)), n_not_scored=int(rows.robust_score.isna().sum()),
                                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                                healthy_flag_rate=float(flag[healthy].mean()) if healthy.any() else None,
                                recall_all_planted=float(tp / (tp + fn)) if tp + fn else None,
                                recall_detectable_24h=float(flag[scorable.scenario.isin(DETECTABLE_AT_24H)].mean())
                                if scorable.scenario.isin(DETECTABLE_AT_24H).any() else None,
                                precision=float(tp / (tp + fp)) if tp + fp else None,
                                roc_auc=_safe(roc_auc_score, scorable.positive, scorable[col]),
                                average_precision=_safe(average_precision_score, scorable.positive, scorable[col]),
                                evidence="SYNTHETIC_ONLY"))
            for sc, g in scorable.groupby("scenario"):
                by_scen.append(dict(split=split, method=m, scenario=sc, n=int(len(g)),
                                    flag_rate=float((g[col] >= thr[m]).mean())))
        # lot-shift check: lots with planted uniform degradation vs others
        lt = lots.merge(rows.groupby("lot_id").lot_wide_degradation_planted.first().reset_index(), on="lot_id")
        lt["flagged"] = lt.lot_codes.map(lambda c: "A_LOT_SHIFT" in c)
        for planted, g in lt.groupby("lot_wide_degradation_planted"):
            summary.append(dict(split=split, split_label=SPLITS[split], method="lot_shift_check",
                                threshold=float(cfg["lot_shift_z_threshold"]), n_scored=int(len(g)),
                                lot_parameter_groups=int(len(g)), planted_lot_degradation=bool(planted),
                                flag_rate=float(g.flagged.mean()), evidence="SYNTHETIC_ONLY"))
        # status of unscorable rows (missing, small lots, zero MAD)
    status = pd.concat([r.assign(split=s) for s, (r, _) in scored.items()]).groupby(
        ["split", "group_status"]).size().rename("n").reset_index()
    return {"summary": pd.DataFrame(summary), "by_scenario": pd.DataFrame(by_scen), "group_status": status,
            "thresholds": thr}


def module_b_synthetic(models) -> pd.DataFrame:
    e, t = data.synthetic_early_inputs(), data.synthetic_targets()
    fc = models.module_b
    d = e.merge(t[["sample_id", "value_168h"]], on="sample_id", validate="one_to_one")
    out = []
    for split in ["validation", "test", "ood_test", "edge_test"]:
        s = d[d.split == split]
        pred = fc.predict(s)
        base = baselines(s)
        s = s.join(pred).join(base)
        for (part, param), g in s.groupby(["part_number", "parameter"]):
            ok = g.prediction.notna()
            gg = g[ok]
            y = gg.value_168h.to_numpy()
            cover = (gg.lower <= y) & (y <= gg.upper)
            lotcov = cover.groupby(gg.lot_id).all()
            ood = gg.b_codes.map(lambda c: "B_OUT_OF_DOMAIN" in c)
            out.append(dict(part_number=part, parameter=param, unit=g.unit.iloc[0], split=split,
                            split_label=SPLITS[split], n=int(len(g)), n_scored=int(ok.sum()),
                            n_escalated_missing=int((~ok).sum()), n_out_of_domain=int(ood.sum()),
                            MAE=float(np.mean(np.abs(gg.prediction - y))),
                            median_AE=float(np.median(np.abs(gg.prediction - y))),
                            persistence_MAE=float(np.mean(np.abs(gg.persistence - y))),
                            linear_extrapolation_MAE=float(np.mean(np.abs(gg.linear_extrapolation - y))),
                            interval_coverage=float(cover.mean()), whole_lot_coverage=float(lotcov.mean()),
                            n_lots=int(lotcov.size), mean_interval_width=float(np.mean(gg.upper - gg.lower)),
                            coverage_in_domain=float(cover[~ood].mean()) if (~ood).any() else None,
                            calibration_lots=fc.calibration[f"{part}__{param}"]["n_calibration_lots"],
                            coverage_target=fc.calibration[f"{part}__{param}"]["coverage_target"],
                            evidence="SYNTHETIC_ONLY"))
    return pd.DataFrame(out)


def decision_synthetic(models) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline on held-out lots vs generator truth at device level."""
    e, gt = data.synthetic_early_inputs(), data.synthetic_ground_truth()
    truth = gt.groupby("device_id").agg(latent=("latent_defect_planted", "any"),
                                        oos168=("true_out_of_spec_168h", "any"),
                                        fault=("measurement_fault_planted", "any"),
                                        lot_deg=("lot_wide_degradation_planted", "any"),
                                        scen=("scenario", lambda s: ",".join(sorted(set(s) - {"healthy_parameter"}))))
    tables, detail = [], []
    cols = ["sample_id", "device_id", "lot_id", "part_number", "parameter", "unit", "value_0h", "value_24h",
            "provenance_class"]
    for split in ["test", "ood_test", "edge_test"]:
        run = run_screening(e[e.split == split][cols], models=models, source_name=f"synthetic {split}")
        dv = run.devices.merge(truth, left_on="device_id", right_index=True)
        dv["truth"] = np.select([dv.oos168, dv.latent, dv.fault, dv.lot_deg],
                                ["out_of_spec_at_168h", "latent_defect_within_spec", "measurement_fault",
                                 "lot_degradation"], "healthy")
        ct = pd.crosstab(dv.truth, dv.decision).reindex(columns=["PASS", "ESCALATE", "FAIL"], fill_value=0)
        ct["split"] = split
        tables.append(ct.reset_index())
        detail.append(dv.assign(split=split)[["split", "device_id", "lot_id", "decision", "basis", "truth", "scen"]])
    return pd.concat(tables, ignore_index=True), pd.concat(detail, ignore_index=True)


def secom() -> dict:
    """Preserved baseline (identical protocol to the dataset package) + ECOD and
    a rank-mean ensemble under the SAME pre-declared protocol. Test scored once."""
    x, splits, labels = data.secom()
    train = x.loc[splits[splits == "train"].index]
    cols = train.columns[(train.notna().sum() > 0) & (train.nunique(dropna=True) > 1)]
    imp = SimpleImputer(strategy="median").fit(train[cols])
    xt = imp.transform(train[cols])
    mu, sd = xt.mean(axis=0), xt.std(axis=0)
    sd[sd == 0] = 1
    iso = IsolationForest(n_estimators=200, random_state=26170, n_jobs=2).fit(xt)
    ecod = ECOD().fit((xt - mu) / sd)

    def scores(ids):
        z = imp.transform(x.loc[ids, cols])
        return {"isolation_forest": -iso.score_samples(z), "ecod": ecod.decision_function((z - mu) / sd)}

    cal_ids = splits[(splits == "calibration") & (labels == 0)].index
    cal = scores(cal_ids)
    ref_train = scores(train.index)
    res = {}
    for split in ["validation", "test"]:
        ids = splits[splits == split].index
        y = labels.loc[ids].to_numpy()
        s = scores(ids)
        # ensemble: mean of percentiles within the TRAIN score distribution
        s["ensemble_rank_mean"] = np.mean([np.searchsorted(np.sort(ref_train[k]), s[k]) / len(ref_train[k])
                                           for k in ("isolation_forest", "ecod")], axis=0)
        calc = dict(cal)
        calc["ensemble_rank_mean"] = np.mean([np.searchsorted(np.sort(ref_train[k]), cal[k]) / len(ref_train[k])
                                              for k in ("isolation_forest", "ecod")], axis=0)
        for m, sc in s.items():
            c = np.sort(calc[m])
            k = int(np.ceil((len(c) + 1) * 0.95))
            thr = float(c[k - 1]) if k <= len(c) else float("inf")
            flag = sc > thr
            tn, fp, fn, tp = confusion_matrix(y, flag, labels=[0, 1]).ravel()
            res.setdefault(split, {})[m] = dict(
                n=int(len(ids)), n_fail=int(y.sum()), threshold=thr, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                recall=float(tp / (tp + fn)) if tp + fn else None, false_positive_rate=float(fp / (fp + tn)),
                roc_auc=float(roc_auc_score(y, sc)), average_precision=float(average_precision_score(y, sc)),
                baseline=(m == "isolation_forest"))
    return {"dataset": "UCI SECOM (CC BY 4.0)", "evidence": "REAL_PROCESS_DATA_NOT_BURN_IN",
            "n_features_used": int(len(cols)), "n_calibration_known_pass": int(len(cal_ids)),
            "protocol": "Features with >0 non-missing values and >1 unique value on TRAIN; median imputation fitted "
                        "on TRAIN; detectors fitted on TRAIN; threshold = finite-sample 95% quantile of scores on "
                        "known-pass CALIBRATION rows; validation and test scored once with no tuning.",
            "caveat": "Chronological drift breaks exchangeability; no false-alarm guarantee. Unsupervised detectors "
                      "on anonymous process features are not expected to find these failures.",
            "results": res}


def real_exploratory(cfg) -> dict:
    f, scope = data.real_drift_view()
    errors, _, v = validate_real(f)
    assert not errors, errors
    out = {"scope": scope, "evidence": "REAL component cohorts; exploratory electrical-drift analysis only",
           "note": "No 168 h burn-in endpoint and no latent-defect labels exist; counts are descriptive, not accuracy."}
    for label, mp in (("default_min_peers_8", cfg["min_peers"]), ("sensitivity_min_peers_5", 5)):
        r = analyze_real(v, dict(cfg, min_peers=mp))
        out[label] = r.groupby(["dataset_id", "exploratory_status"]).size().unstack(fill_value=0).reset_index() \
            .to_dict(orient="records")
        if label.startswith("sensitivity"):
            rr = r[r.dataset_id == "reram"]
            fo = data.reram_outcomes()
            tab = rr[["device_id", "stress_value", "value", "change", "z_change", "exploratory_status"]].merge(
                fo[["device_id", "last_functional_krad", "first_failure_krad"]], on="device_id", how="left")
            out["reram_standby_current_vs_reported_functional_failure"] = json.loads(tab.to_json(orient="records"))
    return out


def main() -> None:
    s = load_settings()
    models = load_models()
    EVAL_DIR.mkdir(exist_ok=True)
    a = module_a_synthetic(models, s.module_a)
    a["summary"].to_csv(EVAL_DIR / "module_a_synthetic_summary.csv", index=False)
    a["by_scenario"].to_csv(EVAL_DIR / "module_a_synthetic_by_scenario.csv", index=False)
    a["group_status"].to_csv(EVAL_DIR / "module_a_synthetic_group_status.csv", index=False)
    print("Module A done")
    b = module_b_synthetic(models)
    b.to_csv(EVAL_DIR / "module_b_synthetic_metrics.csv", index=False)
    print("Module B done")
    ct, detail = decision_synthetic(models)
    ct.to_csv(EVAL_DIR / "decision_synthetic_confusion.csv", index=False)
    detail.to_csv(EVAL_DIR / "decision_synthetic_devices.csv", index=False)
    print("Decisions done")
    sec = secom()
    (EVAL_DIR / "secom_results.json").write_text(json.dumps(sec, indent=1))
    print("SECOM done")
    real = real_exploratory(s.module_a)
    (EVAL_DIR / "real_exploratory.json").write_text(json.dumps(real, indent=1, default=str))
    meta = {"model_version": models.version, "thresholds_module_a": a["thresholds"],
            "evidence_note": "Synthetic metrics are generator-dependent demonstration evidence only."}
    (EVAL_DIR / "evaluation_meta.json").write_text(json.dumps(meta, indent=1))
    print("Saved to", EVAL_DIR)


if __name__ == "__main__":
    main()
