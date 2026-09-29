"""Separate analysis route for the prepared REAL long-format data
(real_measurements / real_drift_view layout).

Exploratory electrical-drift analysis only:
* Module A robust peer comparison at a chosen checkpoint (level and change
  from each device's own stage-0 baseline, in the original unit).
* Peers = same dataset, part, lot, parameter, unit, channel, bias/peer group,
  stress axis and checkpoint. Controls are never peers of stressed devices.
* No PASS/FAIL, no forecast: these cohorts have no 0 h/24 h -> 168 h burn-in
  endpoint, and radiation dose is never converted into burn-in hours.
* Rows whose source flags mark unresolved bias assignment (AD620), sign
  inconsistencies or suspect anneal cells get no trusted automated verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .explain import describe_trajectory
from .module_a import INSUFFICIENT_PEERS, VALID, ZERO_MAD, score_peer_group
from .provenance import REAL_DERIVED, REAL_MEASURED

REQUIRED = ["device_id", "part_number", "lot_id", "parameter", "unit", "channel", "peer_group",
            "measurement_stage", "stress_axis", "stress_value", "value", "provenance_class"]
UNTRUSTED_FLAGS = ("individual_bias_assignment_unverified", "sign_inconsistency", "suspect_anneal")
PEER_KEY = ["dataset_id", "part_number", "lot_id", "parameter", "unit", "channel", "peer_group", "stress_axis",
            "measurement_stage"]
STRESS_PHASES = ("irradiation", "aging", "post_cold_storage")
RADIATION_AXES = ("nominal_dose_checkpoint_krad_Si",)


def validate_real(frame: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame | None]:
    errors, warnings = [], []
    f = frame.copy()
    missing = [c for c in REQUIRED if c not in f]
    if missing:
        return [f"Missing required column(s) for the real-data route: {missing}."], warnings, None
    if "dataset_id" not in f:
        f["dataset_id"] = "upload"
    bad = ~f.provenance_class.isin([REAL_MEASURED, REAL_DERIVED])
    if bad.any():
        errors.append("The real-data route accepts only REAL_MEASURED / REAL_DERIVED rows; synthetic data uses the "
                      "early-input route.")
    if "burn_in_hours" in f:
        rad = f.stress_axis.isin(RADIATION_AXES) & pd.to_numeric(f.burn_in_hours, errors="coerce").notna()
        if rad.any():
            errors.append(f"{int(rad.sum())} radiation-dose row(s) carry burn_in_hours. Radiation dose must not be "
                          "converted into thermal burn-in hours; leave burn_in_hours empty.")
    if "latent_defect_label" in f and f.latent_defect_label.notna().any():
        warnings.append("latent_defect_label values present in real rows; they are ignored (no adjudicated labels).")
    f["value"] = pd.to_numeric(f.value, errors="coerce")
    f["quality_flags"] = f.get("quality_flags", pd.Series("", index=f.index)).fillna("").astype(str)
    key = ["dataset_id", "device_id", "parameter", "channel", "measurement_stage"]
    if f.duplicated(key).any():
        errors.append("Duplicate device/parameter/channel/stage rows; each measurement must appear once.")
    if f.groupby(["dataset_id", "part_number", "parameter"]).unit.nunique().gt(1).any():
        errors.append("Mixed units within a part/parameter; units are never merged (A vs mA vs nA).")
    if f.value.isna().any():
        warnings.append(f"{int(f.value.isna().sum())} missing value(s) kept missing (not imputed).")
    return errors, warnings, (None if errors else f)


def analyze_real(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Robust peer scores per device at every checkpoint (stage > 0)."""
    f = frame.drop(columns=["baseline_value", "delta_from_baseline"], errors="ignore").copy()
    bkey = ["dataset_id", "device_id", "parameter", "channel"]
    base = f[f.measurement_stage == 0][bkey + ["value"]].rename(columns={"value": "baseline_value"})
    f = f.merge(base, on=bkey, how="left", validate="many_to_one")
    f["change"] = f.value - f.baseline_value
    f["is_control"] = f.peer_group.astype(str).str.contains("control", case=False)
    f["untrusted"] = f.quality_flags.str.contains("|".join(UNTRUSTED_FLAGS)) | f.peer_group.eq("UNRESOLVED")
    phase_ok = f.phase.isin(STRESS_PHASES) if "phase" in f else True
    stressed = f[(f.measurement_stage > 0) & ~f.is_control & phase_ok]
    out = []
    for keyv, g in stressed.groupby(PEER_KEY, dropna=False, sort=False):
        lvl, chg = g.value.to_numpy(float), g.change.to_numpy(float)
        mask = np.isfinite(lvl) & np.isfinite(chg) & ~g.untrusted.to_numpy()
        gs, sc = score_peer_group(lvl, chg, mask, cfg)
        sc.index = g.index
        res = g.join(sc)
        res["group_status"], res["n_peers"] = gs.status, gs.n_valid
        res["peer_median_change"], res["peer_mad_change"] = gs.med_change, gs.mad_change
        status = []
        for r in res.itertuples():
            if r.untrusted:
                status.append("UNTRUSTED_NO_VERDICT")
            elif not np.isfinite(r.value) or not np.isfinite(r.change):
                status.append("MISSING")
            elif gs.status == INSUFFICIENT_PEERS:
                status.append("INSUFFICIENT_PEERS")
            elif gs.status == ZERO_MAD:
                status.append("ZERO_MAD")
            elif gs.status == VALID and r.robust_score >= float(cfg["robust_z_threshold"]):
                status.append("UNUSUAL_VS_PEERS")
            else:
                status.append("TYPICAL_VS_PEERS")
        res["exploratory_status"] = status
        out.append(res)
    return pd.concat(out) if out else pd.DataFrame()


def trajectory_shapes(frame: pd.DataFrame) -> pd.DataFrame:
    """Retrospective descriptive shape per device/parameter/channel over stages."""
    rows = []
    for (ds, dev, par, ch), g in frame.sort_values("measurement_stage").groupby(
            ["dataset_id", "device_id", "parameter", "channel"]):
        g = g[g.phase.isin(STRESS_PHASES)] if "phase" in g else g
        rows.append(dict(dataset_id=ds, device_id=dev, parameter=par, channel=ch,
                         shape=describe_trajectory(g.stress_value, g.value), n_points=int(g.value.notna().sum())))
    return pd.DataFrame(rows)
