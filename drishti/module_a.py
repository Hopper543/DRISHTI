"""Module A: lot-relative anomaly detection on early (0 h / 24 h) readings.

Definitions (docs/MODULE_A.md has the rationale and evaluation)
---------------------------------------------------------------
Peer group    devices sharing part_number, lot_id, parameter, unit and
              test_condition. Only rows with both readings present, a known
              specification and no source quality flag act as peers.
Features      level  = value_24h                (current value)
              change = value_24h - value_0h     (change from baseline)
Robust z      z = (x - median) / (1.4826 * MAD) within the peer group, for level
              and change separately. robust_score = max(|z_level|, |z_change|).
Detectors     Isolation Forest and ECOD are fitted ONCE per part/parameter on the
              lot-standardised [z_level, z_change] of historical TRAIN lots, then
              score new devices' lot-standardised features. (Fitting them inside a
              30-device lot makes them rank-only: ECOD in particular cannot say how
              extreme the most extreme device is.)
Normalisation each detector score -> empirical percentile among the same scores
              on held-out CALIBRATION lots.
Ensemble      E = weighted mean of the three percentiles (0..1; weights in config).
Flag rule     config module_a.flag_rule:
              'robust'   (default) robust_score >= robust_z_threshold (3.5, the
                         Iglewicz-Hoaglin modified-z cut-off; not tuned on labels)
              'ensemble' E >= tau, tau set on labelled SYNTHETIC validation lots to
                         a target healthy-parameter flag rate (fallback: the
                         (1 - reference_exceedance) quantile on calibration lots)
              'either'   robust OR ensemble
              On validation lots the robust rule ranked planted anomalies as well
              as IF/ECOD and better than the ensemble at equal flag rates, so it is
              the default; IF/ECOD/ensemble are kept as baselines and evidence.
Guards        < min_peers valid peers -> INSUFFICIENT_PEERS (historical comparison
              shown as evidence only); MAD ~ 0 -> ZERO_MAD (no score invented).
Lot checks    lot median vs historical lot medians (robust z) -> A_LOT_SHIFT;
              lot MAD / historical median within-lot MAD -> A_LOT_SPREAD.

An anomaly score is statistical unusualness, NOT a probability of failure.
Calibration lots contain planted anomalies at generator rates, so tau fixes a
reference exceedance rate, not a guaranteed false-alarm rate on new lots.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from sklearn.ensemble import IsolationForest

from .schema import DQ_MISSING_INPUT, DQ_SOURCE_FLAG, SPEC_UNKNOWN

MAD_K = 1.4826
GROUP_KEY = ["part_number", "lot_id", "parameter", "unit", "test_condition"]
DETECTORS = ("robust", "isolation_forest", "ecod")
SCORE_COL = {"robust": "robust_score", "isolation_forest": "if_score", "ecod": "ecod_score"}

VALID = "VALID"
INSUFFICIENT_PEERS = "INSUFFICIENT_PEERS"
ZERO_MAD = "ZERO_MAD"
UNUSUAL_CODES = ("A_UNUSUAL_CHANGE", "A_UNUSUAL_LEVEL")


def robust_center_scale(x: np.ndarray, rel_tol: float) -> tuple[float, float, bool]:
    """Median, MAD and whether the MAD is (numerically) zero."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, mad, mad <= rel_tol * max(1.0, abs(med))


@dataclass
class GroupScores:
    status: str
    n_valid: int
    med_level: float = np.nan
    mad_level: float = np.nan
    med_change: float = np.nan
    mad_change: float = np.nan
    zero_mad_features: tuple = ()


def score_peer_group(level: np.ndarray, change: np.ndarray, peer_mask: np.ndarray,
                     cfg: dict) -> tuple[GroupScores, pd.DataFrame]:
    """Lot-relative robust standardisation of one peer group."""
    n = len(level)
    out = pd.DataFrame({"z_level": np.nan, "z_change": np.nan, "robust_score": np.nan}, index=range(n))
    nv = int(peer_mask.sum())
    tol = float(cfg.get("zero_mad_rel_tol", 1e-9))
    if nv == 0:
        return GroupScores(INSUFFICIENT_PEERS, 0), out
    ml, sl, zl = robust_center_scale(level[peer_mask], tol)
    mc, sc, zc = robust_center_scale(change[peer_mask], tol)
    g = GroupScores(VALID, nv, ml, sl, mc, sc, tuple(k for k, z in (("level", zl), ("change", zc)) if z))
    # z for non-degenerate features is kept as evidence even without a verdict.
    # A zero MAD is never replaced by a tiny number.
    if not zl:
        out.loc[peer_mask, "z_level"] = (level[peer_mask] - ml) / (MAD_K * sl)
    if not zc:
        out.loc[peer_mask, "z_change"] = (change[peer_mask] - mc) / (MAD_K * sc)
    if nv < int(cfg["min_peers"]):
        g.status = INSUFFICIENT_PEERS
    elif zl or zc:
        g.status = ZERO_MAD
    else:
        out.loc[peer_mask, "robust_score"] = out.loc[peer_mask, ["z_level", "z_change"]].abs().max(axis=1)
    return g, out


def peer_mask(frame: pd.DataFrame) -> np.ndarray:
    finite = np.isfinite(frame[["value_0h", "value_24h"]].to_numpy(dtype=float)).all(axis=1)
    if "row_flags" in frame:
        bad = frame.row_flags.map(lambda fl: DQ_SOURCE_FLAG in fl or SPEC_UNKNOWN in fl).to_numpy(dtype=bool)
        return finite & ~bad
    return finite


def standardise(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-row lot-relative z-scores and per-group summaries."""
    f = frame
    rows, groups = [], []
    for key, g in f.groupby(GROUP_KEY, sort=False, dropna=False):
        lvl = g.value_24h.to_numpy(dtype=float)
        chg = (g.value_24h - g.value_0h).to_numpy(dtype=float)
        gs, sc = score_peer_group(lvl, chg, peer_mask(g), cfg)
        sc.index = g.index
        sc["group_status"], sc["n_peers"] = gs.status, gs.n_valid
        sc["peer_median_level"], sc["peer_mad_level"] = gs.med_level, gs.mad_level
        sc["peer_median_change"], sc["peer_mad_change"] = gs.med_change, gs.mad_change
        sc["zero_mad_features"] = ",".join(gs.zero_mad_features)
        sc["level"], sc["change"] = lvl, chg
        rows.append(sc)
        groups.append(dict(zip(GROUP_KEY, key), status=gs.status, n_peers=gs.n_valid, n_rows=len(g),
                           median_level=gs.med_level, mad_level=gs.mad_level, median_change=gs.med_change,
                           mad_change=gs.mad_change, zero_mad_features=",".join(gs.zero_mad_features)))
    return pd.concat(rows).loc[f.index], pd.DataFrame(groups)


def _quantiles(x, n: int = 401) -> list[float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.quantile(x, np.linspace(0, 1, n)).tolist() if len(x) else []


def percentile(value, ref_q: list[float]) -> np.ndarray:
    """Empirical CDF position of each score within a reference quantile grid."""
    v = np.asarray(value, dtype=float)
    if not ref_q:
        return np.full(v.shape, np.nan)
    q = np.asarray(ref_q)
    out = np.interp(v, q, np.linspace(0, 1, len(q)), left=0.0, right=1.0)
    return np.where(np.isfinite(v), out, np.nan)


def ensemble_score(pcts: np.ndarray, weights: dict) -> np.ndarray:
    """Weighted mean of detector percentiles; columns ordered as DETECTORS."""
    w = np.array([float(weights.get(d, 1.0)) for d in DETECTORS])
    return (np.asarray(pcts, dtype=float) * w).sum(axis=1) / w.sum()


# --------------------------------------------------------------------------
# Reference ("fitted" Module A)
# --------------------------------------------------------------------------

@dataclass
class ModuleAReference:
    by_param: dict                       # "part|parameter" -> JSON-able statistics
    meta: dict
    detectors: dict = field(default_factory=dict)  # "part|parameter" -> {"isolation_forest", "ecod"}

    @staticmethod
    def key(part: str, param: str) -> str:
        return f"{part}|{param}"

    def get(self, part: str, param: str) -> dict | None:
        return self.by_param.get(self.key(part, param))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "reference.json").write_text(json.dumps({"meta": self.meta, "by_param": self.by_param}, indent=1))
        joblib.dump(self.detectors, directory / "detectors.joblib", compress=3)

    @classmethod
    def load(cls, directory: Path) -> "ModuleAReference":
        d = json.loads((Path(directory) / "reference.json").read_text())
        return cls(by_param=d["by_param"], meta=d["meta"], detectors=joblib.load(Path(directory) / "detectors.joblib"))


def _detector_scores(z: np.ndarray, det: dict) -> tuple[np.ndarray, np.ndarray]:
    return -det["isolation_forest"].score_samples(z), det["ecod"].decision_function(z)


def _add_detector_scores(frame: pd.DataFrame, z_rows: pd.DataFrame, reference: "ModuleAReference") -> pd.DataFrame:
    out = z_rows.copy()
    out["if_score"] = np.nan
    out["ecod_score"] = np.nan
    ok = out.robust_score.notna()
    for (part, param), idx in frame[ok].groupby(["part_number", "parameter"]).groups.items():
        det = reference.detectors.get(ModuleAReference.key(part, param))
        if det is None:
            continue
        z = out.loc[idx, ["z_level", "z_change"]].to_numpy(dtype=float)
        out.loc[idx, "if_score"], out.loc[idx, "ecod_score"] = _detector_scores(z, det)
    return out


def build_reference(train: pd.DataFrame, calibration: pd.DataFrame, cfg: dict) -> ModuleAReference:
    """Fit detectors on TRAIN lots; percentile grids and tau on CALIBRATION lots.
    Uses only value_0h and value_24h."""
    ifc = cfg.get("isolation_forest", {})
    tr_rows, tr_groups = standardise(train, cfg)
    ca_rows, _ = standardise(calibration, cfg)
    ref = ModuleAReference({}, {}, {})
    for (part, param), g in train.join(tr_rows).groupby(["part_number", "parameter"]):
        k = ModuleAReference.key(part, param)
        zfit = g.loc[g.robust_score.notna(), ["z_level", "z_change"]].to_numpy(dtype=float)
        iso = IsolationForest(n_estimators=int(ifc.get("n_estimators", 200)),
                              random_state=int(ifc.get("random_state", 26170))).fit(zfit)
        ref.detectors[k] = {"isolation_forest": iso, "ecod": ECOD().fit(zfit)}
        ok = np.isfinite(g[["value_0h", "value_24h"]]).all(axis=1)
        lvl, chg = g.level[ok].to_numpy(), g.change[ok].to_numpy()
        gg = tr_groups[(tr_groups.part_number == part) & (tr_groups.parameter == param)]

        def med_mad(x):
            x = np.asarray(x, dtype=float)
            m = float(np.median(x))
            return [m, float(np.median(np.abs(x - m)))]

        entry = {"unit": str(g.unit.iloc[0]), "n_train_lots": int(g.lot_id.nunique()), "n_train_rows": int(len(zfit)),
                 "pooled_level": med_mad(lvl), "pooled_change": med_mad(chg),
                 "lot_median_level": med_mad(gg.median_level), "lot_median_change": med_mad(gg.median_change),
                 "within_lot_mad_level": float(np.median(gg.mad_level)),
                 "within_lot_mad_change": float(np.median(gg.mad_change))}
        cal = calibration[(calibration.part_number == part) & (calibration.parameter == param)]
        crow = _add_detector_scores(cal, ca_rows.loc[cal.index], ref)
        crow = crow[crow.robust_score.notna()]
        entry["score_quantiles"] = {d: _quantiles(crow[SCORE_COL[d]]) for d in DETECTORS}
        pc = np.column_stack([percentile(crow[SCORE_COL[d]], entry["score_quantiles"][d]) for d in DETECTORS])
        e = ensemble_score(pc, cfg.get("ensemble_weights", {}))
        entry["ensemble_threshold"] = float(np.quantile(e, 1 - float(cfg["reference_exceedance"])))
        entry["n_calibration_rows"] = int(len(e))
        entry["n_calibration_lots"] = int(cal.lot_id.nunique())
        ref.by_param[k] = entry
    ref.meta = {
        "detector_fit_population": "TRAIN split lots: lot-standardised [z_level, z_change] of valid peer groups",
        "normalisation_population": "CALIBRATION split lots (held out from detector fitting)",
        "caveat": "Both populations contain planted anomalies at generator rates; tau is a reference exceedance "
                  "rate on those lots, not a guaranteed false-alarm rate on new lots.",
        "features": ["value_24h (level)", "value_24h - value_0h (change)"],
        "config": {k: cfg[k] for k in ("min_peers", "robust_z_threshold", "reference_exceedance",
                                       "lot_shift_z_threshold", "lot_spread_ratio_threshold")},
        "ensemble": "weighted mean of calibration percentiles of robust, Isolation Forest and ECOD scores",
        "weights": cfg.get("ensemble_weights", {}),
    }
    return ref


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def _hist_z(x, stat):
    med, mad = stat
    x = np.asarray(x, dtype=float)
    return (x - med) / (MAD_K * mad) if mad and mad > 0 else np.full(x.shape, np.nan)


def _unusual(row, rule: str) -> bool:
    """Flag rule: 'robust' (default), 'ensemble' or 'either'. Falls back to the
    robust rule when no ensemble reference exists for the part/parameter."""
    has_ens = np.isfinite(row.ensemble_score) and np.isfinite(row.ensemble_threshold)
    if rule == "robust" or not has_ens:
        return bool(row.a_flag_robust)
    if rule == "ensemble":
        return bool(row.a_flag_ensemble)
    return bool(row.a_flag_robust or row.a_flag_ensemble)


def set_validation_threshold(reference: ModuleAReference, scored: pd.DataFrame, healthy: np.ndarray,
                             target_rate: float) -> float:
    """Global ensemble threshold giving `target_rate` flags among KNOWN-HEALTHY
    parameters of labelled validation lots (synthetic labels). Stored in meta."""
    e = scored.ensemble_score.to_numpy(dtype=float)
    ok = np.isfinite(e) & np.asarray(healthy, dtype=bool)
    thr = float(np.quantile(e[ok], 1 - target_rate))
    reference.meta["ensemble_threshold_validation"] = thr
    reference.meta["ensemble_threshold_note"] = (
        f"Global ensemble threshold = {1 - target_rate:.0%} quantile of the ensemble score over known-healthy "
        f"parameters of SYNTHETIC validation lots (n={int(ok.sum())}). Synthetic labels only; must be re-set on "
        "approved real reference lots before any real use.")
    return thr


def run_module_a(frame: pd.DataFrame, reference: ModuleAReference | None, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score validated early inputs.

    Returns (per-row findings incl. list column `a_codes`, per-lot/parameter table).
    """
    f = frame.copy()
    if "test_condition" not in f:
        f["test_condition"] = "default"
    if "row_flags" not in f:
        f["row_flags"] = [[] for _ in range(len(f))]
    z_rows, groups = standardise(f, cfg)
    scored = _add_detector_scores(f, z_rows, reference) if reference else z_rows.assign(if_score=np.nan, ecod_score=np.nan)
    r = f[["sample_id", "device_id"] + GROUP_KEY].join(scored)
    zt = float(cfg["robust_z_threshold"])
    for c in ("pct_robust", "pct_isolation_forest", "pct_ecod", "ensemble_score", "ensemble_threshold",
              "hist_z_level", "hist_z_change"):
        r[c] = np.nan
    if reference:
        for (part, param), idx in r.groupby(["part_number", "parameter"]).groups.items():
            ref = reference.get(part, param)
            if ref is None:
                continue
            sub = r.loc[idx]
            pc = np.column_stack([percentile(sub[SCORE_COL[d]], ref["score_quantiles"][d]) for d in DETECTORS])
            r.loc[idx, ["pct_robust", "pct_isolation_forest", "pct_ecod"]] = pc
            r.loc[idx, "ensemble_score"] = ensemble_score(pc, reference.meta.get("weights", {}))
            r.loc[idx, "ensemble_threshold"] = ref["ensemble_threshold"]
            r.loc[idx, "hist_z_level"] = _hist_z(sub.level, ref["pooled_level"])
            r.loc[idx, "hist_z_change"] = _hist_z(sub.change, ref["pooled_change"])

    # ---- lot-level shift / spread vs historical lots ----------------------
    g2 = groups.copy()
    for c in ("lot_z_level", "lot_z_change", "spread_ratio_level", "spread_ratio_change"):
        g2[c] = np.nan
    lot_codes_col = []
    for i, g in g2.iterrows():
        ref = reference.get(g.part_number, g.parameter) if reference else None
        codes = []
        if ref is None:
            codes.append("A_NO_HISTORICAL_REFERENCE")
        elif g.n_peers > 0 and np.isfinite(g.median_level):
            zl = float(_hist_z(g.median_level, ref["lot_median_level"]))
            zc = float(_hist_z(g.median_change, ref["lot_median_change"]))
            g2.at[i, "lot_z_level"], g2.at[i, "lot_z_change"] = zl, zc
            # Shift/spread flags only for groups that can give a peer verdict.
            if g.status == VALID:
                if np.nanmax(np.abs([zl, zc])) >= float(cfg["lot_shift_z_threshold"]):
                    codes.append("A_LOT_SHIFT")
                srl = g.mad_level / ref["within_lot_mad_level"] if ref["within_lot_mad_level"] > 0 else np.nan
                src = g.mad_change / ref["within_lot_mad_change"] if ref["within_lot_mad_change"] > 0 else np.nan
                g2.at[i, "spread_ratio_level"], g2.at[i, "spread_ratio_change"] = srl, src
                if np.nanmax([srl, src, 0.0]) >= float(cfg["lot_spread_ratio_threshold"]):
                    codes.append("A_LOT_SPREAD")
        lot_codes_col.append(codes)
    g2["lot_codes"] = lot_codes_col
    lot_lookup = {tuple(g[k] for k in GROUP_KEY): g.lot_codes for _, g in g2.iterrows()}

    # ---- rule outcomes -------------------------------------------------------
    rule = cfg.get("flag_rule", "robust")
    gthr = reference.meta.get("ensemble_threshold_validation") if reference else None
    if gthr is not None:
        r["ensemble_threshold"] = np.where(r.ensemble_score.notna(), gthr, np.nan)
    r["a_flag_robust"] = r.robust_score >= zt
    r["a_flag_ensemble"] = r.ensemble_score >= r.ensemble_threshold

    # ---- per-row reason codes ----------------------------------------------
    codes_col = []
    for i, row in r.iterrows():
        flags = f.at[i, "row_flags"]
        codes = []
        if DQ_MISSING_INPUT in flags:
            codes.append("A_NOT_SCORED_MISSING")
        elif SPEC_UNKNOWN in flags:
            codes.append("A_NOT_SCORED_UNSUPPORTED")
        elif DQ_SOURCE_FLAG in flags:
            codes.append("A_UNTRUSTED_SOURCE_FLAG")
        elif row.group_status == INSUFFICIENT_PEERS:
            codes.append("A_INSUFFICIENT_PEERS")
            if np.nanmax(np.abs([row.hist_z_level, row.hist_z_change, 0.0])) >= zt:
                codes.append("A_HISTORICAL_UNUSUAL")
        elif row.group_status == ZERO_MAD:
            codes.append("A_ZERO_MAD")
        else:
            unusual = _unusual(row, rule)
            if unusual:
                zl, zc = abs(np.nan_to_num(row.z_level)), abs(np.nan_to_num(row.z_change))
                codes.append("A_UNUSUAL_CHANGE" if zc >= zl else "A_UNUSUAL_LEVEL")
        codes.extend(lot_lookup.get(tuple(row[k] for k in GROUP_KEY), []))
        codes_col.append(list(dict.fromkeys(codes)))
    r["a_codes"] = codes_col
    r["a_unusual"] = r.a_codes.map(lambda c: any(x in c for x in UNUSUAL_CODES))
    return r, g2
