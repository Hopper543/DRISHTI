"""Module B: early forecasting of the 168 h value from [value_0h, value_24h].

* Features are EXACTLY [value_0h, value_24h]; target is value_168h, joined by
  sample_id. IDs, lots, scenarios, limits, splits and future readings never
  enter the model.
* One model set per (part_number, parameter): LightGBM quantile regressors at
  q05 / q50 / q95. The q50 model is the central forecast.
* Conformalized quantile regression (Romano et al., 2019) with LOT-level
  calibration: for each calibration lot take the maximum nonconformity
  max(lo - y, y - hi) over its devices, then the finite-sample
  ceil((n_lots + 1) * coverage)-th smallest lot score is the margin added to both
  sides. This targets whole-lot coverage on lots exchangeable with the
  calibration lots; it is NOT a guarantee for shifted or tiny lots, nor a joint
  guarantee across parameters.
* Inference validates units and training domain and escalates unsupported,
  invalid or uncalibrated inputs instead of forecasting.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .schema import FEATURE_COLUMNS, feature_matrix

QNAMES = {0.05: "q05", 0.5: "q50", 0.95: "q95"}


def key(part: str, param: str) -> str:
    return f"{part}__{param}"


def join_targets(inputs: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Attach value_168h by sample_id (one-to-one). Target stays a separate column."""
    return inputs.merge(targets[["sample_id", "value_168h"]], on="sample_id", how="left", validate="one_to_one")


def lot_conformal_margin(lo, hi, y, lots, coverage: float) -> tuple[float, int]:
    s = np.maximum(np.asarray(lo) - y, y - np.asarray(hi))
    lot_scores = pd.DataFrame({"lot": np.asarray(lots), "s": s}).groupby("lot").s.max().to_numpy()
    n = len(lot_scores)
    k = math.ceil((n + 1) * coverage)
    if n == 0 or k > n:
        return float("inf"), n
    return max(0.0, float(np.sort(lot_scores)[k - 1])), n


@dataclass
class Forecaster:
    boosters: dict = field(default_factory=dict)      # key -> {qname: lgb.Booster}
    calibration: dict = field(default_factory=dict)   # key -> dict
    domain: dict = field(default_factory=dict)        # key -> dict
    units: dict = field(default_factory=dict)         # key -> unit
    manifest: dict = field(default_factory=dict)

    # ---- training ----------------------------------------------------------
    @classmethod
    def fit(cls, inputs: pd.DataFrame, targets: pd.DataFrame, cfg: dict) -> "Forecaster":
        d = join_targets(inputs, targets)
        fc = cls()
        dq = cfg.get("domain_quantiles", [0.005, 0.995])
        for (part, param), g in d.groupby(["part_number", "parameter"]):
            k = key(part, param)
            train = g[g.split == "train"]
            train = train[np.isfinite(train[FEATURE_COLUMNS + ["value_168h"]]).all(axis=1)]
            cal = g[g.split == "calibration"]
            cal = cal[np.isfinite(cal[FEATURE_COLUMNS + ["value_168h"]]).all(axis=1)]
            x, y = feature_matrix(train), train.value_168h.to_numpy(dtype=float)
            fc.boosters[k] = {}
            for q in cfg["quantiles"]:
                m = lgb.LGBMRegressor(objective="quantile", alpha=q, **cfg["lightgbm"]).fit(x, y)
                fc.boosters[k][QNAMES[q]] = m.booster_
            fc.units[k] = str(g.unit.iloc[0])
            ch = train.value_24h - train.value_0h
            fc.domain[k] = {
                "value_0h": np.quantile(train.value_0h, dq).tolist(),
                "value_24h": np.quantile(train.value_24h, dq).tolist(),
                "change": np.quantile(ch, dq).tolist(),
                "quantiles": dq, "n_train": int(len(train)), "n_train_lots": int(train.lot_id.nunique())}
            if len(cal):
                p = fc._raw_quantiles(k, feature_matrix(cal))
                margin, n = lot_conformal_margin(p[:, 0], p[:, 2], cal.value_168h.to_numpy(float),
                                                 cal.lot_id, float(cfg["conformal_coverage"]))
                fc.calibration[k] = {"margin": margin, "n_calibration_lots": n,
                                     "n_calibration_rows": int(len(cal)), "coverage_target": cfg["conformal_coverage"],
                                     "method": "CQR, lot-max nonconformity, finite-sample quantile"}
        fc.manifest = {
            "features": FEATURE_COLUMNS, "target": "value_168h", "join_key": "sample_id",
            "model_family": "LightGBM quantile regression (q05, q50, q95) per part_number/parameter",
            "lightgbm_params": cfg["lightgbm"], "training_split": "train", "calibration_split": "calibration",
            "evidence": "SYNTHETIC_ONLY", "lightgbm_version": lgb.__version__,
        }
        return fc

    # ---- inference ---------------------------------------------------------
    def _raw_quantiles(self, k: str, x: np.ndarray) -> np.ndarray:
        b = self.boosters[k]
        p = np.column_stack([b["q05"].predict(x), b["q50"].predict(x), b["q95"].predict(x)])
        return np.sort(p, axis=1)  # prevent quantile crossing

    def contributions(self, part: str, param: str, x: np.ndarray) -> np.ndarray:
        """TreeSHAP contributions of the q50 model: columns [value_0h, value_24h, bias].
        Explains the fitted model's output, not device physics."""
        return self.boosters[key(part, param)]["q50"].predict(np.atleast_2d(x), pred_contrib=True)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-row forecast with b_codes. Never raises for bad rows; escalates them."""
        out = pd.DataFrame(index=frame.index, data={"prediction": np.nan, "lower": np.nan, "upper": np.nan,
                                                    "raw_q05": np.nan, "raw_q95": np.nan, "conformal_margin": np.nan,
                                                    "shap_value_0h": np.nan, "shap_value_24h": np.nan,
                                                    "shap_base": np.nan})
        codes = {i: [] for i in frame.index}
        for (part, param), g in frame.groupby(["part_number", "parameter"]):
            k = key(part, param)
            if k not in self.boosters:
                for i in g.index:
                    codes[i].append("B_UNSUPPORTED_PART")
                continue
            cal = self.calibration.get(k)
            if cal is None or not np.isfinite(cal["margin"]):
                for i in g.index:
                    codes[i].append("B_NO_CALIBRATION")
                continue
            bad_unit = g.unit.astype(str) != self.units[k]
            for i in g.index[bad_unit]:
                codes[i].append("B_UNIT_MISMATCH")
            finite = np.isfinite(g[FEATURE_COLUMNS].to_numpy(dtype=float)).all(axis=1)
            for i in g.index[~finite]:
                codes[i].append("B_NOT_RUN_MISSING_INPUT")
            ok = g[finite & ~bad_unit.to_numpy()]
            if ok.empty:
                continue
            x = feature_matrix(ok)
            p = self._raw_quantiles(k, x)
            m = cal["margin"]
            out.loc[ok.index, ["raw_q05", "prediction", "raw_q95"]] = p
            out.loc[ok.index, "lower"] = p[:, 0] - m
            out.loc[ok.index, "upper"] = p[:, 2] + m
            out.loc[ok.index, "conformal_margin"] = m
            out.loc[ok.index, ["shap_value_0h", "shap_value_24h", "shap_base"]] = self.contributions(part, param, x)
            dom = self.domain[k]
            ch = ok.value_24h - ok.value_0h
            ood = ~(ok.value_0h.between(*dom["value_0h"]) & ok.value_24h.between(*dom["value_24h"])
                    & ch.between(*dom["change"]))
            for i in ok.index[ood.to_numpy()]:
                codes[i].append("B_OUT_OF_DOMAIN")
        out["b_codes"] = [codes[i] for i in frame.index]
        return out

    # ---- persistence -------------------------------------------------------
    def save(self, directory: Path, extra_manifest: dict | None = None) -> dict:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files = {}
        for k, qs in self.boosters.items():
            for qn, b in qs.items():
                name = f"{k}__{qn}.txt"
                b.save_model(str(directory / name))
                files[name] = _sha(directory / name)
        (directory / "calibration.json").write_text(json.dumps(self.calibration, indent=1))
        (directory / "domain.json").write_text(json.dumps({"domain": self.domain, "units": self.units}, indent=1))
        files["calibration.json"] = _sha(directory / "calibration.json")
        files["domain.json"] = _sha(directory / "domain.json")
        man = dict(self.manifest, **(extra_manifest or {}), files=files)
        man["model_version_hash"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
        (directory / "manifest.json").write_text(json.dumps(man, indent=1))
        self.manifest = man
        return man

    @classmethod
    def load(cls, directory: Path, verify: bool = True) -> "Forecaster":
        directory = Path(directory)
        man = json.loads((directory / "manifest.json").read_text())
        if verify:
            for name, h in man["files"].items():
                if _sha(directory / name) != h:
                    raise ValueError(f"Model artifact {name} does not match its manifest hash")
        fc = cls(manifest=man)
        fc.calibration = json.loads((directory / "calibration.json").read_text())
        dd = json.loads((directory / "domain.json").read_text())
        fc.domain, fc.units = dd["domain"], dd["units"]
        for name in man["files"]:
            if name.endswith(".txt"):
                k, qn = name[:-4].rsplit("__", 1)
                fc.boosters.setdefault(k, {})[qn] = lgb.Booster(model_file=str(directory / name))
        return fc


def _sha(path: Path) -> str:
    # Normalise line endings so hashes match across Git checkouts (autocrlf).
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def interval_status(lower: float, upper: float, spec_lo: float, spec_hi: float) -> str:
    """Compare a forecast interval with specification limits."""
    if not all(np.isfinite([lower, upper, spec_lo, spec_hi])) or lower > upper:
        return "B_INVALID_INTERVAL"
    if upper < spec_lo or lower > spec_hi:
        return "B_PREDICTED_VIOLATION"
    if lower >= spec_lo and upper <= spec_hi:
        return "B_INTERVAL_INSIDE"
    return "B_INTERVAL_CROSSES_LIMIT"


def baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """Persistence (y = v24) and linear extrapolation (y = v0 + 7 (v24 - v0))."""
    return pd.DataFrame({"persistence": frame.value_24h,
                         "linear_extrapolation": frame.value_0h + (168 / 24) * (frame.value_24h - frame.value_0h)},
                        index=frame.index)
