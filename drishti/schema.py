"""Early-input contract (0 h / 24 h readings) and actionable validation.

Blocking problems become `errors` (the run is refused); row-level problems
become `row_flags` so the affected parameter is escalated instead of silently
scored. Messages say what is wrong, where, and how to fix it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .provenance import PROVENANCE_CLASSES
from .specs import SpecRegistry
from .units import conversion_factor

FEATURE_COLUMNS = ["value_0h", "value_24h"]
ID_COLUMNS = ["device_id", "lot_id", "part_number", "parameter", "unit"]
REQUIRED = ID_COLUMNS + FEATURE_COLUMNS + ["provenance_class"]
OPTIONAL = ["sample_id", "spec_lower", "spec_upper", "spec_origin", "test_condition", "quality_flags"]
# Endpoint, label and generator-truth columns can never be supplied at inference.
FORBIDDEN = {"value_168h", "true_value_168h", "scenario", "latent_defect_planted", "true_out_of_spec_168h",
             "label", "is_fail", "target", "lot_wide_degradation_planted", "measurement_fault_planted"}
FUTURE_READING = re.compile(r"^value_(\d+(?:\.\d+)?)h$")
IGNORED_ROUTING = {"split"}  # tolerated for audit, never used as a feature
TEMPLATE_COLUMNS = ["sample_id", "device_id", "lot_id", "part_number", "parameter", "unit", "value_0h",
                    "value_24h", "spec_lower", "spec_upper", "spec_origin", "provenance_class",
                    "test_condition", "quality_flags"]

# Row flag codes (data quality). Reason texts are in drishti/reasons.py.
DQ_MISSING_INPUT = "DQ_MISSING_INPUT"
DQ_IDENTICAL_READINGS = "DQ_IDENTICAL_READINGS"
DQ_SOURCE_FLAG = "DQ_SOURCE_QUALITY_FLAG"
DQ_UNIT_CONVERTED = "DQ_UNIT_CONVERTED"
SPEC_UNKNOWN = "SPEC_UNKNOWN"


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    frame: pd.DataFrame | None = None  # normalized frame (None when blocked)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "info": self.info}


def _rows(mask, limit: int = 8) -> str:
    """1-based CSV data-row numbers (header = row 0) for messages."""
    idx = [int(i) + 1 for i in np.flatnonzero(np.asarray(mask, dtype=bool))]
    s = ", ".join(map(str, idx[:limit]))
    return s + (f" (+{len(idx) - limit} more)" if len(idx) > limit else "")


def validate_early_inputs(raw: pd.DataFrame, registry: SpecRegistry) -> ValidationReport:
    rep = ValidationReport()
    f = raw.copy()
    f.columns = [str(c).strip() for c in f.columns]

    # ---- structural checks -------------------------------------------------
    missing = [c for c in REQUIRED if c not in f.columns]
    if missing:
        rep.errors.append(f"Missing required column(s): {missing}. Download the blank template and keep its header row.")
    leaked = sorted(FORBIDDEN.intersection(f.columns))
    future = sorted(c for c in f.columns if (m := FUTURE_READING.match(c)) and float(m.group(1)) > 24)
    if leaked or future:
        rep.errors.append(
            f"Column(s) {leaked + future} contain endpoint, label or post-24 h information. A 24 h screening "
            "decision may only use value_0h and value_24h; remove these columns.")
    if rep.errors:
        return rep
    for c in sorted(IGNORED_ROUTING.intersection(f.columns)):
        rep.info.append(f"Column '{c}' is kept for audit only and is never used as a model feature.")
    unknown = sorted(set(f.columns) - set(REQUIRED) - set(OPTIONAL) - IGNORED_ROUTING)
    if unknown:
        rep.info.append(f"Ignoring unrecognised column(s): {unknown}.")
        f = f.drop(columns=unknown)
    if f.empty:
        rep.errors.append("The file has a header but no data rows.")
        return rep

    for c in ID_COLUMNS + ["provenance_class"]:
        f[c] = f[c].astype("string").str.strip()
        blank = f[c].isna() | f[c].eq("")
        if blank.any():
            rep.errors.append(f"Blank '{c}' in data row(s) {_rows(blank)}. Every row needs a {c}.")
    f["test_condition"] = (f["test_condition"].astype("string").str.strip().fillna("default")
                           if "test_condition" in f else "default")
    f.loc[f.test_condition.eq(""), "test_condition"] = "default"
    if "sample_id" not in f:
        f["sample_id"] = f.device_id + ":" + f.parameter
        rep.info.append("No sample_id column: derived as device_id:parameter.")
    f["sample_id"] = f.sample_id.astype("string").str.strip()
    f["quality_flags"] = f["quality_flags"].astype("string").fillna("") if "quality_flags" in f else ""

    # ---- provenance --------------------------------------------------------
    bad_prov = ~f.provenance_class.isin(PROVENANCE_CLASSES)
    if bad_prov.any():
        rep.errors.append(f"provenance_class must be one of {list(PROVENANCE_CLASSES)}; "
                          f"invalid in row(s) {_rows(bad_prov)}.")
    elif f.provenance_class.nunique() > 1:
        rep.errors.append("One upload must contain a single provenance class; found "
                          f"{sorted(f.provenance_class.unique())}. Real and synthetic data are analysed separately.")

    # ---- numeric parsing: blank = missing (escalate), text = error ----------
    for c in FEATURE_COLUMNS:
        txt = f[c].astype("string").str.strip()
        num = pd.to_numeric(txt, errors="coerce")
        garbage = num.isna() & txt.notna() & ~txt.str.lower().isin(["", "nan", "na", "n/a", "null"])
        if garbage.any():
            rep.errors.append(f"Non-numeric '{c}' in row(s) {_rows(garbage)} (e.g. '{txt[garbage].iloc[0]}'). "
                              "Use plain numbers in the stated unit; leave a missing reading blank.")
        f[c] = num.astype(float)
    inf = ~np.isfinite(f[FEATURE_COLUMNS].fillna(0).to_numpy()).all(axis=1)
    if inf.any():
        rep.warnings.append(f"Infinite reading(s) in row(s) {_rows(pd.Series(inf))}; treated as missing.")
        f.loc[inf, FEATURE_COLUMNS] = f.loc[inf, FEATURE_COLUMNS].where(np.isfinite(f.loc[inf, FEATURE_COLUMNS]))

    # ---- identity ----------------------------------------------------------
    dup = f.sample_id.duplicated(keep=False)
    if dup.any():
        rep.errors.append(f"Duplicate sample_id in row(s) {_rows(dup)}. Each device/parameter needs one row.")
    dup2 = f.duplicated(["device_id", "parameter", "test_condition"], keep=False)
    if dup2.any():
        rep.errors.append(f"Duplicate device/parameter/test_condition in row(s) {_rows(dup2)}. "
                          "Average replicates outside DRISHTI only if your procedure allows it.")
    multi_lot = f.groupby("device_id").lot_id.nunique()
    if (multi_lot > 1).any():
        rep.errors.append(f"Device(s) assigned to more than one lot: {list(multi_lot[multi_lot > 1].index[:5])}.")
    multi_part = f.groupby("lot_id").part_number.nunique()
    if (multi_part > 1).any():
        rep.errors.append(f"Lot(s) with mixed part numbers: {list(multi_part[multi_part > 1].index[:5])}. "
                          "A lot must contain a single part number.")
    if rep.errors:
        return rep

    # ---- units and specification registry ---------------------------------
    f["row_flags"] = [[] for _ in range(len(f))]
    f["spec_revision"] = None
    for c in ("spec_lower", "spec_upper"):
        f[c] = pd.to_numeric(f[c], errors="coerce") if c in f else np.nan
    f["spec_origin"] = f["spec_origin"].astype("string") if "spec_origin" in f else pd.NA
    for (part, param), idx in f.groupby(["part_number", "parameter"]).groups.items():
        entry = registry.lookup(part, param)
        if entry is None:
            for i in idx:
                f.at[i, "row_flags"] = f.at[i, "row_flags"] + [SPEC_UNKNOWN]
            rep.warnings.append(f"{part}/{param}: not in specification registry {registry.revision}; "
                                "these rows are escalated (no limit check, no forecast).")
            continue
        for unit, uidx in f.loc[idx].groupby("unit").groups.items():
            k = conversion_factor(unit, entry["unit"])
            if k is None:
                rep.errors.append(f"{part}/{param}: unit '{unit}' is incompatible with registry unit "
                                  f"'{entry['unit']}' (rows {_rows(f.index.isin(uidx))}). No conversion is guessed.")
                continue
            if k != 1.0:
                f.loc[uidx, FEATURE_COLUMNS] = f.loc[uidx, FEATURE_COLUMNS] * k
                f.loc[uidx, "unit"] = entry["unit"]
                for i in uidx:
                    f.at[i, "row_flags"] = f.at[i, "row_flags"] + [DQ_UNIT_CONVERTED]
                rep.warnings.append(f"{part}/{param}: converted {len(uidx)} row(s) from {unit} to {entry['unit']} "
                                    f"(x{k:g}).")
        lo, hi = float(entry["spec_lower"]), float(entry["spec_upper"])
        given = f.loc[idx, ["spec_lower", "spec_upper"]]
        disagree = given.notna().any(axis=1) & ~(np.isclose(given.spec_lower, lo) & np.isclose(given.spec_upper, hi))
        if disagree.any():
            rep.errors.append(f"{part}/{param}: limits in the file differ from registry {entry['revision']} "
                              f"[{lo:g}, {hi:g}] {entry['unit']}. Remove the limit columns or correct them.")
        f.loc[idx, ["spec_lower", "spec_upper"]] = [lo, hi]
        f.loc[idx, "spec_origin"] = entry["spec_origin"]
        f.loc[idx, "spec_revision"] = entry["revision"]
    if rep.errors:
        return rep

    # ---- row-level data quality --------------------------------------------
    miss = f[FEATURE_COLUMNS].isna().any(axis=1)
    for i in f.index[miss]:
        f.at[i, "row_flags"] = f.at[i, "row_flags"] + [DQ_MISSING_INPUT]
    if miss.any():
        rep.warnings.append(f"{int(miss.sum())} row(s) have a missing 0 h or 24 h reading "
                            f"(row(s) {_rows(miss)}); they are escalated, never imputed.")
    same = ~miss & (f.value_0h == f.value_24h)
    for i in f.index[same]:
        f.at[i, "row_flags"] = f.at[i, "row_flags"] + [DQ_IDENTICAL_READINGS]
    if same.any():
        rep.warnings.append(f"{int(same.sum())} row(s) have bit-identical 0 h and 24 h readings "
                            f"(row(s) {_rows(same)}): possible stuck or quantised measurement.")
    srcflag = f.quality_flags.astype(str).str.strip().ne("")
    for i in f.index[srcflag]:
        f.at[i, "row_flags"] = f.at[i, "row_flags"] + [DQ_SOURCE_FLAG]
    if srcflag.any():
        rep.warnings.append(f"{int(srcflag.sum())} row(s) carry a source quality flag; they receive no trusted "
                            "automated peer verdict.")
    rep.info.append(f"Validated {len(f)} rows: {f.device_id.nunique()} devices, {f.lot_id.nunique()} lots, "
                    f"{f.parameter.nunique()} parameters, provenance {f.provenance_class.iloc[0]}.")
    rep.frame = f.reset_index(drop=True)
    return rep


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """The ONLY model input: [value_0h, value_24h]. Refuses incomplete rows."""
    x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Incomplete measurements must be escalated, not forecast")
    return x


def blank_template() -> pd.DataFrame:
    return pd.DataFrame(columns=TEMPLATE_COLUMNS)
