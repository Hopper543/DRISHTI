"""Versioned specification registry.

The registry is the single source of specification limits. Upload files may
repeat limits, but they must agree with the registry. The shipped registry is
DEMO_ONLY and every row has approved_for_production = False.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_settings

REQUIRED_COLUMNS = ["part_number", "parameter", "unit", "spec_lower", "spec_upper", "spec_origin",
                    "approved_for_production", "revision"]


@dataclass(frozen=True)
class SpecRegistry:
    table: pd.DataFrame
    path: str
    sha256: str

    @property
    def revision(self) -> str:
        revs = sorted(self.table.revision.astype(str).unique())
        return revs[0] if len(revs) == 1 else "+".join(revs)

    @property
    def production_approved(self) -> bool:
        return bool(self.table.approved_for_production.all())

    def lookup(self, part: str, parameter: str) -> dict | None:
        m = self.table[(self.table.part_number == part) & (self.table.parameter == parameter)]
        return None if m.empty else m.iloc[0].to_dict()

    def parameters_for(self, part: str) -> list[str]:
        return sorted(self.table.loc[self.table.part_number == part, "parameter"])

    def parts(self) -> list[str]:
        return sorted(self.table.part_number.unique())


def validate_registry(table: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"Specification registry missing columns: {sorted(missing)}")
    if table.duplicated(["part_number", "parameter"]).any():
        raise ValueError("Duplicate part/parameter entry in specification registry")
    lim = table[["spec_lower", "spec_upper"]].to_numpy(dtype=float)
    if not np.isfinite(lim).all():
        raise ValueError("Specification registry contains non-finite limits")
    if (lim[:, 0] >= lim[:, 1]).any():
        raise ValueError("Specification registry has spec_lower >= spec_upper")


def load_registry(path: str | Path | None = None) -> SpecRegistry:
    p = Path(path) if path else load_settings().spec_registry_path
    raw = p.read_bytes()
    table = pd.read_csv(p, dtype={"part_number": str, "parameter": str, "unit": str, "revision": str})
    table["approved_for_production"] = table.approved_for_production.astype(str).str.lower().eq("true")
    validate_registry(table)
    return SpecRegistry(table=table, path=str(p), sha256=hashlib.sha256(raw).hexdigest())
