"""End-to-end screening: validate -> Module A -> Module B -> decision."""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import load_settings
from .decision import STAGE_LABEL, decide
from .module_a import ModuleAReference, run_module_a
from .module_b import Forecaster, interval_status
from .provenance import REAL_MEASURED
from .schema import ValidationReport, validate_early_inputs
from .specs import SpecRegistry, load_registry


class ProductionGateError(RuntimeError):
    """Raised when production mode is requested without real data and approved specs."""


@dataclass
class Models:
    module_a: ModuleAReference | None
    module_b: Forecaster | None

    @property
    def version(self) -> str:
        mb = self.module_b.manifest.get("model_version_hash", "none") if self.module_b else "none"
        ma = self.module_a.meta.get("reference_hash", "none") if self.module_a else "none"
        return f"drishti-{__version__}/A:{ma}/B:{mb}"


@lru_cache(maxsize=2)
def load_models(models_dir: str | None = None) -> Models:
    d = Path(models_dir) if models_dir else load_settings().models_dir
    a = ModuleAReference.load(d / "module_a") if (d / "module_a" / "reference.json").exists() else None
    b = Forecaster.load(d / "module_b") if (d / "module_b" / "manifest.json").exists() else None
    return Models(a, b)


@dataclass
class ScreeningRun:
    run_id: str
    timestamp_utc: str
    input_sha256: str
    model_version: str
    spec_revision: str
    spec_registry_sha256: str
    mode: str
    provenance_class: str
    validation: ValidationReport
    params: pd.DataFrame = field(default_factory=pd.DataFrame)
    devices: pd.DataFrame = field(default_factory=pd.DataFrame)
    lots: pd.DataFrame = field(default_factory=pd.DataFrame)
    source_name: str = ""

    @property
    def ok(self) -> bool:
        return self.validation.ok

    def summary(self) -> dict:
        counts = self.devices.decision.value_counts().to_dict() if not self.devices.empty else {}
        reason_counts: dict[str, int] = {}
        for codes in self.params.get("reason_codes", []):
            for c in codes:
                reason_counts[c] = reason_counts.get(c, 0) + 1
        return {"run_id": self.run_id, "timestamp_utc": self.timestamp_utc, "input_sha256": self.input_sha256,
                "model_version": self.model_version, "spec_revision": self.spec_revision,
                "spec_registry_sha256": self.spec_registry_sha256, "mode": self.mode, "stage": STAGE_LABEL,
                "provenance_class": self.provenance_class, "source_name": self.source_name,
                "n_devices": int(len(self.devices)), "n_parameters": int(len(self.params)),
                "device_decisions": {k: int(v) for k, v in counts.items()},
                "reason_code_counts": dict(sorted(reason_counts.items())),
                "validation": self.validation.as_dict()}


def hash_input(raw: pd.DataFrame, raw_bytes: bytes | None = None) -> str:
    data = raw_bytes if raw_bytes is not None else raw.to_csv(index=False).encode()
    return hashlib.sha256(data).hexdigest()


def run_screening(raw: pd.DataFrame, *, registry: SpecRegistry | None = None, models: Models | None = None,
                  raw_bytes: bytes | None = None, source_name: str = "", mode: str | None = None) -> ScreeningRun:
    settings = load_settings()
    registry = registry or load_registry()
    models = models or load_models()
    mode = mode or settings.mode
    rep = validate_early_inputs(raw, registry)
    run = ScreeningRun(run_id=uuid.uuid4().hex[:12], timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       input_sha256=hash_input(raw, raw_bytes), model_version=models.version,
                       spec_revision=registry.revision, spec_registry_sha256=registry.sha256, mode=mode,
                       provenance_class="", validation=rep, source_name=source_name)
    if not rep.ok:
        return run
    f = rep.frame
    run.provenance_class = str(f.provenance_class.iloc[0])
    if mode == "production":
        if run.provenance_class != REAL_MEASURED:
            raise ProductionGateError("Production mode requires REAL_MEASURED inputs.")
        if not registry.production_approved:
            raise ProductionGateError("Production mode requires an organisation-approved specification registry.")
        if models.module_b and models.module_b.manifest.get("evidence") == "SYNTHETIC_ONLY":
            raise ProductionGateError("Production mode requires models validated on real lots; "
                                      "shipped models are SYNTHETIC_ONLY.")

    a_rows, lots = run_module_a(f, models.module_a, settings.module_a)
    if models.module_b is not None:
        b = models.module_b.predict(f)
    else:
        b = pd.DataFrame(index=f.index, data={"prediction": np.nan, "lower": np.nan, "upper": np.nan})
        b["b_codes"] = [["B_UNSUPPORTED_PART"] for _ in range(len(f))]
    p = f.join(a_rows.drop(columns=[c for c in a_rows.columns if c in f.columns])).join(b)
    p["interval_code"] = [
        interval_status(r.lower, r.upper, r.spec_lower, r.spec_upper) if np.isfinite(r.lower) else ""
        for r in p.itertuples()]
    if not registry.production_approved:
        p["row_flags"] = [list(fl) + ["DEMO_SPEC_NOT_APPROVED"] for fl in p.row_flags]
    p, devices = decide(p, registry, settings.decision)
    run.params, run.devices, run.lots = p, devices, lots
    return run


def export_frames(run: ScreeningRun) -> dict[str, pd.DataFrame]:
    """Flat, JSON/CSV-friendly result tables."""
    cols = ["sample_id", "device_id", "lot_id", "part_number", "parameter", "unit", "test_condition", "value_0h",
            "value_24h", "change", "spec_lower", "spec_upper", "spec_revision", "n_peers", "peer_median_level",
            "peer_mad_level", "peer_median_change", "peer_mad_change", "z_level", "z_change", "robust_score",
            "if_score", "ecod_score", "ensemble_score", "ensemble_threshold", "prediction", "lower", "upper",
            "conformal_margin", "decision", "basis", "rule", "decisive_codes", "reason_codes", "provenance_class"]
    p = run.params[[c for c in cols if c in run.params.columns]].copy()
    for c in ("decisive_codes", "reason_codes"):
        p[c] = p[c].map(lambda v: ";".join(v))
    d = run.devices.copy()
    d["reason_codes"] = d.reason_codes.map(lambda v: ";".join(v))
    for k, v in {"run_id": run.run_id, "model_version": run.model_version, "spec_revision": run.spec_revision}.items():
        p[k] = v
        d[k] = v
    return {"parameters": p, "devices": d}


def results_json(run: ScreeningRun) -> str:
    fr = export_frames(run)
    doc = {"summary": run.summary(),
           "devices": json.loads(fr["devices"].to_json(orient="records")),
           "parameters": json.loads(fr["parameters"].to_json(orient="records"))}
    return json.dumps(doc, indent=1)
