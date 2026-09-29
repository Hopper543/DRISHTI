"""Train/fit DRISHTI demo models from the SYNTHETIC fixture and save artifacts.

    python scripts/train.py

Module A: detectors fitted on TRAIN lots, percentile grids on CALIBRATION lots,
          optional ensemble threshold on labelled VALIDATION lots.
Module B: LightGBM q05/q50/q95 per part/parameter on TRAIN lots; lot-level CQR
          margin on CALIBRATION lots.
Test, shifted (ood_test) and edge lots are never touched here.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from drishti import __version__, data  # noqa: E402
from drishti.config import load_settings  # noqa: E402
from drishti.module_a import build_reference, run_module_a, set_validation_threshold  # noqa: E402
from drishti.module_b import Forecaster  # noqa: E402


def frame_hash(f) -> str:
    return hashlib.sha256(f.to_csv(index=False).encode()).hexdigest()


def main() -> None:
    s = load_settings()
    e = data.synthetic_early_inputs()
    t = data.synthetic_targets()
    gt = data.synthetic_ground_truth()
    assert set(e.split.unique()) >= {"train", "calibration", "validation"}
    out = s.models_dir
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---- Module A --------------------------------------------------------
    tr, ca, va = (e[e.split == k] for k in ("train", "calibration", "validation"))
    ref = build_reference(tr, ca, s.module_a)
    rows, _ = run_module_a(va, ref, s.module_a)
    lab = rows.merge(gt[["sample_id", "scenario"]], on="sample_id", how="left", validate="one_to_one")
    healthy = lab.scenario.str.startswith("healthy").to_numpy()
    thr = set_validation_threshold(ref, rows, healthy, float(s.module_a["target_healthy_flag_rate"]))
    ref.meta.update(created_utc=now, drishti_version=__version__, evidence="SYNTHETIC_ONLY",
                    train_lots=int(tr.lot_id.nunique()), calibration_lots=int(ca.lot_id.nunique()),
                    validation_lots=int(va.lot_id.nunique()))
    ref.meta["reference_hash"] = hashlib.sha256(json.dumps(ref.by_param, sort_keys=True).encode()).hexdigest()[:16]
    ref.save(out / "module_a")
    print(f"Module A reference saved ({len(ref.by_param)} part/parameters); validation ensemble threshold {thr:.4f}")

    # ---- Module B --------------------------------------------------------
    fit_rows = e[e.split.isin(["train", "calibration"])]
    fc = Forecaster.fit(fit_rows, t, s.module_b)
    man = fc.save(out / "module_b", extra_manifest={
        "created_utc": now, "drishti_version": __version__,
        "training_inputs_sha256": frame_hash(fit_rows[["sample_id", "value_0h", "value_24h", "split"]]),
        "numpy_version": np.__version__})
    for k, c in sorted(fc.calibration.items()):
        print(f"  {k:40s} margin={c['margin']:.4g} ({c['n_calibration_lots']} calibration lots)")
    print(f"Module B saved, model_version_hash={man['model_version_hash']}")


if __name__ == "__main__":
    main()
