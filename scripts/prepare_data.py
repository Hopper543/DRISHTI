"""Copy the DRISHTI dataset package into this repository's data layout.

    python scripts/prepare_data.py --package-dir <path to drishti_dataset_v1>

Writes
  data/demo/    SYNTHETIC fixtures (project generated; committed to Git)
  data/public/  real rows whose sources state public use / CC BY (committed)
  data/local/   full real tables incl. sources without an explicit reuse
                statement, IGBT quarantine (git-ignored; never pushed)

Delivered-file SHA-256 values in the package's docs/file_checksums.json are
verified before anything is copied. Values are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drishti.provenance import DATASETS, redistributable_dataset_ids  # noqa: E402

VERIFY = [
    "data/synthetic_early_inputs.csv", "data/synthetic_measurements.csv", "data/synthetic_lots.csv",
    "evaluation/synthetic_targets_168h.csv", "evaluation/synthetic_ground_truth.csv",
    "data/real_measurements.csv", "data/real_drift_view.csv", "data/real_device_folds.csv",
    "evaluation/reram_functional_outcomes.csv", "data/secom_features.parquet", "data/secom_splits.csv",
    "evaluation/secom_labels.csv", "data/igbt_scalar_quarantined.parquet",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(pkg: Path) -> None:
    listed = {r["path"]: r["sha256"] for r in json.loads((pkg / "docs/file_checksums.json").read_text())}
    for rel in VERIFY:
        if rel not in listed:
            print(f"  [skip] {rel}: no delivered checksum listed")
            continue
        got = sha256(pkg / rel)
        if got != listed[rel]:
            raise SystemExit(f"Checksum mismatch for {rel}; package changed since delivery. Stopping.")
        print(f"  [ok] {rel}")


def read_real(pkg: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(pkg / name, dtype={"source_device_id": str, "lot_id": str}, low_memory=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package-dir", type=Path, required=True)
    ap.add_argument("--skip-local", action="store_true", help="only write committed demo/public data")
    a = ap.parse_args()
    pkg = a.package_dir.resolve()
    print("Verifying delivered checksums ...")
    verify(pkg)

    demo, public, local = ROOT / "data/demo", ROOT / "data/public", ROOT / "data/local"
    for d in (demo, public, local):
        d.mkdir(parents=True, exist_ok=True)

    # SYNTHETIC: project-generated; stored as Parquet to keep the clone small.
    for src, dst in [("data/synthetic_early_inputs.csv", "synthetic_early_inputs"),
                     ("data/synthetic_measurements.csv", "synthetic_measurements"),
                     ("data/synthetic_lots.csv", "synthetic_lots"),
                     ("evaluation/synthetic_targets_168h.csv", "synthetic_targets_168h"),
                     ("evaluation/synthetic_ground_truth.csv", "synthetic_ground_truth")]:
        pd.read_csv(pkg / src, low_memory=False).to_parquet(demo / f"{dst}.parquet", index=False)
    shutil.copy2(pkg / "docs/generator_assumptions.json", demo / "generator_assumptions.json")

    # REAL: split by redistribution basis recorded in drishti/provenance.py.
    ok = redistributable_dataset_ids()
    for name in ["real_measurements", "real_drift_view"]:
        f = read_real(pkg, f"data/{name}.csv")
        f[f.dataset_id.isin(ok)].to_parquet(public / f"{name}.parquet", index=False)
        if not a.skip_local:
            f.to_parquet(local / f"{name}.parquet", index=False)
    folds = pd.read_csv(pkg / "data/real_device_folds.csv", dtype=str)
    folds[folds.dataset_id.isin(ok)].to_csv(public / "real_device_folds.csv", index=False)
    if not a.skip_local:
        folds.to_csv(local / "real_device_folds.csv", index=False)
    shutil.copy2(pkg / "evaluation/reram_functional_outcomes.csv", public / "reram_functional_outcomes.csv")

    # SECOM: CC BY 4.0 (UCI). Attribution in docs/DATA.md.
    pd.read_parquet(pkg / "data/secom_features.parquet").to_parquet(public / "secom_features.parquet", index=False)
    shutil.copy2(pkg / "data/secom_splits.csv", public / "secom_splits.csv")
    shutil.copy2(pkg / "evaluation/secom_labels.csv", public / "secom_labels.csv")

    if not a.skip_local:
        # IGBT: quarantined, no reuse statement found; local analysis only.
        shutil.copy2(pkg / "data/igbt_scalar_quarantined.parquet", local / "igbt_scalar_quarantined.parquet")
        shutil.copy2(pkg / "docs/SOURCES.md", local / "SOURCES_package.md")

    print("\nRedistribution split:")
    for k, v in DATASETS.items():
        print(f"  {k:12s} {'public' if v['in_repo'] else 'LOCAL ONLY':10s} {v['redistribution']}")
    print("Done.")


if __name__ == "__main__":
    main()
