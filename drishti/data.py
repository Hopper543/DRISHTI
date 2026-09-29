"""Dataset loaders. Each loader returns one evidence class; unrelated datasets
are never concatenated into a single training matrix."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .config import DATA_DEMO, DATA_LOCAL, DATA_PUBLIC


def _need(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/prepare_data.py --package-dir <dataset package>")
    return path


@lru_cache(maxsize=1)
def synthetic_early_inputs() -> pd.DataFrame:
    f = pd.read_parquet(_need(DATA_DEMO / "synthetic_early_inputs.parquet"))
    f["test_condition"] = "stress125C_meas25C"  # fixed synthetic context (synthetic_lots.parquet)
    return f


@lru_cache(maxsize=1)
def synthetic_targets() -> pd.DataFrame:
    """168 h endpoint. Evaluation/training target only; never an inference input."""
    return pd.read_parquet(_need(DATA_DEMO / "synthetic_targets_168h.parquet"))


@lru_cache(maxsize=1)
def synthetic_ground_truth() -> pd.DataFrame:
    return pd.read_parquet(_need(DATA_DEMO / "synthetic_ground_truth.parquet"))


@lru_cache(maxsize=1)
def synthetic_measurements() -> pd.DataFrame:
    return pd.read_parquet(_need(DATA_DEMO / "synthetic_measurements.parquet"))


@lru_cache(maxsize=1)
def synthetic_lots() -> pd.DataFrame:
    return pd.read_parquet(_need(DATA_DEMO / "synthetic_lots.parquet"))


def real_drift_view(prefer_local: bool = True) -> tuple[pd.DataFrame, str]:
    """Real component drift view. Returns (frame, scope) where scope says whether
    the full local table or the redistributable public subset was loaded."""
    local = DATA_LOCAL / "real_drift_view.parquet"
    if prefer_local and local.exists():
        return pd.read_parquet(local), "local_full"
    return pd.read_parquet(_need(DATA_PUBLIC / "real_drift_view.parquet")), "public_subset"


def reram_outcomes() -> pd.DataFrame:
    return pd.read_csv(_need(DATA_PUBLIC / "reram_functional_outcomes.csv"), dtype={"device_id": str})


def secom() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    x = pd.read_parquet(_need(DATA_PUBLIC / "secom_features.parquet")).set_index("device_id")
    splits = pd.read_csv(_need(DATA_PUBLIC / "secom_splits.csv")).set_index("device_id").split
    labels = pd.read_csv(_need(DATA_PUBLIC / "secom_labels.csv")).set_index("device_id").is_fail
    return x, splits, labels
