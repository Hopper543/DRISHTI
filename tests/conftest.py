import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drishti.config import DATA_FIXTURES, load_settings  # noqa: E402
from drishti.specs import load_registry  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    """Audit logs and reports from tests never touch the real runtime/ folder."""
    monkeypatch.setenv("DRISHTI_RUNTIME_DIR", str(tmp_path / "runtime"))
    yield


@pytest.fixture(scope="session")
def registry():
    return load_registry()


@pytest.fixture(scope="session")
def cfg_a():
    return dict(load_settings().module_a)


@pytest.fixture(scope="session")
def demo_raw():
    return pd.read_csv(DATA_FIXTURES / "demo_batch_SYNTHETIC.csv")


def make_lot(n=12, part="DEMO_MEMORY", param="leakage_current", unit="uA", lot="T_L1", seed=0,
             v0=10.0, drift=0.3, sd=0.05, prov="SYNTHETIC"):
    """Small, fully specified early-input lot for unit tests (not demo data)."""
    rng = np.random.default_rng(seed)
    a = v0 + rng.normal(0, sd * 4, n)
    b = a + drift + rng.normal(0, sd, n)
    return pd.DataFrame({"sample_id": [f"{lot}_D{i:02d}:{param}" for i in range(n)],
                         "device_id": [f"{lot}_D{i:02d}" for i in range(n)], "lot_id": lot,
                         "part_number": part, "parameter": param, "unit": unit, "value_0h": a, "value_24h": b,
                         "provenance_class": prov})


@pytest.fixture
def lot_factory():
    return make_lot
