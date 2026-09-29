"""Provenance, split isolation, grouping and data-rule checks on shipped data."""
import pandas as pd
import pytest

from drishti import data
from drishti.config import DATA_FIXTURES, DATA_PUBLIC, load_settings
from drishti.provenance import DATASETS, PROVENANCE_CLASSES
from drishti.real import analyze_real, validate_real


def test_whole_lot_split_isolation():
    e = data.synthetic_early_inputs()
    assert e.lot_id.nunique() == 204 and e.device_id.nunique() == 5772 and len(e) == 13468
    assert e.groupby("lot_id").split.nunique().max() == 1
    assert e.groupby("device_id").split.nunique().max() == 1
    counts = e.groupby("split").lot_id.nunique().to_dict()
    assert counts == {"train": 90, "validation": 15, "calibration": 45, "test": 30, "ood_test": 12, "edge_test": 12}


def test_synthetic_is_labelled_synthetic():
    e = data.synthetic_early_inputs()
    assert set(e.provenance_class) == {"SYNTHETIC"}
    assert set(e.spec_origin) == {"DEMONSTRATION_ASSUMPTION"}
    demo = pd.read_csv(DATA_FIXTURES / "demo_batch_SYNTHETIC.csv")
    assert set(demo.provenance_class) == {"SYNTHETIC"}
    ill = demo[~demo.lot_id.str.startswith("SYN_")]
    assert ill.lot_id.str.startswith("ILLUSTRATIVE_").all()  # illustrative fixture is named as such


def test_real_rows_have_no_burn_in_hours_or_labels():
    f, _ = data.real_drift_view()
    assert set(f.provenance_class) <= {"REAL_MEASURED", "REAL_DERIVED"}
    assert f.burn_in_hours.isna().all()
    assert f.latent_defect_label.isna().all()


def test_public_data_contains_only_redistributable_sources():
    pub = pd.read_parquet(DATA_PUBLIC / "real_drift_view.parquet")
    allowed = {k for k, v in DATASETS.items() if v["in_repo"]}
    assert set(pub.dataset_id) <= allowed
    assert not {"nds352", "ad648", "capacitor14", "igbt8"} & set(pub.dataset_id)
    assert not (DATA_PUBLIC / "igbt_scalar_quarantined.parquet").exists()
    assert all(v["evidence"] in PROVENANCE_CLASSES for v in DATASETS.values())


def test_secom_groups_are_dates_not_lots():
    x, splits, labels = data.secom()
    assert x.shape == (1567, 590) and int(labels.sum()) == 104
    s = pd.read_csv(DATA_PUBLIC / "secom_splits.csv")
    assert s.group_kind.eq("calendar_date_NOT_lot").all()
    assert s.groupby("group_id").split.nunique().max() == 1


def test_real_route_keeps_channels_and_blocks_ad620_verdicts():
    f, _ = data.real_drift_view()
    errors, _, v = validate_real(f)
    assert not errors
    r = analyze_real(v, load_settings().module_a)
    ad620 = r[r.dataset_id == "ad620"]
    assert len(ad620) and ad620.exploratory_status.eq("UNTRUSTED_NO_VERDICT").all()
    assert not r.is_control.any()  # controls never act as stressed peers
    op = r[r.dataset_id == "op484"]
    if len(op):  # peers are compared within the same channel only
        assert op.groupby(["measurement_stage", "channel", "peer_group"]).n_peers.max().max() <= 5
    folds = pd.read_csv(DATA_PUBLIC / "real_device_folds.csv")
    assert folds.groupby("device_id").fold.nunique().max() == 1


def test_real_route_rejects_dose_as_burn_in_hours():
    f, _ = data.real_drift_view()
    g = f[f.dataset_id == "u309"].copy()
    g["burn_in_hours"] = g.stress_value
    errors, _, _ = validate_real(g)
    assert any("must not be converted" in e for e in errors)
    h = g.assign(provenance_class="SYNTHETIC", burn_in_hours=None)
    assert any("only REAL" in e for e in validate_real(h)[0])
