"""Module B: features/target separation, no leakage, conformal, serialization."""
import json

import numpy as np
import pandas as pd
import pytest

from drishti import data
from drishti.config import load_settings
from drishti.module_b import Forecaster, interval_status, join_targets, lot_conformal_margin
from drishti.pipeline import load_models


@pytest.fixture(scope="module")
def fc():
    return load_models().module_b


def test_models_use_exactly_two_features(fc):
    assert fc.manifest["features"] == ["value_0h", "value_24h"]
    for qs in fc.boosters.values():
        for b in qs.values():
            assert b.num_feature() == 2


def test_target_joined_by_sample_id_one_to_one():
    e = data.synthetic_early_inputs().head(50)
    t = data.synthetic_targets()
    j = join_targets(e, t)
    assert len(j) == 50 and "value_168h" in j and "value_168h" not in e
    with pytest.raises(Exception):
        join_targets(e, pd.concat([t, t.head(1)]))


def test_inputs_never_contain_future_readings():
    e = data.synthetic_early_inputs()
    assert not {"value_168h", "value_48h", "value_96h", "scenario"}.intersection(e.columns)
    long = data.synthetic_measurements()
    # value_24h is exactly the 24 h observation, not a later one.
    j = e.merge(long[long.burn_in_hours == 24][["device_id", "parameter", "value"]], on=["device_id", "parameter"])
    np.testing.assert_allclose(j.value_24h, j.value, equal_nan=True)


def test_ids_and_limits_do_not_change_predictions(fc):
    e = data.synthetic_early_inputs()
    g = e[(e.split == "test") & (e.parameter == "RDS_on")].head(40).copy()
    a = fc.predict(g)
    g2 = g.copy()
    g2["lot_id"], g2["device_id"], g2["spec_upper"], g2["split"] = "X", "Y", 99.0, "train"
    b = fc.predict(g2)
    np.testing.assert_array_equal(a.prediction, b.prediction)
    np.testing.assert_array_equal(a.upper, b.upper)


def test_serialization_roundtrip(tmp_path):
    s = load_settings()
    e = data.synthetic_early_inputs()
    sub = e[e.part_number == "DEMO_MOSFET"]
    fit = sub[sub.split.isin(["train", "calibration"])]
    cfg = dict(s.module_b, lightgbm=dict(s.module_b["lightgbm"], n_estimators=20))
    f1 = Forecaster.fit(fit, data.synthetic_targets(), cfg)
    f1.save(tmp_path / "m")
    f2 = Forecaster.load(tmp_path / "m")
    test = sub[sub.split == "test"]
    p1, p2 = f1.predict(test), f2.predict(test)
    np.testing.assert_allclose(p1[["prediction", "lower", "upper"]], p2[["prediction", "lower", "upper"]])
    # tampered artifact is refused
    cal = tmp_path / "m" / "calibration.json"
    d = json.loads(cal.read_text())
    next(iter(d.values()))["margin"] = 0.0
    cal.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="manifest hash"):
        Forecaster.load(tmp_path / "m")


def test_training_used_only_train_lots(fc):
    e = data.synthetic_early_inputs()
    for k, dom in fc.domain.items():
        part, param = k.split("__")
        tr = e[(e.part_number == part) & (e.parameter == param) & (e.split == "train")]
        assert dom["n_train"] == int(np.isfinite(tr[["value_0h", "value_24h"]]).all(axis=1).sum())
        assert fc.calibration[k]["n_calibration_lots"] == 15


def test_lot_conformal_margin_finite_sample():
    lo, hi = np.zeros(20), np.ones(20)
    y = np.r_[np.full(10, 0.5), np.full(10, 1.2)]
    lots = np.repeat(np.arange(4), 5)
    m, n = lot_conformal_margin(lo, hi, y, lots, 0.9)
    assert n == 4 and m == float("inf")  # ceil(5*0.9)=5 > 4 lots: no finite guarantee
    m, n = lot_conformal_margin(np.zeros(100), np.ones(100), np.full(100, 1.3), np.repeat(np.arange(20), 5), 0.9)
    assert n == 20 and np.isclose(m, 0.3)


def test_unsupported_unit_missing_and_ood_escalate(fc):
    e = data.synthetic_early_inputs()
    g = e[(e.split == "test") & (e.parameter == "leakage_current")].head(4).copy()
    g.loc[g.index[0], "part_number"] = "NEW_PART"
    g.loc[g.index[1], "unit"] = "nA"
    g.loc[g.index[2], "value_24h"] = np.nan
    g.loc[g.index[3], ["value_0h", "value_24h"]] = [200.0, 260.0]
    out = fc.predict(g)
    assert out.b_codes.iloc[0] == ["B_UNSUPPORTED_PART"]
    assert "B_UNIT_MISMATCH" in out.b_codes.iloc[1] and np.isnan(out.prediction.iloc[1])
    assert "B_NOT_RUN_MISSING_INPUT" in out.b_codes.iloc[2]
    assert "B_OUT_OF_DOMAIN" in out.b_codes.iloc[3]


def test_missing_calibration_escalates(fc):
    e = data.synthetic_early_inputs()
    g = e[(e.split == "test") & (e.parameter == "RDS_on")].head(2)
    broken = Forecaster(boosters=fc.boosters, calibration={}, domain=fc.domain, units=fc.units)
    assert all(c == ["B_NO_CALIBRATION"] for c in broken.predict(g).b_codes)


def test_interval_status():
    assert interval_status(1, 2, 0, 3) == "B_INTERVAL_INSIDE"
    assert interval_status(2, 4, 0, 3) == "B_INTERVAL_CROSSES_LIMIT"
    assert interval_status(4, 5, 0, 3) == "B_PREDICTED_VIOLATION"
    assert interval_status(np.nan, 5, 0, 3) == "B_INVALID_INTERVAL"
