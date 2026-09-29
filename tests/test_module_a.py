"""Module A: robust scores, zero MAD, insufficient peers, lot shift, flags."""
import numpy as np
import pandas as pd

from drishti.module_a import (INSUFFICIENT_PEERS, MAD_K, VALID, ZERO_MAD, run_module_a, score_peer_group)
from drishti.pipeline import load_models
from drishti.schema import validate_early_inputs


def test_robust_z_definition(cfg_a):
    lvl = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    chg = np.arange(10.0)
    g, s = score_peer_group(lvl, chg, np.ones(10, bool), cfg_a)
    med, mad = np.median(lvl), np.median(np.abs(lvl - np.median(lvl)))
    assert g.status == VALID
    assert np.isclose(s.z_level.iloc[-1], (100 - med) / (MAD_K * mad))
    assert np.isclose(s.robust_score.iloc[-1], max(abs(s.z_level.iloc[-1]), abs(s.z_change.iloc[-1])))


def test_zero_mad_never_invents_a_score(cfg_a):
    lvl = np.array([1.8] * 7 + [1.9, 1.7, 2.5])  # quantised: >half identical
    g, s = score_peer_group(lvl, lvl - 1.8, np.ones(10, bool), cfg_a)
    assert g.status == ZERO_MAD and "level" in g.zero_mad_features
    assert s.robust_score.isna().all()
    assert s.z_level.isna().all()  # no division by a tiny substitute MAD


def test_insufficient_peers(cfg_a):
    g, s = score_peer_group(np.arange(5.0), np.arange(5.0), np.ones(5, bool), cfg_a)
    assert g.status == INSUFFICIENT_PEERS and s.robust_score.isna().all()


def test_missing_rows_are_not_peers(cfg_a):
    lvl = np.r_[np.arange(10.0), np.nan]
    mask = np.isfinite(lvl)
    g, s = score_peer_group(lvl, lvl, mask, cfg_a)
    assert g.n_valid == 10 and np.isnan(s.robust_score.iloc[-1])


def _run(frame, registry, cfg_a):
    rep = validate_early_inputs(frame, registry)
    assert rep.ok, rep.errors
    return run_module_a(rep.frame, load_models().module_a, cfg_a)


def test_drifting_device_flagged_as_unusual_change(lot_factory, registry, cfg_a):
    f = lot_factory(n=30, seed=3)
    f.loc[5, "value_24h"] = f.loc[5, "value_0h"] + 3.0  # 10x the lot's typical change
    rows, _ = _run(f, registry, cfg_a)
    assert "A_UNUSUAL_CHANGE" in rows.loc[5, "a_codes"]
    assert rows.a_unusual.sum() <= 3


def test_small_lot_and_zero_mad_codes(lot_factory, registry, cfg_a):
    rows, _ = _run(lot_factory(n=4), registry, cfg_a)
    assert all("A_INSUFFICIENT_PEERS" in c for c in rows.a_codes)
    z = lot_factory(n=12)
    z["value_0h"], z["value_24h"] = 10.0, 10.3
    z.loc[0, "value_24h"] = 12.0
    rows, lots = _run(z, registry, cfg_a)
    assert all("A_ZERO_MAD" in c for c in rows.a_codes)
    assert lots.status.iloc[0] == ZERO_MAD


def test_uniformly_shifted_lot_detected_against_history(lot_factory, registry, cfg_a):
    normal, _ = _run(lot_factory(n=30, seed=1), registry, cfg_a)
    assert not any("A_LOT_SHIFT" in c for c in normal.a_codes)
    shifted = lot_factory(n=30, seed=1, v0=30.0, drift=2.3)  # every device shifted: peers look normal
    rows, lots = _run(shifted, registry, cfg_a)
    assert "A_LOT_SHIFT" in lots.lot_codes.iloc[0]
    assert rows.a_codes.map(lambda c: "A_LOT_SHIFT" in c).all()


def test_scores_use_only_early_readings(lot_factory, registry, cfg_a):
    f = lot_factory(n=20, seed=2)
    a, _ = _run(f, registry, cfg_a)
    g = f.copy()
    g["device_id"] = g.device_id.str.replace("T_L1", "XX")  # IDs are not features
    g["sample_id"] = g.device_id + ":leakage_current"
    b, _ = _run(g, registry, cfg_a)
    np.testing.assert_allclose(a.robust_score, b.robust_score)
    np.testing.assert_allclose(a.ensemble_score, b.ensemble_score)


def test_source_flag_gets_no_trusted_verdict(lot_factory, registry, cfg_a):
    f = lot_factory(n=12)
    f["quality_flags"] = ""
    f.loc[0, "quality_flags"] = "individual_bias_assignment_unverified"
    rows, _ = _run(f, registry, cfg_a)
    assert "A_UNTRUSTED_SOURCE_FLAG" in rows.loc[0, "a_codes"]
    assert np.isnan(rows.loc[0, "robust_score"])


def test_reference_meta_documents_population():
    meta = load_models().module_a.meta
    assert "TRAIN" in meta["detector_fit_population"] and "CALIBRATION" in meta["normalisation_population"]
    assert "not a guaranteed false-alarm rate" in meta["caveat"]
    assert meta["evidence"] == "SYNTHETIC_ONLY"
