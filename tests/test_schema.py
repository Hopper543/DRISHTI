"""Input contract: units, identities, missing values, forbidden/future columns."""
import numpy as np
import pandas as pd
import pytest

from drishti.schema import (DQ_IDENTICAL_READINGS, DQ_MISSING_INPUT, DQ_UNIT_CONVERTED, SPEC_UNKNOWN,
                            feature_matrix, validate_early_inputs)


def test_valid_lot_passes_and_takes_registry_limits(lot_factory, registry):
    rep = validate_early_inputs(lot_factory(), registry)
    assert rep.ok, rep.errors
    assert (rep.frame.spec_upper == 50.0).all() and (rep.frame.spec_revision == "DEMO_ONLY_v1").all()


@pytest.mark.parametrize("col", ["value_168h", "value_48h", "value_96h", "scenario", "latent_defect_planted",
                                 "is_fail", "true_value_168h"])
def test_future_and_label_columns_rejected(lot_factory, registry, col):
    f = lot_factory()
    f[col] = 1.0
    rep = validate_early_inputs(f, registry)
    assert not rep.ok and "post-24 h" in rep.errors[0]


def test_missing_required_column_is_actionable(lot_factory, registry):
    rep = validate_early_inputs(lot_factory().drop(columns=["lot_id"]), registry)
    assert not rep.ok and "lot_id" in rep.errors[0] and "template" in rep.errors[0]


def test_non_numeric_value_is_error_but_blank_is_escalation(lot_factory, registry):
    f = lot_factory().astype({"value_24h": object})
    f.loc[0, "value_24h"] = "abc"
    assert any("Non-numeric" in e for e in validate_early_inputs(f, registry).errors)
    g = lot_factory()
    g.loc[0, "value_24h"] = np.nan
    rep = validate_early_inputs(g, registry)
    assert rep.ok and DQ_MISSING_INPUT in rep.frame.row_flags[0]


def test_duplicates_and_identity_conflicts(lot_factory, registry):
    f = pd.concat([lot_factory(), lot_factory().iloc[:1]])
    assert any("Duplicate sample_id" in e for e in validate_early_inputs(f, registry).errors)
    g = lot_factory()
    g.loc[1, "device_id"] = g.loc[0, "device_id"]
    g.loc[1, "lot_id"] = "OTHER"
    errs = validate_early_inputs(g, registry).errors
    assert any("more than one lot" in e or "Duplicate device" in e for e in errs)
    h = lot_factory()
    h.loc[0, "part_number"] = "DEMO_OPAMP"
    assert any("mixed part numbers" in e for e in validate_early_inputs(h, registry).errors)


def test_si_prefix_conversion_and_incompatible_units(lot_factory, registry):
    f = lot_factory()
    f.loc[:, "unit"] = "nA"
    f[["value_0h", "value_24h"]] *= 1000
    rep = validate_early_inputs(f, registry)
    assert rep.ok
    assert np.allclose(rep.frame.value_0h, lot_factory().value_0h)
    assert all(DQ_UNIT_CONVERTED in fl for fl in rep.frame.row_flags)
    g = lot_factory()
    g.loc[:, "unit"] = "V"
    assert any("incompatible" in e for e in validate_early_inputs(g, registry).errors)


def test_limits_must_match_registry(lot_factory, registry):
    f = lot_factory()
    f["spec_lower"], f["spec_upper"] = 0.0, 999.0
    assert any("differ from registry" in e for e in validate_early_inputs(f, registry).errors)


def test_unknown_part_is_flagged_not_rejected(lot_factory, registry):
    rep = validate_early_inputs(lot_factory(part="UNKNOWN_PART"), registry)
    assert rep.ok and all(SPEC_UNKNOWN in fl for fl in rep.frame.row_flags)


def test_identical_readings_flagged(lot_factory, registry):
    f = lot_factory()
    f.loc[0, "value_24h"] = f.loc[0, "value_0h"]
    rep = validate_early_inputs(f, registry)
    assert DQ_IDENTICAL_READINGS in rep.frame.row_flags[0]


def test_single_provenance_per_upload(lot_factory, registry):
    f = pd.concat([lot_factory(), lot_factory(lot="T_L2", prov="REAL_MEASURED")])
    assert any("single provenance" in e for e in validate_early_inputs(f, registry).errors)
    g = lot_factory(prov="REAL_ISH")
    assert any("provenance_class must be" in e for e in validate_early_inputs(g, registry).errors)


def test_feature_matrix_is_exactly_two_columns(lot_factory):
    f = lot_factory()
    f["spec_upper"], f["split"] = 50.0, "train"
    x = feature_matrix(f)
    assert x.shape == (len(f), 2)
    np.testing.assert_array_equal(x, f[["value_0h", "value_24h"]].to_numpy())
    f.loc[0, "value_24h"] = np.nan
    with pytest.raises(ValueError):
        feature_matrix(f)
