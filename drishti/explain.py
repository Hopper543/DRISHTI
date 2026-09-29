"""Inspectable explanations: measured values, peer statistics, limits, interval,
exact rule. Descriptive drift-shape language only; no failure-mechanism
diagnosis (TDDB, NBTI, electromigration ...) is ever produced."""
from __future__ import annotations

import numpy as np

from . import reasons
from .decision import STAGE_LABEL


def _f(v, unit: str = "", nd: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}g}{(' ' + unit) if unit else ''}"


def parameter_evidence(row) -> list[tuple[str, str]]:
    """(label, value) pairs for an inspector evidence card."""
    u = row.unit
    ev = [("Reading at 0 h", _f(row.value_0h, u)), ("Reading at 24 h", _f(row.value_24h, u)),
          ("Change 0 h -> 24 h", _f(row.value_24h - row.value_0h, u))]
    if np.isfinite(getattr(row, "peer_median_change", np.nan)):
        ev += [(f"Peer group (n={int(row.n_peers)}) median at 24 h", _f(row.peer_median_level, u)),
               ("Peer median change", _f(row.peer_median_change, u)),
               ("Peer spread (MAD) of change", _f(row.peer_mad_change, u)),
               ("Robust z (level / change)", f"{_f(row.z_level, nd=3)} / {_f(row.z_change, nd=3)}")]
    else:
        ev.append(("Peer group", f"n={int(getattr(row, 'n_peers', 0) or 0)} valid peers; status "
                                 f"{getattr(row, 'group_status', 'n/a')}"))
    if np.isfinite(getattr(row, "ensemble_score", np.nan)):
        ev.append(("IF / ECOD / ensemble (calibration percentiles)",
                   f"{_f(row.pct_isolation_forest, nd=3)} / {_f(row.pct_ecod, nd=3)} / {_f(row.ensemble_score, nd=3)}"
                   f" (ensemble threshold {_f(row.ensemble_threshold, nd=3)})"))
    ev.append(("Specification limits", f"[{_f(row.spec_lower)}, {_f(row.spec_upper)}] {u} "
                                       f"({getattr(row, 'spec_revision', '')})"))
    if np.isfinite(getattr(row, "prediction", np.nan)):
        ev += [("Forecast at 168 h (central)", _f(row.prediction, u)),
               ("Calibrated 90% interval", f"[{_f(row.lower)}, {_f(row.upper)}] {u}")]
    else:
        ev.append(("Forecast at 168 h", "not produced (see reasons)"))
    ev.append(("Decision rule applied", f"{row.rule}: {row.decision} ({row.basis})"))
    return ev


def parameter_sentences(row) -> list[str]:
    out = []
    u = row.unit
    ch = row.value_24h - row.value_0h
    if np.isfinite(ch) and np.isfinite(getattr(row, "peer_median_change", np.nan)):
        pm = row.peer_median_change
        rel = f" ({ch / pm:.1f}x the peer median change of {_f(pm, u)})" if pm > 0 and ch > 0 else \
              f" (peer median change {_f(pm, u)})"
        out.append(f"Changed by {_f(ch, u)} between 0 h and 24 h{rel}.")
    if np.isfinite(getattr(row, "prediction", np.nan)):
        out.append(f"Forecast 168 h value {_f(row.prediction, u)}, calibrated interval "
                   f"[{_f(row.lower)}, {_f(row.upper)}] {u}, against limits [{_f(row.spec_lower)}, {_f(row.spec_upper)}].")
    for c in row.reason_codes:
        if reasons.role(c) != "INFO" or c == "DEMO_SPEC_NOT_APPROVED":
            out.append(f"[{c}] {reasons.text(c)}")
    out.append(f"Outcome {row.decision} by rule {row.rule}. {STAGE_LABEL}. Anomaly scores are not failure "
               "probabilities.")
    return out


def describe_trajectory(times, values) -> str:
    """Descriptive shape of a trajectory with >= 3 readings. Retrospective only:
    readings after 24 h are never used for the 24 h decision."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    keep = np.concatenate([[True], np.diff(t) > 0])  # repeated checkpoint (e.g. after storage): keep first
    t, v = t[keep], v[keep]
    if len(t) < 3:
        return "insufficient readings for a shape description"
    span = np.ptp(v)
    scale = max(abs(np.median(v)), 1e-30)
    if span <= 0.01 * scale:
        return "stable (total variation below 1% of the median value)"
    steps = np.diff(v)
    if np.max(np.abs(steps)) >= 0.7 * np.sum(np.abs(steps)) and len(steps) >= 2:
        return "step change (one interval carries most of the change)"
    slopes = steps / np.diff(t)
    first, last = slopes[0], slopes[-1]
    if np.sign(first) == np.sign(last) and abs(last) > 1.5 * abs(first):
        return "accelerating drift"
    if np.sign(first) == np.sign(last) and abs(last) < abs(first) / 1.5:
        return "decelerating drift"
    return "approximately steady drift"


def shap_note() -> str:
    return ("TreeSHAP contributions of the central (q50) LightGBM model to its own output. They explain how the "
            "fitted model uses value_0h and value_24h, not the device physics, and not the interval.")
