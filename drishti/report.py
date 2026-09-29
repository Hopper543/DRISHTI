"""Screening report (PDF) built with reportlab + matplotlib (no browser needed)."""
from __future__ import annotations

import io
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, LongTable, PageBreak, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from . import EVIDENCE_NOTICE, __version__  # noqa: E402
from .decision import STAGE_LABEL  # noqa: E402
from .explain import parameter_evidence, parameter_sentences  # noqa: E402
from .provenance import PROVENANCE_BADGE  # noqa: E402

DECISION_COLOR = {"PASS": colors.HexColor("#1b7f3b"), "FAIL": colors.HexColor("#b3261e"),
                  "ESCALATE": colors.HexColor("#b86e00")}
MAX_DETAIL_DEVICES = 20


def device_chart_png(run, device_id: str, parameter: str) -> bytes:
    """Peers (grey) vs device (blue) at 0/24 h, forecast interval at 168 h, limits."""
    p = run.params
    row = p[(p.device_id == device_id) & (p.parameter == parameter)].iloc[0]
    peers = p[(p.lot_id == row.lot_id) & (p.parameter == parameter) & (p.device_id != device_id)]
    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=150)
    for r in peers.itertuples():
        ax.plot([0, 24], [r.value_0h, r.value_24h], color="#9aa0a6", lw=0.8, alpha=0.6)
    ax.plot([0, 24], [row.value_0h, row.value_24h], color="#1a56db", lw=2.2, marker="o", label=device_id)
    if np.isfinite(row.prediction):
        ax.errorbar([168], [row.prediction], yerr=[[row.prediction - row.lower], [row.upper - row.prediction]],
                    fmt="s", color="#1a56db", capsize=5, label="168 h forecast + 90% interval")
        ax.plot([24, 168], [row.value_24h, row.prediction], ls=":", color="#1a56db", lw=1)
    for lim, name in ((row.spec_lower, "lower limit"), (row.spec_upper, "upper limit")):
        ax.axhline(lim, color="#b3261e", ls="--", lw=1)
        ax.text(170, lim, f" {name}", va="center", fontsize=6, color="#b3261e")
    ax.set_xlabel("Burn-in hours")
    ax.set_ylabel(f"{parameter} [{row.unit}]")
    ax.set_xticks([0, 24, 168])
    ax.set_title(f"{device_id} · {parameter} · lot {row.lot_id} (grey: {len(peers)} peers)", fontsize=8)
    ax.legend(fontsize=6, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("Small", parent=ss["BodyText"], fontSize=7.5, leading=9.5))
    ss.add(ParagraphStyle("Banner", parent=ss["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#5f3700"),
                          backColor=colors.HexColor("#fff4e0"), borderPadding=5))
    return ss


def _table(data, widths, header_bg="#e8eaed", font=7):
    t = LongTable(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), font), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c4c7c5")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def build_pdf(run, focus_device: str | None = None, max_detail: int = MAX_DETAIL_DEVICES) -> bytes:
    ss = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title=f"DRISHTI screening report {run.run_id}", author="DRISHTI")
    el = [Paragraph("DRISHTI — Early Screening Report", ss["Title"]),
          Paragraph(f"{STAGE_LABEL}", ss["Heading4"]),
          Paragraph(f"<b>Evidence class:</b> {PROVENANCE_BADGE.get(run.provenance_class, run.provenance_class)}. "
                    f"{EVIDENCE_NOTICE}", ss["Banner"]), Spacer(1, 6)]
    s = run.summary()
    meta = [["Run ID", run.run_id], ["Timestamp (UTC)", run.timestamp_utc], ["Input SHA-256", run.input_sha256],
            ["Source", run.source_name or "-"], ["Model version", run.model_version],
            ["Specification revision", f"{run.spec_revision} (registry SHA-256 {run.spec_registry_sha256[:16]}…)"],
            ["Mode", run.mode], ["Devices / parameter rows", f"{s['n_devices']} / {len(run.params)}"],
            ["Device decisions", ", ".join(f"{k}: {v}" for k, v in sorted(s["device_decisions"].items()))],
            ["Generated", datetime.now(timezone.utc).isoformat(timespec="seconds") + f" by DRISHTI {__version__}"]]
    el += [_table([["Field", "Value"]] + [[a, Paragraph(str(b), ss["Small"])] for a, b in meta], [45 * mm, 135 * mm]),
           Spacer(1, 8), Paragraph("Device recommendations", ss["Heading3"])]
    rows = [["Device", "Lot", "Part", "Decision", "Basis", "Driving parameter(s)", "Reason codes"]]
    order = {"FAIL": 0, "ESCALATE": 1, "PASS": 2}
    devs = run.devices.assign(_o=run.devices.decision.map(order)).sort_values(["_o", "device_id"])
    for r in devs.itertuples():
        codes = [c for c in r.reason_codes if c not in ("DEMO_SPEC_NOT_APPROVED", "B_INTERVAL_INSIDE")]
        rows.append([r.device_id, r.lot_id, r.part_number, r.decision, r.basis, r.driving_parameters or "-",
                     Paragraph(", ".join(codes) or "-", ss["Small"])])
    t = _table(rows, [30 * mm, 17 * mm, 22 * mm, 17 * mm, 26 * mm, 28 * mm, 42 * mm])
    for i, r in enumerate(devs.itertuples(), start=1):
        t.setStyle(TableStyle([("TEXTCOLOR", (3, i), (3, i), DECISION_COLOR.get(r.decision, colors.black))]))
    el.append(t)

    # ---- details ------------------------------------------------------------
    detail = devs[devs.decision != "PASS"].device_id.tolist()
    if focus_device:
        detail = [focus_device] + [d for d in detail if d != focus_device]
    detail = detail[:max_detail]
    if detail:
        el += [PageBreak(), Paragraph("Evidence for flagged devices", ss["Heading2"])]
        if len(devs[devs.decision != "PASS"]) > max_detail:
            el.append(Paragraph(f"Showing {max_detail} of {int((devs.decision != 'PASS').sum())} non-PASS devices; "
                                "the CSV/JSON export contains all evidence.", ss["Small"]))
    for dev in detail:
        drow = run.devices[run.devices.device_id == dev].iloc[0]
        block = [Paragraph(f"{dev} — <font color='{DECISION_COLOR[drow.decision].hexval()}'>{drow.decision}</font> "
                           f"({drow.basis})", ss["Heading3"])]
        prow = run.params[run.params.device_id == dev]
        for r in prow.itertuples():
            ev = [[a, Paragraph(b, ss["Small"])] for a, b in parameter_evidence(r)]
            sub = [Paragraph(f"<b>{r.parameter}</b> [{r.unit}] → {r.decision} (rule {r.rule})", ss["BodyText"]),
                   _table([["Evidence", "Value"]] + ev, [70 * mm, 110 * mm], font=7)]
            sub += [Paragraph("• " + x, ss["Small"]) for x in parameter_sentences(r)]
            if r.decision != "PASS" and np.isfinite(r.value_0h) and np.isfinite(r.value_24h):
                png = device_chart_png(run, dev, r.parameter)
                sub.append(Image(io.BytesIO(png), width=150 * mm, height=72 * mm))
            sub.append(Spacer(1, 5))
            block.append(KeepTogether(sub))
        el += block

    # ---- appendix -----------------------------------------------------------
    el += [PageBreak(), Paragraph("Decision table (default demo policy)", ss["Heading2"])]
    table = [["Rule", "Condition", "Outcome"],
             ["R1", "Observed 0 h or 24 h reading outside specification", "FAIL (OBSERVED)"],
             ["R2", "Entire calibrated 168 h interval beyond a limit; forecast usable", "FAIL (PREDICTED)"],
             ["R3", "Evidence gap: missing input, unsupported part, no calibration, out of training domain, "
                    "insufficient peers, zero MAD, identical readings, source flag", "ESCALATE"],
             ["R4", "Interval crosses a limit, or Module A unusual level/change, lot shift/spread", "ESCALATE"],
             ["R5", "Interval inside limits and none of the above", "PASS (early screen)"],
             ["Device", "FAIL if any parameter FAIL; ESCALATE if any ESCALATE or missing parameter; else PASS", ""]]
    el.append(_table([[Paragraph(str(c), ss["Small"]) for c in r] for r in table], [18 * mm, 125 * mm, 37 * mm]))
    el += [Spacer(1, 8), Paragraph("Limitations", ss["Heading3"])]
    for line in ["Anomaly scores express statistical unusualness relative to peers; they are not probabilities of failure.",
                 "Forecast intervals are calibrated on held-out lots assumed exchangeable with the screened lot; "
                 "shifted, tiny or new lots are not covered by that argument.",
                 "Shipped models and limits are trained/defined on SYNTHETIC data; they are not qualified for "
                 "acceptance decisions on real hardware.",
                 "Delayed-onset and step defects that are invisible at 24 h cannot be detected at 24 h.",
                 "Validation messages and this report do not replace the organisation's approved screening procedure."]:
        el.append(Paragraph("• " + line, ss["Small"]))

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 6.5)
        canvas.drawString(14 * mm, 8 * mm, f"DRISHTI {run.run_id} · {run.provenance_class} · {STAGE_LABEL}")
        canvas.drawRightString(196 * mm, 8 * mm, f"page {_doc.page}")
        canvas.restoreState()

    doc.build(el, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
