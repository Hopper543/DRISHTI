"""Append-only audit record (JSON Lines).

Each entry stores run ID, input hash, model version, specification revision,
timestamp, decisions and reason codes, plus the SHA-256 of the previous line.
The hash chain makes accidental edits or truncation DETECTABLE by
`verify_chain`; it is ordinary file logging and NOT tamper-proof (anyone with
write access can rewrite the whole file and its chain).
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .config import load_settings

GENESIS = "0" * 64


def audit_path() -> Path:
    return load_settings().runtime_dir / "audit" / "audit_log.jsonl"


def _last_hash(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return GENESIS
    with path.open("rb") as f:
        lines = f.read().splitlines()
    return hashlib.sha256(lines[-1]).hexdigest() if lines else GENESIS


def append(event: str, payload: dict, path: Path | None = None) -> dict:
    path = path or audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"event": event, "logged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "prev_sha256": _last_hash(path), **payload}
    line = json.dumps(entry, sort_keys=True, default=str)
    # "a" mode: existing content is never rewritten by DRISHTI.
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return entry


def record_run(run, path: Path | None = None) -> dict:
    s = run.summary()
    per_device = [{"device_id": r.device_id, "lot_id": r.lot_id, "decision": r.decision, "basis": r.basis,
                   "reason_codes": list(r.reason_codes)} for r in run.devices.itertuples()]
    payload = {k: s[k] for k in ("run_id", "timestamp_utc", "input_sha256", "model_version", "spec_revision",
                                 "spec_registry_sha256", "mode", "stage", "provenance_class", "source_name",
                                 "n_devices", "device_decisions", "reason_code_counts")}
    payload["validation_ok"] = s["validation"]["ok"]
    payload["devices"] = per_device
    return append("SCREENING_RUN", payload, path)


def read(path: Path | None = None) -> list[dict]:
    path = path or audit_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_chain(path: Path | None = None) -> tuple[bool, str]:
    path = path or audit_path()
    if not path.exists():
        return True, "No audit log yet."
    prev = GENESIS
    for i, raw in enumerate(path.read_bytes().splitlines(), start=1):
        if json.loads(raw).get("prev_sha256") != prev:
            return False, f"Chain broken at line {i}: the log was edited, reordered or truncated."
        prev = hashlib.sha256(raw).hexdigest()
    return True, "Hash chain consistent (detects accidental edits; not tamper-proof)."
