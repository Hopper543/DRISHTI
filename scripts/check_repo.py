"""Repository hygiene checks (run in CI):
* no tracked file above 10 MB, no archives/MAT/raw folders, no secrets files
* no local-only (non-redistributable) dataset rows in committed public data
* no virtual environments, caches or runtime logs tracked
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq  # noqa: E402

from drishti.provenance import DATASETS  # noqa: E402

MAX_BYTES = 10 * 1024 * 1024
BANNED_PARTS = (".venv/", "venv/", "__pycache__/", "runtime/", "data/local/", "raw/", ".env")
BANNED_SUFFIX = (".zip", ".mat", ".pyc", ".log", ".key")


def main() -> int:
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    problems = []
    for f in files:
        p = ROOT / f
        if any(b in f for b in BANNED_PARTS) or f.endswith(BANNED_SUFFIX):
            problems.append(f"banned path tracked: {f}")
        if p.exists() and p.stat().st_size > MAX_BYTES:
            problems.append(f"file larger than 10 MB: {f}")
    local_only = {k for k, v in DATASETS.items() if not v["in_repo"]}
    for name in ("real_measurements.parquet", "real_drift_view.parquet"):
        pub = ROOT / "data" / "public" / name
        if pub.exists():
            # A short threaded Parquet read can abort during interpreter shutdown
            # on Linux after printing OK (Arrow issue #34314). This tiny scan does
            # not need worker threads or conversion through pandas.
            ids = pq.read_table(pub, columns=["dataset_id"], use_threads=False)["dataset_id"].to_pylist()
            bad = set(ids) & local_only
            if bad:
                problems.append(f"{name} contains local-only datasets {sorted(bad)}")
    for p in problems:
        print("FAIL:", p)
    print(f"checked {len(files)} tracked files: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
