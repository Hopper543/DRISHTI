"""Build small committed fixtures for the "Load demo" workflow and uploads.

    python scripts/build_demo_fixtures.py

data/fixtures/demo_batch_SYNTHETIC.csv
    Whole held-out SYNTHETIC lots (unchanged values) chosen to show the demo
    cases, plus ONE clearly labelled illustrative lot:
      SYN_L150  ordinary test lot (normal devices)
      SYN_L152  ordinary test lot (SYN_L152_D000 RDS_on interval near 0.5 ohm)
      SYN_L159  ordinary test lot (SYN_L159_D009 abnormal early leakage drift)
      SYN_L180  shifted test lot (planted uniform lot-wide degradation)
      SYN_L192  edge lot, 3 devices, SYN_L192_D000 missing 24 h reading
      ILLUSTRATIVE_Q155  copy of SYN_L155 with threshold_voltage rounded to a
                 0.1 V instrument resolution -> zero MAD. Illustrative fixture,
                 NOT generator output; lot and device IDs say so.
data/fixtures/example_upload_SYNTHETIC.csv   one ordinary lot (upload example)
data/fixtures/early_input_blank.csv          blank template
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from drishti import data  # noqa: E402
from drishti.config import DATA_FIXTURES  # noqa: E402
from drishti.schema import TEMPLATE_COLUMNS, blank_template  # noqa: E402

LOTS = ["SYN_L150", "SYN_L152", "SYN_L159", "SYN_L180", "SYN_L192"]
QUANT_SOURCE, QUANT_LOT, QUANT_PARAM, QUANT_STEP = "SYN_L155", "ILLUSTRATIVE_Q155", "threshold_voltage", 0.1


def main() -> None:
    e = data.synthetic_early_inputs()
    cols = [c for c in TEMPLATE_COLUMNS if c in e.columns]
    batch = e[e.lot_id.isin(LOTS)][cols].copy()
    q = e[e.lot_id == QUANT_SOURCE][cols].copy()
    assert q.part_number.iloc[0] == "DEMO_MOSFET"
    m = q.parameter == QUANT_PARAM
    q.loc[m, ["value_0h", "value_24h"]] = (q.loc[m, ["value_0h", "value_24h"]] / QUANT_STEP).round() * QUANT_STEP
    q["lot_id"] = QUANT_LOT
    q["device_id"] = q.device_id.str.replace(QUANT_SOURCE, QUANT_LOT, regex=False)
    q["sample_id"] = q.device_id + ":" + q.parameter
    q.loc[m, "quality_flags"] = ""
    batch["quality_flags"] = ""
    out = pd.concat([batch, q], ignore_index=True)
    out = out[[c for c in TEMPLATE_COLUMNS if c in out.columns]]
    DATA_FIXTURES.mkdir(parents=True, exist_ok=True)
    out.to_csv(DATA_FIXTURES / "demo_batch_SYNTHETIC.csv", index=False)
    e[e.lot_id == "SYN_L150"][cols].to_csv(DATA_FIXTURES / "example_upload_SYNTHETIC.csv", index=False)
    blank_template().to_csv(DATA_FIXTURES / "early_input_blank.csv", index=False)
    print(f"demo batch: {len(out)} rows, {out.device_id.nunique()} devices, lots {sorted(out.lot_id.unique())}")
    print("quantised MAD check:", float(np.median(np.abs(q.loc[m, 'value_24h'] - q.loc[m, 'value_24h'].median()))))


if __name__ == "__main__":
    main()
