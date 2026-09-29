"""Provenance classes and dataset registry (source, evidence class, reuse basis).

Every table and every result carries one of three evidence classes. These are
never merged into one training matrix or one accuracy number.
"""
from __future__ import annotations

REAL_MEASURED = "REAL_MEASURED"
REAL_DERIVED = "REAL_DERIVED"
SYNTHETIC = "SYNTHETIC"
PROVENANCE_CLASSES = (REAL_MEASURED, REAL_DERIVED, SYNTHETIC)

PROVENANCE_BADGE = {
    REAL_MEASURED: "REAL (measured)",
    REAL_DERIVED: "REAL (derived quantity)",
    SYNTHETIC: "SYNTHETIC (generated demo data)",
}

# in_repo=True means the normalized rows are committed to Git; False means the
# rows stay in data/local (git-ignored) because no explicit reuse statement was
# found in the inspected source. See docs/DATA.md.
DATASETS: dict[str, dict] = {
    "synthetic": dict(
        title="DRISHTI synthetic burn-in fixture (seed 26170)", evidence=SYNTHETIC,
        url=None, redistribution="PROJECT_GENERATED", in_repo=True,
        note="Statistical shapes, not fitted device physics. Demo limits only."),
    "ad620": dict(
        title="NASA NEPP AD620SQ TID report (Burton 2017)", evidence=REAL_MEASURED,
        url="https://nepp.nasa.gov/files/28637/NEPP-TR-2017-Burton-TID-17-046-AD620SQ-2017July-Sept-TN46982.pdf",
        redistribution="NTRS 20170011518: Public Use Permitted", in_repo=True,
        note="Individual device-to-bias assignment unresolved: no trusted automated peer verdict."),
    "reram": dict(
        title="NASA NEPP Fujitsu MB85AS4MT ReRAM TID report (2017)", evidence=REAL_MEASURED,
        url="https://nepp.nasa.gov/files/28380/NEPP-TR-TID-2017Jan-16-041-MB85AS4MT-Fujitsu-ReRAM-TN39593.pdf",
        redistribution="NTRS 20170002638: Public Use Permitted", in_repo=True,
        note="Functional failures are interval-censored in a separate outcome file."),
    "op484": dict(
        title="NASA OP484 TID report (Topper 2018)", evidence=REAL_MEASURED,
        url="https://nepp.nasa.gov/radhome/papers/Topper-TR-17-072-OP484-2018Apr02-TID-NASA-TM-20210018713.pdf",
        redistribution="NTRS 20210018713: Public Use Permitted (PDF also carries a government-purpose notice)",
        in_repo=True, note="Four channels per package stay together."),
    "u309": dict(
        title="NASA InterFET U309 TID report (Osheroff 2019)", evidence=REAL_MEASURED,
        url="https://nepp.nasa.gov/radhome/papers/tid/2019-Osherof-TR-19-009-U309-TID-20205004053.pdf",
        redistribution="NTRS 20205004053: Public Use Permitted", in_repo=True,
        note="All devices within report specifications; no failure truth."),
    "nds352": dict(
        title="NASA NDS352AP TID report", evidence=REAL_MEASURED,
        url="https://nepp.nasa.gov/docuploads/7BF4E3DE-5D63-4D6A-B6422A22FD8E2743/N123101_NDS352A.pdf",
        redistribution="No explicit reuse license found", in_repo=False,
        note="Sign-inconsistent 2.5 krad threshold cells are flagged, not corrected."),
    "ad648": dict(
        title="NASA AD648 TID report (2013)", evidence=REAL_MEASURED,
        url="https://nepp.nasa.gov/radhome/papers/TID/13-005_20130711_AD648_TID.pdf",
        redistribution="No explicit reuse license found", in_repo=False,
        note="Only 66 visually checked cells normalized."),
    "capacitor14": dict(
        title="NASA PCoE capacitor electrical stress #14 (PHM mirror)", evidence=REAL_DERIVED,
        url="https://phm-datasets.s3.amazonaws.com/NASA/14.+Capacitor+Electrical+Stress+-+2.zip",
        redistribution="Data file: no explicit license found (companion paper CC BY 3.0 US)", in_repo=False,
        note="Percent ESR increase / capacitance loss; aging hours, not burn-in hours."),
    "secom": dict(
        title="UCI SECOM (McCann & Johnston 2008), DOI 10.24432/C54305", evidence=REAL_MEASURED,
        url="https://archive.ics.uci.edu/dataset/179/secom",
        redistribution="CC BY 4.0", in_repo=True,
        note="Anonymous process features; calendar dates are not lots; not burn-in data."),
    "igbt8": dict(
        title="NASA PCoE IGBT accelerated aging #8 (PHM mirror)", evidence=REAL_MEASURED,
        url="https://phm-datasets.s3.amazonaws.com/NASA/8.+IGBT+Accelerated+Aging.zip",
        redistribution="No explicit dataset license found", in_repo=False,
        note="QUARANTINED: scaling unverified; excluded from physical-limit decisions and normal training."),
}


def redistributable_dataset_ids() -> list[str]:
    return [k for k, v in DATASETS.items() if v["in_repo"] and k not in ("synthetic", "secom")]


def check_provenance_values(values) -> list[str]:
    """Return unknown provenance labels (empty list if all valid)."""
    return sorted({str(v) for v in values} - set(PROVENANCE_CLASSES))
