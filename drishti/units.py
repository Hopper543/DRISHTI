"""Unit parsing and SI-prefix conversion.

Only conversions between SI prefixes of the SAME base unit are performed
(e.g. nA <-> uA). Anything else (percent, dB, unknown strings) must match the
registry unit exactly; there is no guessing.
"""
from __future__ import annotations

PREFIX = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3, "M": 1e6}
BASES = ("ohm", "A", "V")
ALIASES = {"Ω": "ohm", "ohms": "ohm", "Ohm": "ohm", "mΩ": "mohm", "kΩ": "kohm"}


def parse_unit(unit: str) -> tuple[float, str] | None:
    """Return (scale_to_base, base) or None if not a supported SI unit."""
    u = ALIASES.get(str(unit).strip(), str(unit).strip())
    for base in BASES:
        if u.endswith(base):
            prefix = u[: -len(base)]
            if prefix in PREFIX:
                return PREFIX[prefix], base
    return None


def conversion_factor(from_unit: str, to_unit: str) -> float | None:
    """Multiply values in from_unit by this factor to express them in to_unit."""
    if str(from_unit).strip() == str(to_unit).strip():
        return 1.0
    a, b = parse_unit(from_unit), parse_unit(to_unit)
    if a is None or b is None or a[1] != b[1]:
        return None
    return a[0] / b[0]
