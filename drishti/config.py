"""Configuration loading and repository paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DEMO = ROOT / "data" / "demo"
DATA_PUBLIC = ROOT / "data" / "public"
DATA_LOCAL = ROOT / "data" / "local"
DATA_FIXTURES = ROOT / "data" / "fixtures"
EVAL_DIR = ROOT / "evaluation_results"


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]

    @property
    def mode(self) -> str:
        return self.raw.get("mode", "demo")

    @property
    def module_a(self) -> dict[str, Any]:
        return self.raw["module_a"]

    @property
    def module_b(self) -> dict[str, Any]:
        return self.raw["module_b"]

    @property
    def decision(self) -> dict[str, Any]:
        return self.raw["decision"]

    @property
    def models_dir(self) -> Path:
        return ROOT / self.raw["paths"]["models"]

    @property
    def runtime_dir(self) -> Path:
        # DRISHTI_RUNTIME_DIR lets tests and deployments redirect audit/report output.
        env = os.environ.get("DRISHTI_RUNTIME_DIR")
        return Path(env) if env else ROOT / self.raw["paths"]["runtime"]

    @property
    def spec_registry_path(self) -> Path:
        return ROOT / self.raw["spec_registry"]


@lru_cache(maxsize=4)
def load_settings(path: str | None = None) -> Settings:
    p = Path(path) if path else Path(os.environ.get("DRISHTI_CONFIG", ROOT / "config" / "drishti.yaml"))
    return Settings(yaml.safe_load(p.read_text(encoding="utf-8")))
