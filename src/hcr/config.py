"""Configuration loading with validation at import time.

Config errors should fail loudly and immediately, not silently produce wrong
capital numbers three modules downstream.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


@lru_cache(maxsize=None)
def load(name: str) -> dict[str, Any]:
    """Load a YAML config by stem, e.g. load("uk_basel31")."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    _validate(name, cfg)
    return cfg


def _validate(name: str, cfg: dict[str, Any]) -> None:
    if name == "uk_basel31":
        if cfg["scaling_factor"] != 1.0:
            raise ValueError(
                "scaling_factor must be 1.0 under Basel 3.1 - the 1.06 factor was removed"
            )
        if not 0.99 < cfg["confidence_level"] < 1.0:
            raise ValueError("confidence_level should be 0.999")
        floors = cfg["output_floor"]
        years = sorted(floors)
        vals = [floors[y] for y in years]
        if vals != sorted(vals):
            raise ValueError("output_floor phase-in must be non-decreasing by year")
        if abs(vals[-1] - 0.725) > 1e-9:
            raise ValueError("output_floor must reach 72.5%")

    if name == "ifrs9":
        weights = {k: v["weight"] for k, v in cfg["scenarios"].items()}
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"IFRS 9 scenario weights must sum to 1.0, got {total}")

    if name == "default_definition":
        p = cfg["primary"]
        if p["probation_months"] < 3:
            raise ValueError("probation_months below the 3-month regulatory minimum")
        if p["redefault_treatment"] not in {"new", "continuation"}:
            raise ValueError("redefault_treatment must be 'new' or 'continuation'")
        if p.get("probation_months_distressed_restructuring", 12) < 12:
            raise ValueError("EBA/GL/2016/07 requires at least 12 months' probation after a "
                             "distressed restructuring (lower values belong in a variant)")
        for source in [p, *cfg.get("variants", [])]:
            ft = source.get("forbearance_treatment")
            if ft is not None and ft not in {"suspend_dpd", "count_dpd"}:
                raise ValueError(f"forbearance_treatment '{ft}' is not implemented - "
                                 f"use 'suspend_dpd' or 'count_dpd'")
        known = set((p.get("zero_balance_codes") or {}).keys())
        if known:
            for source in [p, *cfg.get("variants", [])]:
                unknown = set(source.get("utp_zero_balance_codes") or []) - known
                if unknown:
                    raise ValueError(
                        f"UTP codes {sorted(unknown)} in '{source.get('name', 'primary')}' "
                        f"are not in the verified zero_balance_codes list"
                    )
            non_default = {c for c, v in p["zero_balance_codes"].items()
                           if v["treatment"] == "non_default_exit"}
            clash = set(p.get("utp_zero_balance_codes") or []) & non_default
            if clash:
                raise ValueError(f"codes {sorted(clash)} are marked non_default_exit but "
                                 f"listed as UTP triggers")


def path(key: str) -> Path:
    """Resolve a path from data.yaml against the repo root."""
    p = Path(load("data")["paths"][key])
    return p if p.is_absolute() else REPO_ROOT / p
