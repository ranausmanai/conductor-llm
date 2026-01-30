"""Global configuration for Conductor."""

from __future__ import annotations

import copy
import json
import os
from typing import Dict, Any


DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

DEFAULT_MODELS: Dict[str, str] = {
    "baseline": "gpt-4o",
    "cheap": "gpt-4o-mini",
    "quality": "gpt-4o",
}

_pricing: Dict[str, Dict[str, float]] = copy.deepcopy(DEFAULT_PRICING)
_models: Dict[str, str] = copy.deepcopy(DEFAULT_MODELS)


def _parse_json_env(var_name: str) -> Dict[str, Any] | None:
    value = os.getenv(var_name)
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def get_pricing() -> Dict[str, Dict[str, float]]:
    """Return pricing configuration, with optional env override."""
    parsed = _parse_json_env("CONDUCTOR_PRICING_JSON")
    if parsed:
        return parsed
    return _pricing


def set_pricing(pricing: Dict[str, Dict[str, float]]) -> None:
    """Set pricing at runtime."""
    if not isinstance(pricing, dict) or not pricing:
        raise ValueError("pricing must be a non-empty dict")
    for model, rates in pricing.items():
        if not isinstance(rates, dict) or "input" not in rates or "output" not in rates:
            raise ValueError(f"pricing for {model} must include 'input' and 'output'")
    global _pricing
    _pricing = copy.deepcopy(pricing)


def get_models() -> Dict[str, str]:
    """Return model configuration, with optional env override."""
    parsed = _parse_json_env("CONDUCTOR_MODELS_JSON")
    if parsed:
        return parsed
    return _models


def set_models(*, cheap: str | None = None, quality: str | None = None, baseline: str | None = None) -> None:
    """Set model defaults at runtime."""
    global _models
    updated = copy.deepcopy(_models)
    if cheap:
        updated["cheap"] = cheap
    if quality:
        updated["quality"] = quality
    if baseline:
        updated["baseline"] = baseline
    _models = updated
