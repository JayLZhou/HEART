from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from Tuner.BOTuner.lgbo_components.search_space import NumericParamSpec


@dataclass(frozen=True)
class PointPreference:
    values: Dict[str, float]
    confidence: float


@dataclass(frozen=True)
class RegionPreference:
    lower: Dict[str, float]
    upper: Dict[str, float]
    confidence: float


Preference = PointPreference | RegionPreference


class LGBOPreferenceParser:
    """Parse the paper-style LGBO point/region outputs."""

    def parse(self, raw_text: str, specs: Sequence[NumericParamSpec]) -> Preference:
        data = ast.literal_eval(raw_text.strip())
        if not isinstance(data, (list, tuple)) or len(data) != 3:
            raise ValueError("LGBO preference must be [kind, payload, confidence]")
        kind, payload, confidence = data
        confidence = float(confidence)
        if kind == "point":
            values = self._zip_values(payload, specs)
            return PointPreference(values=values, confidence=confidence)
        if kind == "region":
            if not isinstance(payload, (list, tuple)) or len(payload) != 2:
                raise ValueError("Region preference payload must be [lb, ub]")
            lower = self._zip_values(payload[0], specs)
            upper = self._zip_values(payload[1], specs)
            return RegionPreference(lower=lower, upper=upper, confidence=confidence)
        raise ValueError(f"Unsupported LGBO preference kind: {kind}")

    def _zip_values(self, values: Sequence[Any], specs: Sequence[NumericParamSpec]) -> Dict[str, float]:
        if len(values) != len(specs):
            raise ValueError(f"Expected {len(specs)} values, got {len(values)}")
        return {spec.name: float(value) for spec, value in zip(specs, values)}


class LGBOPreferencePlanner:
    """Convert parsed preference into a compact internal plan."""

    def make_plan(self, preference: Preference, specs: Sequence[NumericParamSpec]) -> Dict[str, Any]:
        if isinstance(preference, PointPreference):
            return {
                "mode": "point",
                "point": dict(preference.values),
                "confidence": float(preference.confidence),
            }
        lower = {}
        upper = {}
        for spec in specs:
            lo = min(preference.lower[spec.name], preference.upper[spec.name])
            hi = max(preference.lower[spec.name], preference.upper[spec.name])
            lower[spec.name] = max(spec.low, lo)
            upper[spec.name] = min(spec.high, hi)
        return {
            "mode": "region",
            "lower": lower,
            "upper": upper,
            "confidence": float(preference.confidence),
        }

