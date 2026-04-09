from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from typing import Any, Dict, Sequence

from Tuner.BOTuner.lgbo_components.search_space import CategoricalParamSpec, NumericParamSpec


ParamSpec = NumericParamSpec | CategoricalParamSpec


@dataclass(frozen=True)
class PointPreference:
    values: Dict[str, Any]
    confidence: float


@dataclass(frozen=True)
class RegionPreference:
    lower: Dict[str, Any]
    upper: Dict[str, Any]
    confidence: float


Preference = PointPreference | RegionPreference

FINAL_ANSWER_RE = re.compile(
    r"(?is)\[\s*[\"']?(point|region)[\"']?\s*,\s*(\[[\s\S]*?\])\s*,\s*([01](?:\.\d+)?)\s*\]"
)
THINKING_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:#+\s*)?\[?\s*thinking\s*\]?\s*:?\s*(.*?)"
    r"(?=\n\s*(?:#+\s*)?\[?\s*final\s+answer\s*\]?\s*:|\Z)"
)


class LGBOPreferenceParser:
    """Parse the paper-style LGBO point/region outputs."""

    def parse(self, raw_text: str, specs: Sequence[ParamSpec]) -> Preference:
        preference, _ = self.parse_with_metadata(raw_text, specs)
        return preference

    def parse_with_metadata(self, raw_text: str, specs: Sequence[ParamSpec]) -> tuple[Preference, Dict[str, Any]]:
        final_answer = self.extract_final_answer(raw_text)
        data = self._parse_literal(final_answer)
        thinking = self.extract_thinking(raw_text)
        if not isinstance(data, (list, tuple)) or len(data) != 3:
            raise ValueError("LGBO preference must be [kind, payload, confidence]")
        kind, payload, confidence = data
        kind = str(kind).lower().strip()
        confidence = float(confidence)
        if kind == "point":
            values = self._zip_values(payload, specs)
            preference = PointPreference(values=values, confidence=confidence)
            return preference, {
                "mode": "point",
                "point": dict(values),
                "confidence": confidence,
                "thinking": thinking,
                "raw_preference": final_answer,
            }
        if kind == "region":
            payload = self._coerce_region_payload(payload, specs)
            if not isinstance(payload, (list, tuple)) or len(payload) != 2:
                raise ValueError("Region preference payload must be [lb, ub]")
            lower = self._zip_values(payload[0], specs)
            upper = self._zip_values(payload[1], specs)
            preference = RegionPreference(lower=lower, upper=upper, confidence=confidence)
            return preference, {
                "mode": "region",
                "lower": dict(lower),
                "upper": dict(upper),
                "confidence": confidence,
                "thinking": thinking,
                "raw_preference": final_answer,
            }
        raise ValueError(f"Unsupported LGBO preference kind: {kind}")

    def extract_thinking(self, raw_text: str) -> str | None:
        match = THINKING_RE.search(raw_text or "")
        if not match:
            return None
        thinking = match.group(1).strip()
        return thinking or None

    def extract_final_answer(self, raw_text: str) -> str:
        text = (raw_text or "").strip()
        match = FINAL_ANSWER_RE.search(text)
        if match:
            return match.group(0)
        return text

    def _parse_literal(self, final_answer: str) -> Any:
        try:
            return ast.literal_eval(final_answer)
        except (SyntaxError, ValueError):
            normalized = re.sub(
                r"^\[\s*(point|region)\b",
                lambda match: f'["{match.group(1).lower()}"',
                final_answer.strip(),
                count=1,
                flags=re.IGNORECASE,
            )
            return ast.literal_eval(normalized)

    def _coerce_region_payload(self, payload: Any, specs: Sequence[ParamSpec]) -> Any:
        if not isinstance(payload, (list, tuple)):
            return payload
        if len(payload) == 2 and all(isinstance(item, (list, tuple)) for item in payload):
            return payload
        if len(payload) == len(specs) and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in payload):
            lower = [item[0] for item in payload]
            upper = [item[1] for item in payload]
            return [lower, upper]
        return payload

    def _zip_values(self, values: Sequence[Any], specs: Sequence[ParamSpec]) -> Dict[str, Any]:
        if len(values) != len(specs):
            raise ValueError(f"Expected {len(specs)} values, got {len(values)}")
        out: Dict[str, Any] = {}
        for spec, value in zip(specs, values):
            if isinstance(spec, NumericParamSpec):
                out[spec.name] = float(value)
                continue
            if value not in spec.choices:
                raise ValueError(f"Invalid categorical value {value!r} for {spec.name}; expected one of {list(spec.choices)}")
            out[spec.name] = value
        return out


class LGBOPreferencePlanner:
    """Convert parsed preference into a compact internal plan."""

    def make_plan(self, preference: Preference, specs: Sequence[ParamSpec]) -> Dict[str, Any]:
        numeric_specs = [spec for spec in specs if isinstance(spec, NumericParamSpec)]
        categorical_specs = [spec for spec in specs if isinstance(spec, CategoricalParamSpec)]

        if isinstance(preference, PointPreference):
            confidence = float(preference.confidence)
            lower = {}
            upper = {}
            for spec in numeric_specs:
                width = spec.high - spec.low
                radius = max(width * (0.05 + (1.0 - confidence) * 0.10), 1e-12)
                center = float(preference.values[spec.name])
                lower[spec.name] = max(spec.low, center - radius)
                upper[spec.name] = min(spec.high, center + radius)
            categorical = {
                spec.name: preference.values[spec.name]
                for spec in categorical_specs
                if spec.name in preference.values
            }
            return {
                "mode": "region-soft",
                "point": {spec.name: preference.values[spec.name] for spec in numeric_specs},
                "lower": lower,
                "upper": upper,
                "categorical": categorical,
                "confidence": confidence,
            }
        lower = {}
        upper = {}
        categorical = {}
        for spec in numeric_specs:
            lo = min(preference.lower[spec.name], preference.upper[spec.name])
            hi = max(preference.lower[spec.name], preference.upper[spec.name])
            lower[spec.name] = max(spec.low, lo)
            upper[spec.name] = min(spec.high, hi)
        for spec in categorical_specs:
            lo = preference.lower[spec.name]
            hi = preference.upper[spec.name]
            chosen = lo if lo in spec.choices else hi
            if chosen not in spec.choices:
                raise ValueError(f"Region preference used invalid categorical value for {spec.name}")
            categorical[spec.name] = chosen
        return {
            "mode": "region",
            "lower": lower,
            "upper": upper,
            "categorical": categorical,
            "confidence": float(preference.confidence),
        }

