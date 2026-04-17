from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Sequence

import torch

from Tuner.BOTuner.lgbo_components.search_space import MixedSearchSpaceAdapter, ParamSpec


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

    def __init__(self) -> None:
        self.space = MixedSearchSpaceAdapter()

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
        return {
            spec.name: self.space._coerce_value(value, spec)
            for spec, value in zip(specs, values)
        }


class LGBOPreferencePlanner:
    """Convert parsed preference into a compact internal plan."""

    def __init__(
        self,
        *,
        grid_size: int = 512,
        e_soft_low: float = 2.0,
        e_soft_high: float = 6.0,
    ) -> None:
        self.grid_size = int(grid_size)
        self.e_soft_low = float(e_soft_low)
        self.e_soft_high = float(e_soft_high)

    def _confidence_to_delta(self, confidence: float) -> float:
        p = min(max(float(confidence), 1e-6), 1 - 1e-6)
        return math.sqrt(2.0) * float(torch.erfinv(torch.tensor(2.0 * p - 1.0, dtype=torch.double)))

    @staticmethod
    def _box_width_with_clamp(center_i: float, r: float, eps: float = 1e-6) -> float:
        lb = max(center_i - r, eps)
        ub = min(center_i + r, 1.0 - eps)
        return max(ub - lb, 0.0)

    def _effective_volume_numeric(self, center_norm: Sequence[float], r: float, eps: float = 1e-6) -> float:
        if not center_norm:
            return 1.0
        vol = 1.0
        for ci in center_norm:
            vol *= self._box_width_with_clamp(float(ci), r, eps)
        return max(vol, 0.0)

    def _choose_soft_radius_edge(self, center_norm: Sequence[float], low_margin: float = 0.05, eps: float = 1e-6) -> float:
        if not center_norm:
            return 0.0
        e_target = self.e_soft_low + float(low_margin)
        vol_target = max(e_target / float(self.grid_size), 1e-12)
        lo, hi = 1e-8, 0.5 - eps
        if self._effective_volume_numeric(center_norm, hi, eps) < vol_target:
            return float(hi)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if self._effective_volume_numeric(center_norm, mid, eps) >= vol_target:
                hi = mid
            else:
                lo = mid
        r = 0.5 * (lo + hi)
        e_eff = self.grid_size * self._effective_volume_numeric(center_norm, r, eps)
        if e_eff < self.e_soft_low:
            for _ in range(10):
                r = min(r * 1.2, 0.5 - eps)
                e_eff = self.grid_size * self._effective_volume_numeric(center_norm, r, eps)
                if e_eff >= self.e_soft_low:
                    break
        return float(r)

    def _normalized_region_volume(self, lower: Dict[str, Any], upper: Dict[str, Any], specs: Sequence[ParamSpec]) -> float:
        volume = 1.0
        for spec in specs:
            if spec.kind not in {"float", "int"}:
                continue
            lo = float(lower[spec.name])
            hi = float(upper[spec.name])
            width = max(0.0, float(spec.high) - float(spec.low))
            if width <= 0.0:
                continue
            frac = max(0.0, min(1.0, (hi - lo) / width))
            volume *= frac
        return float(volume)

    def make_plan(self, preference: Preference, specs: Sequence[ParamSpec]) -> Dict[str, Any]:
        if isinstance(preference, PointPreference):
            confidence = float(preference.confidence)
            numeric_specs = [spec for spec in specs if spec.kind in {"float", "int"}]
            center_norm: list[float] = []
            for spec in numeric_specs:
                width = float(spec.high) - float(spec.low)
                center_value = float(preference.values[spec.name])
                if width <= 0.0:
                    center_norm.append(0.5)
                else:
                    center_norm.append((center_value - float(spec.low)) / width)
            r = self._choose_soft_radius_edge(center_norm=center_norm)
            lower = {}
            upper = {}
            for spec in specs:
                if spec.kind in {"float", "int"}:
                    width = float(spec.high) - float(spec.low)
                    center_raw = float(preference.values[spec.name])
                    if width <= 0.0:
                        lower[spec.name] = center_raw
                        upper[spec.name] = center_raw
                    else:
                        center_n = (center_raw - float(spec.low)) / width
                        lo_n = max(1e-6, center_n - r)
                        hi_n = min(1.0 - 1e-6, center_n + r)
                        lower[spec.name] = max(float(spec.low), float(spec.low) + lo_n * width)
                        upper[spec.name] = min(float(spec.high), float(spec.low) + hi_n * width)
                else:
                    lower[spec.name] = preference.values[spec.name]
                    upper[spec.name] = preference.values[spec.name]
            effective_count = self.grid_size * self._effective_volume_numeric(center_norm, r)
            if effective_count < self.e_soft_low:
                mode = "point"
                smooth = None
            elif effective_count < self.e_soft_high:
                mode = "region-soft"
                smooth = 0.08
            else:
                mode = "region"
                smooth = None
            return {
                "mode": mode,
                "point": dict(preference.values),
                "lower": lower,
                "upper": upper,
                "confidence": confidence,
                "delta": self._confidence_to_delta(confidence),
                "x_star": dict(preference.values),
                "region": {
                    "lower": dict(lower),
                    "upper": dict(upper),
                    "grid_size": self.grid_size,
                    "smooth": smooth,
                },
            }
        lower = {}
        upper = {}
        for spec in specs:
            if spec.kind in {"float", "int"}:
                lo = min(float(preference.lower[spec.name]), float(preference.upper[spec.name]))
                hi = max(float(preference.lower[spec.name]), float(preference.upper[spec.name]))
                lower[spec.name] = max(float(spec.low), lo)
                upper[spec.name] = min(float(spec.high), hi)
            else:
                lower[spec.name] = preference.lower[spec.name]
                upper[spec.name] = preference.upper[spec.name]
        confidence = float(preference.confidence)
        delta = self._confidence_to_delta(confidence)
        point = {}
        for spec in specs:
            if spec.kind in {"float", "int"}:
                point[spec.name] = (float(lower[spec.name]) + float(upper[spec.name])) / 2.0
            else:
                point[spec.name] = lower[spec.name]
        effective_count = self.grid_size * self._normalized_region_volume(lower, upper, specs)
        if effective_count < self.e_soft_low:
            mode = "point"
            smooth = 0.08
        elif effective_count < self.e_soft_high:
            mode = "region-soft"
            smooth = 0.08
        else:
            mode = "region"
            smooth = None
        return {
            "mode": mode,
            "point": point,
            "lower": lower,
            "upper": upper,
            "confidence": confidence,
            "delta": delta,
            "x_star": point,
            "region": {
                "lower": dict(lower),
                "upper": dict(upper),
                "grid_size": self.grid_size,
                "smooth": smooth,
            },
        }
