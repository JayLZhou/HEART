from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from typing import Any, Dict, Sequence

import torch

from Tuner.BOTuner.lgbo_components.search_space import CategoricalParamSpec, NumericParamSpec
from Tuner.BOTuner.lgbo_components.unified_surrogate import (
    SurrogateMeta,
    build_surrogate_layers,
    ordered_unit_vector,
    params_to_unit,
)


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
    """Convert parsed preference using ``lgbo/decide.py`` (same as reference repo)."""

    def make_plan(self, preference: Preference, specs: Sequence[ParamSpec]) -> Dict[str, Any]:
        numeric_specs = [spec for spec in specs if isinstance(spec, NumericParamSpec)]
        categorical_specs = [spec for spec in specs if isinstance(spec, CategoricalParamSpec)]
        _, metas = build_surrogate_layers(numeric_specs, categorical_specs)
        if not metas:
            return {"mode": "numeric_v1_fallback", "error": "empty LGBO search space"}

        try:
            from lgbo.decide import decide_preference, decide_preference_tilt_from_expert, parse_expert_input_auto

            conf = float(preference.confidence)
            if isinstance(preference, PointPreference):
                unit = params_to_unit(preference.values, metas)
                vec = ordered_unit_vector(unit, metas)
                expert: list[Any] = ["point", vec, conf]
                plan_raw = decide_preference_tilt_from_expert(
                    expert, d=len(metas), grid_size=512, E_soft_low=2.0, E_soft_high=6.0
                )
                return self._lgbo_raw_plan_to_internal(plan_raw, metas, user_confidence=conf)

            lower_u = params_to_unit(preference.lower, metas)
            upper_u = params_to_unit(preference.upper, metas)
            lb_l = ordered_unit_vector(lower_u, metas)
            ub_l = ordered_unit_vector(upper_u, metas)
            expert_r: list[Any] = ["region", [lb_l, ub_l], conf]
            kw = parse_expert_input_auto(expert_r, d=len(metas), grid_size=512, E_soft_low=2.0, E_soft_high=6.0)
            plan_raw = decide_preference(**kw)
            return self._lgbo_raw_plan_to_internal(plan_raw, metas, user_confidence=conf)
        except Exception:
            return self._make_plan_legacy(preference, numeric_specs, categorical_specs)

    def _lgbo_raw_plan_to_internal(
        self,
        plan_raw: Dict[str, Any],
        metas: Sequence[SurrogateMeta],
        *,
        user_confidence: float,
    ) -> Dict[str, Any]:
        """Map ``lgbo.decide`` output to HEART tilt plan; ``confidence`` is user (0,1)."""
        mode = plan_raw.get("mode")

        def _t_to_unit_dict(t: torch.Tensor) -> Dict[str, float]:
            t = t.detach().flatten()
            return {metas[i].name: float(t[i].item()) for i in range(len(metas))}

        if mode == "point":
            x = plan_raw["x_star"]
            pt = _t_to_unit_dict(x)
            return {"mode": "point", "point": pt, "confidence": user_confidence}

        if mode in ("region", "region-soft"):
            reg = plan_raw.get("region") or {}
            lb = reg.get("lb")
            ub = reg.get("ub")
            if lb is None or ub is None:
                return {"mode": "numeric_v1_fallback", "error": "missing region lb/ub in lgbo plan"}
            lower = _t_to_unit_dict(lb)
            upper = _t_to_unit_dict(ub)
            mid = _t_to_unit_dict(plan_raw["x_star"])
            out: Dict[str, Any] = {
                "mode": "region-soft" if mode == "region-soft" else "region",
                "lower": lower,
                "upper": upper,
                "point": mid,
                "confidence": user_confidence,
            }
            return out

        if mode == "value":
            return {"mode": "numeric_v1_fallback", "error": "lgbo value mode not wired in HEART"}

        return {"mode": "numeric_v1_fallback", "error": f"unsupported lgbo mode {mode!r}"}

    def _make_plan_legacy(
        self,
        preference: Preference,
        numeric_specs: list[NumericParamSpec],
        categorical_specs: list[CategoricalParamSpec],
    ) -> Dict[str, Any]:
        """Previous HEART-local planner (fallback if ``lgbo.decide`` fails)."""

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

