from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Sequence

from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: tuple[Any, ...] = ()


@dataclass(frozen=True)
class NumericParamSpec(ParamSpec):
    pass


class MixedSearchSpaceAdapter:
    """Adapter for the HEART LGBO search space.

    HEART's Optuna space is mixed: ints/floats plus categorical or boolean
    flags. We keep the paper-style LGBO prompt/planner interface, but encode
    mixed parameters into a finite continuous surrogate input space:
    - numeric params -> normalized scalar
    - categorical/bool params -> one-hot block
    """

    def filter_supported_distributions(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        return {
            name: dist
            for name, dist in search_space.items()
            if isinstance(dist, (FloatDistribution, IntDistribution, CategoricalDistribution))
        }

    def build_specs(self, search_space: Dict[str, Any]) -> List[ParamSpec]:
        specs: List[ParamSpec] = []
        for name, dist in self.filter_supported_distributions(search_space).items():
            if isinstance(dist, FloatDistribution):
                specs.append(
                    NumericParamSpec(
                        name=name,
                        kind="float",
                        low=float(dist.low),
                        high=float(dist.high),
                        step=float(dist.step) if dist.step is not None else None,
                        log=bool(dist.log),
                    )
                )
            elif isinstance(dist, IntDistribution):
                specs.append(
                    NumericParamSpec(
                        name=name,
                        kind="int",
                        low=float(dist.low),
                        high=float(dist.high),
                        step=float(dist.step) if dist.step is not None else 1.0,
                        log=bool(dist.log),
                    )
                )
            elif isinstance(dist, CategoricalDistribution):
                choices = tuple(dist.choices)
                kind = "bool" if choices and all(isinstance(choice, bool) for choice in choices) else "categorical"
                specs.append(
                    ParamSpec(
                        name=name,
                        kind=kind,
                        choices=choices,
                    )
                )
        return specs

    def numeric_specs(self, specs: Sequence[ParamSpec]) -> List[NumericParamSpec]:
        return [spec for spec in specs if spec.kind in {"float", "int"}]

    def categorical_specs(self, specs: Sequence[ParamSpec]) -> List[ParamSpec]:
        return [spec for spec in specs if spec.kind in {"categorical", "bool"}]

    def prompt_spec(self, spec: ParamSpec) -> Dict[str, Any]:
        if spec.kind in {"float", "int"}:
            return {
                "name": spec.name,
                "kind": spec.kind,
                "low": spec.low,
                "high": spec.high,
                "step": spec.step,
            }
        return {
            "name": spec.name,
            "kind": spec.kind,
            "choices": list(spec.choices),
        }

    def default_point(self, specs: Sequence[ParamSpec]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for spec in specs:
            if spec.kind == "int":
                out[spec.name] = self._coerce_value((float(spec.low) + float(spec.high)) / 2.0, spec)
            elif spec.kind == "float":
                out[spec.name] = self._coerce_value((float(spec.low) + float(spec.high)) / 2.0, spec)
            else:
                out[spec.name] = spec.choices[0] if spec.choices else None
        return out

    def clip_point(self, point: Dict[str, Any], specs: Sequence[ParamSpec]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for spec in specs:
            if spec.name in point:
                out[spec.name] = self._coerce_value(point[spec.name], spec)
            else:
                out[spec.name] = self.default_point([spec])[spec.name]
        return out

    def normalize_point(self, point: Dict[str, Any], specs: Sequence[ParamSpec]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for spec in self.numeric_specs(specs):
            value = float(self._coerce_value(point[spec.name], spec))
            width = float(spec.high) - float(spec.low)
            out[spec.name] = 0.0 if width <= 0 else (value - float(spec.low)) / width
        return out

    def denormalize_numeric_point(
        self,
        point: Dict[str, float],
        specs: Sequence[ParamSpec],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for spec in self.numeric_specs(specs):
            raw = float(spec.low) + float(point[spec.name]) * (float(spec.high) - float(spec.low))
            out[spec.name] = self._coerce_value(raw, spec)
        return out

    def encoded_dim(self, specs: Sequence[ParamSpec]) -> int:
        total = 0
        for spec in specs:
            total += 1 if spec.kind in {"float", "int"} else max(len(spec.choices), 1)
        return total

    def encode_point(self, point: Dict[str, Any], specs: Sequence[ParamSpec]) -> List[float]:
        encoded: List[float] = []
        clipped = self.clip_point(point, specs)
        for spec in specs:
            value = clipped[spec.name]
            if spec.kind in {"float", "int"}:
                width = float(spec.high) - float(spec.low)
                encoded.append(0.0 if width <= 0 else (float(value) - float(spec.low)) / width)
                continue
            for choice in spec.choices:
                encoded.append(1.0 if value == choice else 0.0)
        return encoded

    def encode_region_bounds(
        self,
        plan: Dict[str, Any],
        specs: Sequence[ParamSpec],
    ) -> tuple[List[float], List[float]]:
        lower_values = plan.get("lower", {})
        upper_values = plan.get("upper", {})
        lower: List[float] = []
        upper: List[float] = []
        for spec in specs:
            if spec.kind in {"float", "int"}:
                lo_raw = lower_values.get(spec.name, spec.low)
                hi_raw = upper_values.get(spec.name, spec.high)
                lo = float(self._coerce_value(lo_raw, spec))
                hi = float(self._coerce_value(hi_raw, spec))
                if hi < lo:
                    lo, hi = hi, lo
                width = float(spec.high) - float(spec.low)
                if width <= 0:
                    lower.append(0.0)
                    upper.append(0.0)
                else:
                    lower.append((lo - float(spec.low)) / width)
                    upper.append((hi - float(spec.low)) / width)
                continue

            lo_val = lower_values.get(spec.name)
            hi_val = upper_values.get(spec.name)
            fixed = None
            if lo_val is not None and hi_val is not None and self._coerce_value(lo_val, spec) == self._coerce_value(hi_val, spec):
                fixed = self._coerce_value(lo_val, spec)
            for choice in spec.choices:
                if fixed is None:
                    lower.append(0.0)
                    upper.append(1.0)
                else:
                    marker = 1.0 if choice == fixed else 0.0
                    lower.append(marker)
                    upper.append(marker)
        return lower, upper

    def enumerate_categorical_assignments(self, specs: Sequence[ParamSpec]) -> List[Dict[str, Any]]:
        categorical = self.categorical_specs(specs)
        if not categorical:
            return [{}]
        names = [spec.name for spec in categorical]
        all_choices = [list(spec.choices) for spec in categorical]
        assignments: List[Dict[str, Any]] = []
        for values in product(*all_choices):
            assignments.append(dict(zip(names, values)))
        return assignments

    def merge_points(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        merged.update(update)
        return merged

    def point_in_region(self, point: Dict[str, Any], plan: Dict[str, Any] | None, specs: Sequence[ParamSpec]) -> bool:
        if not plan or plan.get("mode") not in {"region", "region-soft", "point"}:
            return True
        clipped = self.clip_point(point, specs)
        for spec in specs:
            value = clipped[spec.name]
            if spec.kind in {"float", "int"}:
                lower = self._region_numeric_lower(spec, plan)
                upper = self._region_numeric_upper(spec, plan)
                numeric_value = float(self._coerce_value(value, spec))
                if numeric_value < lower - 1e-9 or numeric_value > upper + 1e-9:
                    return False
                continue
            fixed = self._region_fixed_choice(spec, plan)
            if fixed is not None and value != fixed:
                return False
        return True

    def clip_point_to_region(
        self,
        point: Dict[str, Any],
        plan: Dict[str, Any] | None,
        specs: Sequence[ParamSpec],
    ) -> Dict[str, Any]:
        clipped = self.clip_point(point, specs)
        if not plan or plan.get("mode") not in {"region", "region-soft", "point"}:
            return clipped

        out: Dict[str, Any] = {}
        for spec in specs:
            value = clipped[spec.name]
            if spec.kind in {"float", "int"}:
                lower = self._region_numeric_lower(spec, plan)
                upper = self._region_numeric_upper(spec, plan)
                out[spec.name] = self._coerce_value(min(max(float(value), lower), upper), spec)
                continue
            fixed = self._region_fixed_choice(spec, plan)
            out[spec.name] = fixed if fixed is not None else value
        return out

    def intersect_region_plans(
        self,
        shared_plan: Dict[str, Any] | None,
        query_plan: Dict[str, Any] | None,
        specs: Sequence[ParamSpec],
    ) -> Dict[str, Any] | None:
        if shared_plan is None:
            return query_plan
        if query_plan is None:
            return shared_plan

        lower: Dict[str, Any] = {}
        upper: Dict[str, Any] = {}
        point_hint = self.clip_point(query_plan.get("point", query_plan.get("lower", {})), specs)
        for spec in specs:
            if spec.kind in {"float", "int"}:
                shared_lower = self._region_numeric_lower(spec, shared_plan)
                shared_upper = self._region_numeric_upper(spec, shared_plan)
                query_lower = self._region_numeric_lower(spec, query_plan)
                query_upper = self._region_numeric_upper(spec, query_plan)
                lo = max(shared_lower, query_lower)
                hi = min(shared_upper, query_upper)
                if lo > hi:
                    anchor = float(point_hint.get(spec.name, (shared_lower + shared_upper) / 2.0))
                    anchor = min(max(anchor, shared_lower), shared_upper)
                    lo = hi = anchor
                lower[spec.name] = self._coerce_value(lo, spec)
                upper[spec.name] = self._coerce_value(hi, spec)
                continue

            shared_fixed = self._region_fixed_choice(spec, shared_plan)
            query_fixed = self._region_fixed_choice(spec, query_plan)
            if shared_fixed is not None:
                lower[spec.name] = shared_fixed
                upper[spec.name] = shared_fixed
            elif query_fixed is not None:
                lower[spec.name] = query_fixed
                upper[spec.name] = query_fixed
            else:
                first = spec.choices[0] if spec.choices else None
                last = spec.choices[-1] if spec.choices else None
                lower[spec.name] = first
                upper[spec.name] = last

        merged_point = self.clip_point_to_region(point_hint, shared_plan, specs)
        merged_confidence = min(
            float(shared_plan.get("confidence", 0.5)),
            float(query_plan.get("confidence", 0.5)),
        )
        merged_delta = min(
            float(shared_plan.get("delta", merged_confidence)),
            float(query_plan.get("delta", merged_confidence)),
        )
        mode = "region-soft" if query_plan.get("mode") == "region-soft" else "region"
        region_smooth = None
        for source in (query_plan, shared_plan):
            region = source.get("region") if isinstance(source, dict) else None
            if isinstance(region, dict) and "smooth" in region:
                region_smooth = region.get("smooth")
                break
        region_grid_size = None
        for source in (query_plan, shared_plan):
            region = source.get("region") if isinstance(source, dict) else None
            if isinstance(region, dict) and region.get("grid_size") is not None:
                region_grid_size = int(region["grid_size"])
                break
        return {
            "mode": mode,
            "point": merged_point,
            "lower": lower,
            "upper": upper,
            "confidence": merged_confidence,
            "delta": merged_delta,
            "x_star": merged_point,
            "region": {
                "lower": dict(lower),
                "upper": dict(upper),
                "grid_size": region_grid_size,
                "smooth": region_smooth,
            },
        }

    def _region_numeric_lower(self, spec: ParamSpec, plan: Dict[str, Any]) -> float:
        if plan.get("mode") == "point":
            return float(self._coerce_value(plan["point"][spec.name], spec))
        lower = plan.get("lower", {})
        if spec.name in lower:
            return float(self._coerce_value(lower[spec.name], spec))
        if "point" in plan and spec.name in plan["point"]:
            return float(self._coerce_value(plan["point"][spec.name], spec))
        return float(spec.low)

    def _region_numeric_upper(self, spec: ParamSpec, plan: Dict[str, Any]) -> float:
        if plan.get("mode") == "point":
            return float(self._coerce_value(plan["point"][spec.name], spec))
        upper = plan.get("upper", {})
        if spec.name in upper:
            return float(self._coerce_value(upper[spec.name], spec))
        if "point" in plan and spec.name in plan["point"]:
            return float(self._coerce_value(plan["point"][spec.name], spec))
        return float(spec.high)

    def _region_fixed_choice(self, spec: ParamSpec, plan: Dict[str, Any]) -> Any:
        if plan.get("mode") == "point":
            point = plan.get("point", {})
            if spec.name in point:
                return self._coerce_value(point[spec.name], spec)
            return None
        lower = plan.get("lower", {})
        upper = plan.get("upper", {})
        if spec.name in lower and spec.name in upper:
            lo = self._coerce_value(lower[spec.name], spec)
            hi = self._coerce_value(upper[spec.name], spec)
            if lo == hi:
                return lo
        point = plan.get("point", {})
        if spec.name in point:
            return self._coerce_value(point[spec.name], spec)
        return None

    def _coerce_value(self, raw: Any, spec: ParamSpec) -> Any:
        if spec.kind == "int":
            value = max(float(spec.low), min(float(spec.high), float(raw)))
            step = int(spec.step or 1)
            value = float(spec.low) + round((value - float(spec.low)) / step) * step
            value = max(float(spec.low), min(float(spec.high), value))
            return int(round(value))
        if spec.kind == "float":
            value = max(float(spec.low), min(float(spec.high), float(raw)))
            if spec.step:
                step = float(spec.step)
                value = float(spec.low) + round((value - float(spec.low)) / step) * step
                value = max(float(spec.low), min(float(spec.high), value))
            return float(value)
        if spec.kind == "bool":
            if isinstance(raw, str):
                lowered = raw.strip().lower()
                if lowered in {"true", "1", "yes"}:
                    raw = True
                elif lowered in {"false", "0", "no"}:
                    raw = False
            for choice in spec.choices:
                if raw == choice:
                    return choice
            return bool(raw)
        for choice in spec.choices:
            if raw == choice:
                return choice
            if isinstance(choice, str) and str(raw).strip().lower() == choice.lower():
                return choice
        return spec.choices[0] if spec.choices else raw


class NumericSearchSpaceAdapter(MixedSearchSpaceAdapter):
    """Backward-compatible numeric-only view used by older tests and helpers."""

    def filter_numeric_distributions(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        return {
            name: dist
            for name, dist in search_space.items()
            if isinstance(dist, (FloatDistribution, IntDistribution))
        }

    def build_specs(self, search_space: Dict[str, Any]) -> List[NumericParamSpec]:
        return [spec for spec in super().build_specs(self.filter_numeric_distributions(search_space)) if spec.kind in {"float", "int"}]
