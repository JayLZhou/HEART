from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution


@dataclass(frozen=True)
class NumericParamSpec:
    name: str
    kind: str
    low: float
    high: float
    step: float | None = None
    log: bool = False


@dataclass(frozen=True)
class CategoricalParamSpec:
    name: str
    kind: str
    choices: tuple[Any, ...]


class NumericSearchSpaceAdapter:
    """Keep LGBO V1 focused on numeric parameters only."""

    def filter_numeric_distributions(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        return {
            name: dist
            for name, dist in search_space.items()
            if isinstance(dist, (FloatDistribution, IntDistribution))
        }

    def build_specs(self, search_space: Dict[str, Any]) -> List[NumericParamSpec]:
        specs: List[NumericParamSpec] = []
        for name, dist in self.filter_numeric_distributions(search_space).items():
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
        return specs

    def normalize_point(self, point: Dict[str, Any], specs: Sequence[NumericParamSpec]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for spec in specs:
            value = float(point[spec.name])
            width = spec.high - spec.low
            out[spec.name] = 0.0 if width <= 0 else (value - spec.low) / width
        return out

    def denormalize_point(self, point: Dict[str, float], specs: Sequence[NumericParamSpec]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for spec in specs:
            raw = spec.low + point[spec.name] * (spec.high - spec.low)
            out[spec.name] = self._coerce_value(raw, spec)
        return out

    def clip_point(self, point: Dict[str, Any], specs: Sequence[NumericParamSpec]) -> Dict[str, Any]:
        return {spec.name: self._coerce_value(point[spec.name], spec) for spec in specs}

    def _coerce_value(self, raw: float | int, spec: NumericParamSpec) -> Any:
        value = max(spec.low, min(spec.high, float(raw)))
        if spec.kind == "int":
            step = int(spec.step or 1)
            value = spec.low + round((value - spec.low) / step) * step
            value = max(spec.low, min(spec.high, value))
            return int(round(value))
        if spec.step:
            step = float(spec.step)
            value = spec.low + round((value - spec.low) / step) * step
            value = max(spec.low, min(spec.high, value))
        return float(value)


class CategoricalSearchSpaceAdapter:
    """Filter and describe categorical parameters used by the prompt-guided path."""

    def filter_categorical_distributions(
        self,
        search_space: Dict[str, Any],
        *,
        exclude_names: Iterable[str] | None = None,
    ) -> Dict[str, CategoricalDistribution]:
        excluded = set(exclude_names or [])
        return {
            name: dist
            for name, dist in search_space.items()
            if isinstance(dist, CategoricalDistribution) and name not in excluded
        }

    def build_specs(self, search_space: Dict[str, Any]) -> List[CategoricalParamSpec]:
        specs: List[CategoricalParamSpec] = []
        for name, dist in search_space.items():
            if not isinstance(dist, CategoricalDistribution):
                continue
            kind = "bool" if all(isinstance(choice, bool) for choice in dist.choices) else "categorical"
            specs.append(
                CategoricalParamSpec(
                    name=name,
                    kind=kind,
                    choices=tuple(dist.choices),
                )
            )
        return specs
