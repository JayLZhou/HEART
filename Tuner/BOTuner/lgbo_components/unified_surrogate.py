"""Unified [0, 1]^d surrogate coordinates for all tunable LGBO dimensions.

Mirrors the reference ``lgbo/`` setup where the expert preference and GP live in
the same normalized box. Numeric Optuna params use physical min/max scaling;
categorical / boolean choices map to equally spaced fractions in (0, 1), as in
common one-dimensional relaxations of discrete sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from Tuner.BOTuner.lgbo_components.search_space import (
    CategoricalParamSpec,
    NumericParamSpec,
    NumericSearchSpaceAdapter,
)


@dataclass(frozen=True)
class SurrogateMeta:
    """One row of the joint surrogate; maps to a physical Optuna parameter."""

    name: str
    numeric: NumericParamSpec | None = None
    categorical: CategoricalParamSpec | None = None


def build_surrogate_layers(
    numeric_specs: Sequence[NumericParamSpec],
    categorical_specs: Sequence[CategoricalParamSpec],
) -> tuple[List[NumericParamSpec], List[SurrogateMeta]]:
    """Return per-dimension [0, 1] surrogate specs and metadata for decode."""
    space = NumericSearchSpaceAdapter()
    surrogate_specs: List[NumericParamSpec] = []
    metas: List[SurrogateMeta] = []

    for ns in numeric_specs:
        surrogate_specs.append(
            NumericParamSpec(
                name=ns.name,
                kind="float",
                low=0.0,
                high=1.0,
                step=None,
                log=False,
            )
        )
        metas.append(SurrogateMeta(name=ns.name, numeric=ns, categorical=None))

    for cs in categorical_specs:
        surrogate_specs.append(
            NumericParamSpec(
                name=cs.name,
                kind="float",
                low=0.0,
                high=1.0,
                step=None,
                log=False,
            )
        )
        metas.append(SurrogateMeta(name=cs.name, numeric=None, categorical=cs))

    return surrogate_specs, metas


def params_to_unit(
    params: Dict[str, Any],
    metas: Sequence[SurrogateMeta],
    *,
    space: NumericSearchSpaceAdapter | None = None,
) -> Dict[str, float]:
    """Map a flat trial ``params`` dict to surrogate coordinates in [0, 1]."""
    space = space or NumericSearchSpaceAdapter()
    out: Dict[str, float] = {}
    for m in metas:
        v = params.get(m.name)
        if m.numeric is not None:
            norm = space.normalize_point({m.name: v}, [m.numeric])
            out[m.name] = float(norm[m.name])
        elif m.categorical is not None:
            choices = tuple(m.categorical.choices)
            k = len(choices)
            if k <= 0:
                out[m.name] = 0.5
            elif v not in choices:
                out[m.name] = 0.5
            else:
                idx = choices.index(v)
                out[m.name] = (float(idx) + 0.5) / float(k)
        else:
            out[m.name] = 0.5
    return out


def unit_dict_to_params(
    unit: Dict[str, float],
    metas: Sequence[SurrogateMeta],
    *,
    space: NumericSearchSpaceAdapter | None = None,
) -> Dict[str, Any]:
    """Inverse of :func:`params_to_unit` (clip units, then decode)."""
    space = space or NumericSearchSpaceAdapter()
    out: Dict[str, Any] = {}
    for m in metas:
        u = float(unit.get(m.name, 0.5))
        u = max(0.0, min(1.0, u))
        if m.numeric is not None:
            denorm = space.denormalize_point({m.name: u}, [m.numeric])
            out[m.name] = denorm[m.name]
        elif m.categorical is not None:
            choices = tuple(m.categorical.choices)
            k = len(choices)
            if k <= 0:
                continue
            if k == 1:
                out[m.name] = choices[0]
            else:
                idx = int(round(u * k - 0.5))
                idx = max(0, min(k - 1, idx))
                out[m.name] = choices[idx]
    return out


def ordered_unit_vector(unit: Dict[str, float], metas: Sequence[SurrogateMeta]) -> List[float]:
    return [float(unit.get(m.name, 0.5)) for m in metas]
