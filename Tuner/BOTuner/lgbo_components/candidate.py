from __future__ import annotations

from typing import Any, Dict, Sequence

from Tuner.BOTuner.lgbo_components.search_space import NumericParamSpec, NumericSearchSpaceAdapter
from Tuner.BOTuner.lgbo_components.unified_surrogate import SurrogateMeta, params_to_unit, unit_dict_to_params


class LGBOCandidateGenerator:
    """Propose trial points for LGBO (GP + tilt in ``[0,1]^d`` when ``metas`` is set).

    With ``metas``, ``specs`` are surrogate dimensions ``[0, 1]`` and proposals are
    decoded to physical Optuna params via :func:`unit_dict_to_params`. Without
    ``metas``, uses legacy physical numeric ``specs`` only.
    """

    def __init__(self) -> None:
        self.space = NumericSearchSpaceAdapter()
        self.last_strategy: Dict[str, Any] = {}

    @staticmethod
    def _default_center(spec: NumericParamSpec) -> float:
        return float((spec.low + spec.high) / 2.0)

    def _midpoint_from_region(self, plan: Dict[str, Any], specs: Sequence[NumericParamSpec]) -> Dict[str, float]:
        """Box center per surrogate dimension; missing keys in the plan use spec center."""
        lower = plan.get("lower") or {}
        upper = plan.get("upper") or {}
        point = plan.get("point") or {}
        out: Dict[str, float] = {}
        for spec in specs:
            if spec.name in lower and spec.name in upper:
                out[spec.name] = (float(lower[spec.name]) + float(upper[spec.name])) / 2.0
            elif spec.name in point:
                out[spec.name] = float(point[spec.name])
            else:
                out[spec.name] = self._default_center(spec)
        return out

    def _merge_partial_point(
        self, base: Dict[str, float], partial: Dict[str, Any] | None, specs: Sequence[NumericParamSpec]
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {**base}
        if partial:
            merged.update(partial)
        for spec in specs:
            if spec.name not in merged:
                merged[spec.name] = self._default_center(spec)
        return merged

    def propose(
        self,
        plan: Dict[str, Any] | None,
        observations: Sequence[Dict[str, Any]],
        specs: Sequence[NumericParamSpec],
        *,
        observation_records: Sequence[Any] | None = None,
        higher_is_better: bool = True,
        use_bayesian_surrogate: bool = False,
        metas: Sequence[SurrogateMeta] | None = None,
    ) -> Dict[str, Any]:
        self.last_strategy = {"mode": "heuristic"}
        unified = metas is not None and len(metas) == len(specs)

        if use_bayesian_surrogate and observation_records:
            try:
                from Tuner.BOTuner.lgbo_components.surrogate import LGBONumericBayesGenerator

                candidate = LGBONumericBayesGenerator().propose(
                    plan=plan,
                    observations=observation_records,
                    specs=specs,
                    higher_is_better=higher_is_better,
                    metas=metas if unified else None,
                )
                self.last_strategy = {
                    "mode": "bayes_surrogate",
                    "plan_mode": (plan or {}).get("mode", "none"),
                }
                return candidate
            except Exception as exc:
                self.last_strategy = {
                    "mode": "heuristic_fallback",
                    "error": str(exc),
                    "plan_mode": (plan or {}).get("mode", "none"),
                }

        def _clip_u(d: Dict[str, Any]) -> Dict[str, Any]:
            if unified:
                return unit_dict_to_params(self.space.clip_point(d, specs), metas)
            return self.space.clip_point(d, specs)

        if plan and plan.get("mode") == "point":
            return _clip_u(plan["point"])

        if plan and plan.get("mode") == "region-soft":
            midpoint = self._midpoint_from_region(plan, specs)
            anchor = self.space.clip_point(
                self._merge_partial_point(midpoint, plan.get("point"), specs),
                specs,
            )
            confidence = float(plan.get("confidence", 0.5))
            candidate_u = {}
            for spec in specs:
                candidate_u[spec.name] = confidence * float(anchor[spec.name]) + (1.0 - confidence) * float(
                    midpoint[spec.name]
                )
            return _clip_u(candidate_u)

        if plan and plan.get("mode") == "region":
            midpoint = self._midpoint_from_region(plan, specs)
            return _clip_u(midpoint)

        best = self._best_observation(observations, observation_records=observation_records, higher_is_better=higher_is_better)
        if best:
            if unified:
                u = params_to_unit(best, metas)
                u = self.space.clip_point(u, specs)
                return unit_dict_to_params(u, metas)
            return self.space.clip_point(best, specs)

        mid = {spec.name: self.space._coerce_value((spec.low + spec.high) / 2.0, spec) for spec in specs}
        if unified:
            return unit_dict_to_params(mid, metas)
        return mid

    def _best_observation(
        self,
        observations: Sequence[Dict[str, Any]],
        *,
        observation_records: Sequence[Any] | None,
        higher_is_better: bool,
    ) -> Dict[str, Any] | None:
        if observation_records:
            scored = []
            for record in observation_records:
                params = getattr(record, "params", None)
                objective = getattr(record, "objective", None)
                if params and objective is not None:
                    scored.append((float(objective), params))
            if scored:
                select = max if higher_is_better else min
                return select(scored, key=lambda item: item[0])[1]

        if observations:
            return observations[-1]
        return None
