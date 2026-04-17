from __future__ import annotations

from typing import Any, Dict, Sequence

from Tuner.BOTuner.lgbo_components.search_space import MixedSearchSpaceAdapter, ParamSpec


class LGBOCandidateGenerator:
    """Mixed-space LGBO candidate generator.

    The planner still uses paper-style point / region preferences, while the
    surrogate maps mixed HEART parameters into a continuous encoded space.
    """

    def __init__(self) -> None:
        self.space = MixedSearchSpaceAdapter()
        self.last_strategy: Dict[str, Any] = {}

    def propose(
        self,
        plan: Dict[str, Any] | None,
        observations: Sequence[Dict[str, Any]],
        specs: Sequence[ParamSpec],
        *,
        observation_records: Sequence[Any] | None = None,
        higher_is_better: bool = True,
        use_bayesian_surrogate: bool = False,
    ) -> Dict[str, Any]:
        self.last_strategy = {"mode": "heuristic"}
        if use_bayesian_surrogate and observation_records:
            try:
                from Tuner.BOTuner.lgbo_components.surrogate import LGBOMixedBayesGenerator

                candidate = LGBOMixedBayesGenerator().propose(
                    plan=plan,
                    observations=observation_records,
                    specs=specs,
                    higher_is_better=higher_is_better,
                )
                if plan and plan.get("mode") == "region-soft" and plan.get("point"):
                    anchor = self.space.clip_point_to_region(plan["point"], plan, specs)
                    confidence = float(plan.get("confidence", 0.5))
                    blended = {}
                    for spec in specs:
                        if spec.kind in {"float", "int"}:
                            blended[spec.name] = (
                                (1.0 - confidence) * float(candidate[spec.name])
                                + confidence * float(anchor[spec.name])
                            )
                        else:
                            blended[spec.name] = anchor[spec.name]
                    candidate = blended
                if not (plan and plan.get("mode") == "point"):
                    candidate = self.space.clip_point_to_region(candidate, plan, specs)
                self.last_strategy = {
                    "mode": "bayes_surrogate",
                    "plan_mode": (plan or {}).get("mode", "none"),
                }
                return self.space.clip_point(candidate, specs)
            except Exception as exc:
                self.last_strategy = {
                    "mode": "heuristic_fallback",
                    "error": str(exc),
                    "plan_mode": (plan or {}).get("mode", "none"),
                }

        if plan and plan.get("mode") == "point":
            candidate = self.space.clip_point(plan["point"], specs)
            return self.space.clip_point_to_region(candidate, plan, specs)

        if plan and plan.get("mode") == "region-soft":
            midpoint = self._region_midpoint(plan, specs)
            anchor = self.space.clip_point(plan.get("point", midpoint), specs)
            confidence = float(plan.get("confidence", 0.5))
            candidate = {}
            for spec in specs:
                if spec.kind in {"float", "int"}:
                    candidate[spec.name] = (
                        confidence * float(anchor[spec.name])
                        + (1.0 - confidence) * float(midpoint[spec.name])
                    )
                else:
                    candidate[spec.name] = anchor[spec.name]
            candidate = self.space.clip_point(candidate, specs)
            return self.space.clip_point_to_region(candidate, plan, specs)

        if plan and plan.get("mode") == "region":
            midpoint = self._region_midpoint(plan, specs)
            midpoint = self.space.clip_point(midpoint, specs)
            return self.space.clip_point_to_region(midpoint, plan, specs)

        best = self._best_observation(observations, observation_records=observation_records, higher_is_better=higher_is_better)
        if best:
            best = self.space.clip_point(best, specs)
            return self.space.clip_point_to_region(best, plan, specs)

        default = self.space.default_point(specs)
        return self.space.clip_point_to_region(default, plan, specs)

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

    def _region_midpoint(self, plan: Dict[str, Any], specs: Sequence[ParamSpec]) -> Dict[str, Any]:
        midpoint: Dict[str, Any] = {}
        point_hint = plan.get("point", {})
        for spec in specs:
            if spec.kind in {"float", "int"}:
                midpoint[spec.name] = (
                    float(plan["lower"][spec.name]) + float(plan["upper"][spec.name])
                ) / 2.0
                continue
            if spec.name in point_hint:
                midpoint[spec.name] = point_hint[spec.name]
                continue
            lower = plan["lower"].get(spec.name)
            upper = plan["upper"].get(spec.name)
            if lower is not None and upper is not None and lower == upper:
                midpoint[spec.name] = lower
            else:
                midpoint[spec.name] = spec.choices[0] if spec.choices else None
        return midpoint
