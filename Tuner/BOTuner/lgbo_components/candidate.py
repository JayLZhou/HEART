from __future__ import annotations

from typing import Any, Dict, Sequence

from Tuner.BOTuner.lgbo_components.search_space import NumericParamSpec, NumericSearchSpaceAdapter


class LGBOCandidateGenerator:
    """Lightweight V1 candidate generator for numeric-only LGBO.

    This deliberately avoids heavyweight BO dependencies. It provides a safe,
    testable candidate proposal path that can later be replaced by a full
    surrogate + acquisition implementation.
    """

    def __init__(self) -> None:
        self.space = NumericSearchSpaceAdapter()
        self.last_strategy: Dict[str, Any] = {}

    def propose(
        self,
        plan: Dict[str, Any] | None,
        observations: Sequence[Dict[str, Any]],
        specs: Sequence[NumericParamSpec],
        *,
        observation_records: Sequence[Any] | None = None,
        higher_is_better: bool = True,
        use_bayesian_surrogate: bool = False,
    ) -> Dict[str, Any]:
        self.last_strategy = {"mode": "heuristic"}
        if use_bayesian_surrogate and observation_records:
            try:
                from Tuner.BOTuner.lgbo_components.surrogate import LGBONumericBayesGenerator

                candidate = LGBONumericBayesGenerator().propose(
                    plan=plan,
                    observations=observation_records,
                    specs=specs,
                    higher_is_better=higher_is_better,
                )
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
            return self.space.clip_point(plan["point"], specs)

        if plan and plan.get("mode") == "region-soft":
            midpoint = {
                spec.name: (float(plan["lower"][spec.name]) + float(plan["upper"][spec.name])) / 2.0
                for spec in specs
            }
            anchor = self.space.clip_point(plan.get("point", midpoint), specs)
            confidence = float(plan.get("confidence", 0.5))
            candidate = {}
            for spec in specs:
                candidate[spec.name] = confidence * float(anchor[spec.name]) + (1.0 - confidence) * float(midpoint[spec.name])
            return self.space.clip_point(candidate, specs)

        if plan and plan.get("mode") == "region":
            midpoint = {
                spec.name: (float(plan["lower"][spec.name]) + float(plan["upper"][spec.name])) / 2.0
                for spec in specs
            }
            return self.space.clip_point(midpoint, specs)

        best = self._best_observation(observations, observation_records=observation_records, higher_is_better=higher_is_better)
        if best:
            return self.space.clip_point(best, specs)

        return {
            spec.name: self.space._coerce_value((spec.low + spec.high) / 2.0, spec)
            for spec in specs
        }

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

