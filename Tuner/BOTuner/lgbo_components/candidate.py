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

    def propose(
        self,
        plan: Dict[str, Any] | None,
        observations: Sequence[Dict[str, Any]],
        specs: Sequence[NumericParamSpec],
    ) -> Dict[str, Any]:
        if plan and plan.get("mode") == "point":
            return self.space.clip_point(plan["point"], specs)

        if plan and plan.get("mode") == "region":
            midpoint = {
                spec.name: (float(plan["lower"][spec.name]) + float(plan["upper"][spec.name])) / 2.0
                for spec in specs
            }
            return self.space.clip_point(midpoint, specs)

        if observations:
            best = observations[-1]
            return self.space.clip_point(best, specs)

        return {
            spec.name: self.space._coerce_value((spec.low + spec.high) / 2.0, spec)
            for spec in specs
        }

