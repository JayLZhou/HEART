from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


COMPLETE_TRIAL_STATES = {"COMPLETE", "complete"}


def _trial_state_name(trial: Any) -> str | None:
    state = getattr(trial, "state", None)
    if state is None:
        return None
    return getattr(state, "name", str(state))


@dataclass
class LGBOObservation:
    params: Dict[str, Any]
    objective: float
    flow: Dict[str, Any] | None
    query: Dict[str, Any] | None
    reasoning: str | None


class LGBOHistoryAdapter:
    """Extract lightweight LGBO history from an Optuna-like study."""

    def completed_trials(self, study: Any) -> List[Any]:
        trials = list(study.get_trials(deepcopy=False))
        return [trial for trial in trials if _trial_state_name(trial) in COMPLETE_TRIAL_STATES]

    def observations_from_trials(
        self,
        trials: Sequence[Any],
        numeric_param_names: Sequence[str],
    ) -> List[LGBOObservation]:
        observations: List[LGBOObservation] = []
        for trial in trials:
            params = {name: trial.params[name] for name in numeric_param_names if name in getattr(trial, "params", {})}
            if not params:
                continue
            objective = self._extract_objective(trial)
            user_attrs = getattr(trial, "user_attrs", {}) or {}
            observations.append(
                LGBOObservation(
                    params=params,
                    objective=objective,
                    flow=user_attrs.get("flow"),
                    query=user_attrs.get("query"),
                    reasoning=user_attrs.get("lgbo_reasoning"),
                )
            )
        return observations

    def observed_configs(self, observations: Sequence[LGBOObservation]) -> List[Dict[str, Any]]:
        return [obs.params for obs in observations]

    def observed_objectives(self, observations: Sequence[LGBOObservation]) -> List[float]:
        return [obs.objective for obs in observations]

    def latest_reasoning(self, trials: Sequence[Any]) -> str | None:
        for trial in reversed(list(trials)):
            reasoning = (getattr(trial, "user_attrs", {}) or {}).get("lgbo_reasoning")
            if reasoning:
                return reasoning
        return None

    def _extract_objective(self, trial: Any) -> float:
        values = getattr(trial, "values", None)
        if values:
            return float(values[0])
        value = getattr(trial, "value", None)
        if value is None:
            raise ValueError("Completed trial is missing objective value(s)")
        return float(value)

