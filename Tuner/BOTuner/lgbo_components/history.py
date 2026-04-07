from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence
import json


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
            flow = self._extract_flow(trial)
            params = self._extract_params(trial, flow, numeric_param_names)
            if not params:
                continue
            objective = self._extract_objective(trial)
            user_attrs = getattr(trial, "user_attrs", {}) or {}
            observations.append(
                LGBOObservation(
                    params=params,
                    objective=objective,
                    flow=flow,
                    query=user_attrs.get("query"),
                    reasoning=user_attrs.get("lgbo_reasoning"),
                )
            )
        return observations

    def observed_configs(self, observations: Sequence[LGBOObservation]) -> List[Dict[str, Any]]:
        return [obs.params for obs in observations]

    def observed_objectives(self, observations: Sequence[LGBOObservation]) -> List[float]:
        return [obs.objective for obs in observations]

    def best_observation(
        self,
        observations: Sequence[LGBOObservation],
        *,
        higher_is_better: bool = True,
    ) -> LGBOObservation | None:
        if not observations:
            return None
        key_fn = lambda obs: obs.objective
        return max(observations, key=key_fn) if higher_is_better else min(observations, key=key_fn)

    def latest_reasoning(self, trials: Sequence[Any], *, query: Any | None = None) -> str | None:
        target_query_key = self._query_key(query)
        for trial in reversed(list(trials)):
            if target_query_key is not None:
                trial_query_key = self._query_key((getattr(trial, "user_attrs", {}) or {}).get("query"))
                if trial_query_key != target_query_key:
                    continue
            reasoning = (getattr(trial, "user_attrs", {}) or {}).get("lgbo_reasoning")
            if reasoning:
                return reasoning
        return None

    def latest_query_text(self, study: Any, trials: Sequence[Any]) -> str:
        study_query = (getattr(study, "user_attrs", {}) or {}).get("query")
        if study_query:
            return self._stringify_query(study_query)

        for trial in reversed(list(trials)):
            query = (getattr(trial, "user_attrs", {}) or {}).get("query")
            if query:
                return self._stringify_query(query)
        return "Optimize the objective using the available numeric search space."

    def build_history_lines(
        self,
        observations: Sequence[LGBOObservation],
        *,
        higher_is_better: bool = True,
        max_items: int = 6,
    ) -> List[str]:
        if not observations:
            return []

        ordered = list(observations)
        ordered = ordered[-max_items:]
        if not higher_is_better:
            best = min(obs.objective for obs in observations)
        else:
            best = max(obs.objective for obs in observations)

        lines: List[str] = []
        for idx, obs in enumerate(reversed(ordered), start=1):
            params = ", ".join(f"{name}={value}" for name, value in obs.params.items())
            mark = " best_so_far" if obs.objective == best else ""
            query_tag = ""
            query_key = self._query_key(obs.query)
            if query_key is not None:
                query_tag = f"; query_id={query_key}"
            lines.append(f"recent_{idx}: objective={obs.objective}; params=({params}){query_tag}{mark}")
        return lines

    def _extract_objective(self, trial: Any) -> float:
        values = getattr(trial, "values", None)
        if values:
            return float(values[0])
        value = getattr(trial, "value", None)
        if value is None:
            raise ValueError("Completed trial is missing objective value(s)")
        return float(value)

    def _stringify_query(self, query: Any) -> str:
        if isinstance(query, str):
            return query
        if isinstance(query, dict):
            for key in ("question", "query", "task", "input", "prompt", "id"):
                value = query.get(key)
                if value:
                    return str(value)
            return json.dumps(query, ensure_ascii=False, sort_keys=True)
        return str(query)

    def _query_key(self, query: Any) -> str | None:
        if query is None:
            return None
        if isinstance(query, dict):
            for key in ("id", "_id", "source_id", "question"):
                value = query.get(key)
                if value is not None:
                    return str(value)
            return json.dumps(query, ensure_ascii=False, sort_keys=True)
        return str(query)

    def _extract_flow(self, trial: Any) -> Dict[str, Any] | None:
        flow = (getattr(trial, "user_attrs", {}) or {}).get("flow")
        if flow is None:
            return None
        if isinstance(flow, dict):
            return flow
        if isinstance(flow, str):
            try:
                parsed = json.loads(flow)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    def _extract_params(
        self,
        trial: Any,
        flow: Dict[str, Any] | None,
        numeric_param_names: Sequence[str],
    ) -> Dict[str, Any]:
        trial_params = getattr(trial, "params", {}) or {}
        params = {name: trial_params[name] for name in numeric_param_names if name in trial_params}
        if params:
            return params

        if flow:
            return {name: flow[name] for name in numeric_param_names if name in flow}

        user_attrs = getattr(trial, "user_attrs", {}) or {}
        suggested_prefix = "suggested:"
        return {
            name: user_attrs[f"{suggested_prefix}{name}"]
            for name in numeric_param_names
            if f"{suggested_prefix}{name}" in user_attrs
        }

