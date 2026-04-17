from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable


class ParameterCoverageTracker:
    def __init__(self, parameter_names: Iterable[str]):
        self.parameter_names = list(parameter_names)
        self.values_by_param: Dict[str, set[Any]] = {name: set() for name in self.parameter_names}

    def observe_trial(self, trial: Any) -> None:
        attrs = getattr(trial, "user_attrs", {}) or {}
        for name in self.parameter_names:
            key = f"suggested:{name}"
            if key in attrs:
                self.values_by_param[name].add(self._hashable(attrs[key]))

    def summary(self) -> Dict[str, Any]:
        per_param = {}
        exercised = []
        for name in self.parameter_names:
            values = sorted(self.values_by_param[name], key=str)
            exercised_flag = len(values) > 1
            if exercised_flag:
                exercised.append(name)
            per_param[name] = {
                "num_values": len(values),
                "values": [self._restore(value) for value in values],
                "exercised": exercised_flag,
            }
        return {
            "parameter_count": len(self.parameter_names),
            "exercised_count": len(exercised),
            "exercised_parameters": exercised,
            "per_parameter": per_param,
        }

    def _hashable(self, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(self._hashable(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((k, self._hashable(v)) for k, v in value.items()))
        return value

    def _restore(self, value: Any) -> Any:
        if isinstance(value, tuple):
            if value and all(isinstance(item, tuple) and len(item) == 2 for item in value):
                return {k: self._restore(v) for k, v in value}
            return [self._restore(item) for item in value]
        return value
