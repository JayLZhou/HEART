from __future__ import annotations

from typing import Any, Dict

from optuna.distributions import BaseDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial

from Option.Config2 import Config
from Tuner.BOTuner.lgbo_components.candidate import LGBOCandidateGenerator
from Tuner.BOTuner.lgbo_components.history import LGBOHistoryAdapter
from Tuner.BOTuner.lgbo_components.preference import LGBOPreferencePlanner
from Tuner.BOTuner.lgbo_components.search_space import NumericSearchSpaceAdapter
from Tuner.BOTuner.lgbo_components.trace_store import LGBOTraceStore


class LGBOSampler:
    """Dependency-light V1 LGBO sampler.

    This initial version focuses on the numeric parameter subspace only and
    keeps candidate generation lightweight so it can be developed without
    requiring the full HEART runtime stack.
    """

    def __init__(self, config: Config):
        self.config = config
        self.search_space = config.tuner.search_space
        self.history = LGBOHistoryAdapter()
        self.numeric_space = NumericSearchSpaceAdapter()
        self.preference_planner = LGBOPreferencePlanner()
        self.trace_store = LGBOTraceStore()
        self.candidates = LGBOCandidateGenerator()

    def infer_relative_search_space(self, study: Study | None, trial: FrozenTrial | None) -> Dict[str, BaseDistribution]:
        def flatten_dict(d: dict) -> dict:
            flat = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    flat.update(flatten_dict(value))
                else:
                    flat[key] = value
            return flat

        search_space = self.config.tuner.search_space.build_distributions(self.config.tuner.tuner_params)
        return flatten_dict(search_space)

    def sample_relative(self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]) -> Dict[str, Any]:
        numeric_search_space = self.numeric_space.filter_numeric_distributions(search_space)
        numeric_specs = self.numeric_space.build_specs(numeric_search_space)
        if not numeric_specs:
            return self.search_space.sample_from_distributions(dists=search_space)

        completed_trials = self.history.completed_trials(study)
        observations = self.history.observations_from_trials(
            completed_trials,
            numeric_param_names=[spec.name for spec in numeric_specs],
        )
        observed_configs = self.history.observed_configs(observations)

        # V1: keep the LGBO interface in place while defaulting to lightweight
        # candidate generation. Full preference-driven prompting can be added on
        # top of this structure later without changing the Optuna integration.
        plan = None
        candidate = self.candidates.propose(
            plan=plan,
            observations=observed_configs,
            specs=numeric_specs,
        )

        params = self.search_space.sample_from_distributions(dists=search_space)
        params.update(candidate)
        self.trace_store.write(
            trial,
            raw=None,
            parsed=None,
            plan={"mode": "numeric_v1_fallback", "candidate": dict(candidate)},
            reasoning="LGBO V1 numeric-only fallback candidate generation.",
        )
        return params

    def sample_independent(self, study: Study, trial: FrozenTrial, name: str, distribution: BaseDistribution):
        raise NotImplementedError("LGBOSampler only supports relative sampling")

