from __future__ import annotations

import asyncio
from typing import Any, Dict

from optuna.distributions import BaseDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial

from Option.Config2 import Config
from Prompt.LGBOPrompt import LGBO_NUMERIC_SYSTEM_PROMPT, build_lgbo_numeric_prompt
from Provider.LLMProviderRegister import create_llm_instance
from Tuner.BOTuner.lgbo_components.candidate import LGBOCandidateGenerator
from Tuner.BOTuner.lgbo_components.history import LGBOHistoryAdapter
from Tuner.BOTuner.lgbo_components.preference import LGBOPreferenceParser, LGBOPreferencePlanner
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
        self.preference_parser = LGBOPreferenceParser()
        self.preference_planner = LGBOPreferencePlanner()
        self.trace_store = LGBOTraceStore()
        self.candidates = LGBOCandidateGenerator()
        self.llm = None

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
        objective_name = getattr(self.config.tuner.optimization, "objective_1_name", "objective")
        higher_is_better = True
        history_lines = self.history.build_history_lines(
            observations,
            higher_is_better=higher_is_better,
        )
        query_text = self.history.latest_query_text(study, completed_trials)
        current_query = (getattr(study, "user_attrs", {}) or {}).get("query")
        previous_reasoning = self.history.latest_reasoning(completed_trials, query=current_query)

        raw_response = None
        parsed_trace = None
        plan = None
        reasoning = previous_reasoning

        try:
            prompt = build_lgbo_numeric_prompt(
                query_text=query_text,
                objective_name=objective_name,
                param_specs=[spec.__dict__ for spec in numeric_specs],
                history_lines=history_lines,
                previous_reasoning=previous_reasoning,
            )
            raw_response = self._call_llm(prompt)
            if raw_response:
                preference, parsed_trace = self.preference_parser.parse_with_metadata(raw_response, numeric_specs)
                reasoning = parsed_trace.get("thinking") or previous_reasoning
                plan = self.preference_planner.make_plan(preference, numeric_specs)
        except Exception as exc:
            plan = {
                "mode": "numeric_v1_fallback",
                "error": str(exc),
            }

        candidate = self.candidates.propose(
            plan=plan,
            observations=observed_configs,
            specs=numeric_specs,
            observation_records=observations,
            higher_is_better=higher_is_better,
            use_bayesian_surrogate=True,
        )

        params = self.search_space.sample_from_distributions(dists=search_space)
        params.update(candidate)
        self.trace_store.write(
            trial,
            raw=raw_response,
            parsed=parsed_trace,
            plan={
                **(plan or {"mode": "numeric_v1_fallback"}),
                "candidate": dict(candidate),
                "candidate_strategy": dict(self.candidates.last_strategy),
            },
            reasoning=reasoning or "LGBO V1 numeric-only fallback candidate generation.",
        )
        return params

    def sample_independent(self, study: Study, trial: FrozenTrial, name: str, distribution: BaseDistribution):
        raise NotImplementedError("LGBOSampler only supports relative sampling")

    def before_trial(self, study: Study, trial: FrozenTrial):
        pass

    def after_trial(self, study: Study, trial: FrozenTrial, state, values):
        pass

    def _call_llm(self, user_prompt: str) -> str | None:
        llm = self._get_llm()
        if llm is None:
            return None

        messages = [
            {"role": "system", "content": LGBO_NUMERIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self._run_coro(llm.acompletion_text(messages=messages, stream=False))

    def _get_llm(self):
        if self.llm is None and getattr(self.config, "llms", None):
            self.llm = create_llm_instance(self.config.llms[0])
        return self.llm

    def _run_coro(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

