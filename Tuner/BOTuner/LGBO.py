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
from Tuner.BOTuner.lgbo_components.search_space import (
    CategoricalSearchSpaceAdapter,
    NumericSearchSpaceAdapter,
)
from Tuner.BOTuner.lgbo_components.trace_store import LGBOTraceStore
from Tuner.BOTuner.lgbo_components.unified_surrogate import build_surrogate_layers


class LGBOSampler:
    """HEART **Tuner LGBO** — lives in ``Tuner/BOTuner/`` (this file + ``lgbo_components/``).

    This is the sampler selected by ``optimization.sampler == "lgbo"``. It is not
    the standalone reference package under the repo-root ``lgbo/`` folder; that
    tree is optional reference code (e.g. ``lgbo.decide`` may be imported for
    planning).

    All tunable dimensions share one normalized surrogate in ``[0,1]^d``
    (:mod:`Tuner.BOTuner.lgbo_components.unified_surrogate`). Proposals use
    :class:`Tuner.BOTuner.lgbo_components.surrogate.LGBONumericBayesGenerator`.
    """

    def __init__(self, config: Config, use_llm_guidance: bool = True):
        self.config = config
        self.use_llm_guidance = use_llm_guidance
        self.search_space = config.tuner.search_space
        self.history = LGBOHistoryAdapter()
        self.numeric_space = NumericSearchSpaceAdapter()
        self.categorical_space = CategoricalSearchSpaceAdapter()
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
        categorical_search_space = self.categorical_space.filter_categorical_distributions(
            search_space,
            exclude_names=set(),
        )
        categorical_specs = self.categorical_space.build_specs(categorical_search_space)
        surrogate_specs, metas = build_surrogate_layers(numeric_specs, categorical_specs)
        if not surrogate_specs:
            return self.search_space.sample_from_distributions(dists=search_space)

        completed_trials = self.history.completed_trials(study)
        budget_ctx = (getattr(study, "user_attrs", {}) or {}).get("lgbo_budget_context", {}) or {}
        current_cluster_id = budget_ctx.get("cluster_id")
        gamma = float(budget_ctx.get("gamma", 0.0))
        utility = float(budget_ctx.get("utility", 0.0))
        warm_start = budget_ctx.get("warm_start_candidate")

        def _trial_in_cluster(t: Any) -> bool:
            if current_cluster_id is None:
                return True
            q = (getattr(t, "user_attrs", {}) or {}).get("query") or {}
            return str(q.get("__cluster_id")) == str(current_cluster_id)

        selected_param_names = [spec.name for spec in numeric_specs] + [spec.name for spec in categorical_specs]
        observations = self.history.observations_from_trials(
            completed_trials,
            param_names=selected_param_names,
            trial_filter=_trial_in_cluster,
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

        # Warm-start transfer for cold-start clusters: if no local observation exists,
        # directly evaluate the source cluster's best candidate as the first local point.
        if warm_start and not observations:
            candidate = {k: v for k, v in warm_start.items() if k in search_space}
            if candidate:
                params = self.search_space.sample_from_distributions(dists=search_space)
                params.update(candidate)
                self.trace_store.write(
                    trial,
                    raw=None,
                    parsed={"mode": "warm_start_transfer", "cluster_id": current_cluster_id},
                    plan={
                        "mode": "warm_start_transfer",
                        "candidate": dict(candidate),
                        "candidate_strategy": {"mode": "warm_start_transfer"},
                    },
                    reasoning="Budget-aware warm-start transfer from high-synergy cluster best.",
                )
                return params

        if self.use_llm_guidance:
            try:
                param_specs = [spec.__dict__ for spec in numeric_specs] + [spec.__dict__ for spec in categorical_specs]
                prompt = build_lgbo_numeric_prompt(
                    query_text=query_text,
                    objective_name=objective_name,
                    param_specs=param_specs,
                    history_lines=history_lines,
                    previous_reasoning=previous_reasoning,
                )
                raw_response = self._call_llm(prompt)
                if raw_response:
                    preference, parsed_trace = self.preference_parser.parse_with_metadata(
                        raw_response,
                        [*numeric_specs, *categorical_specs],
                    )
                    reasoning = parsed_trace.get("thinking") or previous_reasoning
                    plan = self.preference_planner.make_plan(preference, [*numeric_specs, *categorical_specs])
                    if plan is not None:
                        plan["utility"] = utility
                        plan["adaptive_gamma"] = gamma
            except Exception as exc:
                plan = {
                    "mode": "numeric_v1_fallback",
                    "error": str(exc),
                    "utility": utility,
                    "adaptive_gamma": gamma,
                }

        candidate = self.candidates.propose(
            plan=plan,
            observations=observed_configs,
            specs=surrogate_specs,
            observation_records=observations,
            higher_is_better=higher_is_better,
            use_bayesian_surrogate=True,
            metas=metas,
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
            reasoning=reasoning or "LGBO unified surrogate candidate generation.",
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
