from __future__ import annotations

import asyncio
from typing import Any, Dict, Sequence

from optuna.distributions import BaseDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial

from Option.Config2 import Config
from Prompt.LGBOPrompt import LGBO_SYSTEM_PROMPT, build_lgbo_prompt
from Provider.LLMProviderRegister import create_llm_instance
from Tuner.BOTuner.lgbo_components.candidate import LGBOCandidateGenerator
from Tuner.BOTuner.lgbo_components.history import LGBOHistoryAdapter
from Tuner.BOTuner.lgbo_components.preference import LGBOPreferenceParser, LGBOPreferencePlanner
from Tuner.BOTuner.lgbo_components.search_space import MixedSearchSpaceAdapter, ParamSpec
from Tuner.BOTuner.lgbo_components.trace_store import LGBOTraceStore


class LGBOSampler:
    """Mixed-space LGBO sampler for HEART."""

    MAX_TOTAL_REQUEST_CONCURRENCY = 28
    MAX_KEY_ATTEMPTS_PER_REQUEST = 3

    def __init__(self, config: Config):
        self.config = config
        self.search_space = config.tuner.search_space
        self.history = LGBOHistoryAdapter()
        self.space = MixedSearchSpaceAdapter()
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
        result = self.sample_relative_batch(
            study=study,
            trial_contexts=[
                {
                    "trial": trial,
                    "query": (getattr(study, "user_attrs", {}) or {}).get("query"),
                }
            ],
            search_space=search_space,
        )
        return result[0]

    def sample_relative_batch(
        self,
        *,
        study: Study,
        trial_contexts: Sequence[Dict[str, Any]],
        search_space: Dict[str, BaseDistribution],
    ) -> list[Dict[str, Any]]:
        supported_search_space = self.space.filter_supported_distributions(search_space)
        specs = self.space.build_specs(supported_search_space)
        if not specs:
            return [self.search_space.sample_from_distributions(dists=search_space) for _ in trial_contexts]

        llm_inputs = [self._prepare_trial_input(study, ctx, specs) for ctx in trial_contexts]
        raw_responses = self._call_llm_batch([item["prompt"] for item in llm_inputs])

        results: list[Dict[str, Any]] = []
        for ctx, prepared, raw_response in zip(trial_contexts, llm_inputs, raw_responses):
            results.append(
                self._finalize_candidate(
                    trial=ctx["trial"],
                    raw_response=raw_response,
                    prepared=prepared,
                    search_space=search_space,
                    specs=specs,
                )
            )
        return results

    def sample_independent(self, study: Study, trial: FrozenTrial, name: str, distribution: BaseDistribution):
        raise NotImplementedError("LGBOSampler only supports relative sampling")

    def before_trial(self, study: Study, trial: FrozenTrial):
        pass

    def after_trial(self, study: Study, trial: FrozenTrial, state, values):
        pass

    def _prepare_trial_input(self, study: Study, ctx: Dict[str, Any], specs: Sequence[ParamSpec]) -> Dict[str, Any]:
        completed_trials = ctx.get("completed_trials")
        if completed_trials is None:
            completed_trials = self.history.completed_trials(study)

        observations = self.history.observations_from_trials(
            completed_trials,
            numeric_param_names=[spec.name for spec in specs],
        )
        observed_configs = self.history.observed_configs(observations)
        objective_name = getattr(self.config.tuner.optimization, "objective_1_name", "objective")
        higher_is_better = True
        history_lines = self.history.build_lgbo_history_entries(
            observations,
            max_items=ctx.get("history_max_items"),
            objective_name=objective_name,
        )
        query = ctx.get("query")
        query_text = self._query_text(query) or self.history.latest_query_text(study, completed_trials)
        previous_reasoning = self.history.latest_reasoning(completed_trials, query=query)
        prompt = build_lgbo_prompt(
            query_text=query_text,
            objective_name=objective_name,
            param_specs=[self.space.prompt_spec(spec) for spec in specs],
            history_lines=history_lines,
            previous_reasoning=previous_reasoning,
            round_id=ctx.get("round_id"),
            cluster_id=ctx.get("cluster_id"),
            shared_region=ctx.get("shared_region"),
        )
        return {
            "completed_trials": completed_trials,
            "observations": observations,
            "observed_configs": observed_configs,
            "objective_name": objective_name,
            "higher_is_better": higher_is_better,
            "query": query,
            "query_text": query_text,
            "previous_reasoning": previous_reasoning,
            "prompt": prompt,
            "shared_region": ctx.get("shared_region"),
        }

    def _finalize_candidate(
        self,
        *,
        trial: FrozenTrial,
        raw_response: str | None,
        prepared: Dict[str, Any],
        search_space: Dict[str, BaseDistribution],
        specs: Sequence[ParamSpec],
    ) -> Dict[str, Any]:
        parsed_trace = None
        plan = None
        reasoning = prepared.get("previous_reasoning")

        try:
            if raw_response:
                preference, parsed_trace = self.preference_parser.parse_with_metadata(raw_response, specs)
                reasoning = parsed_trace.get("thinking") or reasoning
                plan = self.preference_planner.make_plan(preference, specs)
        except Exception as exc:
            plan = {
                "mode": "mixed_v1_fallback",
                "error": str(exc),
            }

        shared_region = prepared.get("shared_region")
        active_plan = self.space.intersect_region_plans(shared_region, plan, specs)
        if shared_region is not None and active_plan is None:
            active_plan = shared_region

        candidate = self.candidates.propose(
            plan=active_plan,
            observations=prepared["observed_configs"],
            specs=specs,
            observation_records=prepared["observations"],
            higher_is_better=prepared["higher_is_better"],
            use_bayesian_surrogate=True,
        )

        params = self.search_space.sample_from_distributions(dists=search_space)
        params.update(candidate)
        self.trace_store.write(
            trial,
            raw=raw_response,
            parsed=parsed_trace,
            plan={
                **(active_plan or {"mode": "mixed_v1_fallback"}),
                "candidate": dict(candidate),
                "candidate_strategy": dict(self.candidates.last_strategy),
                "shared_region": shared_region,
            },
            reasoning=reasoning or "LGBO V1 mixed-space fallback candidate generation.",
        )
        return params

    def _query_text(self, query: Any) -> str | None:
        if query is None:
            return None
        if isinstance(query, dict):
            for key in ("question", "query", "task", "input", "prompt", "id"):
                value = query.get(key)
                if value:
                    return str(value)
        if isinstance(query, str):
            return query
        return str(query)

    def _call_llm(self, user_prompt: str) -> str | None:
        llm = self._get_llm()
        if llm is None:
            return None
        messages = [
            {"role": "system", "content": LGBO_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self._run_coro(llm.acompletion_text(messages=messages, stream=False))

    def _call_llm_batch(self, prompts: Sequence[str]) -> list[str | None]:
        llm = self._get_llm()
        if llm is None:
            return [None for _ in prompts]
        return self._run_coro(self._acall_llm_batch(prompts))

    async def _acall_llm_batch(self, prompts: Sequence[str]) -> list[str | None]:
        llm = self._get_llm()
        if llm is None:
            return [None for _ in prompts]
        if not prompts:
            return []
        max_parallel = min(self.MAX_TOTAL_REQUEST_CONCURRENCY, len(prompts))
        semaphore = asyncio.Semaphore(max_parallel)

        async def _one(prompt: str) -> str | None:
            async with semaphore:
                return await self._acall_prompt_with_immediate_failover(llm, prompt)

        return await asyncio.gather(*(_one(prompt) for prompt in prompts))

    async def _acall_prompt_with_immediate_failover(
        self,
        llm: Any,
        prompt: str,
    ) -> str | None:
        messages = [
            {"role": "system", "content": LGBO_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if not hasattr(llm, "acquire_batch_slot") or not hasattr(llm, "release_batch_slot") or not hasattr(
            llm, "acompletion_text_with_slot"
        ):
            return await llm.acompletion_text(messages=messages, stream=False)

        slot_count_fn = getattr(llm, "batch_slot_count", None)
        slot_attempts = max(1, int(slot_count_fn())) if callable(slot_count_fn) else 1
        slot_attempts = min(slot_attempts, self.MAX_KEY_ATTEMPTS_PER_REQUEST)
        last_error: Exception | None = None

        for _ in range(slot_attempts):
            slot_idx = await llm.acquire_batch_slot()
            try:
                return await llm.acompletion_text_with_slot(slot_idx=slot_idx, messages=messages, stream=False)
            except Exception as exc:
                # immediate key failover: no extra sleep here
                last_error = exc
            finally:
                llm.release_batch_slot(slot_idx)

        # Give up this prompt after N key attempts so one bad prompt does not block the round.
        if last_error is not None:
            return None
        return None

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
