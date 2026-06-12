"""LLM-enhanced TPE sampler for RAG hyperparameter optimization.

Inspired by LLAMBO (Ma et al., 2024, arXiv:2402.03921):
- LLM replaces the Parzen estimator (density model) in TPE
- Given observed (config, score) history, LLM suggests the next config
- Conditions on target threshold γ = top-25% score (LLAMBO-style)
- Falls back to random sampling when insufficient history
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any, Dict, List, Sequence, Tuple

from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.study import Study
from optuna.trial import FrozenTrial

from Common.Logger import logger
from Option.Config2 import Config
from Provider.LLMProviderRegister import create_llm_instance


LLM_TPE_SYSTEM_PROMPT = """You are a hyperparameter optimization expert for RAG (Retrieval-Augmented Generation) pipelines.

Given a history of evaluated configurations and their F1 scores, suggest ONE new configuration
that is likely to achieve a higher F1 score than the target threshold.

Rules:
- Return ONLY a valid JSON object with the exact parameter names listed in the search space
- All values must be within the specified bounds / choices
- Do not add explanations or extra text outside the JSON
- Integer parameters must be integers, not floats
"""


class LLMTPESampler:
    """LLAMBO-inspired LLM-TPE sampler: LLM as config generator conditioned on target score.

    At each step (after n_startup random trials), the LLM receives:
    - The RAG search space definition
    - Top-20 past observations sorted by score (best first)
    - Target threshold γ = 75th percentile of observed scores

    It returns a JSON config to evaluate next.
    """

    def __init__(self, config: Config, n_startup: int = 10):
        self.config = config
        self.n_startup = n_startup
        self.llm = None

    def infer_relative_search_space(
        self, study: Study | None, trial: FrozenTrial | None
    ) -> Dict[str, BaseDistribution]:
        def flatten_dict(d: dict) -> dict:
            flat = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    flat.update(flatten_dict(value))
                else:
                    flat[key] = value
            return flat

        raw = self.config.tuner.search_space.build_distributions(self.config.tuner.tuner_params)
        return flatten_dict(raw)

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        observations = self._get_observations(study, search_space)

        if len(observations) < self.n_startup:
            logger.debug("[LLM-TPE] Startup phase (%d/%d), random sample", len(observations), self.n_startup)
            return self._random_sample(search_space)

        # γ = 75th percentile score (top 25% threshold, per LLAMBO)
        scores = [s for s, _ in observations]
        import statistics
        gamma = sorted(scores)[int(len(scores) * 0.75)]

        prompt = self._build_prompt(search_space, observations, gamma)
        try:
            raw_response = self._call_llm(prompt)
            if raw_response:
                parsed = self._parse_response(raw_response, search_space)
                if parsed:
                    logger.debug("[LLM-TPE] LLM suggestion accepted")
                    return parsed
        except Exception as exc:
            logger.warning("[LLM-TPE] LLM call failed: %s — falling back to random", exc)

        return self._random_sample(search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        name: str,
        distribution: BaseDistribution,
    ) -> Any:
        raise NotImplementedError("LLMTPESampler only supports relative sampling")

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        pass

    def after_trial(self, study: Study, trial: FrozenTrial, state: Any, values: Any) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_observations(
        self,
        study: Study,
        search_space: Dict[str, BaseDistribution],
    ) -> List[Tuple[float, Dict[str, Any]]]:
        observations = []
        for t in study.get_trials(deepcopy=False):
            state_name = getattr(getattr(t, "state", None), "name", "")
            if state_name not in ("COMPLETE", "complete"):
                continue
            values = getattr(t, "values", None)
            if not values:
                continue
            score = float(values[0])

            # Extract params from trial.params, then flow JSON, then user_attrs suggested:*
            params = self._extract_params(t, search_space)
            if not params:
                continue
            observations.append((score, params))

        observations.sort(key=lambda x: x[0], reverse=True)
        return observations

    def _extract_params(
        self,
        trial: Any,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        # Try trial.params first (standard Optuna)
        trial_params = getattr(trial, "params", {}) or {}
        params = {k: v for k, v in trial_params.items() if k in search_space}
        if params:
            return params

        # Try user_attrs["suggested:*"]
        user_attrs = getattr(trial, "user_attrs", {}) or {}
        params = {
            k: user_attrs[f"suggested:{k}"]
            for k in search_space
            if f"suggested:{k}" in user_attrs
        }
        if params:
            return params

        # Try flow JSON
        flow_raw = user_attrs.get("flow")
        if flow_raw:
            try:
                flow = json.loads(flow_raw) if isinstance(flow_raw, str) else flow_raw
                params = {k: v for k, v in flow.items() if k in search_space}
                if params:
                    return params
            except Exception:
                pass

        return {}

    def _build_prompt(
        self,
        search_space: Dict[str, BaseDistribution],
        observations: List[Tuple[float, Dict[str, Any]]],
        gamma: float,
    ) -> str:
        lines = [
            "Optimize a RAG pipeline. Suggest a config likely to score above the target.",
            "",
            "Search space:",
        ]
        for name, dist in search_space.items():
            if isinstance(dist, CategoricalDistribution):
                lines.append(f"  {name}: one of {list(dist.choices)}")
            elif isinstance(dist, IntDistribution):
                lines.append(f"  {name}: int in [{dist.low}, {dist.high}]")
            elif isinstance(dist, FloatDistribution):
                lines.append(f"  {name}: float in [{dist.low:.4f}, {dist.high:.4f}]")

        lines += ["", f"Target threshold (gamma): {gamma:.4f}", "", "Past observations (best first):"]
        for i, (score, cfg) in enumerate(observations[:20]):
            cfg_str = ", ".join(f"{k}={v}" for k, v in list(cfg.items())[:8])
            lines.append(f"  {i+1}. score={score:.4f} | {cfg_str}")

        lines += [
            "",
            f"Suggest ONE config that would likely achieve score > {gamma:.4f}.",
            "Return ONLY a JSON object with ALL parameters from the search space.",
        ]
        return "\n".join(lines)

    def _parse_response(
        self,
        response: str,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any] | None:
        # Extract first JSON object from response
        match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
        if not match:
            logger.warning("[LLM-TPE] No JSON found in response: %s", response[:200])
            return None

        try:
            raw = json.loads(match.group())
        except json.JSONDecodeError as e:
            logger.warning("[LLM-TPE] JSON parse error: %s", e)
            return None

        params: Dict[str, Any] = {}
        for name, dist in search_space.items():
            val = raw.get(name)
            if val is None:
                # Fall back to random for missing key
                params[name] = self._sample_one(dist)
                continue
            try:
                if isinstance(dist, CategoricalDistribution):
                    params[name] = val if val in dist.choices else random.choice(list(dist.choices))
                elif isinstance(dist, IntDistribution):
                    params[name] = max(dist.low, min(dist.high, int(float(val))))
                elif isinstance(dist, FloatDistribution):
                    params[name] = max(dist.low, min(dist.high, float(val)))
                else:
                    params[name] = val
            except Exception:
                params[name] = self._sample_one(dist)

        return params

    def _random_sample(self, search_space: Dict[str, BaseDistribution]) -> Dict[str, Any]:
        return {name: self._sample_one(dist) for name, dist in search_space.items()}

    @staticmethod
    def _sample_one(dist: BaseDistribution) -> Any:
        if isinstance(dist, CategoricalDistribution):
            return random.choice(list(dist.choices))
        elif isinstance(dist, IntDistribution):
            return random.randint(dist.low, dist.high)
        elif isinstance(dist, FloatDistribution):
            return random.uniform(dist.low, dist.high)
        return None

    def _call_llm(self, user_prompt: str) -> str | None:
        llm = self._get_llm()
        if llm is None:
            return None
        messages = [
            {"role": "system", "content": LLM_TPE_SYSTEM_PROMPT},
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
