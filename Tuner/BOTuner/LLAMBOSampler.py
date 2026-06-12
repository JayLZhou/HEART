"""LLAMBO sampler: LLM as zero-shot surrogate for RAG hyperparameter optimization.

Based on LLAMBO (Ma et al., 2024, arXiv:2402.03921):
- Generate K random candidate configs per round
- Use LLM to predict the score of each candidate given observed history
- Select the candidate with highest predicted score
- Falls back to random when insufficient history (n_startup)
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any, Dict, List, Tuple

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


LLAMBO_SYSTEM_PROMPT = """You are a hyperparameter optimization expert for RAG (Retrieval-Augmented Generation) pipelines.
Your task is to predict the accuracy score of a given RAG configuration based on observed history.
Respond with ONLY a single float between 0.0 and 1.0 — no explanation, no extra text."""


class LLAMBOSampler:
    """LLAMBO: LLM as zero-shot surrogate for RAG hyperparameter search.

    At each step (after n_startup random trials):
    1. Sample k_candidates random configs from the search space
    2. For each candidate, ask the LLM to predict its score given history
    3. Return the candidate with the highest predicted score

    This is the LLAMBO zero-shot surrogate variant (Ma et al., 2024).
    """

    def __init__(self, config: Config, n_startup: int = 1, k_candidates: int = 5):
        self.config = config
        self.n_startup = n_startup
        self.k_candidates = k_candidates
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
        history = self._get_history(study, search_space)

        if len(history) < self.n_startup:
            logger.info("[LLAMBO] Startup phase ({}/{}) random sample".format(len(history), self.n_startup))
            return self._random_sample(search_space)

        candidates = [self._random_sample(search_space) for _ in range(self.k_candidates)]

        predicted_scores = []
        for i, cfg in enumerate(candidates):
            try:
                score = self._predict_score(history, search_space, cfg)
            except Exception as exc:
                logger.warning("[LLAMBO] Prediction failed for candidate %d: %s — using 0.0", i, exc)
                score = 0.0
            predicted_scores.append(score)
            logger.debug("[LLAMBO] Candidate %d predicted score: %.4f", i, score)

        best_idx = max(range(len(candidates)), key=lambda i: predicted_scores[i])
        logger.info(
            "[LLAMBO] Predicted scores: %s → selecting candidate #%d (%.4f)",
            [f"{s:.3f}" for s in predicted_scores],
            best_idx,
            predicted_scores[best_idx],
        )
        return candidates[best_idx]

    def sample_independent(self, study, trial, name, distribution):
        raise NotImplementedError("LLAMBOSampler only supports relative sampling")

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        pass

    def after_trial(self, study: Study, trial: FrozenTrial, state: Any, values: Any) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_history(
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
        trial_params = getattr(trial, "params", {}) or {}
        params = {k: v for k, v in trial_params.items() if k in search_space}
        if params:
            return params

        user_attrs = getattr(trial, "user_attrs", {}) or {}
        params = {
            k: user_attrs[f"suggested:{k}"]
            for k in search_space
            if f"suggested:{k}" in user_attrs
        }
        if params:
            return params

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

    def _predict_score(
        self,
        history: List[Tuple[float, Dict[str, Any]]],
        search_space: Dict[str, BaseDistribution],
        candidate: Dict[str, Any],
    ) -> float:
        prompt = self._build_prediction_prompt(history, search_space, candidate)
        response = self._call_llm(prompt)
        return self._parse_score(response)

    def _build_prediction_prompt(
        self,
        history: List[Tuple[float, Dict[str, Any]]],
        search_space: Dict[str, BaseDistribution],
        candidate: Dict[str, Any],
    ) -> str:
        lines = [
            "Past evaluated RAG configurations (accuracy score, higher is better):",
            "",
        ]
        for score, cfg in history[:20]:
            key_params = ["rag_method", "reranker_choice", "rag_top_k", "rag_query_decomposition_enabled"]
            summary = ", ".join(
                f"{k}={cfg[k]}" for k in key_params if k in cfg
            )
            rest = ", ".join(f"{k}={v}" for k, v in cfg.items() if k not in key_params)
            if rest:
                summary += f", {rest[:80]}"
            lines.append(f"  score={score:.4f} | {summary}")

        lines += [
            "",
            "Candidate configuration to score:",
            "  " + ", ".join(f"{k}={v}" for k, v in candidate.items()),
            "",
            "Based on the history above, predict the accuracy score (0.0–1.0) for this candidate.",
            "Return ONLY a single float.",
        ]
        return "\n".join(lines)

    def _parse_score(self, response: str | None) -> float:
        if not response:
            return 0.0
        response = response.strip()
        match = re.search(r'\b(1\.0+|0\.\d+|\.\d+|[01])\b', response)
        if match:
            try:
                val = float(match.group())
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        logger.warning("[LLAMBO] Could not parse score from: %s", response[:100])
        return 0.0

    def _call_llm(self, user_prompt: str) -> str | None:
        llm = self._get_llm()
        if llm is None:
            return None
        messages = [
            {"role": "system", "content": LLAMBO_SYSTEM_PROMPT},
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
