import json
import os
import time
import unittest
from pathlib import Path

import optuna

from Option.Config2 import Config
from Prompt.LGBOPrompt import build_lgbo_prompt
from Tuner.BOTuner.LGBO import LGBOSampler


class LGBOLiveLLMTimingTests(unittest.TestCase):
    """Opt-in live test for the single LGBO LLM call.

    This is intentionally a live/integration test, not a strict unit test:
    it hits the configured remote LLM endpoint once, measures latency, and
    records the exact prompt / response pair that LGBO would use.
    """

    @unittest.skipUnless(os.getenv("RUN_LGBO_LIVE") == "1", "Set RUN_LGBO_LIVE=1 to run live LLM timing test")
    def test_single_lgbo_llm_call_timing(self):
        config = Config.parse(
            Path("Option/LGBO_qwen_internal_api.yaml"),
            dataset_name="hotpotqa_100",
        )
        sampler = LGBOSampler(config=config)

        search_space = sampler.infer_relative_search_space(study=None, trial=None)
        supported = sampler.space.filter_supported_distributions(search_space)
        specs = sampler.space.build_specs(supported)

        storage = (
            "sqlite:////home/yingli/Youran/HEART/agent_workspace/runs/"
            "hotpotqa_100/lgbo_qwen_internal_api_1query_5trials/optuna_study/study.db"
        )
        study = optuna.load_study(
            study_name="lgbo_qwen_internal_api_1query_5trials__lgbo_shared_history",
            storage=storage,
        )

        completed_trials = sampler.history.completed_trials(study)
        observations = sampler.history.observations_from_trials(
            completed_trials,
            numeric_param_names=[spec.name for spec in specs],
        )
        history_lines = sampler.history.build_history_lines(
            observations,
            higher_is_better=True,
        )
        query_text = sampler.history.latest_query_text(study, completed_trials)
        current_query = (getattr(study, "user_attrs", {}) or {}).get("query")
        previous_reasoning = sampler.history.latest_reasoning(
            completed_trials,
            query=current_query,
        )

        prompt = build_lgbo_prompt(
            query_text=query_text,
            objective_name=getattr(config.tuner.optimization, "objective_1_name", "objective"),
            param_specs=[sampler.space.prompt_spec(spec) for spec in specs],
            history_lines=history_lines,
            previous_reasoning=previous_reasoning,
        )

        llm = sampler._get_llm()
        start = time.perf_counter()
        response = sampler._call_llm(prompt)
        elapsed = time.perf_counter() - start

        payload = {
            "model": getattr(llm, "model", None),
            "base_url": getattr(getattr(llm, "config", None), "base_url", None),
            "prompt_chars": len(prompt),
            "elapsed_s": elapsed,
            "response_chars": len(response or ""),
            "prompt": prompt,
            "response": response,
        }

        out = Path("agent_workspace/qwen_local_debug/test_lgbo_live_llm_timing.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

        self.assertEqual(payload["model"], "Qwen/Qwen3-8B")
        self.assertGreater(payload["prompt_chars"], 500)
        self.assertTrue(response and response.strip())


if __name__ == "__main__":
    unittest.main()
