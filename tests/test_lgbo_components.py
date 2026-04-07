import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from optuna.distributions import FloatDistribution, IntDistribution


def _ensure_package(name: str) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = []  # mark as package
    sys.modules[name] = module


def _load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ensure_package("Prompt")
_ensure_package("Tuner")
_ensure_package("Tuner.BOTuner")
_ensure_package("Tuner.BOTuner.lgbo_components")

prompt_module = _load_module("Prompt.LGBOPrompt", "Prompt/LGBOPrompt.py")
search_space_module = _load_module(
    "Tuner.BOTuner.lgbo_components.search_space",
    "Tuner/BOTuner/lgbo_components/search_space.py",
)
history_module = _load_module(
    "Tuner.BOTuner.lgbo_components.history",
    "Tuner/BOTuner/lgbo_components/history.py",
)
preference_module = _load_module(
    "Tuner.BOTuner.lgbo_components.preference",
    "Tuner/BOTuner/lgbo_components/preference.py",
)
trace_store_module = _load_module(
    "Tuner.BOTuner.lgbo_components.trace_store",
    "Tuner/BOTuner/lgbo_components/trace_store.py",
)
candidate_module = _load_module(
    "Tuner.BOTuner.lgbo_components.candidate",
    "Tuner/BOTuner/lgbo_components/candidate.py",
)

build_lgbo_numeric_prompt = prompt_module.build_lgbo_numeric_prompt
LGBOCandidateGenerator = candidate_module.LGBOCandidateGenerator
LGBOHistoryAdapter = history_module.LGBOHistoryAdapter
LGBOPreferenceParser = preference_module.LGBOPreferenceParser
LGBOPreferencePlanner = preference_module.LGBOPreferencePlanner
NumericSearchSpaceAdapter = search_space_module.NumericSearchSpaceAdapter
LGBOTraceStore = trace_store_module.LGBOTraceStore


class FakeTrial:
    def __init__(self, params, value, state="COMPLETE", user_attrs=None):
        self.params = params
        self.value = value
        self.state = types.SimpleNamespace(name=state)
        self.user_attrs = user_attrs or {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class FakeStudy:
    def __init__(self, trials):
        self._trials = trials

    def get_trials(self, deepcopy=False):
        return list(self._trials)


class LGBOComponentTests(unittest.TestCase):
    def test_numeric_search_space_filters_float_and_int(self):
        adapter = NumericSearchSpaceAdapter()
        dists = {
            "rag_top_k": IntDistribution(2, 10, step=1),
            "rag_hybrid_bm25_weight": FloatDistribution(0.1, 0.9, step=0.1),
            "category": object(),
        }
        filtered = adapter.filter_numeric_distributions(dists)
        self.assertEqual(set(filtered), {"rag_top_k", "rag_hybrid_bm25_weight"})

    def test_history_adapter_extracts_completed_numeric_observations(self):
        adapter = LGBOHistoryAdapter()
        study = FakeStudy(
            [
                FakeTrial({"rag_top_k": 5, "rag_hybrid_bm25_weight": 0.4}, 0.81, state="COMPLETE"),
                FakeTrial({"rag_top_k": 7}, 0.73, state="FAIL"),
            ]
        )
        trials = adapter.completed_trials(study)
        observations = adapter.observations_from_trials(trials, ["rag_top_k", "rag_hybrid_bm25_weight"])
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].params["rag_top_k"], 5)
        self.assertAlmostEqual(observations[0].objective, 0.81)

    def test_history_adapter_can_filter_reasoning_to_current_query(self):
        adapter = LGBOHistoryAdapter()
        trials = [
            FakeTrial({"rag_top_k": 5}, 0.81, user_attrs={"query": {"id": 0}, "lgbo_reasoning": "first-query"}),
            FakeTrial({"rag_top_k": 7}, 0.73, user_attrs={"query": {"id": 1}, "lgbo_reasoning": "second-query"}),
        ]
        reasoning = adapter.latest_reasoning(trials, query={"id": 0})
        self.assertEqual(reasoning, "first-query")

    def test_preference_parser_and_planner_support_point(self):
        specs = NumericSearchSpaceAdapter().build_specs(
            {
                "rag_top_k": IntDistribution(2, 10, step=1),
                "rag_hybrid_bm25_weight": FloatDistribution(0.1, 0.9, step=0.1),
            }
        )
        parser = LGBOPreferenceParser()
        pref, meta = parser.parse_with_metadata(
            "Thinking:\nPick a compact promising neighborhood.\n\nFinal Answer:\n[\"point\", [6, 0.5], 0.8]",
            specs,
        )
        plan = LGBOPreferencePlanner().make_plan(pref, specs)
        self.assertEqual(meta["mode"], "point")
        self.assertEqual(plan["mode"], "region-soft")
        self.assertEqual(plan["point"]["rag_top_k"], 6.0)
        self.assertIn("thinking", meta)

    def test_preference_parser_accepts_bareword_point(self):
        specs = NumericSearchSpaceAdapter().build_specs(
            {
                "rag_top_k": IntDistribution(2, 10, step=1),
                "rag_hybrid_bm25_weight": FloatDistribution(0.1, 0.9, step=0.1),
            }
        )
        parser = LGBOPreferenceParser()
        pref, meta = parser.parse_with_metadata(
            "Final Answer:\n[point, [6, 0.5], 0.8]",
            specs,
        )
        self.assertEqual(meta["mode"], "point")
        self.assertEqual(pref.values["rag_top_k"], 6.0)

    def test_preference_parser_accepts_transposed_region_pairs(self):
        specs = NumericSearchSpaceAdapter().build_specs(
            {
                "rag_top_k": IntDistribution(2, 10, step=1),
                "rag_hybrid_bm25_weight": FloatDistribution(0.1, 0.9, step=0.1),
                "rag_query_decomposition_num_queries": IntDistribution(2, 20, step=1),
                "reranker_top_k": IntDistribution(2, 128, step=1),
            }
        )
        parser = LGBOPreferenceParser()
        pref, meta = parser.parse_with_metadata(
            "Final Answer:\n[region, [[2, 10], [0.1, 0.9], [2, 20], [10, 128]], 0.3]",
            specs,
        )
        self.assertEqual(meta["mode"], "region")
        self.assertEqual(pref.lower["rag_top_k"], 2.0)
        self.assertEqual(pref.upper["rag_hybrid_bm25_weight"], 0.9)

    def test_candidate_generator_fallback_uses_best_objective(self):
        generator = LGBOCandidateGenerator()
        specs = NumericSearchSpaceAdapter().build_specs(
            {
                "rag_top_k": IntDistribution(2, 10, step=1),
                "rag_hybrid_bm25_weight": FloatDistribution(0.1, 0.9, step=0.1),
            }
        )
        records = [
            types.SimpleNamespace(
                params={"rag_top_k": 3, "rag_hybrid_bm25_weight": 0.2},
                objective=0.51,
            ),
            types.SimpleNamespace(
                params={"rag_top_k": 7, "rag_hybrid_bm25_weight": 0.6},
                objective=0.82,
            ),
        ]
        candidate = generator.propose(
            plan=None,
            observations=[record.params for record in records],
            specs=specs,
            observation_records=records,
            higher_is_better=True,
        )
        self.assertEqual(candidate["rag_top_k"], 7)
        self.assertAlmostEqual(candidate["rag_hybrid_bm25_weight"], 0.6)

    def test_candidate_generator_can_use_bayesian_surrogate(self):
        generator = LGBOCandidateGenerator()
        specs = NumericSearchSpaceAdapter().build_specs(
            {
                "rag_top_k": IntDistribution(2, 10, step=1),
                "rag_hybrid_bm25_weight": FloatDistribution(0.1, 0.9, step=0.1),
            }
        )
        records = [
            types.SimpleNamespace(
                params={"rag_top_k": 2, "rag_hybrid_bm25_weight": 0.2},
                objective=0.20,
            ),
            types.SimpleNamespace(
                params={"rag_top_k": 5, "rag_hybrid_bm25_weight": 0.5},
                objective=0.65,
            ),
            types.SimpleNamespace(
                params={"rag_top_k": 8, "rag_hybrid_bm25_weight": 0.8},
                objective=0.90,
            ),
        ]
        plan = {
            "mode": "region-soft",
            "point": {"rag_top_k": 8, "rag_hybrid_bm25_weight": 0.8},
            "lower": {"rag_top_k": 7, "rag_hybrid_bm25_weight": 0.7},
            "upper": {"rag_top_k": 9, "rag_hybrid_bm25_weight": 0.9},
            "confidence": 0.8,
        }
        candidate = generator.propose(
            plan=plan,
            observations=[record.params for record in records],
            specs=specs,
            observation_records=records,
            higher_is_better=True,
            use_bayesian_surrogate=True,
        )
        self.assertIn(generator.last_strategy["mode"], {"bayes_surrogate", "heuristic_fallback"})
        self.assertGreaterEqual(candidate["rag_top_k"], 2)
        self.assertLessEqual(candidate["rag_top_k"], 10)
        self.assertGreaterEqual(candidate["rag_hybrid_bm25_weight"], 0.1)
        self.assertLessEqual(candidate["rag_hybrid_bm25_weight"], 0.9)

    def test_prompt_builder_includes_history_and_reasoning(self):
        prompt = build_lgbo_numeric_prompt(
            query_text="Answer the user question accurately.",
            objective_name="accuracy",
            param_specs=[
                {"name": "rag_top_k", "low": 2, "high": 10, "kind": "int"},
                {"name": "rag_hybrid_bm25_weight", "low": 0.1, "high": 0.9, "kind": "float"},
            ],
            history_lines=["recent_1: objective=0.82; params=(rag_top_k=7) best_so_far"],
            previous_reasoning="Hybrid retrieval seems promising.",
        )
        self.assertIn("Completed trial history:", prompt)
        self.assertIn("Hybrid retrieval seems promising.", prompt)
        self.assertIn("Answer the user question accurately.", prompt)

    def test_trace_store_writes_attrs(self):
        trial = FakeTrial({}, 0.0)
        store = LGBOTraceStore()
        store.write(
            trial,
            raw="[\"point\", [6], 0.9]",
            parsed={"mode": "point"},
            plan={"mode": "point"},
            reasoning="test reasoning",
        )
        self.assertEqual(trial.user_attrs["lgbo_reasoning"], "test reasoning")
        self.assertEqual(trial.user_attrs["lgbo_plan"]["mode"], "point")


if __name__ == "__main__":
    unittest.main()

