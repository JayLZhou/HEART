import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from optuna.distributions import FloatDistribution, IntDistribution
from Tuner.BOTuner.lgbo_components.history import LGBOHistoryAdapter
from Tuner.BOTuner.lgbo_components.preference import LGBOPreferenceParser, LGBOPreferencePlanner
from Tuner.BOTuner.lgbo_components.search_space import NumericSearchSpaceAdapter
from Tuner.BOTuner.lgbo_components.trace_store import LGBOTraceStore


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

    def test_preference_parser_and_planner_support_point(self):
        specs = NumericSearchSpaceAdapter().build_specs(
            {
                "rag_top_k": IntDistribution(2, 10, step=1),
                "rag_hybrid_bm25_weight": FloatDistribution(0.1, 0.9, step=0.1),
            }
        )
        parser = LGBOPreferenceParser()
        pref = parser.parse("[\"point\", [6, 0.5], 0.8]", specs)
        plan = LGBOPreferencePlanner().make_plan(pref, specs)
        self.assertEqual(plan["mode"], "point")
        self.assertEqual(plan["point"]["rag_top_k"], 6.0)

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

