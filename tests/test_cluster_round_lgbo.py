import importlib.util
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_package(name: str) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = []
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
_ensure_package("Tuner.BOTuner.Cluster")
_ensure_package("Tuner.BOTuner.lgbo_components")

_load_module("Prompt.LGBOPrompt", "Prompt/LGBOPrompt.py")
_load_module("Tuner.BOTuner.lgbo_components.search_space", "Tuner/BOTuner/lgbo_components/search_space.py")
_load_module("Tuner.BOTuner.lgbo_components.history", "Tuner/BOTuner/lgbo_components/history.py")
_load_module("Tuner.BOTuner.lgbo_components.preference", "Tuner/BOTuner/lgbo_components/preference.py")
_load_module("Tuner.BOTuner.lgbo_components.trace_store", "Tuner/BOTuner/lgbo_components/trace_store.py")
_load_module("Tuner.BOTuner.lgbo_components.candidate", "Tuner/BOTuner/lgbo_components/candidate.py")
_load_module("Tuner.BOTuner.lgbo_components.surrogate", "Tuner/BOTuner/lgbo_components/surrogate.py")
_load_module("Tuner.BOTuner.lgbo_components.monitoring", "Tuner/BOTuner/lgbo_components/monitoring.py")
_load_module("Tuner.BOTuner.Cluster.cluster_state", "Tuner/BOTuner/Cluster/cluster_state.py")
lgbo_module = _load_module("Tuner.BOTuner.LGBO", "Tuner/BOTuner/LGBO.py")
cluster_module = _load_module("Tuner.BOTuner.Cluster.ClusterRoundLGBO", "Tuner/BOTuner/Cluster/ClusterRoundLGBO.py")

from Config.SearchSpace import SearchSpace


LGBOSampler = lgbo_module.LGBOSampler
ClusterRoundLGBORunner = cluster_module.ClusterRoundLGBORunner


TARGETS = {
    0: {
        "template_name": "default",
        "response_synthesizer_llm": "fixed-llm",
        "faiss_hnsw_m": 16,
        "faiss_hnsw_ef_search": 32,
        "faiss_hnsw_ef_construction": 40,
        "faiss_metric": "l2",
        "rag_method": "dense",
        "rag_query_decomposition_enabled": False,
        "rag_top_k": 12,
        "rag_hybrid_bm25_weight": 0.1,
        "rag_query_decomposition_llm_name": "fixed-llm",
        "rag_query_decomposition_num_queries": 2,
        "rag_fusion_mode": "simple",
        "reranker_name": "flashrank",
        "reranker_top_k": 8,
    },
    1: {
        "template_name": "cot",
        "response_synthesizer_llm": "fixed-llm",
        "faiss_hnsw_m": 32,
        "faiss_hnsw_ef_search": 128,
        "faiss_hnsw_ef_construction": 160,
        "faiss_metric": "inner_product",
        "rag_method": "hybrid",
        "rag_query_decomposition_enabled": False,
        "rag_top_k": 48,
        "rag_hybrid_bm25_weight": 0.5,
        "rag_query_decomposition_llm_name": "fixed-llm",
        "rag_query_decomposition_num_queries": 2,
        "rag_fusion_mode": "simple",
        "reranker_name": "flashrank",
        "reranker_top_k": 24,
    },
    2: {
        "template_name": "default",
        "response_synthesizer_llm": "fixed-llm",
        "faiss_hnsw_m": 64,
        "faiss_hnsw_ef_search": 256,
        "faiss_hnsw_ef_construction": 320,
        "faiss_metric": "l2",
        "rag_method": "sparse",
        "rag_query_decomposition_enabled": False,
        "rag_top_k": 96,
        "rag_hybrid_bm25_weight": 0.9,
        "rag_query_decomposition_llm_name": "fixed-llm",
        "rag_query_decomposition_num_queries": 2,
        "rag_fusion_mode": "simple",
        "reranker_name": "flashrank",
        "reranker_top_k": 40,
    },
}


class FakeLLM:
    def __init__(self):
        self.prompt_records = []

    async def acompletion_text(self, messages, stream=False):
        prompt = messages[-1]["content"]
        record = self._parse_prompt(prompt)
        self.prompt_records.append(record)
        return self._response(record)

    def _parse_prompt(self, prompt: str):
        lines = prompt.splitlines()
        round_id = int(next(line.split(":", 1)[1].strip() for line in lines if line.startswith("Round:")))
        task_line = next(line for line in lines if line.startswith("- group="))
        group = int(task_line.split("group=")[1].split()[0])
        query_id = int(task_line.split("query=")[1].split()[0])
        param_names = []
        capture = False
        for line in lines:
            if line.strip() == "Parameter order:":
                capture = True
                continue
            if capture and not line.strip():
                break
            if capture and line.startswith("- "):
                param_names.append(line.split(":")[0][2:])
        shared_region_lines = []
        capture = False
        for line in lines:
            if line.strip() == "Shared cluster region:":
                capture = True
                continue
            if capture and not line.strip():
                break
            if capture:
                shared_region_lines.append(line)
        return {
            "prompt": prompt,
            "round_id": round_id,
            "group": group,
            "query_id": query_id,
            "param_names": param_names,
            "shared_region_lines": shared_region_lines,
        }

    def _response(self, record):
        group = record["group"]
        round_id = record["round_id"]
        query_id = record["query_id"]
        target = dict(TARGETS[group])
        offset = (query_id % 3) - 1
        if round_id == 1:
            values = {
                **target,
                "template_name": "cot" if query_id % 2 else "default",
                "faiss_hnsw_m": [16, 32, 64][query_id % 3],
                "faiss_hnsw_ef_search": [32, 128, 256][query_id % 3],
                "faiss_hnsw_ef_construction": [40, 160, 320][query_id % 3],
                "faiss_metric": "inner_product" if query_id % 2 else "l2",
                "rag_method": ["dense", "hybrid", "sparse"][query_id % 3],
                "rag_top_k": min(128, max(2, target["rag_top_k"] + offset * 18)),
                "rag_hybrid_bm25_weight": round(min(0.9, max(0.1, target["rag_hybrid_bm25_weight"] + offset * 0.3)), 1),
                "reranker_top_k": min(128, max(2, target["reranker_top_k"] + offset * 8)),
            }
        else:
            values = {
                **target,
                "rag_top_k": min(128, max(2, target["rag_top_k"] + offset * 2)),
                "reranker_top_k": min(128, max(2, target["reranker_top_k"] + offset)),
                "rag_hybrid_bm25_weight": round(min(0.9, max(0.1, target["rag_hybrid_bm25_weight"] + offset * 0.1)), 1),
            }
        ordered_values = [values[name] for name in record["param_names"]]
        return f'Thinking:\nquery {query_id} round {round_id}\n\nFinal Answer:\n["point", {repr(ordered_values)}, 0.8]'


class FakeFlow:
    def __init__(self, flat_params):
        self.flat_params = flat_params

    def query(self, query: str) -> str:
        time.sleep(0.01)
        return json.dumps(self.flat_params)


class FakeBuilder:
    def build_flow(self, params):
        flat = {}
        for key, value in params.items():
            if key == "rag_retriever":
                for sub_key, sub_value in value.items():
                    flat[f"rag_{sub_key}"] = sub_value
            elif key == "reranker":
                for sub_key, sub_value in value.items():
                    if sub_key in {"top_k", "reranker_top_k"}:
                        flat["reranker_top_k"] = sub_value
                    elif sub_key in {"reranker_name", "llm"}:
                        flat["reranker_name"] = sub_value
            else:
                flat[key] = value
        return FakeFlow(flat)


class FakeEvaluator:
    def evaluate_single(self, query, persist=False):
        params = json.loads(query["output"])
        target = TARGETS[int(query["group"])]
        score = 100.0
        for name in (
            "template_name",
            "faiss_metric",
            "rag_method",
        ):
            if params.get(name) != target[name]:
                score -= 12.0
        score -= abs(float(params["faiss_hnsw_m"]) - float(target["faiss_hnsw_m"])) / 4.0
        score -= abs(float(params["faiss_hnsw_ef_search"]) - float(target["faiss_hnsw_ef_search"])) / 24.0
        score -= abs(float(params["faiss_hnsw_ef_construction"]) - float(target["faiss_hnsw_ef_construction"])) / 30.0
        score -= abs(float(params["rag_top_k"]) - float(target["rag_top_k"])) / 3.0
        score -= abs(float(params["rag_hybrid_bm25_weight"]) - float(target["rag_hybrid_bm25_weight"])) * 20.0
        score -= abs(float(params["reranker_top_k"]) - float(target["reranker_top_k"])) / 2.0
        return {"accuracy": max(0.0, round(score, 4))}


class ClusterRoundLGBOTests(unittest.TestCase):
    def _make_config(self, working_dir: str):
        search_space = SearchSpace()
        search_space.response_synthesizer_llms = ["fixed-llm"]
        search_space.rag_retriever.query_decomposition_enabled = [False]
        search_space.rag_retriever.query_decomposition.llm_names = ["fixed-llm"]
        search_space.rag_retriever.fusion.fusion_modes = ["simple"]
        search_space.reranker.llms = ["flashrank"]
        tuner = types.SimpleNamespace(
            search_space=search_space,
            tuner_params=["template_name", "response_synthesizer_llm", "reranker", "rag_retriever"],
            optimization=types.SimpleNamespace(
                objective_1_name="accuracy",
                max_concurrent_trials=9,
            ),
            cluster_k=3,
            cluster_random_seed=7,
        )
        return types.SimpleNamespace(
            tuner=tuner,
            llms=[],
            num_trials=3,
            working_dir=working_dir,
            exp_name="cluster_round_test",
            embedding=types.SimpleNamespace(api_type="dummy"),
        )

    def _make_queries(self):
        queries = []
        for group in range(3):
            for local_idx in range(3):
                query_id = group * 3 + local_idx
                queries.append(
                    {
                        "id": query_id,
                        "group": group,
                        "question": f"group={group} query={query_id} synthetic question",
                        "answer": f"answer-{query_id}",
                    }
                )
        return queries

    def _query_embedder(self, texts):
        vectors = []
        for text in texts:
            group = int(text.split("group=")[1].split()[0])
            base = [0.0, 0.0, 0.0]
            base[group] = 1.0
            vectors.append(base)
        return vectors

    def test_query_clustering_is_stable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            runner = ClusterRoundLGBORunner(
                config=config,
                builder=FakeBuilder(),
                evaluator=FakeEvaluator(),
                query_embedder=self._query_embedder,
                sampler=LGBOSampler(config=config),
            )
            queries = self._make_queries()
            first = [(item.query_id, item.cluster_id) for item in runner._cluster_queries(queries)]
            second = [(item.query_id, item.cluster_id) for item in runner._cluster_queries(queries)]
        self.assertEqual(first, second)

    def test_round_synchronous_cluster_runner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            sampler = LGBOSampler(config=config)
            fake_llm = FakeLLM()
            sampler.llm = fake_llm
            runner = ClusterRoundLGBORunner(
                config=config,
                builder=FakeBuilder(),
                evaluator=FakeEvaluator(),
                query_embedder=self._query_embedder,
                sampler=sampler,
            )
            summary = runner.run(self._make_queries())

        self.assertEqual(summary["num_queries"], 9)
        self.assertEqual(summary["num_rounds"], 3)
        self.assertEqual(len(set(summary["query_to_cluster"].values())), 3)
        self.assertEqual(len(summary["per_round_average_acc"]), 3)
        self.assertGreaterEqual(summary["per_round_average_acc"][-1], summary["per_round_average_acc"][0])
        self.assertGreater(summary["average_wall_time_per_query_s"], 0.0)

        coverage = summary["parameter_coverage"]
        self.assertEqual(coverage["parameter_count"], 9)
        self.assertEqual(coverage["exercised_count"], 9)
        self.assertTrue(all(item["exercised"] for item in coverage["per_parameter"].values()))

        by_round = {}
        for record in summary["trial_records"]:
            by_round.setdefault(record["round_id"], []).append(record)
        self.assertEqual({round_id: len(records) for round_id, records in by_round.items()}, {1: 9, 2: 9, 3: 9})

        for round_id in (2, 3):
            prompts = [item for item in fake_llm.prompt_records if item["round_id"] == round_id]
            self.assertEqual(len(prompts), 9)
            self.assertTrue(all("round_id=1" in item["prompt"] for item in prompts))
            self.assertTrue(all(f"round_id={round_id}" not in item["prompt"] for item in prompts))

        round2_records = by_round[2]
        for cluster_id in {record["cluster_id"] for record in round2_records}:
            cluster_records = [record for record in round2_records if record["cluster_id"] == cluster_id]
            self.assertEqual(len(cluster_records), 3)
            unique_points = {
                (
                    record["params"]["rag_top_k"],
                    record["params"]["reranker_top_k"],
                    record["params"]["rag_hybrid_bm25_weight"],
                )
                for record in cluster_records
            }
            self.assertGreater(len(unique_points), 1)

        round2_prompts = [item for item in fake_llm.prompt_records if item["round_id"] == 2]
        self.assertTrue(all(item["shared_region_lines"] for item in round2_prompts))


if __name__ == "__main__":
    unittest.main()
