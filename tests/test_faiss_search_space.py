import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from Config.FaissConfig import FaissSearchSpace
from Config.SearchSpace import SearchSpace


class FakeTrial:
    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_int(self, name, low, high, step=1, log=False):
        return low

    def suggest_float(self, name, low, high, step=None, log=False):
        return low


class FaissSearchSpaceTests(unittest.TestCase):
    def test_build_distributions_exposes_faiss_params_when_requested(self):
        search_space = SearchSpace()
        dists = search_space.build_distributions(["rag_retriever", "faiss"])
        self.assertIn("faiss_hnsw_m", dists)
        self.assertIn("faiss_hnsw_ef_search", dists)
        self.assertIn("faiss_hnsw_ef_construction", dists)
        self.assertIn("faiss_metric", dists)

    def test_sample_uses_faiss_defaults_when_not_requested(self):
        search_space = SearchSpace()
        params = search_space.sample(FakeTrial(), ["rag_retriever"])
        self.assertEqual(params["faiss_hnsw_m"], 32)
        self.assertEqual(params["faiss_hnsw_ef_search"], 64)
        self.assertEqual(params["faiss_hnsw_ef_construction"], 40)
        self.assertEqual(params["faiss_metric"], "l2")

    def test_faiss_search_space_defaults_match_runtime_defaults(self):
        faiss = FaissSearchSpace()
        defaults = faiss.defaults()
        self.assertEqual(defaults["faiss_hnsw_m"], 32)
        self.assertEqual(defaults["faiss_hnsw_ef_search"], 64)
        self.assertEqual(defaults["faiss_hnsw_ef_construction"], 40)
        self.assertEqual(defaults["faiss_metric"], "l2")


if __name__ == "__main__":
    unittest.main()
