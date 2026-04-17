from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence

import numpy as np
import optuna

from Common.Logger import logger
from Option.Config2 import Config
from Storage.NameSpace import Namespace, Workspace
from Storage.OptunaStorage import OptunaStorage
from Tuner.BOTuner.LGBO import LGBOSampler
from Tuner.BOTuner.Cluster.cluster_state import LGBOClusterStateStore
from Tuner.BOTuner.lgbo_components.monitoring import ParameterCoverageTracker

if TYPE_CHECKING:
    from Pipeline.FlowBuild import FlowBuilder
    from Utils.Evaluation import Evaluator


def wrap_params(params: dict):
    out = {}
    rag = {}
    reranker = {}

    for key, value in params.items():
        if key.startswith("rag_"):
            rag[key[len("rag_"):]] = value
        elif key.startswith("reranker_"):
            reranker[key] = value
        else:
            out[key] = value

    out["rag_retriever"] = rag
    out["reranker"] = reranker
    return out


@dataclass
class QueryClusterAssignment:
    query_id: int
    cluster_id: int


class ClusterRoundLGBORunner:
    def __init__(
        self,
        *,
        config: Config,
        builder: "FlowBuilder",
        evaluator: "Evaluator",
        query_embedder: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
        sampler: LGBOSampler | None = None,
        history_max_items: int | None = None,
    ):
        self.config = config
        self.builder = builder
        self.evaluator = evaluator
        self.history_max_items = None if history_max_items is None else max(1, int(history_max_items))
        self.workspace = Workspace(self.config.working_dir, self.config.exp_name)
        self.namespace = Namespace(self.workspace)
        self.storage = OptunaStorage(self.namespace)
        self.sampler = sampler or LGBOSampler(config=self.config)
        self.query_embedder = query_embedder
        self.cluster_state = LGBOClusterStateStore(
            Path(self.workspace.make_for("cluster_round_state").get_save_path()) / "cluster_state.json"
        )
        self.parameter_names = self._active_parameter_names()
        self.progress_log_path = Path(self.workspace.make_for("cluster_round_reports").get_save_path()) / "cluster_round_progress.log"

    def run(self, queries: Sequence[dict]) -> Dict[str, Any]:
        if not queries:
            return {}

        run_started = time.perf_counter()
        queries = [dict(query) for query in queries]
        assignments = self._cluster_queries(queries)
        query_to_cluster = {item.query_id: item.cluster_id for item in assignments}
        study = self._create_or_load_study()
        coverage = ParameterCoverageTracker(self.parameter_names)
        search_space = self.sampler.infer_relative_search_space(study=None, trial=None)
        per_round_average_acc: list[float] = []
        per_query_wall_times: dict[str, float] = {str(query["id"]): 0.0 for query in queries}
        per_query_best_acc: dict[str, float] = {str(query["id"]): float("-inf") for query in queries}
        round_summaries: list[dict[str, Any]] = []
        trial_records: list[dict[str, Any]] = []
        max_workers = min(len(queries), max(1, self.config.tuner.optimization.max_concurrent_trials))
        total_trials = len(queries) * self.config.num_trials
        completed_trials = 0
        self.progress_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_log_path.write_text("", encoding="utf-8")
        self._append_progress(
            f"RUN_START queries={len(queries)} rounds={self.config.num_trials} total_trials={total_trials} clusters={self.config.tuner.cluster_k}"
        )

        for round_id in range(1, self.config.num_trials + 1):
            logger.info("Cluster-round LGBO round %s/%s", round_id, self.config.num_trials)
            round_started = time.perf_counter()
            self._append_progress(
                f"ROUND_START round={round_id}/{self.config.num_trials} completed_trials={completed_trials}/{total_trials}"
            )
            history_snapshot = self.sampler.history.completed_trials(study)
            trials = [study.ask() for _ in queries]
            trial_contexts = []
            for query, trial in zip(queries, trials):
                cluster_id = query_to_cluster[int(query["id"])]
                shared_region = self.cluster_state.get(cluster_id).shared_plan
                trial.set_user_attr("query", query)
                trial.set_user_attr("query_id", int(query["id"]))
                trial.set_user_attr("cluster_id", cluster_id)
                trial.set_user_attr("round_id", round_id)
                trial_contexts.append(
                    {
                        "trial": trial,
                        "query": query,
                        "cluster_id": cluster_id,
                        "round_id": round_id,
                        "shared_region": shared_region,
                        "completed_trials": history_snapshot,
                        "history_max_items": self.history_max_items,
                    }
                )

            params_batch = self.sampler.sample_relative_batch(
                study=study,
                trial_contexts=trial_contexts,
                search_space=search_space,
            )
            for trial, params in zip(trials, params_batch):
                for key, value in params.items():
                    trial.set_user_attr(f"suggested:{key}", value)

            round_results = self._execute_round(
                queries=queries,
                trials=trials,
                params_batch=params_batch,
                max_workers=max_workers,
                round_id=round_id,
                total_trials=total_trials,
                completed_trials=completed_trials,
                run_started=run_started,
            )

            round_accs = []
            for item in round_results:
                trial = item["trial"]
                metrics = item["metrics"]
                query = item["query"]
                query_id = str(query["id"])
                acc = float(metrics.get(self.config.tuner.optimization.objective_1_name, 0.0))
                round_accs.append(acc)
                per_query_wall_times[query_id] += float(item["elapsed_s"])
                per_query_best_acc[query_id] = max(per_query_best_acc[query_id], acc)
                self._set_trial_metadata(trial=trial, metrics=metrics, params=item["params"], query=query)
                study.tell(trial, [acc])
                coverage.observe_trial(trial)
                trial_records.append(
                    {
                        "trial_number": int(getattr(trial, "number", -1)),
                        "round_id": int((getattr(trial, "user_attrs", {}) or {}).get("round_id", round_id)),
                        "cluster_id": int((getattr(trial, "user_attrs", {}) or {}).get("cluster_id")),
                        "query_id": int(query["id"]),
                        "acc": acc,
                        "params": item["params"],
                        "shared_plan_mode": ((getattr(trial, "user_attrs", {}) or {}).get("lgbo_plan") or {}).get("mode"),
                    }
                )
                completed_trials += 1

            self._update_cluster_state(round_results=round_results, round_id=round_id)
            avg_acc = float(sum(round_accs) / len(round_accs)) if round_accs else 0.0
            per_round_average_acc.append(avg_acc)
            round_summaries.append(
                {
                    "round_id": round_id,
                    "avg_acc": avg_acc,
                    "query_ids": [int(item["query"]["id"]) for item in round_results],
                    "trial_numbers": [int(getattr(item["trial"], "number", -1)) for item in round_results],
                }
            )
            self._append_progress(
                f"ROUND_DONE round={round_id}/{self.config.num_trials} avg_acc={avg_acc:.2f} round_elapsed_s={time.perf_counter() - round_started:.2f} completed_trials={completed_trials}/{total_trials}"
            )

        summary = {
            "num_queries": len(queries),
            "num_rounds": self.config.num_trials,
            "cluster_k": self.config.tuner.cluster_k,
            "query_to_cluster": {str(item.query_id): item.cluster_id for item in assignments},
            "per_round_average_acc": per_round_average_acc,
            "per_query_best_acc": per_query_best_acc,
            "per_query_wall_time_s": per_query_wall_times,
            "average_wall_time_per_query_s": float(sum(per_query_wall_times.values()) / len(per_query_wall_times)),
            "parameter_coverage": coverage.summary(),
            "round_summaries": round_summaries,
            "trial_records": trial_records,
        }
        self._write_summary(summary)
        self._append_progress(
            f"RUN_DONE total_elapsed_s={time.perf_counter() - run_started:.2f} average_wall_time_per_query_s={summary['average_wall_time_per_query_s']:.2f}"
        )
        return summary

    def _create_or_load_study(self):
        storage = self.storage.get_storage()
        study_name = f"{self.config.exp_name}__cluster_round_lgbo"
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except KeyError:
            pass
        return optuna.create_study(
            study_name=study_name,
            directions=["maximize"],
            sampler=self.sampler,
            storage=storage,
        )

    def _cluster_queries(self, queries: Sequence[dict]) -> list[QueryClusterAssignment]:
        query_texts = [str(query["question"]) for query in queries]
        embeddings = np.asarray(self._embed_queries(query_texts), dtype=float)
        cluster_k = min(max(1, int(self.config.tuner.cluster_k)), len(queries))
        labels = self._kmeans(embeddings, k=cluster_k, seed=int(self.config.tuner.cluster_random_seed))
        return [
            QueryClusterAssignment(query_id=int(query["id"]), cluster_id=int(label))
            for query, label in zip(queries, labels)
        ]

    def _embed_queries(self, query_texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self.query_embedder is not None:
            return self.query_embedder(query_texts)
        from Index import get_rag_embedding

        embedder = get_rag_embedding(self.config.embedding.api_type, self.config)
        return embedder._get_text_embeddings(list(query_texts))

    def _kmeans(self, X: np.ndarray, *, k: int, seed: int, max_iters: int = 25) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if len(X) < k:
            k = len(X)
        centroids = X[rng.choice(len(X), size=k, replace=False)].copy()
        labels = np.zeros(len(X), dtype=int)
        for _ in range(max_iters):
            distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for idx in range(k):
                mask = labels == idx
                if np.any(mask):
                    centroids[idx] = X[mask].mean(axis=0)
                else:
                    centroids[idx] = X[rng.integers(0, len(X))]
        return labels

    def _execute_round(
        self,
        *,
        queries: Sequence[dict],
        trials: Sequence[Any],
        params_batch: Sequence[Dict[str, Any]],
        max_workers: int,
        round_id: int,
        total_trials: int,
        completed_trials: int,
        run_started: float,
    ) -> list[dict]:
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_context: dict[Any, tuple[dict, Any, Dict[str, Any]]] = {}
            for query, trial, params in zip(queries, trials, params_batch):
                future = executor.submit(
                    self._run_single_query_trial,
                    dict(query),
                    trial,
                    params,
                )
                future_context[future] = (dict(query), trial, params)
            for future in as_completed(future_context):
                query, trial, params = future_context[future]
                try:
                    item = future.result()
                except Exception as exc:
                    # Keep the round progressing even if one query trial fails.
                    logger.exception(
                        "Cluster-round trial failed; continuing with fallback metrics. "
                        "round=%s query_id=%s trial_number=%s error=%s",
                        round_id,
                        query.get("id"),
                        getattr(trial, "number", -1),
                        exc,
                    )
                    item = {
                        "trial": trial,
                        "query": query,
                        "params": params,
                        "metrics": {
                            self.config.tuner.optimization.objective_1_name: 0.0,
                            "error": str(exc),
                        },
                        "elapsed_s": 0.0,
                    }
                results.append(item)
                finished = completed_trials + len(results)
                progress = finished / total_trials if total_trials else 0.0
                elapsed = time.perf_counter() - run_started
                eta = (elapsed / finished) * (total_trials - finished) if finished else 0.0
                acc = float(item["metrics"].get(self.config.tuner.optimization.objective_1_name, 0.0))
                self._append_progress(
                    f"TRIAL_DONE round={round_id}/{self.config.num_trials} query_id={item['query']['id']} trial_number={getattr(item['trial'], 'number', -1)} acc={acc:.2f} trial_elapsed_s={item['elapsed_s']:.2f} completed_trials={finished}/{total_trials} progress={progress:.2%} eta_s={eta:.2f}"
                )
        results.sort(key=lambda item: int(item["query"]["id"]))
        return results

    def _run_single_query_trial(self, query: dict, trial: Any, params: Dict[str, Any]) -> dict:
        started = time.perf_counter()
        flow = self.builder.build_flow(wrap_params(params))
        response = flow.query(query["question"])
        query["output"] = response
        metrics = self.evaluator.evaluate_single(query, persist=False)
        elapsed = time.perf_counter() - started
        return {
            "trial": trial,
            "query": query,
            "params": params,
            "metrics": metrics,
            "elapsed_s": elapsed,
        }

    def _set_trial_metadata(self, *, trial: Any, metrics: Dict[str, Any], params: Dict[str, Any], query: dict) -> None:
        for metric_name, score in metrics.items():
            if isinstance(score, bool):
                stored_score = score
            elif isinstance(score, (int, float)):
                stored_score = score * 0.01
            else:
                stored_score = score
            trial.set_user_attr("metric_" + metric_name, stored_score)
        trial.set_user_attr("flow", json.dumps(params))
        trial.set_user_attr("query", query)

    def _update_cluster_state(self, *, round_results: Sequence[dict], round_id: int) -> None:
        grouped: Dict[int, list[dict]] = {}
        for item in round_results:
            cluster_id = int(item["trial"].user_attrs["cluster_id"])
            grouped.setdefault(cluster_id, []).append(item)

        for cluster_id, items in grouped.items():
            best_item = max(
                items,
                key=lambda item: float(item["metrics"].get(self.config.tuner.optimization.objective_1_name, 0.0)),
            )
            trial = best_item["trial"]
            plan = (getattr(trial, "user_attrs", {}) or {}).get("lgbo_plan")
            reasoning = (getattr(trial, "user_attrs", {}) or {}).get("lgbo_reasoning")
            self.cluster_state.update(
                cluster_id,
                shared_plan=plan,
                shared_reasoning=reasoning,
                round_id=round_id,
                trial_numbers=[int(getattr(item["trial"], "number", -1)) for item in items],
            )

    def _active_parameter_names(self) -> list[str]:
        dists = self.sampler.infer_relative_search_space(study=None, trial=None)
        names = []
        for name, dist in dists.items():
            choices = getattr(dist, "choices", None)
            if choices is not None:
                if len(choices) > 1:
                    names.append(name)
                continue
            low = getattr(dist, "low", None)
            high = getattr(dist, "high", None)
            if low is None or high is None or low != high:
                names.append(name)
        return names

    def _write_summary(self, summary: Dict[str, Any]) -> None:
        out = Path(self.workspace.make_for("cluster_round_reports").get_save_path()) / "cluster_round_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    def _append_progress(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.progress_log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
