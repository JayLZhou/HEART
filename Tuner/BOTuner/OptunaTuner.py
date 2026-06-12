from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
import optuna
import typing as T
import hashlib
import json
import traceback
import copy
from datetime import datetime, timezone
from optuna.exceptions import DuplicatedStudyError
from optuna.study import Study
from optuna.trial import TrialState
from Option.Config2 import Config
from Common.Logger import logger
from Tuner.BOTuner.HierarchicalTPE import HierarchicalTPESampler
from Tuner.BOTuner.BasicBOTuner import BasicBOTuner
from Tuner.BOTuner.LLMBO import LLMBOSampler
from Tuner.BOTuner.LGBO import LGBOSampler
from Tuner.BOTuner.LLMTPESampler import LLMTPESampler
import importlib
from Pipeline.FlowBuild import FlowBuilder
from Utils.Evaluation import Evaluator
from Storage.NameSpace import Workspace, Namespace
from Storage.OptunaStorage import OptunaStorage


def wrap_params(params: dict):
    # If already structured (SearchSpace.sample format), return as-is
    if isinstance(params.get("rag_retriever"), dict) and "method" in params.get("rag_retriever", {}):
        return params
    out = {}
    rag = {}
    reranker = {}

    for key, value in params.items():
        if key.startswith("rag_"):
            rest = key[len("rag_"):]    # 去掉 rag_ 前缀
            rag[rest] = value
        elif key.startswith("reranker_"):
            reranker[key] = value
        else:
            out[key] = value

    out["rag_retriever"] = rag
    out["reranker"] = reranker
    return out     


class OptunaTuner(BasicBOTuner):
    def __init__(self, config: Config, builder: FlowBuilder, evaluator: Evaluator, query: dict):
        self.config = config
        self._align_llm_choices_to_config()
        self.builder = builder
        self.evaluator = evaluator
        self.workspace = Workspace(self.config.working_dir, self.config.exp_name)
        self.namespace = Namespace(self.workspace)
        print("Namespace: ", self.namespace)
        self.storage = OptunaStorage(self.namespace)
        # self._tuner = self._create_tuner()
        
        self._tuner = self._create_tuner(query)

    def _align_llm_choices_to_config(self) -> None:
        available_llm_names = [llm.model for llm in self.config.llms]
        if not available_llm_names:
            return

        def keep_available(names):
            filtered = [name for name in names if name in available_llm_names]
            return filtered or available_llm_names.copy()

        self.config.tuner.search_space.response_synthesizer_llms = keep_available(
            self.config.tuner.search_space.response_synthesizer_llms
        )
        self.config.tuner.search_space.rag_retriever.query_decomposition.llm_names = keep_available(
            self.config.tuner.search_space.rag_retriever.query_decomposition.llm_names
        )
        self.config.retriever.query_decomposition.llm_names = keep_available(
            self.config.retriever.query_decomposition.llm_names
        )
        self.config.query.subquestion_engine_llms = keep_available(
            self.config.query.subquestion_engine_llms
        )
        self.config.query.subquestion_response_synthesizer_llms = keep_available(
            self.config.query.subquestion_response_synthesizer_llms
        )

    def get_sampler(self) -> optuna.samplers.BaseSampler:
        if self.config.tuner.optimization.sampler == "tpe":
            return optuna.samplers.TPESampler(
                n_startup_trials=self.config.tuner.optimization.num_random_trials,
                constant_liar=True,
                multivariate=True,
            )
        elif self.config.tuner.optimization.sampler == "hierarchical":
            return HierarchicalTPESampler(
                constant_liar=True,
                n_startup_trials=self.config.optimization.num_random_trials,
            )
        elif self.config.tuner.optimization.sampler == "llmbo":
            return LLMBOSampler(
                config=self.config,
            )
        elif self.config.tuner.optimization.sampler == "lgbo":
            return LGBOSampler(
                config=self.config,
            )
        elif self.config.tuner.optimization.sampler == "gpbo":
            return LGBOSampler(
                config=self.config,
                use_llm_guidance=False,
            )
        elif self.config.tuner.optimization.sampler == "llm_tpe":
            return LLMTPESampler(
                config=self.config,
                n_startup=self.config.tuner.optimization.num_random_trials or 10,
            )
        elif self.config.tuner.optimization.sampler == "llambo":
            from Tuner.BOTuner.LLAMBOSampler import LLAMBOSampler
            return LLAMBOSampler(
                config=self.config,
                n_startup=self.config.tuner.optimization.num_random_trials or 1,
                k_candidates=int(getattr(self.config.tuner.optimization, "llambo_k_candidates", 5)),
            )
        else:
            raise ValueError("Invalid sampler")

    def _share_history_across_queries(self) -> bool:
        return self.config.tuner.optimization.sampler in {"lgbo", "gpbo", "llm_tpe", "llambo"}

    def _study_name_for_query(self, query: dict) -> str:
        sampler = self.config.tuner.optimization.sampler
        if sampler == "lgbo":
            return f"{self.config.exp_name}__lgbo_shared_history"
        elif sampler == "gpbo":
            return f"{self.config.exp_name}__gpbo_shared_history"
        elif sampler == "llm_tpe":
            return f"{self.config.exp_name}__llm_tpe_shared_history"
        return str(query["id"])

    def _create_tuner(self, query: dict) -> Study:
        """Get a study instance for optuna"""
        study_name = self._study_name_for_query(query)
        storage = self.storage.get_storage()
        sampler = self.get_sampler()
        print(query, study_name, storage)
        if self._share_history_across_queries():
            try:
                study = optuna.load_study(
                    study_name=study_name,
                    storage=storage,
                    sampler=sampler,
                )
            except KeyError:
                study = optuna.create_study(
                    study_name=study_name,
                    directions=["maximize"],
                    sampler=sampler,
                    storage=storage,
                )
        else:
            try:
                optuna.delete_study(
                    study_name=study_name,
                    storage=storage,
                )
            except KeyError:
                pass
            study = optuna.create_study(
                study_name=study_name,
                directions=["maximize"],
                sampler=sampler,
                storage=storage,
            )
        study.set_user_attr("query", query)
        if self._share_history_across_queries():
            study.set_user_attr("history_scope", "cross_query_shared")

        # self.save_config(study, self.study_config)
        return study




    def save_config(self, study: Study, config: Config):
        """Save study config to database"""
        attrs = config.model_dump(mode="json")
        logger.info("Saving study config of %s to the database", study.study_name)
        for attr, value in attrs.items():
            study.set_user_attr(attr, value)





    def __call__(self, query, cluster_context: dict | None = None):
        return self._run_single_trial(query=query, cluster_context=cluster_context)

    def run_cluster_trial(
        self,
        *,
        cluster_queries: list[dict],
        cluster_context: dict | None = None,
    ) -> dict:
        """Sample one config and evaluate it on multiple queries from the same cluster.

        This implements the budget-aware document semantics:
        one trial config x_k(t), evaluated on N_k query samples, then aggregate.
        """
        if not cluster_queries:
            raise ValueError("cluster_queries must not be empty")

        rep_query = copy.deepcopy(cluster_queries[0])
        trial = self._tuner.ask()
        params = self._suggest_params(trial=trial, query=rep_query, cluster_context=cluster_context)
        print(f"TRIAL: {params}")
        for k, v in params.items():
            trial.set_user_attr(f"suggested:{k}", v)
        trial.set_user_attr(
            "cluster_query_ids",
            [q.get("id") for q in cluster_queries],
        )

        metric_rows: list[dict] = []
        failed_query_ids: list[T.Any] = []
        failed_messages: list[str] = []
        try:
            flow = self.builder.build_flow(wrap_params(params))
            eval_workers = int(getattr(getattr(self.config.tuner, "optimization", None), "eval_parallel_workers", 1))

            def _eval_one_query(q):
                q_eval = copy.deepcopy(q)
                response = flow.query(q_eval["question"])
                q_eval["output"] = response
                return self.evaluator.evaluate_single(q_eval)

            if eval_workers <= 1:
                for q in cluster_queries:
                    try:
                        metric_rows.append(_eval_one_query(q))
                    except Exception as ex:
                        failed_query_ids.append(q.get("id"))
                        failed_messages.append(f"{q.get('id')}: {ex.__class__.__name__}: {ex}")
                        logger.warning(
                            "Cluster query eval failed for query_id=%s in trial=%s: %s",
                            q.get("id"), getattr(trial, "number", "unknown"), ex,
                        )
            else:
                import concurrent.futures as _cf
                # per-query timeout: LLM_API_TIMEOUT(30s) * 3 retries + backoff = ~150s max
                _per_query_timeout = 150
                _total_timeout = min(3600, max(300, _per_query_timeout * len(cluster_queries) // max(eval_workers, 1)))
                # Use shutdown(wait=False) so stuck threads don't block after timeout
                _pool = ThreadPoolExecutor(max_workers=eval_workers)
                try:
                    future_to_q = {_pool.submit(_eval_one_query, q): q for q in cluster_queries}
                    done, not_done = _cf.wait(future_to_q.keys(), timeout=_total_timeout)
                    for fut in not_done:
                        q = future_to_q[fut]
                        failed_query_ids.append(q.get("id"))
                        failed_messages.append(f"{q.get('id')}: TimeoutError: query eval exceeded total round timeout")
                        logger.warning(
                            "Cluster query eval total-timeout for query_id=%s in trial=%s",
                            q.get("id"), getattr(trial, "number", "unknown"),
                        )
                    for fut in done:
                        q = future_to_q[fut]
                        try:
                            metric_rows.append(fut.result())
                        except Exception as ex:
                            failed_query_ids.append(q.get("id"))
                            failed_messages.append(f"{q.get('id')}: {ex.__class__.__name__}: {ex}")
                            logger.warning(
                                "Cluster query eval failed for query_id=%s in trial=%s: %s",
                                q.get("id"), getattr(trial, "number", "unknown"), ex,
                            )
                finally:
                    _pool.shutdown(wait=False)  # abandon stuck threads, don't block
        except Exception as ex:
            logger.exception("Cluster objective had an unhandled exception: %s", ex)
            aggregated = {
                "failed": True,
                "exception_message": str(ex),
                "exception_stacktrace": traceback.format_exc(),
                "exception_class": ex.__class__.__name__,
            }
            aggregated["cluster_eval_size"] = len(cluster_queries)
            aggregated["cluster_success_count"] = 0
            aggregated["cluster_failure_count"] = len(cluster_queries)
            if failed_query_ids:
                trial.set_user_attr("cluster_failed_query_ids", failed_query_ids)
            if failed_messages:
                trial.set_user_attr("cluster_failed_messages", failed_messages)
            self._set_trial(
                trial=trial,
                metrics=aggregated,
                flow_json=json.dumps(params),
                query=rep_query,
            )
            self._tuner.tell(trial, state=TrialState.FAIL)
            return aggregated

        if metric_rows:
            aggregated = self._aggregate_metrics(metric_rows)
        else:
            aggregated = {
                "failed": True,
                "exception_message": "No successful query evaluations in cluster batch.",
                "exception_class": "NoSuccessfulEval",
            }
        aggregated["cluster_eval_size"] = len(cluster_queries)
        aggregated["cluster_success_count"] = len(metric_rows)
        aggregated["cluster_failure_count"] = len(cluster_queries) - len(metric_rows)
        if failed_query_ids:
            trial.set_user_attr("cluster_failed_query_ids", failed_query_ids)
        if failed_messages:
            trial.set_user_attr("cluster_failed_messages", failed_messages)
        self._set_trial(
            trial=trial,
            metrics=aggregated,
            flow_json=json.dumps(params),
            query=rep_query,
        )

        if not metric_rows:
            self._tuner.tell(trial, state=TrialState.FAIL)
            return aggregated

        objective_name = self.config.tuner.optimization.objective_1_name
        obj = float(aggregated.get(objective_name, 0.0))
        self._tuner.tell(trial, [obj])
        return aggregated

    def _run_single_trial(self, *, query: dict, cluster_context: dict | None = None) -> dict:
        trial = self._tuner.ask()
        params = self._suggest_params(trial=trial, query=query, cluster_context=cluster_context)
        print(f"TRIAL: {params}")

        for k, v in params.items():
            trial.set_user_attr(f"suggested:{k}", v)

        try:   
            # import pdb
            # pdb.set_trace()
            flow = self.builder.build_flow(wrap_params(params))
            response = flow.query(query["question"])
            query["output"] = response
            metrics = self.evaluator.evaluate_single(query)

        except Exception as ex:
            logger.exception("Objective had an unhandled exception: %s", ex)
            metrics = {
                "failed": True,
                "exception_message": str(ex),
                "exception_stacktrace": traceback.format_exc(),
                "exception_class": ex.__class__.__name__,
            }
        finally:
            self._set_trial(
                trial=trial,
                metrics=metrics,
                flow_json=json.dumps(params),
                query=query
            )
        if metrics.get("failed"):
            self._tuner.tell(trial, state=TrialState.FAIL)
            return metrics
        self._tuner.tell(trial, [metrics[self.config.tuner.optimization.objective_1_name]])
        return metrics

    def _suggest_params(self, *, trial, query: dict, cluster_context: dict | None = None) -> dict:
        params = trial.params
        if self.config.tuner.optimization.sampler in {"llmbo", "lgbo", "gpbo", "llm_tpe", "llambo"}:
            sampler = self.get_sampler()
            search_space = sampler.infer_relative_search_space(study=None, trial=None)
            study_name = self._study_name_for_query(query)
            print(query, study_name, self.storage.get_storage())
            study = optuna.load_study(
                study_name=study_name,
                storage=self.storage.get_storage(),
            )
            study.set_user_attr("query", query)
            if cluster_context:
                study.set_user_attr("lgbo_budget_context", cluster_context)
            params = sampler.sample_relative(study, trial, search_space)
        else:
            search_space = self.config.tuner.search_space
            params = search_space.sample(trial, self.config.tuner.tuner_params)
        return params

    def _aggregate_metrics(self, metric_rows: list[dict]) -> dict:
        agg: dict = {}
        keys = set()
        for row in metric_rows:
            keys.update(row.keys())
        for k in keys:
            vals = [row[k] for row in metric_rows if k in row]
            if not vals:
                continue
            if all(isinstance(v, (int, float, bool)) for v in vals):
                agg[k] = float(sum(float(v) for v in vals) / len(vals))
            else:
                agg[k] = vals[-1]
        return agg

    def get_study(self) -> Study:
        return self._tuner

    def completed_trials(self):
        return [t for t in self._tuner.get_trials(deepcopy=False) if getattr(getattr(t, "state", None), "name", str(getattr(t, "state", None))) in {"COMPLETE", "complete"}]


    def _set_trial(self, trial: optuna.trial.FrozenTrial | optuna.trial.Trial, metrics: T.Dict[str, float] | None = None, flow_json: str | None = None, query: dict | None = None):
        if metrics:
            for metric_name, score in metrics.items():
                if isinstance(score, bool):
                    stored_score = score
                elif isinstance(score, (int, float)):
                    stored_score = score * 0.01
                else:
                    stored_score = score
                trial.set_user_attr("metric_" + metric_name, stored_score)
        if flow_json:
                trial.set_user_attr("flow", flow_json)
        if query:
                trial.set_user_attr("query", query)
       
    def _trial_exists(self,
    study_name: str,
    params: T.Dict[str, T.Any],
    storage: str) -> bool:
        storage = storage or self.config.database.get_optuna_storage()
        logger.debug("Loading '%s' from storage: %s", study_name, storage)
        study = optuna.load_study(study_name=study_name, storage=storage)
        for trial in study.get_trials():
            if params == trial.params:
                return True
        return False

 
