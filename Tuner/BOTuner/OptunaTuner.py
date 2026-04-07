import optuna
import typing as T
import hashlib
import json
import traceback
from datetime import datetime, timezone
from optuna.exceptions import DuplicatedStudyError
from optuna.study import Study
from Option.Config2 import Config
from Common.Logger import logger
from Tuner.BOTuner.HierarchicalTPE import HierarchicalTPESampler
from Tuner.BOTuner.BasicBOTuner import BasicBOTuner
from Tuner.BOTuner.LLMBO import LLMBOSampler
from Tuner.BOTuner.LGBO import LGBOSampler
from Pipeline.FlowBuild import FlowBuilder
from Utils.Evaluation import Evaluator
from Storage.NameSpace import Workspace, Namespace
from Storage.OptunaStorage import OptunaStorage


def wrap_params(params: dict):
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
        else:
            raise ValueError("Invalid sampler")

    def _share_history_across_queries(self) -> bool:
        return self.config.tuner.optimization.sampler == "lgbo"

    def _study_name_for_query(self, query: dict) -> str:
        if self._share_history_across_queries():
            return f"{self.config.exp_name}__lgbo_shared_history"
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





    def __call__(self, query):
        trial = self._tuner.ask()
        params = trial.params
        if self.config.tuner.optimization.sampler in {"llmbo", "lgbo"}:
            sampler = self.get_sampler()
            search_space = sampler.infer_relative_search_space(study=None, trial=None)
            study_name = self._study_name_for_query(query)
            print(query, study_name, self.storage.get_storage())
            # import pdb
            # pdb.set_trace()
            study = optuna.load_study(
                study_name=study_name,
                storage=self.storage.get_storage(),
            )
            # import pdb
            # pdb.set_trace()
            params = sampler.sample_relative(study, trial, search_space)
        else:
            search_space = self.config.tuner.search_space
            params = search_space.sample(trial, self.config.tuner.tuner_params)


        # import pdb
        # pdb.set_trace()

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
            
            raise ex
        finally:
            self._set_trial(
                trial=trial,
                metrics=metrics,
                flow_json=json.dumps(params),
                query=query
            )
        self._tuner.tell(trial, [metrics[self.config.tuner.optimization.objective_1_name]])
        # import pdb
        # pdb.set_trace()
        return metrics


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

 


