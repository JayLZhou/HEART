import optuna
import typing as T
import json
import traceback
from datetime import datetime, timezone
from Option.Config2 import Config
from Common.Logger import logger
from Tuner.BOTuner.HierarchicalTPE import HierarchicalTPESampler
from Tuner.BOTuner.BasicBOTuner import BasicBOTuner
from Pipeline.FlowBuild import FlowBuilder
from Utils.Evaluation import Evaluator


     


class OptunaTuner(BasicBOTuner):
    def __init__(self, config: Config, builder: FlowBuilder, evaluator: Evaluator):
        self.config = config
        self.builder = builder
        self.evaluator = evaluator
        self._tuner = self._create_tuner()

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
        else:
            raise ValueError("Invalid sampler")


       


    def _create_tuner(self) -> optuna.Study:
        """Get a study instance for optuna"""
        study_name = self.config.tuner.name

        
        if self.config.tuner.reuse_study:
            logger.info(
                "Reusing study '%s' or creating new one", study_name
            )
            if self.config.tuner.recreate_study:
                self.recreate_with_completed_trials(self.config, storage)
     

        sampler = self.get_sampler()
        study = optuna.create_study(
            study_name=study_name,
            directions=["maximize"],
            sampler=sampler,
        )
        # self.save_config(study, self.study_config)
        return study




    def save_config(self, study: optuna.Study, config: Config):
        """Save study config to database"""
        attrs = config.model_dump(mode="json")
        logger.info("Saving study config of %s to the database", study.study_name)
        for attr, value in attrs.items():
            study.set_user_attr(attr, value)


    



    def __call__(self, query):
        trial = self._tuner.ask()
        search_space = self.config.tuner.search_space

        params = search_space.sample(trial, self.config.tuner.tuner_params)
        try:   
            flow = self.builder.build_flow(params)
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
            )
        self._tuner.tell(trial, [metrics[self.config.tuner.optimization.objective_1_name]])
        import pdb
        pdb.set_trace()
        return metrics


    def _set_trial(self, trial: optuna.trial.FrozenTrial | optuna.trial.Trial, metrics: T.Dict[str, float] | None = None, flow_json: str | None = None):
        if metrics:
            for metric_name, score in metrics.items():
                trial.set_user_attr("metric_" + metric_name, score)   
        if flow_json:
                trial.set_user_attr("flow", flow_json)
       
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

 


