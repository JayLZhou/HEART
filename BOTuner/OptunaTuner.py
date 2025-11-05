import optuna
import typing as T
import json
import traceback
import numpy as np
from datetime import datetime, timezone
from Config import StudyConfig
from Common.Logger import logger
from Common.Utils import user_confirm_delete, recreate_with_completed_trials
from BOTuner.HierarchicalTPE import HierarchicalTPESampler
from BOTuner.BasicTuner import BasicTuner

class OptunaTuner(BasicTuner):
    def __init__(self, study_config: StudyConfig):
        self.study_config = study_config

    def get_sampler(self) -> optuna.samplers.BaseSampler:
        if self.study_config.optimization.sampler == "tpe":
            return optuna.samplers.TPESampler(
                n_startup_trials=self.study_config.optimization.num_random_trials,
                constant_liar=True,
                multivariate=True,
            )
        elif self.study_config.optimization.sampler == "hierarchical":
            return HierarchicalTPESampler(
                constant_liar=True,
                n_startup_trials=self.study_config.optimization.num_random_trials,
            )
        else:
            raise ValueError("Invalid sampler")

    def create_instance(self) -> optuna.Study:
        """Get a study instance for optuna"""
        study_name = self.study_config.name
        storage = cfg.database.get_optuna_storage()
        
        if self.study_config.reuse_study:
            logger.info(
                "Reusing study '%s' or creating new one", study_name
            )
            if self.study_config.recreate_study:
                recreate_with_completed_trials(self.study_config, storage)
        elif user_confirm_delete(self.study_config):
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
                logger.info("Study '%s' deleted", study_name)
            except KeyError:
                logger.info(
                    "Study '%s' does not exist, creating new", study_name
                )

        sampler = self.get_sampler()
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=self.study_config.reuse_study,
            directions=["maximize"],
            sampler=sampler,
        )
        self.save_config(study, self.study_config)
        return study

    def evaluate(
        self,
        params: T.Dict,
    ) -> T.Tuple[float, float, T.Dict[str, T.Any], str]:
        flow_start = datetime.now(timezone.utc).timestamp()
        logger.info("Evaluating flow with config: %s", params)
        flow_json = json.dumps(params)
 
        obj, results = self._evaluate(params)

        results["failed"] = False
        results["flow_start"] = flow_start
        results["flow_end"] = datetime.now(timezone.utc).timestamp()
        results["flow_duration"] = float(results["flow_end"]) - float(results["flow_start"])
        logger.info("Evaluation finished. Finalizing trial report. %s", results)
        return obj, results, flow_json

    def _evaluate(
        self,
        params: T.Dict,
    ) -> T.Tuple[float, float, T.Dict[str, float | str]]:
        flow = build_flow(params, self.study_config)
        results: T.Dict[str, T.Any] = eval_dataset(
            study_config=self.study_config,
            dataset_iter=self.study_config.dataset,
            flow=flow,
            evaluation_mode=self.study_config.evaluation.mode,
        )

        obj1 = results[self.study_config.optimization.objective_1_name]
        obj2 = results[self.study_config.optimization.objective_2_name]

   
      
        return obj1, obj2, results

    def save_config(self, study: optuna.Study, study_config: StudyConfig):
        """Save study config to database"""
        attrs = study_config.model_dump(mode="json")
        logger.info("Saving study config of %s to the database", study.study_name)
        for attr, value in attrs.items():
            study.set_user_attr(attr, value)

    def objective(
        self,
        trial: optuna.Trial,
        components: T.List[str],
    ) -> T.Tuple[float, float]:  # objective function for optuna trials
        from syftr.tuner.core import set_trial

        search_space = self.study_config.search_space
        params: dict[str, str | bool | int | float]
        for i in range(self.study_config.optimization.num_retries_unique_params):
            params = search_space.sample(trial, components)
            if not self.study_config.optimization.skip_existing:
                logger.info("Using generated parameter combination without check")
                break
            if not trial_exists(self.study_config.name, params):
                logger.info(
                    "Found novel parameter combination after %i retries: %s",
                    i,
                    str(params),
                )
                break
        try:
            obj1, obj2, metrics, flow_json = self.evaluate(params)
        except Exception as ex:
            logger.exception("Objective had an unhandled exception: %s", ex)
            metrics = {
                "failed": True,
                "exception_message": str(ex),
                "exception_stacktrace": traceback.format_exc(),
                "exception_class": ex.__class__.__name__,
            }
            flow_json = json.dumps(params)
            raise ex
        finally:
            set_trial(
                trial=trial,
                study_config=self.study_config,
                params=params,
                is_seeding=False,
                metrics=metrics,
                flow_json=flow_json,
            )

        return obj1, obj2

# 兼容性函数包装器，用于向后兼容
def get_study(study_config: StudyConfig) -> optuna.Study:
    """兼容性函数包装器，用于向后兼容"""
    tuner = OptunaTuner(study_config)
    return tuner.get_study()


def objective(
    trial: optuna.Trial,
    study_config: StudyConfig,
    components: T.List[str],
) -> T.Tuple[float, float]:
    """兼容性函数包装器，用于向后兼容"""
    tuner = OptunaTuner(study_config)
    return tuner.objective(trial, components)