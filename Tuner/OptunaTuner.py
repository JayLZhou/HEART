import optuna
from Common.Config import StudyConfig
from Common.Logger import logger
from Common.Utils import user_confirm_delete, recreate_with_completed_trials
from Common.Database import save_study_config_to_db
from Tuner.HierarchicalTPE import HierarchicalTPESampler
from Common.Database import cfg

def get_sampler(study_config: StudyConfig) -> optuna.samplers.BaseSampler:
    if study_config.optimization.sampler == "tpe":
        return optuna.samplers.TPESampler(
            n_startup_trials=study_config.optimization.num_random_trials,
            constant_liar=True,
            multivariate=True,
        )
    elif study_config.optimization.sampler == "hierarchical":
        return HierarchicalTPESampler(
            constant_liar=True,
            n_startup_trials=study_config.optimization.num_random_trials,
        )
    else:
        raise ValueError("Invalid sampler")

def get_study(study_config: StudyConfig) -> optuna.Study:
    """Get a study instance for optuna"""
    study_name = study_config.name
    storage = cfg.database.get_optuna_storage()
    
    if study_config.reuse_study:
        logger.info(
            "Reusing study '%s' or creating new one", study_name
        )
        if study_config.recreate_study:
            recreate_with_completed_trials(study_config, storage)
    elif user_confirm_delete(study_config):
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            logger.info("Study '%s' deleted", study_name)
        except KeyError:
            logger.info(
                "Study '%s' does not exist, creating new", study_name
            )

    sampler = get_sampler(study_config)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=study_config.reuse_study,
        directions=["maximize"],
        sampler=sampler,
    )
    save_study_config_to_db(study, study_config)
    return study
def evaluate(
    params: T.Dict,
    study_config: StudyConfig,
) -> T.Tuple[float, float, T.Dict[str, T.Any], str]:

    flow_start = datetime.now(timezone.utc).timestamp()
    logger.info("Evaluating flow with config: %s", params)
    flow_json = json.dumps(params)
    # if study_config.evaluation.use_tracing_metrics:
    #     span_exporter = get_span_exporter()
    obj1, obj2, results = _evaluate(params, study_config)
    # if study_config.evaluation.use_tracing_metrics:
    #     set_tracing_metrics(span_exporter, results)
    results["failed"] = False
    results["flow_start"] = flow_start
    results["flow_end"] = datetime.now(timezone.utc).timestamp()
    results["flow_duration"] = float(results["flow_end"]) - float(results["flow_start"])
    logger.info("Evaluation finished. Finalizing trial report. %s", results)
    return obj1, obj2, results, flow_json

def _evaluate(
    params: T.Dict,
    study_config: StudyConfig,
) -> T.Tuple[float, float, T.Dict[str, float | str]]:

    flow = build_flow(params, study_config)
    results: T.Dict[str, T.Any] = eval_dataset(
        study_config=study_config,
        dataset_iter=study_config.dataset,
        flow=flow,
        evaluation_mode=study_config.evaluation.mode,
    )

    obj1 = results[study_config.optimization.objective_1_name]
    obj2 = results[study_config.optimization.objective_2_name]

    if np.isnan(obj2):
        logger.fatal(
            "%s value is NaN and the trial will crash.\nParams: %s",
            study_config.optimization.objective_2_name,
            json.dumps(params, indent=2),
        )
    pdb.set_trace()
    return obj1, obj2, results

def objective(
    trial: optuna.Trial,
    study_config: StudyConfig,
    components: T.List[str],
) -> T.Tuple[float, float]: # objective function for optuna trials
    from syftr.tuner.core import set_trial


    search_space = study_config.search_space
    params: dict[str, str | bool | int | float]
    for i in range(study_config.optimization.num_retries_unique_params):
        params = search_space.sample(trial, components)
        if not study_config.optimization.skip_existing:
            logger.info("Using generated parameter combination without check")
            break
        if not trial_exists(study_config.name, params):
            logger.info(
                "Found novel parameter combination after %i retries: %s",
                i,
                str(params),
            )
            break
    try:
        obj1, obj2, metrics, flow_json = evaluate(params, study_config)
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
            study_config=study_config,
            params=params,
            is_seeding=False,
            metrics=metrics,
            flow_json=flow_json,
        )

    return obj1, obj2    