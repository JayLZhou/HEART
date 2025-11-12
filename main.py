
from Option.Config2 import Config
import argparse
import os
import random
import numpy as np
import torch
from pathlib import Path
from shutil import copyfile
from Data.DataLoader import RAGDataset
import pandas as pd
# from Utils.Evaluation import Evaluator
from Common.Utils import welcome_message
from tqdm import tqdm
# from Tuner.OptunaTuner import get_study, objective
from Common.Logger import logger
import optuna
from Pipeline.FlowBuild import FlowBuilder

parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
parser.add_argument("-dataset_name", type=str, help="Name of the dataset.")
args = parser.parse_args()

opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name)
builder = FlowBuilder(config=opt)
    
dataset = RAGDataset(
        data_dir=os.path.join(opt.data_root, opt.dataset_name)
)




def check_dirs(opt):
    # For each query, save the results in a separate directory
    result_dir = os.path.join(opt.working_dir, opt.exp_name, "Results")
    # Save the current used config in a separate directory
    config_dir = os.path.join(opt.working_dir, opt.exp_name, "Configs")
    # Save the metrics of entire experiment in a separate directory
    metric_dir = os.path.join(opt.working_dir, opt.exp_name, "Metrics")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    opt_name = args.opt[args.opt.rindex("/") + 1 :]
    basic_name = os.path.join(args.opt.split("/")[0], "Config2.yaml")
    copyfile(args.opt, os.path.join(config_dir, opt_name))
    copyfile(basic_name, os.path.join(config_dir, "Config2.yaml"))
    return result_dir


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wrapper_tuning(opt, study_config, components, num_trials):
 
    logger.info("Starting sequential optimization")

    results = []
    
    for i in tqdm(range(num_trials), desc="Running trials"):
        logger.info("Running trial %d/%d", i+1, num_trials)
        try:
            trial = tuner.start()
            result = tuner(study_config, components)
            tuner.backward(trial)

            results.append({
                study_config.optimization.objective_1_name: obj_1,
            })
        except Exception as e:
            logger.error(f"Trial %d failed with error: {str(e)}", i+1)
            continue
    


if __name__ == "__main__":
    welcome_message()
    seed_everything(42)
    result_dir = check_dirs(opt)

    # Offline indexing
    corpus = dataset.get_corpus()
    builder.build_indexing(corpus)

    # Online RAG tuning
    wrapper_tuning(opt, builder)


