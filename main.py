
from Option.Config2 import Config
import argparse
import os
import random
import numpy as np
import torch
from pathlib import Path
from shutil import copyfile
from Data.DataLoader import RAGDataset
from Common.Utils import welcome_message
from tqdm import tqdm
from Common.Logger import logger
from Pipeline.FlowBuild import FlowBuilder
from Tuner.TunerFactory import get_tuner
from Utils.Evaluation import Evaluator
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
parser.add_argument("-dataset_name", type=str, help="Name of the dataset.")
args = parser.parse_args()

opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name)
builder = FlowBuilder(config=opt)
dataset = RAGDataset(data_dir=os.path.join(opt.data_root, opt.dataset_name))
num_trials = opt.num_trials




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


def wrapper_tuning():
 
    logger.info("Starting RAG tuning: query level")
    dataset_len = len(dataset)
    record_limit = os.getenv("HEART_RECORD_LIMIT")
    if record_limit:
        dataset_len = min(dataset_len, max(0, int(record_limit)))
    for _, idx in enumerate(range(dataset_len)):
        results = []
        query = dataset[idx]
        tuner = get_tuner(config=opt, builder=builder, evaluator=evaluator, query=query)
        for i in tqdm(range(num_trials), desc="Running trials"):
            logger.info(f"Running trial {i+1}/{num_trials}")
            try:
                result = tuner(query = query)
            except Exception as e:
                logger.error(f"Trial {i+1} failed with error: {str(e)}")
                raise
                continue
        results.append(result)        


if __name__ == "__main__":
    welcome_message()
    seed_everything(42)
    result_dir = check_dirs(opt)

    # Offline indexing
    corpus = dataset.get_corpus()
    builder.build_indexing(corpus)
    evaluator = Evaluator(eval_path=os.path.join(opt.working_dir, opt.exp_name, "Results", "results.json"), dataset_name=opt.dataset_name)

    # Online RAG tuning
    wrapper_tuning()

