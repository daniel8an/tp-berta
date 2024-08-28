import os
import sys

sys.path.append(os.getcwd())  # to correctly import bin & lib
import json
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from lib import DataConfig, data_preproc, calculate_metrics
from lib import BIN_CHECKPOINT as CHECKPOINT_DIR


def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, choices=["xgboost", "catboost", "tabnet"]
    )
    parser.add_argument("--output", type=str, default="finetune_outputs")
    parser.add_argument(
        "--dataset", type=str, default="train_1811_Pokemon-with-stats-Generation-8"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["binclass", "regression", "multiclass"],
        required=True,
    )
    args = parser.parse_args()

    args.output = f"{args.output}/{args.task}/{args.model}-tuned/{args.dataset}"
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    # some basic model configuration
    cfg_file = f"configs/tuned/{args.task}/{args.model}/{args.dataset}/cfg.json"
    if not os.path.exists(cfg_file):
        shutil.rmtree(args.output)
        raise AssertionError(f"{args.model}-{args.dataset} tuned config missing")
    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    return args, cfg


def save_result(args, model_cfgs, best_ev, final_test, suffix):
    saved_results = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "configs": model_cfgs,
        "best_eval_score": best_ev,
        "final_test_score": final_test,
    }
    with open(Path(args.output) / f"{suffix}.json", "w") as f:
        json.dump(saved_results, f, indent=4)


def seed_everything(seed=42):
    """
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    """
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


"""args"""
device = torch.device("cuda")
args, cfg = get_training_args()
seed_everything(seed=42)

""" prepare Datasets and Dataloaders """
if args.task == "binclass":
    from lib import FINETUNE_BIN_DATA as FINETUNE_DATA
elif args.task == "regression":
    from lib import FINETUNE_REG_DATA as FINETUNE_DATA
elif args.task == "multiclass":
    from lib import FINETUNE_MUL_DATA as FINETUNE_DATA

preproc_type = "ftt" if args.model == "tabnet" else args.model
data_config = DataConfig.from_pretrained(
    CHECKPOINT_DIR,
    data_dir=FINETUNE_DATA,
    batch_size=64,
    train_ratio=0.8,
    preproc_type=preproc_type,
    pre_train=False,
)
dataset = data_preproc(args.dataset, data_config, no_str=True, tt=args.task)

if dataset.X_cat is None:
    Xs = {k: dataset.X_num[k] for k in ["train", "val", "test"]}
else:
    if args.model == "tabnet":
        Xs = {
            k: np.concatenate((dataset.X_cat[k], dataset.X_num[k]), axis=1)
            for k in ["train", "val", "test"]
        }
    else:
        Xs = {
            k: np.concatenate((dataset.X_num[k], dataset.X_cat[k]), axis=1)
            for k in ["train", "val", "test"]
        }
ys = {k: dataset.y[k] for k in ["train", "val", "test"]}

if args.model == "tabnet":
    cat_idxs = [] if dataset.X_cat is None else list(range(dataset.n_cat_features))
    cat_dims = [] if dataset.X_cat is None else dataset.get_category_sizes("train")
    if args.task_type == "regression":
        ys = {k: ys[k].reshape(-1, 1) for k in ["train", "val", "test"]}

""" Tree Model """
if args.model == "xgboost":
    if dataset.is_regression:
        model = XGBRegressor(**cfg["model"], random_state=42)
        predict = model.predict
    else:
        model = XGBClassifier(
            **cfg["model"], random_state=42, disable_default_eval_metric=True
        )
        if dataset.is_multiclass:
            predict = model.predict_proba
            cfg["training"]["eval_metric"] = "merror"
        else:
            predict = lambda x: model.predict_proba(x)[:, 1]
            cfg["training"]["eval_metric"] = "error"

    model.fit(
        Xs["train"],
        dataset.y["train"],
        eval_set=[(Xs["val"], dataset.y["val"])],
        **cfg["training"],
    )
elif args.model == "catboost":
    cfg["model"]["task_type"] = "GPU"
    cfg["model"]["devices"] = "0"
    if dataset.is_regression:
        model = CatBoostRegressor(**cfg["model"])
        predict = model.predict
    else:
        model = CatBoostClassifier(**cfg["model"], eval_metric="Accuracy")
        predict = (
            model.predict_proba
            if dataset.is_multiclass
            else lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]
        )
    model.fit(
        Xs["train"],
        dataset.y["train"],
        **cfg["training"],
        eval_set=(Xs["val"], dataset.y["val"]),
    )
elif args.model == "tabnet":
    cfg["model"]["cat_idxs"] = cat_idxs
    cfg["model"]["cat_dims"] = cat_dims
    cfg["model"]["optimizer_params"] = {
        "lr": cfg["model"].pop("lr"),
    }
    if dataset.is_regression:
        model = TabNetRegressor(**cfg["model"])
        predict = model.predict
    else:
        model = TabNetClassifier(**cfg["model"])
        predict = (
            model.predict_proba
            if dataset.is_multiclass
            else lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]
        )
    model.fit(
        Xs["train"], ys["train"], eval_set=[(Xs["val"], ys["val"])], **cfg["training"],
    )
else:
    raise NotImplementedError

prediction = {k: predict(v) for k, v in Xs.items()}
prediction_type = None if dataset.is_regression else "probs"
metric_key = {"regression": "rmse", "binclass": "roc_auc", "multiclass": "accuracy"}[
    dataset.task_type.value
]
scale = 1 if not dataset.is_regression else -1
scores = {
    k: calculate_metrics(
        dataset.y[k],
        prediction[k],
        dataset.task_type.value,
        "probs" if not dataset.is_regression else None,
        dataset.y_info,
    )[metric_key]
    * scale
    for k in ["train", "val", "test"]
}

for k in scores:
    print(k, scores[k])


"""Record Exp Results"""
save_result(args, cfg, scores["val"], scores["test"], "finish")
