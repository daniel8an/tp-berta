import numpy as np
import torch
import os
import sys

sys.path.append(os.getcwd())  # to correctly import bin & lib
import json

import wandb
import shutil
import random
import argparse
import uuid

from tqdm import trange, tqdm
from pathlib import Path

from bin import build_default_model
from lib import (
    DataConfig,
    Regulator,
    prepare_tpberta_loaders,
    magnitude_regloss,
    calculate_metrics,
    make_tpberta_optimizer,
)


def load_single_dataset(dataset_name, data_config, task_type):
    data_loader, dataset = prepare_tpberta_loaders(
        [dataset_name], data_config, tt=task_type
    )
    return data_loader[0], dataset[0]


def save_result(
    args,
    best_ev,
    final_test,
    tr_losses,
    reg_losses,
    tr_gating_losses,
    ev_losses,
    ev_metrics,
    test_metrics,
    suffix,
):
    saved_results = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "best_eval_score": best_ev,
        "final_test_score": final_test,
        "ev_metric": ev_metrics,
        "test_metric": test_metrics,
        "tr_loss": tr_losses,
        "ev_loss": ev_losses,
        "tr_gating_loss": tr_gating_losses,
    }
    if args.lamb > 0:
        saved_results["reg_loss"] = reg_losses
    with open(Path(args.result_dir) / f"{suffix}.json", "w") as f:
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


def generate_run_id(args):
    components = [
        f"task-{args.task[:3]}",  # task abbreviation
        f"data-{args.dataset[:3]}",  # dataset abbreviation
        f"gating-{'on' if args.is_gating else 'off'}",  # is_gating
        f"gloss-{args.gating_loss_weight:.4f}",  # gating loss weight
        f"gstart-{args.gating_start_epoch:02d}",  # gating start epoch
        f"hidden-{args.hidden_size}",  # hidden size
        f"gapply-{args.apply_gating[:3]}",  # apply gating abbreviation
        f"temp-{args.gating_temperature:.2f}",  # initial temperature
        f"mintemp-{args.min_temperature:.2f}",  # min temperature
        f"tdecay-{args.temperature_decay:.2f}",  # temperature decay
        f"keep-{args.min_keep_ratio:.2f}",  # min keep ratio
        f"lr-{args.lr:.6f}",  # learning rate
        f"batch-{args.batch_size}",  # batch size
        f"epochs-{args.max_epochs:02d}",  # max epochs
    ]
    return "__".join(components)


def make_optimizer(model, args):
    gating_params = [p for n, p in model.named_parameters() if 'gating' in n]
    base_params = [p for n, p in model.named_parameters() if 'gating' not in n]
    
    return torch.optim.AdamW([
        {'params': base_params, 'lr': args.lr},
        {'params': gating_params, 'lr': args.lr}  # Faster learning for gates
    ], weight_decay=args.weight_decay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="finetune_outputs")
    parser.add_argument("--model_suffix", type=str, default="pytorch_models/best")
    parser.add_argument("--dataset", type=str, default="HR Employee Attrition")
    parser.add_argument(
        "--task",
        type=str,
        choices=["binclass", "regression", "multiclass"],
        required=True,
    )
    parser.add_argument("--lr", type=float, default=1e-5)  # fixed learning rate
    parser.add_argument(
        "--weight_decay", type=float, default=0.0
    )  # no weight decay in default
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--lamb", type=float, default=0.0
    )  # no regularization in finetune
    parser.add_argument("--wandb", action='store_true')
    # Add new gating-related arguments
    parser.add_argument("--is_gating", action="store_true", help="Use gating mechanism or not")
    parser.add_argument("--gating_loss_weight", type=float, default=0.001, help="Weight for gating loss")
    parser.add_argument("--gating_start_epoch", type=int, default=5, help="Epoch to start gating warmup")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size for the model")
    parser.add_argument("--apply_gating", type=str, default="input", choices=["input", "embedding", "none"], help="Where to apply gating mechanism")
    # Add gating regularization parameters
    parser.add_argument("--lambda1", type=float, default=0.01,
                       help="L0 regularization strength for gating")
    parser.add_argument("--lambda2", type=float, default=0.01,
                       help="Stability regularization strength for gating")
    parser.add_argument("--gating_temperature", type=float, default=0.5,
                       help="Temperature for gating mechanism")
    # Add the missing arguments
    parser.add_argument("--min_temperature", type=float, default=0.1,
                       help="Minimum temperature for gating annealing")
    parser.add_argument("--temperature_decay", type=float, default=0.99,
                       help="Temperature decay rate for gating annealing")
    parser.add_argument("--entropy_weight", type=float, default=0.01,
                       help="Weight for entropy regularization")
    parser.add_argument("--min_keep_ratio", type=float, default=0.3,
                       help="Minimum ratio of tokens to keep")
    parser.add_argument("--sigma", type=float, default=0.1,
                       help="Standard deviation of noise for gating")

    args = parser.parse_args()

    # Generate run_id
    run_id = generate_run_id(args)
    print(f"Generated run_id: {run_id}")

    # keep default settings
    args.result_dir = f"{args.result_dir}/{args.task}/TPBerta-default/{run_id}"
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if args.task == "binclass":
        from lib import FINETUNE_BIN_DATA as FINETUNE_DATA
        from lib import BIN_CHECKPOINT as CHECKPOINT_DIR
    elif args.task == "regression":
        from lib import FINETUNE_REG_DATA as FINETUNE_DATA
        from lib import REG_CHECKPOINT as CHECKPOINT_DIR
    elif args.task == "multiclass":
        from lib import FINETUNE_MUL_DATA as FINETUNE_DATA
        from lib import BIN_CHECKPOINT as CHECKPOINT_DIR

    seed_everything(seed=42)
    """ Data Preparation """
    data_config = DataConfig.from_pretrained(
        CHECKPOINT_DIR,
        data_dir=FINETUNE_DATA,
        batch_size=64,
        train_ratio=0.8,
        preproc_type="lm",
        pre_train=False,
    )
    (data_loader, _), dataset = load_single_dataset(
        args.dataset, data_config, args.task
    )

    """ Model Preparation """
    device = torch.device("cuda")
    args.pretrain_dir = str(CHECKPOINT_DIR)  # pre-trained TPBerta dir
    model_config, model = build_default_model(
        args, data_config, dataset.n_classes, device, pretrain=True
    )
    model.tpberta.is_gating = args.is_gating
    model.tpberta.apply_gating = args.apply_gating
    if hasattr(model.tpberta, 'gating'):
        model.tpberta.gating.lambda1 = args.lambda1
        model.tpberta.gating.lambda2 = args.lambda2
    model_config.is_gating = args.is_gating
    model_config.apply_gating = args.apply_gating
    print(f"Model config after build_default_model: {model_config}")
    print("Args after parsing:")
    print(json.dumps(vars(args), indent=4))

    optimizer = make_optimizer(model, args)

    tot_step = 0
    best_metric = -np.inf
    final_test_metric = 0
    no_improvement = 0
    tr_task_losses, tr_reg_losses, tr_gating_losses = [], [], []
    ev_task_losses, ev_metrics = [], []
    test_metrics = []
    metric_key = {
        "regression": "rmse",
        "binclass": "roc_auc",
        "multiclass": "accuracy",
    }[dataset.task_type.value]
    scale = 1 if not dataset.is_regression else -1
    steps_per_save = 200

    if args.wandb:
        wandb.init(
            project="TPBerta-Finetune",
            config=vars(args),
            id=run_id,
            name=run_id,  # Also set the run name to be the same as the ID
            resume="allow"
        )
        print(f"Wandb run ID: {wandb.run.id}")

    for epoch in trange(args.max_epochs, desc="Finetuning"):
        cur_step = 0
        tr_loss = 0.0  # train loss
        reg_loss = 0.0  # regularization loss
        gating_loss = 0.0  # gating loss
        model.train()

        if epoch < args.gating_start_epoch:
            is_epoch_train_gates = False
        else:
            is_epoch_train_gates = True

        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "is_epoch_train_gates": is_epoch_train_gates
            })

        for batch in tqdm(data_loader["train"], desc=f"epoch-{epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            optimizer.zero_grad()

            logits, outputs, batch_gating_loss, _ = model(**batch, is_epoch_train_gates=is_epoch_train_gates)

            # In your training loop
            # if args.is_gating and is_epoch_train_gates:
            #     sparsity = model.tpberta.gating.get_sparsity_stats()
            #     print(f"Current sparsity: {sparsity:.3f}")
            #
            #     # Optional: Log gate distribution
            #     if hasattr(outputs, 'stochastic_gate'):
            #         gate_dist = outputs.stochastic_gate
            #         print(f"Gate distribution: mean={gate_dist.mean():.3f}, std={gate_dist.std():.3f}")
            
            loss = Regulator.compute_loss(logits, labels, dataset.task_type.value)
            tr_loss += loss.cpu().item()
            if args.lamb > 0:  # triplet loss used in pre-training
                reg = magnitude_regloss(labels.shape[0], data_config.num_encoder, model)
                reg_loss += reg.cpu().item()
                loss = loss + args.lamb * reg
            
            # Add gating loss if is_gating is True
            if args.is_gating and is_epoch_train_gates and batch_gating_loss is not None:
                gating_loss += batch_gating_loss.cpu().item()
                loss = loss + args.gating_loss_weight * batch_gating_loss

            loss.backward()
            optimizer.step()
            print(
                f"\repoch [{epoch+1}/{args.max_epochs}] | step {cur_step+1} | avg tr loss: {tr_loss / (cur_step+1)} | avg gating loss: {gating_loss / (cur_step+1)} | avg reg loss: {reg_loss / (cur_step+1)}",
                end="",
            )
            cur_step += 1
            tot_step += 1
            if tot_step % steps_per_save == 0:
                print(f"[STEP] {tot_step}: saving tmp results")
                save_result(
                    args,
                    best_metric,
                    final_test_metric,
                    tr_task_losses,
                    tr_reg_losses,
                    tr_gating_losses,
                    ev_task_losses,
                    ev_metrics,
                    test_metrics,
                    "tmp",
                )

            if args.wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/task_loss": tr_loss / (cur_step + 1),
                    "train/reg_loss": reg_loss / (cur_step + 1),
                    "train/gating_loss": gating_loss / (cur_step + 1),
                    "epoch": epoch,
                    "step": tot_step,
                })

            if args.wandb and args.is_gating:
                gate_stats = model.tpberta.gating.get_gate_statistics()
                wandb.log({
                    "train/mean_gate": gate_stats['mean_gate'],
                    "train/gate_std": gate_stats['gate_std'],
                    "train/active_gates": gate_stats['active_gates'],
                    # ... other metrics ...
                })

        tr_task_losses.append(tr_loss / cur_step)
        tr_reg_losses.append(reg_loss / cur_step)
        tr_gating_losses.append(gating_loss / cur_step)

        # evaluating
        preds, golds, ev_loss = [], [], []
        model.eval()
        for batch in tqdm(data_loader["val"], desc="evaluate"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.no_grad():
                logits, _, _, _ = model(**batch, train_gates=False)
                loss = Regulator.compute_loss(logits, labels, dataset.task_type.value)
            preds.append(logits.cpu())
            golds.append(labels.cpu())
            ev_loss.append(loss.cpu().item())

        ev_task_losses.append(sum(ev_loss) / len(ev_loss))
        score = (
            calculate_metrics(
                torch.cat(golds).numpy(),
                torch.cat(preds).numpy(),
                dataset.task_type.value,
                "logits" if not dataset.is_regression else None,
                dataset.y_info,
            )[metric_key]
            * scale
        )
        ev_metrics.append(score)

        # testing
        preds, golds = [], []
        for batch in tqdm(data_loader["test"], desc="testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.no_grad():
                logits, _, _, _ = model(**batch, train_gates=False)
            preds.append(logits.cpu())
            golds.append(labels.cpu())
        test_score = (
            calculate_metrics(
                torch.cat(golds).numpy(),
                torch.cat(preds).numpy(),
                dataset.task_type.value,
                "logits" if not dataset.is_regression else None,
                dataset.y_info,
            )[metric_key]
            * scale
        )
        test_metrics.append(test_score)

        print()
        print(f"[Eval] {metric_key}: {score} | [Test] {metric_key}: {test_score}")
        if score > best_metric:
            best_metric = score
            final_test_metric = test_score
            no_improvement = 0
            print("best result")
        else:
            no_improvement += 1
        if args.early_stop > 0 and no_improvement == args.early_stop:
            print("early stopping")
            break

        if args.wandb:
            wandb.log({
                "eval/score": score,
                "eval/loss": sum(ev_loss) / len(ev_loss),
                "test/score": test_score,
                "epoch": epoch,
            })

    print(f"best_metric: {best_metric}, final_test_metric: {final_test_metric}")
    save_result(
        args,
        best_metric,
        final_test_metric,
        tr_task_losses,
        tr_reg_losses,
        tr_gating_losses,
        ev_task_losses,
        ev_metrics,
        test_metrics,
        "finish",
    )

    if args.wandb:
        wandb.log({
            "best_eval_score": best_metric,
            "final_test_score": final_test_metric,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
