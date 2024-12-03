import subprocess
import json
import os
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime

DATA_DIR = "/home/danieleitan/tp-berta/data/finetune-bin"

# Configuration for experiments
DATASETS = [
    x for x in os.listdir(DATA_DIR) if 'csv' in x
]

TASKS = {
    x: "binclass" for x in DATASETS
}

# Experiment configurations
GATING_CONFIGS = [
    # {
    #     "is_gating": False,
    #     "gating_loss_weight": 0,
    # },
    # # Conservative configuration - focuses on preserving performance
    # {
    #     "is_gating": True,
    #     "gating_loss_weight": 0.005,  # Reduced weight to be more conservative
    #     "gating_temperature": 3.0,    # Higher temperature for softer decisions
    #     "min_temperature": 0.5,       # Higher minimum to prevent hard gating
    #     "temperature_decay": 0.998,   # Much slower decay
    #     "hidden_size": 768,
    #     "apply_gating": "input",
    #     "gating_start_epoch": 15,     # Later start to establish baseline
    #     "lambda1": 0.0001,           # Much lower entropy penalty
    #     "lambda2": 0.0005,           # Lower smoothness penalty
    #     "entropy_weight": 0.001,     # Lower entropy weight
    #     "min_keep_ratio": 0.7,       # Higher minimum keep ratio
    #     "sigma": 0.03                # Lower noise for stability
    # },
    # # # Moderate configuration - balanced approach
    # # {
    # #     "is_gating": True,
    # #     "gating_loss_weight": 0.008,  # Moderate weight
    # #     "gating_temperature": 2.5,
    # #     "min_temperature": 0.3,
    # #     "temperature_decay": 0.997,
    # #     "hidden_size": 768,
    # #     "apply_gating": "input",
    # #     "gating_start_epoch": 10,
    # #     "lambda1": 0.0003,
    # #     "lambda2": 0.0008,
    # #     "entropy_weight": 0.003,
    # #     "min_keep_ratio": 0.6,
    # #     "sigma": 0.04
    # # },
    # Performance-focused configuration
    {
        "is_gating": True,
        "gating_loss_weight": 0.005,  # Reduced from 0.01
        "gating_temperature": 3.0,  # Increased from 2.0
        "min_temperature": 0.3,  # Increased from 0.2
        "temperature_decay": 0.997,  # Slowed down from 0.996
        "hidden_size": 768,
        "apply_gating": "input",
        "gating_start_epoch": 300,  # Increased from 5
        "lambda1": 0.0002,  # Reduced from 0.0005
        "lambda2": 0.0005,  # Reduced from 0.001
        "entropy_weight": 0.002,  # Reduced from 0.005
        "min_keep_ratio": 0.7,  # Increased from 0.5
        "sigma": 0.03
    }
]

# Modified base configuration for better training
BASE_CONFIG = {
    "max_epochs": 500,        # Increased to allow more time for convergence
    "early_stop": 999999,        # Reduced to be more responsive
    "batch_size": 32,
    "lr": 2e-5,             # Slightly increased
    "wandb": False,
    "weight_decay": 0.0001,  # Reduced further
    "max_sequence_length": 512,
}


def run_experiment(dataset, task, config):
    """Run a single experiment with given configuration"""
    cmd = ["/opt/conda/envs/tpberta/bin/python3.9", "/home/danieleitan/tp-berta/scripts/finetune/default/run_default_config_tpberta.py"]

    # Add base arguments
    for key, value in BASE_CONFIG.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    # Add dataset and task
    cmd.extend(["--dataset", dataset[:-4]])
    cmd.extend(["--task", task])

    # Add gating configuration
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    # Run the experiment
    process = subprocess.run(cmd, capture_output=True, text=True)

    # Extract results
    result_dir = None
    for line in process.stdout.split('\n'):
        print(line)
        if "Generated run_id:" in line:
            run_id = line.split(": ")[1].strip()
            result_dir = f"finetune_outputs/{task}/TPBerta-default/{run_id}"

    print('*'*50)

    if result_dir and os.path.exists(result_dir):
        # Read the results
        with open(os.path.join(result_dir, "finish.json"), 'r') as f:
            results = json.load(f)

        # Clean up
        shutil.rmtree(result_dir)

        return {
            'dataset': dataset[:-4],
            'best_eval_score': results['best_eval_score'],
            'final_test_score': results['final_test_score'],
            **config  # Include configuration in results
        }
    else:
        print(f"Warning: Could not find results directory for {dataset}")
        return None


def main():
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"experiment_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)

    # Store all results
    all_results = []

    # Run experiments
    for dataset in DATASETS:
        task = TASKS[dataset]
        print(f"\nRunning experiments for {dataset} ({task})")

        for config in GATING_CONFIGS:
            print(f"\nConfiguration: {config}")
            result = run_experiment(dataset, task, config)
            if result:
                all_results.append(result)

                # Save intermediate results
                df = pd.DataFrame(all_results)
                df.to_csv(results_dir / "results.csv", index=False)

                # Also save as formatted markdown for easy viewing
                with open(results_dir / "results.md", 'w') as f:
                    f.write(df.to_markdown())

    print(f"\nAll experiments completed. Results saved in {results_dir}")


if __name__ == "__main__":
    main()
