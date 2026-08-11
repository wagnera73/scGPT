import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import wandb

from scripts.analyses import load_default_hyperparameters, run_batch, run_finetune, run_perturbation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name")
    parser.add_argument("--task", default="batch_correction")
    parser.add_argument("--model", default="scGPT")
    parser.add_argument("--load_model", default=str(REPO_ROOT / "save" / "scGPT_human"))
    parser.add_argument("--label_column", default="celltype")
    parser.add_argument("--project")
    parser.add_argument(
        "--balance_classes",
        action="store_true",
        help="Stratify the train/valid split by label and use inverse-frequency "
        "class weights in the finetune loss.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    hyperparameter_defaults = load_default_hyperparameters()[args.task]
    hyperparameter_defaults.update(vars(args))

    run = wandb.init(
        config=hyperparameter_defaults,
        project=args.project or "scGPT",
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    config = wandb.config
    print(config)

    if config.model == "scGPT":
        if config.task == "batch_correction":
            run_batch(config)
        elif config.task == "annotation":
            print("Annotation --- WIP")
        elif config.task == "grn":
            print("GRN -- WIP")
        elif config.task == "integration":
            print("Integration -- WIP")
        elif config.task == "perturbation":
            run_perturbation(config)
        elif config.task == "finetune":
            run_finetune(config)

    run.finish()
    wandb.finish()
