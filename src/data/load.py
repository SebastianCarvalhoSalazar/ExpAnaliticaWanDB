import argparse
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import wandb
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=.8):
    """
    Load the Wisconsin Breast Cancer dataset.
    """
    # Load data
    wbcd = wisconsin_breast_cancer_data = datasets.load_breast_cancer()
    feature_names = wbcd.feature_names
    labels = wbcd.target_names
    X_train, X_test, y_train, y_test = train_test_split(wbcd.data, wbcd.target, test_size=0.2)
    return [X_train, X_test, y_train, y_test, feature_names, labels]

def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project="expanaliticawandb",
        name=f"Load Raw Data ExecId-{args.IdExecution}", job_type="load-data") as run:
        
        datasets = load()  # separate code for loading the datasets
        print(datasets[0])

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="raw wisconsin_breast_cancer_dataset, split into train/val/test",
            metadata={"source": "sklearn.datasets.load_breast_cancer",
                      "sizes": [len(dataset) for dataset in datasets]})

        # for name, data in zip(names, datasets):
        #     # 🐣 Store a new file in the artifact, and write something into its contents.
        #     with raw_data.new_file(name + ".pt", mode="wb") as file:
        #         x, y = data.tensors
        #         torch.save((x, y), file)

        # # ✍️ Save the artifact to W&B.
        # run.log_artifact(raw_data)

# testing
load_and_log()