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

def load(train_size=.8, test_size=.1):
    """
    Load the Wisconsin Breast Cancer dataset.
    """
    # Load data
    wbcd = wisconsin_breast_cancer_data = datasets.load_breast_cancer()
    feature_names = wbcd.feature_names
    labels = wbcd.target_names
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(wbcd.data, wbcd.target, test_size=(1-train_size))
    # Further split test set into test and validation sets
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=test_size)

    return {
        'Data': {
            'Train': {'X': X_train, 'y': y_train},
            'Validation': {'X': X_validation, 'y': y_validation},
            'Test': {'X': X_test, 'y': y_test}
        },
        'Metadata': {'Feature_names': feature_names, 'Labels': labels}
    }

def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project="expanaliticawandb",
        name=f"Load Raw Data ExecId-{args.IdExecution}", job_type="load-data") as run:
        datasets = load()  # separate code for loading the datasets

        # Create a W&B Artifact
        raw_data = wandb.Artifact(
            "wisconsin_breast_cancer_dataset-raw", type="dataset",
            description="raw Wisconsin Breast Cancer dataset, split into train/val/test",
            metadata={
                "source": "sklearn.datasets.load_breast_cancer",
                "sizes": {key: (len(value['X']),len(value['y'])) for key, value in datasets['Data'].items()},
                "features": len(datasets['Metadata']['Feature_names']),
                "labels": len(datasets['Metadata']['Labels'])
            })
        
        # Save each dataset split to the artifact
        for name, data in datasets['Data'].items():
            with raw_data.new_file(name + "_X.npy", mode="wb") as file:
                np.save(file, data['X'])
            with raw_data.new_file(name + "_y.npy", mode="wb") as file:
                np.save(file, data['y'])        

        # Save the artifact to W&B
        run.log_artifact(raw_data)                

# testing
load_and_log()