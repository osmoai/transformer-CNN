import configparser
import subprocess
import argparse
import os
import pandas as pd
from sklearn.model_selection import KFold

def modify_config(config_file, train_mode, data=None, gpu=0, result=None, model_name=None):
    """
    Modifies the config.cfg file for training or applying the model.

    Parameters:
        - config_file (str): Path to the config file.
        - train_mode (bool): Whether to run in training mode (True) or apply mode (False).
        - data (str): Path to data file (train or apply).
        - gpu (int): GPU ID.
        - result (str): Path to result file.
        - model_name (str): Name of the model file for saving checkpoints.
    """
    
    config = configparser.ConfigParser()
    config.read(config_file)

    # Update [Task] section
    config.set("Task", "train_mode", str(train_mode))
    
    if train_mode:
        config.set("Task", "train_data_file", data if data else "train.csv")
    else:
        config.set("Task", "apply_data_file", data if data else "apply.csv")
        config.set("Task", "result_file", result if result else "result.csv")

    # Set GPU device
    config.set("Details", "gpu", str(gpu))

    # Update model name to save different versions for each CV fold
    if model_name:
        config.set("Task", "model_file", model_name)

    with open(config_file, "w") as configfile:
        config.write(configfile)

    print(f"Updated {config_file} successfully for {model_name}.")

def run_model(script_path, config_file):
    """
    Executes the model using the modified config file.

    Parameters:
        - script_path (str): Path to the script that runs the model.
        - config_file (str): Path to the modified config file.
    """
    
    command = f"python {script_path} {config_file}"
    
    try:
        print(f"Executing command: {command}")
        subprocess.run(command, shell=True, check=True)
        print("Model execution completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing model: {e}")

def cross_validation(data_file, script, config_file, output_folder, n_splits=5, naug=10, gpu=0):
    """
    Performs 5-fold cross-validation using index-based grouping and trains models accordingly.

    Parameters:
        - data_file (str): Path to full dataset.
        - script (str): Path to model script.
        - config_file (str): Path to config file.
        - output_folder (str): Folder to store models and results.
        - n_splits (int): Number of cross-validation splits.
        - naug (int): Augmentation factor used for grouping indices.
        - gpu (int): GPU ID.
    """
    
    os.makedirs(output_folder, exist_ok=True)

    # Load data
    df = pd.read_csv(data_file)
    df["augid"] = df.index // naug
    print(df.head())
    print(df.shape)
    # Create CV splits using index-based grouping
    unique_groups = df["augid"].unique()  # Get unique molecule groups

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(unique_groups)):
        print(f"\n--- Running Fold {fold+1}/{n_splits} ---")
        
        # Extract train/val and test sets
        train_val_groups = unique_groups[train_idx]
        test_groups = unique_groups[test_idx]

        train_val_data = df[df["augid"].isin(train_val_groups)]
        test_data = df[df["augid"].isin(test_groups)]
        print(train_val_data.shape,test_data.shape)
        
        train_file = os.path.join(output_folder, f"train_fold_{fold+1}.csv")
        test_file = os.path.join(output_folder, f"test_fold_{fold+1}.csv")
        result_file = os.path.join(output_folder, f"result_fold_{fold+1}.csv")
        model_name = os.path.join(output_folder, f"model_fold_{fold+1}.tar")

        train_val_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)

        print(f"Training data saved: {train_file}")
        print(f"Testing data saved: {test_file}")

        # Train the model
        modify_config(config_file, train_mode=True, data=train_file, gpu=gpu, model_name=model_name)
        run_model(script, config_file)

        # Apply model on test set
        modify_config(config_file, train_mode=False, data=test_file, gpu=gpu, result=result_file, model_name=model_name)
        run_model(script, config_file)

        print(f"Results saved: {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform cross-validation and train multiple models.")

    parser.add_argument("--config", type=str, default="config.cfg", help="Path to the config file.")
    parser.add_argument("--script", type=str, default="transformer-cnnv6.py", help="Path to the model script.")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV file.")
    parser.add_argument("--output", type=str, default="cv_results", help="Folder to store CV results and models.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (default: 0).")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV splits (default: 5).")
    parser.add_argument("--naug", type=int, default=10, help="Augmentation factor for grouping indices.")

    args = parser.parse_args()

    cross_validation(
        data_file=args.data,
        script=args.script,
        config_file=args.config,
        output_folder=args.output,
        n_splits=args.n_splits,
        naug=args.naug,
        gpu=args.gpu
    )
