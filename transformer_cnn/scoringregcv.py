import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def cross_validate_scoring(cv_results_dir, target_columns, augmented_number):
    """
    Perform cross-validation by iterating over multiple fold result files and compute performance.

    Args:
        cv_results_dir (str): Directory where result_fold_i.csv and test_fold_i.csv files are stored.
        target_columns (list): List of target column names (e.g., ['LRI', 'B1', 'B9']).
        augmented_number (int): Number of augmentations to align the files.

    Returns:
        dict: Performance metrics for each fold and the full dataset.
    """
    
    # Identify available fold files
    result_files = sorted([f for f in os.listdir(cv_results_dir) if f.startswith("result_fold_") and f.endswith(".csv")])
    test_files = sorted([f for f in os.listdir(cv_results_dir) if f.startswith("test_fold_") and f.endswith(".csv")])

    assert len(result_files) == len(test_files), "Mismatch between result and test files!"

    fold_metrics = {target: [] for target in target_columns}  # Store metrics per target
    dfs  = [] 
    for fold, (result_file, test_file) in enumerate(zip(result_files, test_files), start=1):
        print(f"\nProcessing Fold {fold}:")
        
        result_path = os.path.join(cv_results_dir, result_file)
        test_path = os.path.join(cv_results_dir, test_file)

        # Compute metrics for each target variable
        for target in target_columns:
            
            metrics, dfi = compute_statistics(result_path, augmented_number, test_path, target)
            dfs.append(dfi)
            fold_metrics[target].append(metrics)
            print(f"Fold {fold} - {target} Metrics: {metrics}")

    # Compute mean & std across folds
    cv_results = {}
    for target in target_columns:
        avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics[target]]) for metric in fold_metrics[target][0]}
        std_metrics = {metric: np.std([fold[metric] for fold in fold_metrics[target]]) for metric in fold_metrics[target][0]}
        cv_results[target] = {"cv_mean": avg_metrics, "cv_std": std_metrics}

        print(f"\nCross-validation Results for {target}:")
        for metric in avg_metrics:
            print(f"{metric}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    # Compute overall performance on full dataset
    print("\nComputing performance on full dataset...")
    full_pred_dfs = [pd.read_csv(os.path.join(cv_results_dir, f)) for f in result_files]
    full_true_dfs = [pd.read_csv(os.path.join(cv_results_dir, f)) for f in test_files]

    full_pred_df = pd.concat(full_pred_dfs, ignore_index=True)
    full_true_df = pd.concat(full_true_dfs, ignore_index=True)

    full_metrics = {}
    for target in target_columns:
        full_metrics[target], _ = compute_statistics(full_pred_df, augmented_number, full_true_df, target)
        print(f"\nFull Dataset Performance for {target}:")
        for metric, value in full_metrics[target].items():
            print(f"{metric}: {value:.4f}")
    pd.concat(dfs,axis=0).sort_values('augid').reset_index(drop=True).to_csv('globalres.csv',index=False)
    return {
        "folds": fold_metrics,
        "cv_results": cv_results,
        "full_data": full_metrics
    }


def compute_statistics(augmented_file_pred, augmented_number, augmented_file_true=None, target_column='target'):
    """
    Compute statistics for a given target variable:
    - Average predictions per molecule (group by `augid`)
    - Compute R², RMSE, MSE, and MAE
    """

    # Load augmented dataset
    if isinstance(augmented_file_pred, str):
        y_pred = pd.read_csv(augmented_file_pred)
    else:
        y_pred = augmented_file_pred  # If full dataset is passed

    if isinstance(augmented_file_true, str):
        y_true = pd.read_csv(augmented_file_true)
    else:
        y_true = augmented_file_true  # If full dataset is passed

    # Remove unnecessary columns
    for df in [y_pred, y_true]:
        df.drop(columns=[col for col in ['Unnamed: 1'] if col in df.columns], inplace=True)

    # Automatically rename Result0, Result1, Result2, ... to correct target names
    result_columns = [f"Result{i}" for i in range(len(y_pred.columns))]  # Dynamically find results
    rename_map = {result_columns[i]: target_column for i in range(len(result_columns)) if result_columns[i] in y_pred.columns}

    y_pred.rename(columns=rename_map, inplace=True)
    y_true.rename(columns=rename_map, inplace=True)

    if target_column not in y_pred.columns:
        raise ValueError(f"Error: Target column '{target_column}' missing in dataset.")

    if target_column not in y_true.columns:
        raise ValueError(f"Error: Target column '{target_column}' missing in dataset.")

    # Ensure augid exists in y_true
    if 'augid' not in y_true.columns:
        print("Adding 'augid' to y_true dataset...")
        y_true['augid'] = y_true.index // augmented_number
        assert len(y_true) / augmented_number == round(len(y_true) / augmented_number), "Incorrect 'augid' alignment!"

    # Merge true and predicted values
    df = pd.concat([y_true[['augid', target_column]], y_pred[[target_column]]], axis=1)
    df.columns = ['augid', 'true_value', 'predicted_value']

    # Compute the mean value of the target variable for each molecule (grouped by `augid`)
    df_avg_true = df.groupby('augid')['true_value'].mean()
    df_avg_pred = df.groupby('augid')['predicted_value'].mean()


    y_true_vals = df_avg_true.values
    y_pred_vals = df_avg_pred.values
    print(len(y_true_vals),len(y_pred_vals))

    # Compute metrics
    metrics = {
        "R²": r2_score(y_true_vals, y_pred_vals),
        "RMSE": np.sqrt(mean_squared_error(y_true_vals, y_pred_vals)),
        "MSE": mean_squared_error(y_true_vals, y_pred_vals),
        "MAE": mean_absolute_error(y_true_vals, y_pred_vals)
    }

    return metrics, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Validation Performance Evaluation")
    parser.add_argument("--cv_results_dir", "-d", type=str, required=True, help="Path to the directory containing result_fold_i.csv and test_fold_i.csv")
    parser.add_argument("--target_columns", "-t", type=str, nargs="+", required=True, help="List of target columns (e.g., LRI B1 B9)")
    parser.add_argument("--augmented_number", "-n", type=int, required=True, help="Number of augmentations per molecule")

    args = parser.parse_args()

    results = cross_validate_scoring(args.cv_results_dir, args.target_columns, args.augmented_number)