import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_statistics(augmented_file_pred, augmented_file_true=None, target_column='target'):
    """
    Compute augmentation statistics:
    - Average predictions per molecule (group by `augid`)
    - Compute R², RMSE, MSE, and MAE using original (augid=0) vs. averaged augmented values

    Args:
        augmented_file (str): Path to the augmented CSV file.
        target_column (str): Column containing the target values.

    Returns:
        dict: Computed statistics.
    """
    
    # Load augmented dataset
    if augmented_file_true == None:
        print('cannot compute Statistics there is not true dataset')

    else:

        y_pred = pd.read_csv(augmented_file_pred)
        y_true = pd.read_csv(augmented_file_true)

        if target_column not in y_pred.columns:
            raise ValueError("Error: 'augid' or target column missing in dataset.")

        if 'augid'  in  y_true.columns and target_column in y_true.columns :
            print('data structure matching perfectly')
            y_true = y_true[[target_column,'augid']]
            y_pred = y_pred[[target_column]]
            y_pred.columns = ['prediction']
        else:
            print('data structure error we cannot merge properly')
            y_true = y_true[['augid']]

            y_pred = y_pred[[target_column]]

        df = pd.concat([y_true, y_pred], axis=1) 

        # Compute the mean value of the target variable for each molecule (grouped by `augid`)
        if 'augid'  in  y_true.columns and target_column in y_true.columns :
            df_avg_true = df.groupby('augid')[target_column].mean().reset_index()
            df_avg_pred = df.groupby('augid')['prediction'].mean().reset_index()        
            df_avg_std = df.groupby('augid')['prediction'].std().reset_index()        
            df_avg_std.columns = ['augid','std']
        
        else:

            df_avg_true = df.groupby('augid')[target_column].mean().reset_index()
            df_avg_pred = df.groupby('augid')[target_column].mean().reset_index()
            df_avg_std = df.groupby('augid')[target_column].std().reset_index()        
            df_avg_std.columns = ['augid','std']
        # Compute metrics
        y_true_ = df_avg_true.values
        y_pred_ = df_avg_pred.values

        r2 = r2_score(y_true_, y_pred_)
        mse = mean_squared_error(y_true_, y_pred_)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_, y_pred_)

        # Print results
        if 'augid'  in  y_true.columns and target_column in y_true.columns :

            print(f"R²: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"MAE: {mae:.4f}")
        else:

            print('FAKE values we cannot make Statistics on new data!')
            print(f"R²: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"MAE: {mae:.4f}")

        print(pd.concat([df_avg_true,df_avg_pred,df_avg_std], axis=1))


        return {"R²": r2, "RMSE": rmse, "MSE": mse, "MAE": mae}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Augmentation Statistics")
    parser.add_argument("--augmented_file_pred", '-o', type=str, required=True, help="Path to the augmented dataset CSV")
    parser.add_argument("--augmented_file_true", '-i', type=str, default=None, required=False, help="Path to the augmented dataset CSV")
    parser.add_argument("--target_column", '-n', type=str, required=True, help="Name of the target column (e.g., 'property')")

    args = parser.parse_args()

    compute_statistics(args.augmented_file_pred, args.augmented_file_true, args.target_column)