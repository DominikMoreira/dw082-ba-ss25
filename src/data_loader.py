import pandas as pd
import os

def load_data(file_path='data/raw/test_split.csv'):
    # Loads the data from the CSV file.
    df = pd.read_csv(file_path)
    return df

def export_dataframes_to_csv(train_df, test_df, val_df, output_path='data/processed/'):
    """
    Export train, test and validation dataframes to CSV files.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        val_df: Validation dataframe
        output_path: Path where CSV files should be saved
    """

    # Create directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Export each dataframe
    train_df.to_csv(os.path.join(output_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test.csv'), index=False)
    val_df.to_csv(os.path.join(output_path, 'val.csv'), index=False)

    print(f"Dataframes exported to {output_path}")
