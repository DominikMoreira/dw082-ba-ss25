# Lädt und verarbeitet die Daten aus "cleaned_reviews.csv".

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path='data/cleaned_reviews.csv'):
    """ Loads the data from the CSV file. """
    df = pd.read_csv(file_path)
    return df

def split_data(df, random_state=42):
    """
    Splits the dataset into training (70%), test (20%), and validation (10%) sets.
    Maintains the same distribution of classes (Rating) across all splits.

    Args:
        df: pandas DataFrame containing the data
        random_state: random seed for reproducibility

    Returns:
        train_df: training set (70% of data)
        test_df: test set (20% of data)
        val_df: validation set (10% of data)
    """
    # First split: 80% train+val, 20% test
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['Rating'],
        random_state=random_state
    )

    # Second split: Split train_val into 87.5% train, 12.5% val (0.875 * 80% = 70% of total)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.125,  # 0.125 * 80% = 10% of total data
        stratify=train_val_df['Rating'],
        random_state=random_state
    )

    print(f"Training set size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test set size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"Validation set size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")

    return train_df, test_df, val_df

def get_data():
    """Main function for loading and preprocessing the data."""
    df = load_data()
    train_df, test_df, val_df = split_data(df)
    export_dataframes_to_csv(train_df, test_df, val_df)
    return train_df, test_df, val_df

def export_dataframes_to_csv(train_df, test_df, val_df, output_path='data/processed/'):
    """
    Export train, test and validation dataframes to CSV files.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        val_df: Validation dataframe
        output_path: Path where CSV files should be saved
    """
    import os

    # Create directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Export each dataframe
    train_df.to_csv(os.path.join(output_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test.csv'), index=False)
    val_df.to_csv(os.path.join(output_path, 'val.csv'), index=False)

    print(f"Dataframes exported to {output_path}")
