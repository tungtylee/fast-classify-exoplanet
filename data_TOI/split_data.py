

import pandas as pd
import numpy as np
import os

def split_data_train_test(seed, file_path, output_dir):
    """
    Reads the TOI data, splits it into a training and a test set
    based on the unique star ID (tid), and saves them to separate CSV files.
    Validation should be performed on the training set using cross-validation.
    """

    print(f"Reading data from {file_path}...")

    # Read the CSV file, skipping the initial comment lines
    try:
        df = pd.read_csv(file_path, skiprows=92)
        print("File read successfully!")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # 1. Get all unique tids
    unique_tids = df['tid'].unique()
    print(f"Found {len(unique_tids)} unique star IDs (tid).")

    # 2. Shuffle the tids for random splitting
    np.random.seed(seed)  # for reproducibility
    np.random.shuffle(unique_tids)

    # 3. Define an 80/20 split ratio and partition the tids
    train_ratio = 0.8
    train_split_idx = int(len(unique_tids) * train_ratio)

    train_tids = unique_tids[:train_split_idx]
    test_tids = unique_tids[train_split_idx:]

    # 4. Create the datasets based on the partitioned tids
    train_df = df[df['tid'].isin(train_tids)]
    test_df = df[df['tid'].isin(test_tids)]

    print("\n--- Data Split Summary ---")
    print(f"Total rows: {len(df)}")
    print(f"Training set rows: {len(train_df)} (for training and cross-validation)")
    print(f"Test set rows: {len(test_df)} (for final model evaluation)")
    print("--------------------------\n")

    # 5. Analyze class balance
    analyze_class_balance(train_df, test_df)

    # 6. Save the split datasets to new CSV files
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        train_df.to_csv(os.path.join(output_dir, 'train_set.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_set.csv'), index=False)
        
        print(f"Successfully saved split files to: {output_dir}")
        print("- train_set.csv")
        print("- test_set.csv")

    except Exception as e:
        print(f"An error occurred while saving the files: {e}")


def analyze_class_balance(train_df, test_df):
    """
    Analyzes and prints the class balance of the training and testing sets.
    """
    print("\n--- Class Balance Analysis (CP, KP vs. FP) ---")
    
    def print_balance(dataset_name, df):
        counts = df['tfopwg_disp'].value_counts()
        
        cp = counts.get('CP', 0)
        kp = counts.get('KP', 0)
        fp = counts.get('FP', 0)
        pc = counts.get('PC', 0)
        
        positive = cp + kp
        negative = fp
        
        total_classified = positive + negative
        
        print(f"{dataset_name}:")
        if total_classified > 0:
            print(f"  - Positive (CP+KP): {positive} ({positive/total_classified:.2%})")
            print(f"  - Negative (FP):    {negative} ({negative/total_classified:.2%})")
        else:
            print("  - No Positive or Negative classes found.")
            
        print(f"  - Other (PC):       {pc}")
        print(f"  - Total Classified: {total_classified}")

    print_balance("Training Set", train_df)
    print_balance("Test Set", test_df)
        
    print("----------------------------------------------------\n")


if __name__ == '__main__':
    # Relative path to the source file from the script's location
    file_path = 'TOI_2025.10.03_22.49.45.csv'
    # Relative path to the output directory
    output_dir = './'

    seed = 1004
    split_data_train_test(seed, file_path, output_dir)
