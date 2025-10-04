import pandas as pd
import numpy as np
import argparse

def analyze_csv(file_path, missing_only=False):
    """
    Analyzes a CSV file and prints a summary of each column.

    For numeric columns, it prints descriptive statistics.
    For categorical columns, it prints the value distribution.
    If missing_only is True, it only shows columns with missing data.
    """
    try:
        # Try to read the CSV, skipping comment lines that might be present
        df = pd.read_csv(file_path, comment='#')
        print(f"Successfully loaded {file_path}")
        print(f"Shape of the dataframe: {df.shape}")
        print("-" * 30)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    columns_displayed = 0
    total_rows = len(df)
    for col in df.columns:
        missing_values = df[col].isnull().sum()

        # If the flag is on, skip columns with no missing values
        if missing_only and missing_values == 0:
            continue

        columns_displayed += 1
        print(f"Column: {col}")
        print(f"Data Type: {df[col].dtype}")

        # Uniqueness analysis
        unique_values = df[col].nunique()
        # A column is considered unique if every row has a distinct value.
        is_fully_unique = unique_values == total_rows
        print(f"Unique Values: {unique_values}")
        if is_fully_unique:
            print("Is Column Unique: Yes")
        else:
            print("Is Column Unique: No")

        # Missing values
        if missing_values > 0:
            print(f"Missing Values: {missing_values} ({missing_values/total_rows:.2%})")
        else:
            print("Missing Values: 0")

        # Check if column is numeric or categorical
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numeric column
            print("Type: Numeric")
            print("Statistics:")
            print(df[col].describe())
        else:
            # Categorical column
            print("Type: Categorical")
            # Heuristic to decide whether to print all unique values
            if unique_values > 50:
                 print("Value Counts: (Too many to display)")
            else:
                print("Value Counts:")
                print(df[col].value_counts())

        print("-" * 30)

    if missing_only and columns_displayed == 0:
        print("No columns with missing values found.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze the columns of a CSV file.')
    parser.add_argument('file_path', type=str, help='The path to the CSV file to analyze.')
    parser.add_argument('--missing-only', action='store_true', help='Only show columns with missing values.')
    args = parser.parse_args()

    analyze_csv(args.file_path, args.missing_only)