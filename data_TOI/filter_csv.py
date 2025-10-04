import pandas as pd
import argparse
import os

def filter_csv_by_columns(input_csv, columns_file, output_csv):
    """
    Filters a CSV file to keep only the columns specified in a text file.
    """
    # 1. Read the list of columns to keep
    try:
        with open(columns_file, 'r') as f:
            selected_columns = [line.strip() for line in f.readlines()]
        if not selected_columns:
            print(f"Error: The columns file '{columns_file}' is empty.")
            return
        print(f"Read {len(selected_columns)} columns from {columns_file}")
    except FileNotFoundError:
        print(f"Error: The columns file '{columns_file}' was not found.")
        print("Please run interactive_filter.py first to generate it.")
        return

    # 2. Read the source CSV
    try:
        df = pd.read_csv(input_csv, comment='#')
        print(f"Loaded source CSV: {input_csv}")
    except FileNotFoundError:
        print(f"Error: The source CSV file '{input_csv}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading {input_csv}: {e}")
        return

    # 3. Filter the DataFrame
    original_columns = df.columns.tolist()
    # Find which of the selected columns are actually in the dataframe
    columns_to_keep = [col for col in selected_columns if col in original_columns]
    missing_cols = set(selected_columns) - set(columns_to_keep)

    if not columns_to_keep:
        print("Error: None of the selected columns were found in the input CSV.")
        return

    if missing_cols:
        print(f"Warning: The following {len(missing_cols)} columns from your list were not found in the CSV and will be ignored:")
        for col in missing_cols:
            print(f"- {col}")

    filtered_df = df[columns_to_keep]

    # 4. Save the new CSV
    try:
        filtered_df.to_csv(output_csv, index=False)
        print(f"\nSuccessfully created filtered file: \033[1m{output_csv}\033[0m")
        print(f"New shape: {filtered_df.shape}")
    except Exception as e:
        print(f"\nError saving new CSV file: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter a CSV file based on a list of selected columns.')
    parser.add_argument('input_csv', type=str, help='Path to the source CSV file (e.g., train_set.csv).')
    parser.add_argument('columns_file', type=str, help='Path to the text file containing column names to keep.')
    parser.add_argument('output_csv', type=str, help='Path for the output filtered CSV.')
    args = parser.parse_args()

    filter_csv_by_columns(args.input_csv, args.columns_file, args.output_csv)
