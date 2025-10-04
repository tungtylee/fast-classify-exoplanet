import pandas as pd
import argparse
import json
import os

def filter_out_dontcare(input_csv, config_path, output_csv):
    """
    Filters a CSV file to remove rows based on "dontcare" values specified in a config file.
    """
    # 1. Read the configuration file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        target_column = config.get("target_column")
        mapping = config.get("mapping")
        if not target_column or not mapping:
            print(f"Error: Config file '{config_path}' is missing 'target_column' or 'mapping'.")
            return
    except FileNotFoundError:
        print(f"Error: The config file '{config_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{config_path}'.")
        return

    dontcare_values = [key for key, value in mapping.items() if value == "dontcare"]
    if not dontcare_values:
        print("No 'dontcare' values found in the config mapping. Nothing to filter.")
        return
    print(f"Found {len(dontcare_values)} 'dontcare' values to filter out: {dontcare_values}")

    # 2. Read the source CSV
    try:
        df = pd.read_csv(input_csv, comment='#')
        print(f"Loaded source CSV: {input_csv} with shape {df.shape}")
    except FileNotFoundError:
        print(f"Error: The source CSV file '{input_csv}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading {input_csv}: {e}")
        return

    # 3. Filter the DataFrame
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the input CSV.")
        return

    original_rows = len(df)
    filtered_df = df[~df[target_column].isin(dontcare_values)]
    rows_removed = original_rows - len(filtered_df)

    if rows_removed == 0:
        print(f"No rows were removed. The '{target_column}' column did not contain any of the 'dontcare' values.")
    else:
        print(f"Removed {rows_removed} rows where '{target_column}' was in {dontcare_values}.")

    # 4. Save the new CSV
    try:
        filtered_df.to_csv(output_csv, index=False)
        print(f"\nSuccessfully created filtered file: [1m{output_csv}[0m")
        print(f"New shape: {filtered_df.shape}")
    except Exception as e:
        print(f"\nError saving new CSV file: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter a CSV file to remove "dontcare" rows.')
    parser.add_argument('input_csv', type=str, help='Path to the source CSV file (e.g., train_set.csv).')
    parser.add_argument('output_csv', type=str, help='Path for the output filtered CSV.')
    parser.add_argument('--config', type=str, default='scorer_conf.json', help='Path to the scorer configuration file (default: scorer_conf.json).')
    args = parser.parse_args()

    # Ensure the config path is absolute if it's not already
    config_path = args.config
    if not os.path.isabs(config_path):
        # Assuming the script is run from the project root or data_TOI, let's be robust
        script_dir = os.path.dirname(os.path.realpath(__file__))
        potential_path = os.path.join(script_dir, config_path)
        if os.path.exists(potential_path):
            config_path = potential_path
        # If not found, we'll let the function handle the FileNotFoundError

    filter_out_dontcare(args.input_csv, config_path, args.output_csv)
