import pandas as pd
import argparse

def analyze_column_summary(df, col):
    """Returns a summary string for a single column."""
    summary = []
    summary.append(f"Data Type: {df[col].dtype}")

    missing_values = df[col].isnull().sum()
    if missing_values > 0:
        summary.append(f"Missing Values: {missing_values} ({missing_values/len(df):.2%})")
    else:
        summary.append("Missing Values: 0")

    if pd.api.types.is_numeric_dtype(df[col]):
        summary.append("Type: Numeric")
        desc = df[col].describe()
        summary.append(f"Mean: {desc['mean']:.2f}, Std: {desc['std']:.2f}")
        summary.append(f"Min: {desc['min']:.2f}, Max: {desc['max']:.2f}")
    else:
        summary.append("Type: Categorical")
        unique_values = df[col].nunique()
        summary.append(f"Unique Values: {unique_values}")
        if unique_values < 10:
            summary.append(f"Values: {df[col].unique().tolist()}")

    return ' | '.join(summary)

def interactive_filter(file_path, output_file):
    """
    Interactively prompts the user to select columns from a CSV file and saves the selection.
    """
    try:
        df = pd.read_csv(file_path, comment='#')
        print(f"Loaded {file_path}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    selected_columns = []
    columns = df.columns.tolist()

    for i, col in enumerate(columns):
        print("\n" + "-"*50)
        print(f"Column {i+1}/{len(columns)}: \033[1m{col}\033[0m")
        summary = analyze_column_summary(df, col)
        print(summary)
        
        while True:
            choice = input("Action -> (p)ick | (x)discard | (q)uit: ").lower()
            if choice in ['p', 'x', 'q']:
                break
            print("Invalid input. Please enter 'p', 'x', or 'q'.")

        if choice == 'p':
            selected_columns.append(col)
            print(f"Picked: {col}")
        elif choice == 'x':
            print(f"Discarded: {col}")
        elif choice == 'q':
            print("Quitting interactive session.")
            break

    if not selected_columns:
        print("\nNo columns were selected. Exiting.")
        return

    print("\n" + "-"*50)
    print(f"Picked {len(selected_columns)} columns:")
    for col in selected_columns:
        print(f"- {col}")

    try:
        with open(output_file, 'w') as f:
            for col in selected_columns:
                f.write(f"{col}\n")
        print(f"\nSuccessfully saved selected columns to \033[1m{output_file}\033[0m")
    except Exception as e:
        print(f"\nError saving to file: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactively select columns from a CSV file.')
    parser.add_argument('file_path', type=str, help='Path to the input CSV file.')
    parser.add_argument('output_file', type=str, help='Path to save the list of selected columns.')
    args = parser.parse_args()

    interactive_filter(args.file_path, args.output_file)
