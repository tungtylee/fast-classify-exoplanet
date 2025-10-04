import pandas as pd
import json
import argparse
from collections import defaultdict

def calculate_scores(gt_csv, pred_csv, config_path):
    """
    Compares a prediction CSV to a ground truth CSV and calculates classification metrics.
    """
    # 1. Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        id_col = config['id_column']
        target_col = config['target_column']
        mapping = config['mapping']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Could not load or parse config file '{config_path}'. Details: {e}")
        return

    # 2. Load CSVs
    try:
        gt_df = pd.read_csv(gt_csv)
        pred_df = pd.read_csv(pred_csv)
    except FileNotFoundError as e:
        print(f"Error loading CSV file: {e}")
        return

    # 3. Merge dataframes on the specified ID column
    merged_df = pd.merge(
        gt_df[[id_col, target_col]],
        pred_df[[id_col, target_col]],
        on=id_col,
        how='inner',
        suffixes=('_gt', '_pred')
    )

    if merged_df.empty:
        print(f"Error: No common rows found between the two CSVs on column '{id_col}'. Cannot score.")
        return

    # 4. Calculate confusion matrix based on the mapping
    scores = defaultdict(int)
    positive_label = 'positive'
    negative_label = 'negative'

    target_gt = f"{target_col}_gt"
    target_pred = f"{target_col}_pred"

    for _, row in merged_df.iterrows():
        gt_val = str(row[target_gt])
        pred_val = str(row[target_pred])

        gt_class = mapping.get(gt_val, 'dontcare')
        pred_class = mapping.get(pred_val, 'dontcare')

        # Per user request, only ignore if the ground truth is 'dontcare'
        if gt_class == 'dontcare':
            scores['ignored'] += 1
            continue

        # If gt is positive, a 'positive' prediction is a TP, anything else is an FN.
        if gt_class == positive_label:
            if pred_class == positive_label:
                scores['TP'] += 1
            else:  # Includes 'negative' and 'dontcare' predictions
                scores['FN'] += 1
        # If gt is negative, a 'positive' prediction is an FP, anything else is a TN.
        elif gt_class == negative_label:
            if pred_class == positive_label:
                scores['FP'] += 1
            else:  # Includes 'negative' and 'dontcare' predictions
                scores['TN'] += 1

    # 5. Calculate and print metrics
    tp, fp, tn, fn = scores['TP'], scores['FP'], scores['TN'], scores['FN']
    total_scored = tp + fp + tn + fn

    if total_scored == 0:
        print("No scorable pairs found after applying the mapping. All pairs were mapped to 'dontcare'.")
        return

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total_scored

    print("--- Scoring Results ---")
    print(f"Scored on column: '{target_col}'")
    print(f"Total items matched by ID '{id_col}': {len(merged_df)}")
    print(f"Total items scored (after 'dontcare' filter): {total_scored}")
    print(f"Ignored items: {scores['ignored']}")
    print("\n--- Confusion Matrix ---")
    print(f"{'':17} | {'Predicted Positive':^18} | {'Predicted Negative':^18}")
    print("-" * 60)
    print(f"{'Actual Positive':<17} | {f'{tp:^6} (TP)':^18} | {f'{fn:^6} (FN)':^18}")
    print(f"{'Actual Negative':<17} | {f'{fp:^6} (FP)':^18} | {f'{tn:^6} (TN)':^18}")
    print("\n--- Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a prediction CSV against a ground truth CSV based on a JSON config.')
    parser.add_argument('gt_csv', help='Path to the ground truth CSV file.')
    parser.add_argument('pred_csv', help='Path to the prediction CSV file.')
    parser.add_argument('config_json', nargs='?', default='scorer_conf.json', help='Path to the scoring config JSON file (default: scorer_conf.json).')
    args = parser.parse_args()

    calculate_scores(args.gt_csv, args.pred_csv, args.config_json)
