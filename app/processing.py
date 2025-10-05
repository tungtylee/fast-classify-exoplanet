import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import shap
import matplotlib.pyplot as plt

def split_data(df, train_ratio=0.8, seed=42):
    # ... (existing function)
    if 'tid' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'tid' column for splitting.")
    
    unique_tids = df['tid'].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_tids)
    
    train_split_idx = int(len(unique_tids) * train_ratio)
    train_tids = unique_tids[:train_split_idx]
    
    train_df = df[df['tid'].isin(train_tids)].copy()
    test_df = df[~df['tid'].isin(train_tids)].copy()
    
    return train_df, test_df

def create_binary_dataset(df, target_column, mapping):
    # ... (existing function)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    df_processed = df.copy()
    df_processed['target_binary'] = df_processed[target_column].map(mapping)
    valid_targets = ['positive', 'negative']
    df_processed = df_processed[df_processed['target_binary'].isin(valid_targets)]
    df_processed['target_binary'] = df_processed['target_binary'].apply(lambda x: 1 if x == 'positive' else 0)
    return df_processed

def train_xgboost_cv(df, features, target_col, n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0, reg_alpha=0.0, seed=42, folds=5, eval_metric='auc'):
    
    X = df[features].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=int)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_metrics = []
    best_model = None
    best_score = -np.inf
    
    medians = np.nanmedian(X, axis=0)
    X_imputed = np.nan_to_num(X, nan=medians)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_imputed, y), 1):
        X_train, y_train = X_imputed[train_idx], y[train_idx]
        X_val, y_val = X_imputed[val_idx], y[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=subsample, colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
            reg_lambda=reg_lambda, reg_alpha=reg_alpha, random_state=seed, n_jobs=-1,
            use_label_encoder=False, eval_metric=eval_metric
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        
        y_val_prob = model.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_prob > 0.5).astype(int)

        metrics = {
            'fold': fold, 'auc': roc_auc_score(y_val, y_val_prob),
            'accuracy': accuracy_score(y_val, y_val_pred), 'f1': f1_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred), 'recall': recall_score(y_val, y_val_pred)
        }
        fold_metrics.append(metrics)

        if metrics[eval_metric] > best_score:
            best_score = metrics[eval_metric]
            best_model = model

    metrics_df = pd.DataFrame(fold_metrics)
    return best_model, metrics_df, medians

def run_inference(model, df, features, medians, threshold=0.5):
    X_infer = df[features].to_numpy(dtype=float)
    X_infer = np.nan_to_num(X_infer, nan=medians)
    probs = model.predict_proba(X_infer)[:, 1]
    predictions = (probs >= threshold).astype(int)
    results_df = df.copy()
    results_df['prediction_label'] = predictions
    results_df['prediction_score'] = probs
    return results_df

def generate_shap_summary_plot(model, df, features, medians):
    """
    Generates a SHAP summary plot (beeswarm).
    """
    X = df[features].to_numpy(dtype=float)
    X_imputed = np.nan_to_num(X, nan=medians)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_imputed)
    
    # Create a figure and plot on it
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, pd.DataFrame(X_imputed, columns=features), show=False, plot_size=None)
    plt.tight_layout()
    return fig
