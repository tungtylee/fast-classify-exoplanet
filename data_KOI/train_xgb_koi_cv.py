#!/usr/bin/env python3
"""XGBoost classifier for KOI cumulative catalog with CV & inference support."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

CONFIG_TYPE = Mapping[str, object]


def load_conf(path: str) -> CONFIG_TYPE:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric dtype (in a copy)."""

    numeric = df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return numeric


def numeric_feature_cols(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
    explicit: Sequence[str] | None = None,
    extra_drop: Iterable[str] | None = None,
) -> List[str]:
    """Identify usable numeric feature columns for training."""

    if explicit:
        missing = [col for col in explicit if col not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {', '.join(missing)}")
        return list(explicit)

    drop = {id_col, target_col}
    if extra_drop:
        drop.update(extra_drop)

    features: List[str] = []
    for col in df.columns:
        if col in drop:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            features.append(col)

    if not features:
        raise ValueError("No usable numeric feature columns detected; consider specifying --feats explicitly.")

    return features


def map_labels_for_training(df: pd.DataFrame, conf: CONFIG_TYPE) -> Tuple[pd.DataFrame, np.ndarray]:
    target = conf["target_column"]
    mapping = conf["mapping"]
    mapped = df[target].astype(str).str.strip().map(mapping)
    mask = mapped.isin(["positive", "negative"])
    train_df = df.loc[mask].copy()
    y = mapped.loc[mask].map({"negative": 0, "positive": 1}).to_numpy()
    return train_df, y


def median_impute(X: np.ndarray, medians: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if medians is None:
        medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    if inds[0].size:
        X[inds] = np.take(medians, inds[1])
    return X, medians


def make_model(args: argparse.Namespace, pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        gamma=args.gamma,
        objective="binary:logistic",
        tree_method=args.tree_method,
        max_bin=args.max_bin if args.max_bin > 0 else None,
        random_state=args.seed,
        n_jobs=0,
        scale_pos_weight=pos_weight,
    )


def metrics_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out: Dict[str, float] = {}
    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["auc"] = float("nan")
    out["acc"] = accuracy_score(y_true, y_pred)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    return out


def prepare_matrix(df: pd.DataFrame, feats: Sequence[str]) -> np.ndarray:
    numeric = _to_numeric(df, feats)
    return numeric.to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scorer_conf.json")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--inference_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--model_out", default="xgb_koi_model.joblib")

    parser.add_argument("--feats", default="", help="Comma-separated feature list; auto-detect numeric columns when empty.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--scale_pos_weight", type=float, default=0.0, help=">0 uses the provided value; 0 with --balance auto computes from data.")
    parser.add_argument("--balance", choices=["none", "auto"], default="auto")

    parser.add_argument("--n_estimators", type=int, default=600)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--tree_method", default="hist")
    parser.add_argument("--max_bin", type=int, default=256)

    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    parser.add_argument("--eval_metric", default="auc")
    parser.add_argument("--save_best_by", choices=["auc", "acc", "f1", "precision", "recall"], default="auc")
    parser.add_argument("--cv_only", action="store_true")

    args = parser.parse_args()

    conf = load_conf(args.config)
    id_col = conf["id_column"]
    target_col = conf["target_column"]

    df_train_all = pd.read_csv(args.train_csv)
    df_infer = pd.read_csv(args.inference_csv)

    feats = numeric_feature_cols(
        df_train_all,
        id_col=id_col,
        target_col=target_col,
        explicit=[c.strip() for c in args.feats.split(",") if c.strip()],
        extra_drop=["kepler_name"],
    )

    df_train, y = map_labels_for_training(df_train_all, conf)
    X_all = prepare_matrix(df_train, feats)

    pos_weight = args.scale_pos_weight
    if args.balance == "auto" and pos_weight <= 0:
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        pos_weight = neg / max(1.0, pos)

    best_pack = None
    best_score = -np.inf
    cv_records: List[Dict[str, float]] = []

    if args.cv:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X_all, y), start=1):
            X_train, y_train = X_all[train_idx], y[train_idx]
            X_valid, y_valid = X_all[valid_idx], y[valid_idx]

            X_train, medians = median_impute(X_train, None)
            X_valid, _ = median_impute(X_valid, medians)

            model = make_model(args, pos_weight)
            model.set_params(eval_metric=args.eval_metric)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

            y_valid_prob = model.predict_proba(X_valid)[:, 1]
            metrics = metrics_dict(y_valid, y_valid_prob, threshold=args.threshold)
            metrics["fold"] = fold_idx
            cv_records.append(metrics)

            crit = metrics.get(args.save_best_by, float("nan"))
            if np.isnan(crit):
                crit = -np.inf
            if crit > best_score:
                best_score = crit
                best_pack = {"model": model, "feature_cols": feats, "medians": medians}

        if cv_records:
            cv_df = pd.DataFrame(cv_records)
            cv_df.loc["mean"] = cv_df.mean(numeric_only=True)
            cv_path = os.path.splitext(args.model_out)[0] + "_cv_metrics.csv"
            cv_df.to_csv(cv_path, index=False)

    if best_pack is None:
        X_all_imp, med_all = median_impute(X_all, None)
        model = make_model(args, pos_weight)
        model.set_params(eval_metric=args.eval_metric)
        model.fit(X_all_imp, y, eval_set=[(X_all_imp, y)], verbose=False)
        pack = {"model": model, "feature_cols": feats, "medians": med_all}
    else:
        pack = best_pack

    joblib.dump(pack, args.model_out)

    if args.cv_only:
        return

    X_inf = prepare_matrix(df_infer, feats)
    X_inf, _ = median_impute(X_inf, medians=pack["medians"])
    prob = pack["model"].predict_proba(X_inf)[:, 1]
    pred = np.where(prob >= args.threshold, "positive", "negative")

    out = pd.DataFrame({
        "kepid": df_infer[id_col],
        "kepler_name": df_infer.get("kepler_name", pd.Series(["" for _ in range(len(df_infer))])),
        "koi_disposition": pred,
    })
    out.to_csv(args.out_csv, index=False)

    enriched = out.copy()
    enriched["score_positive"] = prob
    enriched_path = os.path.splitext(args.out_csv)[0] + "_with_score.csv"
    enriched.to_csv(enriched_path, index=False)


if __name__ == "__main__":
    main()
