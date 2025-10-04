#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost classifier for TOI table with CV & early stopping

標籤由 scorer_conf.json 控制：
  CP/KP -> positive, FP/FA -> negative, PC/APC -> dontcare(不參與訓練)
輸入：small_filtered2.csv 類似格式
輸出：三欄 (toi, tid, tfopwg_disp)；另輸出 *_with_score.csv（含 score_positive）
"""

import argparse, json, os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import joblib

# ------------------------- Utils -------------------------

def load_conf(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def numeric_feature_cols(df: pd.DataFrame, id_col: str, target_col: str, explicit: List[str] = None) -> List[str]:
    if explicit:
        for c in explicit:
            if c not in df.columns:
                raise ValueError(f"--feats 指定的欄位不存在：{c}")
        return explicit
    drop = {id_col, target_col, "tid"}
    feats = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    if not feats:
        raise ValueError("找不到可用的數值特徵欄位！請確認 CSV 內有數值欄位（如 st_* / pl_*）。")
    return feats

def map_labels_for_training(df: pd.DataFrame, conf: Dict) -> Tuple[pd.DataFrame, np.ndarray]:
    target = conf["target_column"]
    mapping = conf["mapping"]
    mapped = df[target].astype(str).map(mapping)
    mask = mapped.isin(["positive", "negative"])
    train_df = df.loc[mask].copy()
    y = mapped.loc[mask].map({"negative":0, "positive":1}).values
    return train_df, y

def median_impute(X: np.ndarray, medians=None):
    if medians is None:
        medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    if inds[0].size:
        X[inds] = np.take(medians, inds[1])
    return X, medians

def make_model(args, spw: float = 1.0) -> XGBClassifier:
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
        scale_pos_weight=spw
    )

def metrics_dict(y_true, y_prob, thr: float) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    out = {}
    # 分數可能遇到單一類別時要保護
    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["auc"] = np.nan
    out["acc"] = accuracy_score(y_true, y_pred)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    return out

# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scorer_conf.json")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--inference_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model_out", default="xgb_model.joblib")

    # features & labels
    ap.add_argument("--feats", default="", help="逗號分隔的特徵欄位清單；留空則自動挑數值欄位")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)

    # imbalance
    ap.add_argument("--scale_pos_weight", type=float, default=0.0, help=">0 則使用此值；=0 且 --balance auto 時自動計算")
    ap.add_argument("--balance", choices=["none","auto"], default="auto")

    # xgb params
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--min_child_weight", type=float, default=1.0)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--reg_alpha", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--tree_method", default="hist")
    ap.add_argument("--max_bin", type=int, default=256)

    # CV / early stopping
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--early_stopping_rounds", type=int, default=50)
    ap.add_argument("--eval_metric", default="auc")
    ap.add_argument("--save_best_by", choices=["auc","acc","f1","precision","recall"], default="auc")
    ap.add_argument("--cv_only", action="store_true")

    args = ap.parse_args()

    conf = load_conf(args.config)
    id_col = conf["id_column"]
    target_col = conf["target_column"]

    df_train_all = pd.read_csv(args.train_csv)
    df_infer = pd.read_csv(args.inference_csv)

    feats = numeric_feature_cols(
        df_train_all, id_col, target_col,
        explicit=[c.strip() for c in args.feats.split(",")] if args.feats.strip() else None
    )

    # 準備訓練資料（排除 dontcare）
    df_train, y = map_labels_for_training(df_train_all, conf)
    X_all = df_train[feats].to_numpy(dtype=float)

    # class imbalance
    spw = args.scale_pos_weight
    if args.balance == "auto" and spw <= 0:
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        spw = float(neg) / max(1.0, float(pos))

    # ---- CV ----
    best_pack = None
    if args.cv:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_rows = []
        best_score = -np.inf

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y), 1):
            X_tr, y_tr = X_all[tr_idx], y[tr_idx]
            X_va, y_va = X_all[va_idx], y[va_idx]

            # 中位數填補：以 training fold 的統計填補 train/valid
            X_tr, med = median_impute(X_tr, None)
            X_va, _ = median_impute(X_va, med)

            model = make_model(args, spw)
            model.set_params(**{"eval_metric": args.eval_metric})

            model.set_params(**{"eval_metric": args.eval_metric})
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )



            # 以最佳迭代的預測算分
            y_va_prob = model.predict_proba(X_va)[:, 1]
            m = metrics_dict(y_va, y_va_prob, thr=args.threshold)
            m["fold"] = fold
            fold_rows.append(m)

            # 依指定指標挑最佳 fold 的模型（保存 medians/feats）
            crit = m[args.save_best_by]
            if np.isnan(crit):
                crit = -np.inf
            if crit > best_score:
                best_score = crit
                best_pack = {"model": model, "feature_cols": feats, "medians": med}

        cv_df = pd.DataFrame(fold_rows)
        cv_df.loc["mean"] = cv_df.mean(numeric_only=True)
        cv_path = os.path.splitext(args.model_out)[0] + "_cv_metrics.csv"
        cv_df.to_csv(cv_path, index=False)

    # ---- 最終訓練 or 用 CV 最佳模型 ----
    if best_pack is None:
        # 全資料再訓練一個（常見做法）
        X_all_imp, med_all = median_impute(X_all, None)
        model = make_model(args, spw)
        model.set_params(**{"eval_metric": args.eval_metric})
        model.fit(X_all_imp, y, eval_set=[(X_all_imp, y)], verbose=False)
        
        pack = {"model": model, "feature_cols": feats, "medians": med_all}
    else:
        pack = best_pack

    # 存模型
    joblib.dump(pack, args.model_out)

    if args.cv_only:
        return

    # ---- 推論 ----
    X_inf = df_infer[feats].to_numpy(dtype=float)
    X_inf, _ = median_impute(X_inf, medians=pack["medians"])
    prob = pack["model"].predict_proba(X_inf)[:, 1]
    pred = np.where(prob >= args.threshold, "positive", "negative")

    out = pd.DataFrame({
        "toi": df_infer[id_col],
        "tid": df_infer["tid"],
        "tfopwg_disp": pred
    })
    out.to_csv(args.out_csv, index=False)

    out2 = out.copy()
    out2["score_positive"] = prob
    rich = os.path.splitext(args.out_csv)[0] + "_with_score.csv"
    out2.to_csv(rich, index=False)

if __name__ == "__main__":
    main()