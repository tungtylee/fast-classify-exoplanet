#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, joblib
import pandas as pd
import numpy as np

def median_impute(X, med):
    X = X.astype(float)
    idx = np.where(np.isnan(X))
    if idx[0].size:
        X[idx] = np.take(med, idx[1])
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="訓練後的 ckpt，例如 xgb_model.joblib")
    ap.add_argument("--inference_csv", required=True, help="要推論的 CSV")
    ap.add_argument("--out_csv", required=True, help="輸出檔名，例如 test_set_pred.csv")
    ap.add_argument("--id_column", default="toi")
    ap.add_argument("--tid_column", default="tid")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    pack = joblib.load(args.model)  # {"model", "feature_cols", "medians"}
    model = pack["model"]
    feats = pack["feature_cols"]
    med = pack["medians"]

    df = pd.read_csv(args.inference_csv)
    X = df[feats].to_numpy(dtype=float)
    X = median_impute(X, med)

    prob = model.predict_proba(X)[:, 1]
    pred = np.where(prob >= args.threshold, "positive", "negative")

    out = pd.DataFrame({
        "toi": df[args.id_column],
        "tid": df[args.tid_column],
        "tfopwg_disp": pred
    })
    out.to_csv(args.out_csv, index=False)

    out2 = out.copy()
    out2["score_positive"] = prob
    rich = os.path.splitext(args.out_csv)[0] + "_with_score.csv"
    out2.to_csv(rich, index=False)

if __name__ == "__main__":
    main()
