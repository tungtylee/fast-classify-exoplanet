#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, joblib
import numpy as np, pandas as pd
import shap
import matplotlib.pyplot as plt

def median_impute(X, med):
    X = X.copy().astype(float)
    inds = np.where(np.isnan(X))
    if inds[0].size:
        X[inds] = np.take(med, inds[1])
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgb_model.joblib")
    ap.add_argument("--train_csv", required=True)   # 用來取特徵名與做 SHAP（只要有同欄位即可）
    ap.add_argument("--id_column", default="toi")
    ap.add_argument("--target_column", default="tfopwg_disp")
    ap.add_argument("--out_prefix", default="shap_report")
    args = ap.parse_args()

    pack = joblib.load(args.model)
    model = pack["model"]
    feats = pack["feature_cols"]
    med = pack["medians"]

    df = pd.read_csv(args.train_csv)
    X = df[feats].to_numpy(dtype=float)
    X = median_impute(X, med)

    # 用 TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # (n_samples, n_features)
    base_value = explainer.expected_value

    # 1) 匯出 mean(|SHAP|) CSV
    mean_abs = np.nanmean(np.abs(shap_values), axis=0)
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs}) \
            .sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(f"{args.out_prefix}_importance.csv", index=False)

    # 2) Bar 圖（mean|SHAP|）
    topk = min(20, len(feats))
    top = imp.head(topk)
    plt.figure(figsize=(8, max(4, topk*0.35)))
    plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    plt.xlabel("mean(|SHAP value|)")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_bar.png", dpi=160)
    plt.close()

    # 3) Beeswarm 摘要圖（全特徵或前 20）
    # shap.summary_plot 需要 dataframe 以顯示特徵值分佈
    shap_df = pd.DataFrame(X, columns=feats)
    shap.summary_plot(shap_values, shap_df, max_display=topk, show=False)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    # 4) 依賴圖：對前 3 名重要特徵各出一張
    for f in top["feature"].head(3).tolist():
        shap.dependence_plot(f, shap_values, shap_df, show=False)
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_depend_{f}.png", dpi=160, bbox_inches="tight")
        plt.close()

    print("Saved:",
          f"{args.out_prefix}_importance.csv",
          f"{args.out_prefix}_bar.png",
          f"{args.out_prefix}_beeswarm.png",
          "and up to 3 dependence plots.")

if __name__ == "__main__":
    main()