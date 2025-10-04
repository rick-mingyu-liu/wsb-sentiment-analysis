#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge model predictions into a labeled sample using NORMALIZED IDs (lowercase, strip t1_/t3_),
then evaluate and plot.

Example:
python merge_and_compare_normalized.py \
  --sample human_check_sample.csv \
  --preds results_comments/ensemble_predictions.csv \
  --gold-col gold_sentiment \
  --pred-col sentiment_label \
  --sample-id comment_id \
  --pred-id comment_id \
  --outdir artifacts_from_compare
"""
import argparse, os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def norm_id(series: pd.Series) -> pd.Series:
    """Lowercase, trim, and strip leading t{digit}_ prefixes (e.g., t1_, t3_)."""
    s = series.astype(str).str.strip().str.lower()
    s = s.str.replace(r"^t[0-9]_", "", regex=True)
    return s

def map_to_int(series):
    # Try numeric first
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().any():
        return s_num.map({-1:-1, 0:-1, 1:0, 2:1})
    # Strings
    s = series.astype(str).str.lower()
    mapping = {
        "label_0": -1, "negative": -1, "neg": -1,
        "label_1": 0,  "neutral": 0,   "neu": 0,
        "label_2": 1,  "positive": 1,  "pos": 1,
        "positive (sarcastic bullish)": 1,
        "negative (sarcastic bearish)": -1,
        "neutral (sarcastic unclear)": 0,
    }
    out = s.map(lambda x: mapping.get(x, np.nan))
    has_num = s.str.contains(r"\d", regex=True, na=False)
    if has_num.any():
        nums = s[has_num].str.extract(r"(-?\d+)")[0].astype(float).map({0:-1, 1:0, 2:1})
        out.loc[has_num] = nums
    return out

def guess_id(df):
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["comment_id","post_id","id","link_id","id_str"]):
            return c
    return None

def guess_pred_col(df):
    for name in ["sentiment_pred","sentiment_label","label","sentiment","predicted_label"]:
        for c in df.columns:
            if c.lower() == name:
                return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, help="Path to human_check_sample.csv")
    ap.add_argument("--preds", required=True, help="Path to ensemble_predictions.csv")
    ap.add_argument("--gold-col", default="gold_sentiment")
    ap.add_argument("--pred-col", default=None, help="Prediction column in preds CSV (e.g., sentiment_label)")
    ap.add_argument("--sample-id", default=None, help="ID column in sample (e.g., comment_id)")
    ap.add_argument("--pred-id", default=None, help="ID column in preds (e.g., comment_id)")
    ap.add_argument("--outdir", default="artifacts_from_compare")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    S = pd.read_csv(args.sample)
    P = pd.read_csv(args.preds)

    sid = args.sample_id or guess_id(S)
    pid = args.pred_id or guess_id(P)
    pcol = args.pred_col or guess_pred_col(P)

    if sid is None or pid is None:
        raise SystemExit(f"[ERROR] Could not detect ID columns (sample-id={sid}, preds-id={pid}). Pass --sample-id/--pred-id.")
    if pcol is None:
        raise SystemExit(f"[ERROR] Could not detect prediction column in {args.preds}. Pass --pred-col.")

    # Normalize IDs and merge
    S["_id_norm"] = norm_id(S[sid])
    P["_id_norm"] = norm_id(P[pid])
    keep = ["_id_norm", pcol]
    M = S.merge(P[keep].rename(columns={pcol: "sentiment_pred"}), on="_id_norm", how="left").drop(columns=["_id_norm"])

    # Save merged sample with predictions filled
    merged_csv = os.path.join(args.outdir, "human_check_sample_with_preds.csv")
    M.to_csv(merged_csv, index=False)

    # Evaluate
    y_true = pd.to_numeric(M[args.gold_col], errors="coerce")
    y_pred = map_to_int(M["sentiment_pred"])
    mask = y_true.notna() & y_pred.notna()
    gold = y_true[mask].astype(int)
    pred = y_pred[mask].astype(int)

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
    labels = [-1,0,1]; names = ["neg","neu","pos"]
    prec, rec, f1, sup = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    acc = accuracy_score(gold, pred)
    macro_f1 = float(np.mean(f1))

    metrics = {
        "n_eval": int(len(gold)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": {names[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])} for i in range(3)},
        "sample_id": sid, "pred_id": pid, "pred_col": "sentiment_pred",
        "merged_sample_path": merged_csv
    }
    with open(os.path.join(args.outdir, "metrics_report.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(gold, pred, labels=labels)
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    plt.title("Confusion Matrix (gold vs model)"); plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=10)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "confusion_matrix.png"), dpi=150); plt.close(fig)

    # Distribution comparison
    def counts(v): return {k:int((v==k).sum()) for k in labels}
    c_true = counts(gold); c_pred = counts(pred)
    fig = plt.figure()
    x = np.arange(len(labels)); w = 0.35
    plt.bar(x - w/2, [c_true[k] for k in labels], width=w, label="human")
    plt.bar(x + w/2, [c_pred[k] for k in labels], width=w, label="model")
    plt.xticks(x, names); plt.ylabel("count"); plt.title("Label distribution (human vs model)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "label_distribution_compare.png"), dpi=150); plt.close(fig)

    # Error examples (preserve text columns if present)
    txt_cols = [c for c in S.columns if any(k in c.lower() for k in ["title","selftext","body","text"])]
    keep_cols = [sid, args.gold_col, "sentiment_pred"] + txt_cols[:2]
    M.loc[mask].loc[gold.index[gold.values != pred.values], keep_cols].head(50)\
        .to_csv(os.path.join(args.outdir, "error_examples.csv"), index=False)

    print(f"[OK] Evaluated {len(gold)} rows. Artifacts in {args.outdir}")
    print(" - human_check_sample_with_preds.csv")
    print(" - metrics_report.json")
    print(" - confusion_matrix.png")
    print(" - label_distribution_compare.png")
    print(" - error_examples.csv")

if __name__ == "__main__":
    main()
