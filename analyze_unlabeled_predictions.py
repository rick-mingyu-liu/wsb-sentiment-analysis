#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def safe_get(df, names, default=np.nan):
    for n in names:
        if n in df.columns:
            return df[n].astype(float)
    return pd.Series([default]*len(df))

def pick_datetime(df):
    for c in df.columns:
        cl = c.lower()
        if "created" in cl or cl.endswith("_dt") or cl.endswith("_date") or "timestamp" in cl:
            try:
                return pd.to_datetime(df[c], errors="coerce", utc=True).dt.date
            except Exception:
                pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--outdir", default="unlabeled_analysis")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    P = pd.read_csv(args.preds)

    def col(names): return next((n for n in names if n in P.columns), None)

    neg = col(["sentiment_proba_neg","prob_neg","p_neg"])
    neu = col(["sentiment_proba_neu","prob_neu","p_neu"])
    pos = col(["sentiment_proba_pos","prob_pos","p_pos"])
    if not all([neg, neu, pos]):
        raise SystemExit("[ERROR] Could not find probability columns in predictions.")

    p_neg = P[neg].astype(float); p_neu = P[neu].astype(float); p_pos = P[pos].astype(float)
    s_sarc = safe_get(P, ["sarcasm_proba_sarcastic","p_sarcastic","sarcasm_prob"], default=np.nan)

    # Core scores (no human labels required)
    sentiment_score = p_pos - p_neg
    confidence_max = np.max(np.stack([p_neg, p_neu, p_pos], axis=1), axis=1)
    neutrality_bias = p_neu
    sorted_probs = np.sort(np.stack([p_neg, p_neu, p_pos], axis=1), axis=1)
    margin = sorted_probs[:,-1] - sorted_probs[:,-2]
    score_log1p = np.log1p(P["score"].clip(lower=0)) if "score" in P.columns else pd.Series([np.nan]*len(P))

    # Save per-row analysis
    out = P.copy()
    out["sentiment_score"] = sentiment_score
    out["sarcasm_score"] = s_sarc
    out["confidence_max"] = confidence_max
    out["neutrality_bias"] = neutrality_bias
    out["margin"] = margin
    if "score" in P.columns: out["log1p_score"] = score_log1p
    out_csv = os.path.join(args.outdir, "per_row_with_scores.csv"); out.to_csv(out_csv, index=False)

    # Simple hist helper
    def hist(series, title, fname, bins=50):
        fig = plt.figure()
        plt.hist(pd.Series(series).dropna(), bins=bins)
        plt.title(title); plt.xlabel(title); plt.ylabel("count")
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir, fname), dpi=150); plt.close(fig)

    # Plots
    hist(sentiment_score, "Sentiment Score (P_pos - P_neg)", "hist_sentiment_score.png")
    hist(s_sarc, "Sarcasm Score (P_sarcastic)", "hist_sarcasm_score.png")
    hist(neutrality_bias, "Neutrality Bias (P_neu)", "hist_neutrality_bias.png")
    hist(confidence_max, "Max Class Probability (confidence)", "hist_confidence_max.png")
    hist(margin, "Margin (max prob - 2nd max)", "hist_margin.png")

    # Scatter: sentiment vs sarcasm
    fig = plt.figure()
    plt.scatter(sentiment_score, s_sarc, s=6, alpha=0.4)
    plt.xlabel("Sentiment Score (Ppos - Pneg)"); plt.ylabel("Sarcasm Score (Psarcastic)")
    plt.title("Sentiment vs Sarcasm"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "scatter_sentiment_vs_sarcasm.png"), dpi=150); plt.close(fig)

    # Engagement stratification (if score exists)
    if "score" in P.columns:
        q = pd.qcut(score_log1p.fillna(0), q=4, labels=["Q1","Q2","Q3","Q4"])
        groups = [pd.Series(sentiment_score)[q==lab].dropna() for lab in ["Q1","Q2","Q3","Q4"]]
        fig = plt.figure()
        plt.boxplot(groups, labels=["Q1","Q2","Q3","Q4"])
        plt.title("Sentiment Score by Engagement Quartile (log1p(score))")
        plt.ylabel("Sentiment Score"); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "box_sentiment_by_engagement.png"), dpi=150); plt.close(fig)

    # Optional daily rollups (if a date/timestamp column exists)
    dt = pick_datetime(P)
    if dt is not None:
        agg = pd.DataFrame({"date": dt})
        w = score_log1p.fillna(1.0) if "score" in P.columns else pd.Series([1.0]*len(P))
        agg["sentiment_score"] = sentiment_score
        agg["sarcasm_score"] = s_sarc
        agg["neutrality_bias"] = neutrality_bias
        grp = pd.concat([agg, w.rename("w")], axis=1).groupby("date")
        daily = pd.DataFrame({
            "sentiment_score_mean": grp.apply(lambda g: np.average(g["sentiment_score"], weights=g["w"])),
            "sarcasm_score_mean": grp.apply(lambda g: np.average(g["sarcasm_score"], weights=g["w"])),
            "neutrality_bias_mean": grp.apply(lambda g: np.average(g["neutrality_bias"], weights=g["w"])),
            "n_items": grp.size(),
            "w_mean": grp["w"].mean()
        }).reset_index()
        daily.to_csv(os.path.join(args.outdir, "daily_unlabeled_features.csv"), index=False)

        def lineplot(x, y, title, fname):
            fig = plt.figure()
            plt.plot(x, y); plt.title(title); plt.xlabel("date"); plt.xticks(rotation=45, ha="right")
            plt.tight_layout(); plt.savefig(os.path.join(args.outdir, fname), dpi=150); plt.close(fig)

        lineplot(daily["date"], daily["sentiment_score_mean"], "Daily Sentiment Score (weighted mean)", "daily_sentiment_score.png")
        lineplot(daily["date"], daily["sarcasm_score_mean"], "Daily Sarcasm Score (weighted mean)", "daily_sarcasm_score.png")
        lineplot(daily["date"], daily["neutrality_bias_mean"], "Daily Neutrality Bias (weighted mean)", "daily_neutrality_bias.png")

    print("[OK] Analysis complete:", args.outdir)

if __name__ == "__main__":
    main()
