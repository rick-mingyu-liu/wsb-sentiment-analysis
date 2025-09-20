#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score a SQLite table using TWO HF models (sentiment + sarcasm), save all outputs to a folder,
and optionally evaluate against gold labels and generate matplotlib plots.

Examples:

# POSTS (text in 'title')
python score_sqlite_with_ensemble.py \
  --db wsb.sqlite --table reddit_raw_posts --textcol title --idcol post_id \
  --sentiment_dir ./deberta-financial --sarcasm_dir ./sarcasm_detector \
  --sentiment_label sentiment_gold --sarcasm_label sarcasm_gold \
  --outdir results_posts --write

# COMMENTS (text in 'body')
python score_sqlite_with_ensemble.py \
  --db wsb.sqlite --table reddit_raw_comments --textcol body --idcol comment_id \
  --sentiment_dir ./deberta-financial --sarcasm_dir ./sarcasm_detector \
  --sentiment_label sentiment_gold --sarcasm_label sarcasm_gold \
  --outdir results_comments --write
"""

import argparse, os, sys, sqlite3, re, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Score SQLite with sentiment+sarcasm ensemble and make plots.")
    p.add_argument("--db", required=True, help="Path to SQLite file")
    p.add_argument("--table", required=True, help="Table to read")
    p.add_argument("--textcol", required=True, help="Text column")
    p.add_argument("--idcol", default="", help="Optional id column")
    p.add_argument("--where", default="", help="Optional WHERE (omit 'WHERE')")
    p.add_argument("--limit", type=int, default=0, help="LIMIT (0=no limit)")
    p.add_argument("--sentiment_dir", required=True, help="HF sentiment model dir")
    p.add_argument("--sarcasm_dir", required=True, help="HF sarcasm model dir")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    p.add_argument("--max_length", type=int, default=256, help="Tokenizer max_length")
    p.add_argument("--outdir", required=True, help="Output folder for CSV/plots/metrics")
    p.add_argument("--write", action="store_true", help="Write predictions back to SQLite (table: model_predictions_ensemble)")
    p.add_argument("--sentiment_label", default="", help="Gold label column for sentiment (optional)")
    p.add_argument("--sarcasm_label", default="", help="Gold label column for sarcasm (optional)")
    return p.parse_args()

# ---------- Heuristics ----------
def contains_sarcasm_clues(text: str) -> bool:
    text_lower = text.lower()
    sarcasm_keywords = ["yeah right","sure","great job","nice job","amazing","love it",
                        "brilliant","fantastic","wonderful","oh great","perfect"]
    sarcasm_emojis   = ["🙄","😂","😉","🙂","🤣"]
    if any(kw in text_lower for kw in sarcasm_keywords): return True
    if any(e in text for e in sarcasm_emojis): return True
    if "..." in text or "!!!" in text: return True
    if re.search(r"i'd .* if", text_lower): return True
    if "just buy in. or don't" in text_lower: return True
    if "i am tipsy" in text_lower or "i'm drunk" in text_lower: return True
    return False

def bull_bear_counts(text_lower: str):
    bullish = ["to the moon","buy in","long","calls","bullish","diamond hands","🚀"]
    bearish = ["at the top","crash","bagholder","lose","drop","tank","losing"]
    b = sum(1 for c in bullish if c in text_lower)
    r = sum(1 for c in bearish if c in text_lower)
    return b, r

# ---------- HF helpers ----------
@torch.no_grad()
def hf_probs(model, tok, texts, device, max_length=256):
    enc = tok(list(map(str, texts)), padding=True, truncation=True,
              max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k,v in enc.items()}
    logits = model(**enc).logits
    return torch.softmax(logits, dim=-1).cpu().numpy()

# ---------- Plot helpers (matplotlib only) ----------
def plot_bar_counts(values, title, outpath):
    labels, counts = np.unique(values, return_counts=True)
    fig = plt.figure()
    plt.bar(labels, counts)
    plt.title(title); plt.ylabel("count"); plt.xticks(rotation=20, ha="right")
    plt.tight_layout(); fig.savefig(outpath); plt.close(fig)

def plot_box_by_group(scores, groups, title, outpath):
    # boxplot of scores grouped by categorical groups
    uniq = [g for g in np.unique(groups) if pd.notna(g)]
    data = [scores[groups==g] for g in uniq]
    fig = plt.figure()
    plt.boxplot(data, labels=uniq, vert=True)
    plt.title(title); plt.ylabel("log1p(score)"); plt.xticks(rotation=20, ha="right")
    plt.tight_layout(); fig.savefig(outpath); plt.close(fig)

def plot_confusion(true, pred, title, outpath):
    # simple confusion matrix using matplotlib only
    uniq = sorted(np.unique(np.concatenate([np.array(true), np.array(pred)])))
    idx = {l:i for i,l in enumerate(uniq)}
    cm = np.zeros((len(uniq), len(uniq)), dtype=int)
    for t,p in zip(true, pred):
        cm[idx[t], idx[p]] += 1
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(uniq))); ax.set_xticklabels(uniq, rotation=45, ha="right")
    ax.set_yticks(range(len(uniq))); ax.set_yticklabels(uniq)
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(len(uniq)):
        for j in range(len(uniq)):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); fig.savefig(outpath); plt.close(fig)

# ---------- Main ----------
def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if not Path(args.db).exists(): sys.exit(f"[ERROR] DB not found: {args.db}")
    if not Path(args.sentiment_dir).is_dir(): sys.exit(f"[ERROR] sentiment_dir not found")
    if not Path(args.sarcasm_dir).is_dir(): sys.exit(f"[ERROR] sarcasm_dir not found")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    # Load models
    sent_model = AutoModelForSequenceClassification.from_pretrained(args.sentiment_dir).to(device)
    sent_tok   = AutoTokenizer.from_pretrained(args.sentiment_dir)
    sarc_model = AutoModelForSequenceClassification.from_pretrained(args.sarcasm_dir).to(device)
    sarc_tok   = AutoTokenizer.from_pretrained(args.sarcasm_dir)

    id2label_sent = getattr(sent_model.config, "id2label", {0:"negative",1:"neutral",2:"positive"})
    id2label_sarc = getattr(sarc_model.config, "id2label", {0:"not sarcastic",1:"sarcastic"})

    # Read table
    con = sqlite3.connect(args.db)
    q = f"SELECT * FROM {args.table}"
    if args.where.strip(): q += f" WHERE {args.where}"
    if args.limit > 0: q += f" LIMIT {args.limit}"
    df = pd.read_sql_query(q, con)
    if df.empty: 
        print("[WARN] no rows."); con.close(); return
    if args.textcol not in df.columns:
        con.close(); sys.exit(f"[ERROR] text column '{args.textcol}' not found.")

    texts = df[args.textcol].fillna("").astype(str)

    # Predict
    sent_probs = hf_probs(sent_model, sent_tok, texts, device, args.max_length)   # Nx3
    sarc_probs = hf_probs(sarc_model, sarc_tok, texts, device, args.max_length)   # Nx2
    sent_idx = sent_probs.argmax(axis=1)
    sarc_idx = sarc_probs.argmax(axis=1)
    sent_pred = np.vectorize(lambda i: id2label_sent.get(int(i), str(int(i))))(sent_idx)
    sarc_pred = np.vectorize(lambda i: id2label_sarc.get(int(i), str(int(i))))(sarc_idx)

    # Fusion with heuristics
    final = []
    for text, s_label, s_idx_val, s_row, sarc_idx_val, sarc_row in zip(texts, sent_pred, sent_idx, sent_probs, sarc_idx, sarc_probs):
        low = text.lower()
        bull, bear = bull_bear_counts(low)
        sarc_score = float(sarc_row[1])
        raw_sarc = "sarcastic" if sarc_idx_val == 1 else "not sarcastic"
        if sarc_score < 0.1 and contains_sarcasm_clues(text):
            raw_sarc, sarc_score = "sarcastic", 0.95
        if bull > 0 and sarc_score < 0.99:
            raw_sarc = "not sarcastic"
        if contains_sarcasm_clues(text):
            raw_sarc = "sarcastic"

        if raw_sarc == "sarcastic":
            if s_label == "positive":
                label = "positive (sarcastic bullish)"
            elif s_label == "negative":
                label = "negative (sarcastic bearish)"
            else:
                label = "negative (sarcastic bearish)" if bear > bull else ("positive (sarcastic bullish)" if bull>0 else "neutral (sarcastic unclear)")
        else:
            if s_label == "positive": label = "positive"
            elif s_label == "negative": label = "negative"
            else: label = "positive" if bull>0 else ("negative" if bear>0 else "neutral")
        final.append(label)

    # Build output DF
    out_cols = [c for c in [args.idcol] if c and c in df.columns]
    out = df[out_cols].copy() if out_cols else pd.DataFrame(index=df.index)
    out["sentiment_pred"] = sent_pred
    out["sarcasm_pred"] = np.where(sarc_idx==1, "sarcastic", "not sarcastic")
    out["sentiment_proba_neg"] = sent_probs[:,0]
    out["sentiment_proba_neu"] = sent_probs[:,1]
    out["sentiment_proba_pos"] = sent_probs[:,2]
    out["sarcasm_proba_sarcastic"] = sarc_probs[:,1]
    out["final_interpretation"] = final

    # Save CSV
    csv_path = outdir / "ensemble_predictions.csv"
    out.to_csv(csv_path, index=False)
    print(f"[INFO] saved CSV -> {csv_path}")

    # Optional write-back
    if args.write:
        out.to_sql("model_predictions_ensemble", con, if_exists="replace", index=False)
        print("[INFO] wrote table -> model_predictions_ensemble")

    # ---------- EVAL + METRICS ----------
    metrics = {}
    if args.sentiment_label and args.sentiment_label in df.columns:
        try:
            from sklearn.metrics import classification_report
            rep = classification_report(df[args.sentiment_label].astype(str), out["sentiment_pred"].astype(str), output_dict=True, zero_division=0)
            metrics["sentiment_report"] = rep
            # confusion
            plot_confusion(
                df[args.sentiment_label].astype(str).values,
                out["sentiment_pred"].astype(str).values,
                "Sentiment Confusion Matrix",
                outdir / "sentiment_confusion.png"
            )
        except Exception as e:
            print(f"[WARN] sentiment eval failed: {e}")

    if args.sarcasm_label and args.sarcasm_label in df.columns:
        try:
            from sklearn.metrics import classification_report
            rep = classification_report(df[args.sarcasm_label].astype(str), out["sarcasm_pred"].astype(str), output_dict=True, zero_division=0)
            metrics["sarcasm_report"] = rep
            plot_confusion(
                df[args.sarcasm_label].astype(str).values,
                out["sarcasm_pred"].astype(str).values,
                "Sarcasm Confusion Matrix",
                outdir / "sarcasm_confusion.png"
            )
        except Exception as e:
            print(f"[WARN] sarcasm eval failed: {e}")

    # ---------- PLOTS ----------
    # class distributions
    plot_bar_counts(out["sentiment_pred"].values, "Sentiment Predictions (count)", outdir / "sentiment_counts.png")
    plot_bar_counts(out["sarcasm_pred"].values, "Sarcasm Predictions (count)", outdir / "sarcasm_counts.png")

    # score vs sentiment (if a 'score' column exists)
    if "score" in df.columns:
        # log1p to tame heavy tail, ignore negatives for log plot but show as 0
        score_log1p = np.log1p(np.clip(df["score"].fillna(0).astype(float).values, a_min=0, a_max=None))
        plot_box_by_group(score_log1p, out["sentiment_pred"].values, "Reddit Score vs Sentiment (log1p)", outdir / "score_by_sentiment.png")

    # save metrics JSON if any
    if metrics:
        with open(outdir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] saved metrics -> {outdir / 'metrics.json'}")

    con.close()
    print(f"[DONE] All artifacts in: {outdir.resolve()}")

if __name__ == "__main__":
    main()
