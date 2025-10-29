#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentiment + Sarcasm scorer.
- Outputs a single decimal in [0,1] (0=very negative, 0.5=neutral, 1=very positive).
- Can score raw texts or a CSV.
- Optional 'eval' mode: merge with a human-labeled CSV and report metrics + plots.

CLI examples:
1) Score texts:
   python combined_scorer.py score \
     --sent ./deberta-financial --sarc ./sarcasm_detector \
     --texts "Love the earnings beat" "Guidance cut; not great."

2) Score a CSV:
   python combined_scorer.py score \
     --sent ./deberta-financial --sarc ./sarcasm_detector \
     --in-csv human_checked_sample.csv --text-col body --id-col cid --out scored.csv

3) Evaluate against human_pred in the same CSV (creates plots in outdir):
   python combined_scorer.py eval \
     --sent ./deberta-financial --sarc ./sarcasm_detector \
     --human-csv human_checked_sample.csv --text-col body --id-col cid \
     --neg-t 0.40 --pos-t 0.60 --outdir artifacts_combo
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Union
import argparse, os, json

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------- Config & Model ----------------
@dataclass
class ComboCfg:
    sent_source: str
    sarc_source: str
    sent_sub: Optional[str] = None
    sarc_sub: Optional[str] = None
    max_len: int = 128
    device: Optional[str] = None            # "mps" | "cuda" | "cpu" | None(auto)
    w_posneg: float = 0.8                   # sentiment mapping weights
    w_neu: float = 0.2
    beta: float = 0.6                       # sarcasm flip strength (0..1)


class SentSarcasmScorer:
    def __init__(self, cfg: ComboCfg):
        self.cfg = cfg
        self.device = self._pick_device(cfg.device)
        self.sent_model, self.sent_tok = self._load(cfg.sent_source, cfg.sent_sub)
        self.sarc_model, self.sarc_tok = self._load(cfg.sarc_source, cfg.sarc_sub)
        self.sent_model.eval(); self.sarc_model.eval()

    def _pick_device(self, device: Optional[str]) -> str:
        if device: return device
        if torch.backends.mps.is_available(): return "mps"
        if torch.cuda.is_available(): return "cuda"
        return "cpu"

    def _load(self, src: str, sub: Optional[str]):
        if sub:
            tok = AutoTokenizer.from_pretrained(src, subfolder=sub, use_fast=True)
            mdl = AutoModelForSequenceClassification.from_pretrained(src, subfolder=sub).to(self.device)
        else:
            tok = AutoTokenizer.from_pretrained(src, use_fast=True)
            mdl = AutoModelForSequenceClassification.from_pretrained(src).to(self.device)
        return mdl, tok

    @torch.inference_mode()
    def _base_sentiment(self, texts: List[str]) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        batch = self.sent_tok(texts, return_tensors="pt", padding=True, truncation=True,
                              max_length=cfg.max_len).to(self.device)
        probs = softmax(self.sent_model(**batch).logits, dim=-1).cpu().numpy()
        p_neg, p_neu, p_pos = probs[:,0], probs[:,1], probs[:,2]
        term1 = 0.5 * (p_pos - p_neg + 1.0) * cfg.w_posneg
        term2 = (0.5 + (p_neu - 0.5)) * cfg.w_neu
        base = np.clip(term1 + term2, 0.0, 1.0)
        return {"base": base, "p_neg": p_neg, "p_neu": p_neu, "p_pos": p_pos}

    @torch.inference_mode()
    def _sarcasm_prob(self, texts: List[str]) -> np.ndarray:
        cfg = self.cfg
        batch = self.sarc_tok(texts, return_tensors="pt", padding=True, truncation=True,
                              max_length=cfg.max_len).to(self.device)
        probs = softmax(self.sarc_model(**batch).logits, dim=-1).cpu().numpy()
        return probs[:,1]  # p(sarcastic)

    @torch.inference_mode()
    def score(self, texts: Union[str, Iterable[str]], return_details: bool=True) -> List[Dict[str, float]]:
        if isinstance(texts, str): texts = [texts]
        texts = list(texts)

        s = self._base_sentiment(texts)
        p_sarc = self._sarcasm_prob(texts)

        beta = self.cfg.beta
        base = s["base"]
        final = (1 - beta * p_sarc) * base + (beta * p_sarc) * (1 - base)
        final = np.clip(final, 0.0, 1.0)

        if not return_details:
            return [{"score": float(x)} for x in final]

        out = []
        for i, t in enumerate(texts):
            out.append({
                "text": t,
                "score": float(final[i]),
                "base_score": float(base[i]),
                "p_neg": float(s["p_neg"][i]),
                "p_neu": float(s["p_neu"][i]),
                "p_pos": float(s["p_pos"][i]),
                "p_sarcasm": float(p_sarc[i]),
            })
        return out


# ---------------- Helpers ----------------
def _ensure_dir(p: str):
    if p and not os.path.isdir(p): os.makedirs(p, exist_ok=True)

def _to_class(x: float, neg_t: float, pos_t: float) -> str:
    if x < neg_t: return "neg"
    if x > pos_t: return "pos"
    return "neu"


# ---------------- CLI actions ----------------
def cli_score(args):
    cfg = ComboCfg(
        sent_source=args.sent, sarc_source=args.sarc,
        sent_sub=args.sent_sub, sarc_sub=args.sarc_sub,
        max_len=args.max_len, device=args.device,
        w_posneg=args.w_posneg, w_neu=args.w_neu, beta=args.beta
    )
    scorer = SentSarcasmScorer(cfg)

    if args.texts:
        res = scorer.score(args.texts, return_details=True)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    # CSV path
    df = pd.read_csv(args.in_csv)
    if args.text_col not in df.columns:
        raise SystemExit(f"text col '{args.text_col}' not in {list(df.columns)}")
    texts = df[args.text_col].astype(str).fillna("")
    out_rows = []
    B = args.batch
    for i in range(0, len(texts), B):
        chunk = texts.iloc[i:i+B].tolist()
        out = scorer.score(chunk, return_details=True)
        out_rows.extend(out)
    out_df = pd.DataFrame(out_rows)
    if args.id_col and args.id_col in df.columns:
        out_df.insert(0, args.id_col, df[args.id_col])
    out_path = args.out or "scored_combo.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}  ({len(out_df)} rows)")


def cli_eval(args):
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.human_csv)
    if args.text_col not in df.columns:
        raise SystemExit(f"text col '{args.text_col}' not in {list(df.columns)}")
    if "human_pred" not in df.columns:
        raise SystemExit("human_csv must contain column 'human_pred' with decimals in [0,1].")

    df["human_pred"] = pd.to_numeric(df["human_pred"], errors="coerce")
    eval_df = df[df["human_pred"].notna()].copy()
    if eval_df.empty:
        raise SystemExit("No rows with human_pred. Fill some values and re-run.")

    # score with model (if no combo_score yet)
    if "combo_score" not in eval_df.columns:
        cfg = ComboCfg(
            sent_source=args.sent, sarc_source=args.sarc,
            sent_sub=args.sent_sub, sarc_sub=args.sarc_sub,
            max_len=args.max_len, device=args.device,
            w_posneg=args.w_posneg, w_neu=args.w_neu, beta=args.beta
        )
        scorer = SentSarcasmScorer(cfg)
        B = args.batch
        scores = []
        for i in range(0, len(eval_df), B):
            chunk = eval_df[args.text_col].iloc[i:i+B].astype(str).tolist()
            out = scorer.score(chunk, return_details=True)
            scores.extend([r["score"] for r in out])
        eval_df["combo_score"] = scores

    # regression metrics
    y = eval_df["human_pred"].to_numpy()
    yhat = eval_df["combo_score"].to_numpy()
    mae = float(np.mean(np.abs(yhat - y)))
    rmse = float(np.sqrt(np.mean((yhat - y)**2)))
    corr = float(np.corrcoef(y, yhat)[0,1]) if len(y) > 1 else float("nan")
    print(json.dumps({"n": int(len(y)), "mae": mae, "rmse": rmse, "pearson_r": corr}, indent=2))

    # classification report
    eval_df["human_cls"] = eval_df["human_pred"].map(lambda x: _to_class(x, args.neg_t, args.pos_t))
    eval_df["combo_cls"] = eval_df["combo_score"].map(lambda x: _to_class(x, args.neg_t, args.pos_t))
    print("\nClassification report (combo vs human):")
    print(classification_report(eval_df["human_cls"], eval_df["combo_cls"], digits=3))

    # plots
    _ensure_dir(args.outdir)
    # scatter
    plt.figure(figsize=(5,5))
    plt.scatter(eval_df["human_pred"], eval_df["combo_score"], s=12, alpha=0.5)
    plt.plot([0,1],[0,1])
    plt.title("Human vs Combo (sentiment + sarcasm)")
    plt.xlabel("human_pred"); plt.ylabel("combo_score"); plt.tight_layout()
    sp = os.path.join(args.outdir, "scatter_human_vs_combo.png")
    plt.savefig(sp, dpi=150); plt.close()
    # confusion
    labels = ["neg","neu","pos"]
    cm = confusion_matrix(eval_df["human_cls"], eval_df["combo_cls"], labels=labels)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(3), labels); plt.yticks(range(3), labels)
    for i in range(3):
        for j in range(3):
            plt.text(j,i,str(cm[i,j]),ha="center",va="center")
    plt.title("Confusion (combo)"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    cp = os.path.join(args.outdir, "confusion_combo.png")
    plt.savefig(cp, dpi=150); plt.close()
    print(f"[OK] artifacts → {args.outdir}")
    # save merged
    eval_df.to_csv(os.path.join(args.outdir, "human_with_combo.csv"), index=False)


# ---------------- main ----------------
def build_parser():
    p = argparse.ArgumentParser(description="Sentiment+Sarcasm scorer")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sent", required=True, help="sentiment model dir/repo (e.g., ./deberta-financial)")
    common.add_argument("--sarc", required=True, help="sarcasm model dir/repo (e.g., ./sarcasm_detector)")
    common.add_argument("--sent-sub", default=None)
    common.add_argument("--sarc-sub", default=None)
    common.add_argument("--max-len", type=int, default=128)
    common.add_argument("--device", default=None, choices=[None,"cpu","cuda","mps"])
    common.add_argument("--w-posneg", type=float, default=0.8)
    common.add_argument("--w-neu", type=float, default=0.2)
    common.add_argument("--beta", type=float, default=0.6)
    common.add_argument("--batch", type=int, default=256)

    # score
    ps = sub.add_parser("score", parents=[common], help="score texts or a CSV")
    ps.add_argument("--texts", nargs="+", help="texts to score")
    ps.add_argument("--in-csv", dest="in_csv", help="input CSV (if not using --texts)")
    ps.add_argument("--text-col", default="body")
    ps.add_argument("--id-col", default=None)
    ps.add_argument("--out", default=None)
    ps.set_defaults(func=cli_score)

    # eval
    pe = sub.add_parser("eval", parents=[common], help="evaluate vs human_pred in CSV")
    pe.add_argument("--human-csv", required=True)
    pe.add_argument("--text-col", default="body")
    pe.add_argument("--id-col", default="cid")
    pe.add_argument("--neg-t", type=float, default=0.40)
    pe.add_argument("--pos-t", type=float, default=0.60)
    pe.add_argument("--outdir", default="artifacts_combo")
    pe.set_defaults(func=cli_eval)

    return p

def main():
    args = build_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
