#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentiment-only scorer → single decimal in [0,1]
0 = very negative, 0.5 = neutral, 1 = very positive
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Union

import torch
import numpy as np
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class SentConfig:
    source: str = "./deberta-financial"       # HF repo or local dir
    subfolder: Optional[str] = None           # e.g. "deberta-financial" if loading from a repo root
    max_len: int = 128
    device: Optional[str] = None              # "mps" | "cuda" | "cpu" | None(auto)
    # scoring weights: tweak if you want to bias toward neutral
    w_posneg: float = 0.8                     # weight for (pos vs neg) part
    w_neu: float = 0.2                        # small smoothing using neutral prob (w_posneg + w_neu = 1.0)


class SentimentScorer:
    def __init__(self, cfg: SentConfig):
        self.cfg = cfg
        self.device = self._choose_device(cfg.device)
        self.model, self.tok = self._load_classifier(cfg.source, cfg.subfolder)
        self.model.eval()

    def _choose_device(self, device: Optional[str]) -> str:
        if device: return device
        if torch.backends.mps.is_available(): return "mps"
        if torch.cuda.is_available(): return "cuda"
        return "cpu"

    def _load_classifier(self, repo_or_dir: str, subfolder: Optional[str]):
        if subfolder:
            tok = AutoTokenizer.from_pretrained(repo_or_dir, subfolder=subfolder, use_fast=True)
            mdl = AutoModelForSequenceClassification.from_pretrained(repo_or_dir, subfolder=subfolder).to(self.device)
        else:
            tok = AutoTokenizer.from_pretrained(repo_or_dir, use_fast=True)
            mdl = AutoModelForSequenceClassification.from_pretrained(repo_or_dir).to(self.device)
        return mdl, tok


    @torch.inference_mode()
    def score(
        self,
        texts: Union[str, Iterable[str]],
        return_details: bool = True
    ) -> List[Dict[str, float]]:
        """
        Returns list of dicts.
        Each dict has:
          - score: float in [0,1]
          - p_neg, p_neu, p_pos (if return_details=True)
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        cfg = self.cfg

        batch = self.tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_len
        ).to(self.device)
        probs = softmax(self.model(**batch).logits, dim=-1).cpu().numpy()  # [N,3]: (p_neg, p_neu, p_pos)
        p_neg, p_neu, p_pos = probs[:, 0], probs[:, 1], probs[:, 2]

        # Base mapping to [0,1]:
        #   term1: map (p_pos - p_neg) from [-1,1] → [0,1] via 0.5*(x+1)
        #   term2: mild neutral smoothing around 0.5
        term1 = 0.5 * (p_pos - p_neg + 1.0) * cfg.w_posneg
        term2 = (0.5 + (p_neu - 0.5)) * cfg.w_neu
        final = np.clip(term1 + term2, 0.0, 1.0)

        if not return_details:
            return [{"score": float(x)} for x in final]

        out = []
        for i, t in enumerate(texts):
            out.append({
                "text": t,
                "score": float(final[i]),
                "p_neg": float(p_neg[i]),
                "p_neu": float(p_neu[i]),
                "p_pos": float(p_pos[i]),
            })
        return out


# ----------------- CLI -----------------
def _main():
    import argparse, json
    ap = argparse.ArgumentParser(description="Sentiment-only scorer → single decimal [0,1].")
    ap.add_argument("--model", default="./deberta-financial", help="HF repo or local dir for sentiment model")
    ap.add_argument("--sub",   default=None, help="subfolder if loading from a HF repo root (e.g., 'deberta-financial')")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "mps"])
    ap.add_argument("--w-posneg", type=float, default=0.8)
    ap.add_argument("--w-neu",    type=float, default=0.2)
    ap.add_argument("--text", nargs="+", help="texts to score (space-separated or quoted)")
    ap.add_argument("--json", action="store_true", help="print JSON output")
    args = ap.parse_args()

    cfg = SentConfig(
        source=args.model,
        subfolder=args.sub,
        max_len=args.max_len,
        device=args.device,
        w_posneg=args.w_posneg,
        w_neu=args.w_neu,
    )
    scorer = SentimentScorer(cfg)

    texts = args.text or [
        "Revenue and EPS beat expectations; guidance raised.",
        "Results were in line with expectations.",
        "Regulatory fines and declining margins weigh on outlook."
    ]
    res = scorer.score(texts, return_details=True)

    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        for r in res:
            print(f"{r['text']}\n  score={r['score']:.3f}  "
                  f"(p_pos={r['p_pos']:.2f}, p_neu={r['p_neu']:.2f}, p_neg={r['p_neg']:.2f})\n")


if __name__ == "__main__":
    _main()
