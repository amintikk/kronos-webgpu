#!/usr/bin/env python3
"""Regenerate tests/fixtures/greedy.json: upstream PyTorch forecasts with greedy decoding (top_k=1).

Usage: make_reference.py --upstream <Kronos repo> --fixtures tests/fixtures
"""
import argparse
import json
import sys

import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--upstream", required=True)
ap.add_argument("--fixtures", required=True)
a = ap.parse_args()
sys.path.insert(0, a.upstream)
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

k = json.load(open(f"{a.fixtures}/klines.json"))
cols = ["open", "high", "low", "close", "volume", "amount"]
df = pd.DataFrame([[r[0], *(float(r[i]) for i in (1, 2, 3, 4, 5, 7))] for r in k], columns=["t", *cols])
df["ts"] = pd.to_datetime(df["t"], unit="ms")
out = {}
CASES = [("mini", "NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048, 600, 24),
         ("small", "NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512, 500, 40)]
for key, repo, tok, max_context, T, P in CASES:
    pred = KronosPredictor(Kronos.from_pretrained(repo).eval(), KronosTokenizer.from_pretrained(tok).eval(), device="cpu", max_context=max_context)
    x, y = df.iloc[:T], df.iloc[T:T + P]
    torch.manual_seed(0)
    r = pred.predict(x[cols], x["ts"], y["ts"], P, T=1.0, top_k=1, top_p=1.0, sample_count=1, verbose=False)
    out[key] = {"T": T, "P": P, "pred": r.values.tolist()}
json.dump(out, open(f"{a.fixtures}/greedy.json", "w"))
