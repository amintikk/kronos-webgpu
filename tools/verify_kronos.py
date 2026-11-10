#!/usr/bin/env python3
"""Check the exported ONNX graphs against the upstream PyTorch implementation."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--upstream", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--models", nargs="+", default=None)
a = ap.parse_args()
sys.path.insert(0, a.upstream)
from model import Kronos, KronosTokenizer  # noqa: E402

out = Path(a.out)
manifest = json.loads((out / "manifest.json").read_text())
torch.manual_seed(1)
rng = np.random.default_rng(1)
worst = 0.0


def sess(name):
    return ort.InferenceSession(str(out / name), providers=["CPUExecutionProvider"])


def diff(tag, got, ref):
    global worst
    d = float(np.max(np.abs(got - ref)))
    worst = max(worst, d)
    print(f"  {tag}: max abs diff {d:.2e}")


for key in a.models or manifest["models"]:
    m = manifest["models"][key]
    print(key)
    tok = KronosTokenizer.from_pretrained(manifest["tokenizers"][m["tokenizer"]]["source"]).eval()
    model = Kronos.from_pretrained(m["source"]).eval()
    T = 96
    x = torch.clip(torch.randn(1, T, 6).cumsum(1) * 0.3, -5, 5)
    with torch.no_grad():
        ref_s1, ref_s2 = tok.encode(x, half=True)
    z = sess(m["tokenizer"] + "_enc.onnx").run(None, {"x": x.numpy()})[0]
    bits = z > 0
    w = 2 ** np.arange(10)
    s1 = (bits[..., :10] * w).sum(-1)
    s2 = (bits[..., 10:] * w).sum(-1)
    print("  tokens equal:", bool((s1 == ref_s1.numpy()).all() and (s2 == ref_s2.numpy()).all()))
    assert (s1 == ref_s1.numpy()).all() and (s2 == ref_s2.numpy()).all()

    with torch.no_grad():
        ref_y = tok.decode([ref_s1, ref_s2], half=True).numpy()
    y = sess(m["tokenizer"] + "_dec.onnx").run(None, {"bits": np.where(bits, 1.0, -1.0).astype(np.float32)})[0]
    diff("tokenizer decode", y, ref_y)

    stamp = torch.stack([torch.randint(0, 60, (1, T + 1)), torch.randint(0, 24, (1, T + 1)), torch.randint(0, 7, (1, T + 1)),
                         torch.randint(1, 32, (1, T + 1)), torch.randint(1, 13, (1, T + 1))], dim=-1).float()
    lm, s2h = sess(key + "_lm.onnx"), sess(key + "_s2.onnx")
    L, H, hd, CH, chd = m["n_layers"], m["n_heads"], m["head_dim"], m["cross_heads"], m["cross_head_dim"]
    feed = {"s1": s1.astype(np.int32), "s2": s2.astype(np.int32), "stamp": stamp[:, :T].numpy().astype(np.int32),
            "ck": np.zeros((1, CH, 0, chd), np.float32), "cv": np.zeros((1, CH, 0, chd), np.float32)}
    for i in range(L):
        feed[f"pk{i}"] = np.zeros((1, H, 0, hd), np.float32)
        feed[f"pv{i}"] = np.zeros((1, H, 0, hd), np.float32)
    names = [o.name for o in lm.get_outputs()]
    res = dict(zip(names, lm.run(None, feed)))
    with torch.no_grad():
        ref_logits, ctx = model.decode_s1(ref_s1, ref_s2, stamp[:, :T])
        samp = torch.tensor([[int(rng.integers(0, 1024))]])
        ref_l2 = model.decode_s2(ctx, samp)[:, -1, :].numpy()
    diff("s1 logits (prefix)", res["logits"], ref_logits[:, -1, :].numpy())
    l2 = s2h.run(None, {"h_last": res["h_last"], "s1": samp.numpy().astype(np.int32).reshape(1), "ck": res["nck"], "cv": res["ncv"]})[0]
    diff("s2 logits (prefix)", l2, ref_l2)

    # cached step with a broadcast batch of 3
    B = 3
    n1 = rng.integers(0, 1024, (B, 1)).astype(np.int32)
    n2 = rng.integers(0, 1024, (B, 1)).astype(np.int32)
    feed = {"s1": n1, "s2": n2, "stamp": np.repeat(stamp[:, T:].numpy().astype(np.int32), B, 0), "ck": res["nck"], "cv": res["ncv"]}
    for i in range(L):
        feed[f"pk{i}"] = res[f"nk{i}"]
        feed[f"pv{i}"] = res[f"nv{i}"]
    res2 = dict(zip(names, lm.run(None, feed)))
    with torch.no_grad():
        f1 = torch.cat([ref_s1.repeat(B, 1), torch.from_numpy(n1).long()], 1)
        f2 = torch.cat([ref_s2.repeat(B, 1), torch.from_numpy(n2).long()], 1)
        ref_logits, ctx = model.decode_s1(f1, f2, stamp.repeat(B, 1, 1))
        samp = torch.from_numpy(rng.integers(0, 1024, (B, 1)))
        ref_l2 = model.decode_s2(ctx, samp)[:, -1, :].numpy()
    diff("s1 logits (cached step)", res2["logits"], ref_logits[:, -1, :].numpy())
    l2 = s2h.run(None, {"h_last": res2["h_last"], "s1": samp.numpy().astype(np.int32).reshape(B), "ck": res2["nck"], "cv": res2["ncv"]})[0]
    diff("s2 logits (cached step)", l2, ref_l2)

print("worst diff", worst)
sys.exit(0 if worst < 2e-3 else 1)
