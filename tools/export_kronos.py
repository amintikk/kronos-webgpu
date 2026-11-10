#!/usr/bin/env python3
"""Export Kronos (https://github.com/shiyu-coder/Kronos) to ONNX graphs built for the browser.

The upstream inference loop re-runs the whole context on every predicted candle. Here the
language model is re-implemented functionally (same weights) with a key/value cache, so each
autoregressive step only processes one token:

  tok_enc.onnx   x[1,T,6]                       -> z[1,T,20]            (sign bits = tokens)
  tok_dec.onnx   bits[B,T,20] (+-1)             -> y[B,T,6]
  lm.onnx        s1,s2[B,Tn] stamp[B,Tn,5] past -> logits_s1[B,V] h_last[B,D] present
  s2.onnx        h_last[B,D] s1[B] ck,cv        -> logits_s2[B,V]

Usage: export_kronos.py --upstream <Kronos repo> --out <dir> [--models mini small base]
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = {
    "mini": {"repo": "NeoQuasar/Kronos-mini", "tok": "NeoQuasar/Kronos-Tokenizer-2k", "tok_id": "tok2k", "max_context": 2048, "label": "Kronos-mini", "params": "4.1M"},
    "small": {"repo": "NeoQuasar/Kronos-small", "tok": "NeoQuasar/Kronos-Tokenizer-base", "tok_id": "tokbase", "max_context": 512, "label": "Kronos-small", "params": "24.7M"},
    "base": {"repo": "NeoQuasar/Kronos-base", "tok": "NeoQuasar/Kronos-Tokenizer-base", "tok_id": "tokbase", "max_context": 512, "label": "Kronos-base", "params": "102.3M"},
}
OPSET = 17
NEG = -1e9


def rms(norm, x):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + norm.eps) * norm.weight


def rope(x, pos, inv_freq):
    # x [B,H,T,hd], pos [T] float
    freqs = pos[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rot * sin


def heads(t, n_heads):
    b, s, d = t.shape[0], t.shape[1], t.shape[2]
    return t.reshape(b, s, n_heads, d // n_heads).transpose(1, 2)


def unheads(t):
    b, s = t.shape[0], t.shape[2]
    return t.transpose(1, 2).reshape(b, s, -1)


def ffn(f, x):
    return f.w2(F.silu(f.w1(x)) * f.w3(x))


def causal_block(layer, x, pos, past_k=None, past_v=None):
    """Pre-norm block with causal RoPE attention. Returns (x, k_all, v_all)."""
    att = layer.self_attn
    h = rms(layer.norm1, x)
    q = rope(heads(att.q_proj(h), att.n_heads), pos, att.rotary.inv_freq)
    k = rope(heads(att.k_proj(h), att.n_heads), pos, att.rotary.inv_freq)
    v = heads(att.v_proj(h), att.n_heads)
    tn = x.shape[1]
    if past_k is not None:
        b = x.shape[0]
        k = torch.cat([past_k.expand(b, -1, -1, -1), k], dim=2)
        v = torch.cat([past_v.expand(b, -1, -1, -1), v], dim=2)
    tk = k.shape[2]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(att.head_dim)
    key_idx = torch.arange(tk, device=x.device)[None, :]
    qry_idx = (torch.arange(tn, device=x.device) + (tk - tn))[:, None]
    scores = scores + (key_idx > qry_idx).to(scores.dtype)[None, None] * NEG
    out = unheads(torch.matmul(torch.softmax(scores, dim=-1), v))
    x = x + att.out_proj(out)
    x = x + ffn(layer.ffn, rms(layer.norm2, x))
    return x, k, v


class TokEncode(nn.Module):
    def __init__(self, tok):
        super().__init__()
        self.tok = tok

    def forward(self, x):
        t = self.tok
        z = t.embed(x)
        pos = torch.arange(x.shape[1], device=x.device).to(z.dtype)
        for layer in t.encoder:
            z, _, _ = causal_block(layer, z, pos)
        return t.quant_embed(z)


class TokDecode(nn.Module):
    def __init__(self, tok):
        super().__init__()
        self.tok = tok

    def forward(self, bits):
        t = self.tok
        z = t.post_quant_embed(bits * (1.0 / math.sqrt(t.codebook_dim)))
        pos = torch.arange(bits.shape[1], device=bits.device).to(z.dtype)
        for layer in t.decoder:
            z, _, _ = causal_block(layer, z, pos)
        return t.head(z)


class LMStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, s1, s2, stamp, ck, cv, *past):
        m = self.m
        d = m.d_model
        e = m.embedding
        x = e.fusion_proj(torch.cat([e.emb_s1(s1) * math.sqrt(d), e.emb_s2(s2) * math.sqrt(d)], dim=-1))
        te = m.time_emb
        x = x + (te.hour_embed(stamp[:, :, 1]) + te.weekday_embed(stamp[:, :, 2]) + te.day_embed(stamp[:, :, 3])
                 + te.month_embed(stamp[:, :, 4]) + te.minute_embed(stamp[:, :, 0]))
        tp = past[0].shape[2]
        pos = (torch.arange(x.shape[1], device=x.device) + tp).to(x.dtype)
        present = []
        for i, layer in enumerate(m.transformer):
            x, k, v = causal_block(layer, x, pos, past[2 * i], past[2 * i + 1])
            present += [k, v]
        x = rms(m.norm, x)
        logits = m.head.proj_s1(x[:, -1, :])
        ca = m.dep_layer.cross_attn
        b = x.shape[0]
        ck = torch.cat([ck.expand(b, -1, -1, -1), heads(ca.k_proj(x), ca.n_heads)], dim=2)
        cv = torch.cat([cv.expand(b, -1, -1, -1), heads(ca.v_proj(x), ca.n_heads)], dim=2)
        return (logits, x[:, -1, :], ck, cv, *present)


class S2Head(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, h_last, s1, ck, cv):
        m = self.m
        ca = m.dep_layer.cross_attn
        # Upstream queries with a single sibling token, so RoPE is the identity rotation here.
        q = heads(ca.q_proj(m.embedding.emb_s1(s1))[:, None, :], ca.n_heads)
        scores = torch.matmul(q, ck.transpose(-1, -2)) / math.sqrt(ca.head_dim)
        out = unheads(torch.matmul(torch.softmax(scores, dim=-1), cv))[:, 0, :]
        x2 = rms(m.dep_layer.norm, h_last + ca.out_proj(out))
        return m.head.proj_s2(x2)


def export(module, args, path, input_names, output_names, dynamic_axes):
    torch.onnx.export(module, args, str(path), input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, opset_version=OPSET, do_constant_folding=True, dynamo=False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_meta(path):
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def lm_io(model):
    n = model.n_layers
    past = [f"p{kv}{i}" for i in range(n) for kv in ("k", "v")]
    present = [f"n{kv}{i}" for i in range(n) for kv in ("k", "v")]
    ins = ["s1", "s2", "stamp", "ck", "cv", *past]
    outs = ["logits", "h_last", "nck", "ncv", *present]
    axes = {"s1": {0: "b", 1: "tn"}, "s2": {0: "b", 1: "tn"}, "stamp": {0: "b", 1: "tn"},
            "ck": {0: "bp", 2: "tp"}, "cv": {0: "bp", 2: "tp"}, "logits": {0: "b"}, "h_last": {0: "b"},
            "nck": {0: "b", 2: "t"}, "ncv": {0: "b", 2: "t"}}
    for name in past:
        axes[name] = {0: "bp", 2: "tp"}
    for name in present:
        axes[name] = {0: "b", 2: "t"}
    return ins, outs, axes


def lm_dummy(model, b, tn, bp, tp):
    hd = model.d_model // model.n_heads
    ca = model.dep_layer.cross_attn
    s1 = torch.randint(0, 2 ** model.s1_bits, (b, tn), dtype=torch.int32)
    s2 = torch.randint(0, 2 ** model.s2_bits, (b, tn), dtype=torch.int32)
    stamp = torch.stack([torch.randint(0, 60, (b, tn)), torch.randint(0, 24, (b, tn)), torch.randint(0, 7, (b, tn)),
                         torch.randint(1, 32, (b, tn)), torch.randint(1, 13, (b, tn))], dim=-1).to(torch.int32)
    ck = torch.randn(bp, ca.n_heads, tp, ca.head_dim)
    cv = torch.randn(bp, ca.n_heads, tp, ca.head_dim)
    past = [torch.randn(bp, model.n_heads, tp, hd) for _ in range(2 * model.n_layers)]
    return (s1, s2, stamp, ck, cv, *past)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    a = ap.parse_args()
    sys.path.insert(0, a.upstream)
    from model import Kronos, KronosTokenizer  # noqa: E402

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"models": {}, "tokenizers": {}}
    torch.manual_seed(0)

    for key in a.models:
        spec = MODELS[key]
        tok_id = spec["tok_id"]
        if tok_id not in manifest["tokenizers"] or not (out / f"{tok_id}_enc.onnx").exists():
            tok = KronosTokenizer.from_pretrained(spec["tok"]).eval()
            enc_p, dec_p = out / f"{tok_id}_enc.onnx", out / f"{tok_id}_dec.onnx"
            with torch.no_grad():
                export(TokEncode(tok), (torch.randn(1, 64, tok.d_in),), enc_p, ["x"], ["z"],
                       {"x": {0: "b", 1: "t"}, "z": {0: "b", 1: "t"}})
                export(TokDecode(tok), (torch.sign(torch.randn(2, 64, tok.codebook_dim)),), dec_p, ["bits"], ["y"],
                       {"bits": {0: "b", 1: "t"}, "y": {0: "b", 1: "t"}})
            manifest["tokenizers"][tok_id] = {"source": spec["tok"], "s1_bits": tok.s1_bits, "s2_bits": tok.s2_bits,
                                              "d_in": tok.d_in, "enc": file_meta(enc_p), "dec": file_meta(dec_p)}
            print("exported tokenizer", tok_id)

        model = Kronos.from_pretrained(spec["repo"]).eval()
        lm_p, s2_p = out / f"{key}_lm.onnx", out / f"{key}_s2.onnx"
        ins, outs, axes = lm_io(model)
        ca = model.dep_layer.cross_attn
        with torch.no_grad():
            export(LMStep(model), lm_dummy(model, 3, 2, 1, 5), lm_p, ins, outs, axes)
            export(S2Head(model), (torch.randn(3, model.d_model), torch.randint(0, 2 ** model.s1_bits, (3,), dtype=torch.int32),
                                   torch.randn(3, ca.n_heads, 7, ca.head_dim), torch.randn(3, ca.n_heads, 7, ca.head_dim)),
                   s2_p, ["h_last", "s1", "ck", "cv"], ["logits"],
                   {"h_last": {0: "b"}, "s1": {0: "b"}, "ck": {0: "b", 2: "t"}, "cv": {0: "b", 2: "t"}, "logits": {0: "b"}})
        manifest["models"][key] = {
            "label": spec["label"], "params": spec["params"], "source": spec["repo"], "tokenizer": tok_id,
            "max_context": spec["max_context"], "n_layers": model.n_layers, "n_heads": model.n_heads,
            "d_model": model.d_model, "head_dim": model.d_model // model.n_heads,
            "cross_heads": ca.n_heads, "cross_head_dim": ca.head_dim,
            "s1_vocab": 2 ** model.s1_bits, "s2_vocab": 2 ** model.s2_bits,
            "lm": file_meta(lm_p), "s2": file_meta(s2_p),
        }
        print("exported model", key)
        manifest_path.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
