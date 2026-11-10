#!/usr/bin/env python3
"""Export Chronos-Bolt (amazon/chronos-bolt-*) to a single ONNX graph: context[B,T] -> quantiles[B,9,64].

T must be a multiple of the patch size (16) and contain no NaN; the browser guarantees both.
Usage: export_chronos.py --out <dir> [--models tiny small]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from chronos import ChronosBoltPipeline

SIZES = {"tiny": "9M", "mini": "21M", "small": "48M", "base": "205M"}


class Wrap(torch.nn.Module):
    """ChronosBoltModelForForecasting.forward without the NaN handling (inputs are complete)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, context):
        m = self.model
        cfg = m.chronos_config
        b = context.shape[0]
        loc = context.mean(dim=-1, keepdim=True)
        scale = (context - loc).square().mean(dim=-1, keepdim=True).sqrt()
        scale = torch.where(scale == 0, torch.full_like(scale, m.instance_norm.eps), scale)
        x = ((context - loc) / scale).reshape(b, -1, cfg.input_patch_size)
        embeds = m.input_patch_embedding(torch.cat([x, torch.ones_like(x)], dim=-1))
        reg = m.shared(torch.full((b, 1), m.config.reg_token_id, dtype=torch.long))
        embeds = torch.cat([embeds, reg], dim=-2)
        mask = torch.ones_like(embeds[:, :, 0])
        hidden = m.encoder(attention_mask=mask, inputs_embeds=embeds)[0]
        start = torch.full((b, 1), m.config.decoder_start_token_id, dtype=torch.long)
        seq = m.decoder(input_ids=start, encoder_hidden_states=hidden, encoder_attention_mask=mask, return_dict=True).last_hidden_state
        q = m.output_patch_embedding(seq).view(b, m.num_quantiles, cfg.prediction_length)
        return q * scale[:, :, None] + loc[:, :, None]


class Ref(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, context):
        return self.model(context).quantile_preds


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--models", nargs="+", default=["tiny", "small"])
a = ap.parse_args()
out = Path(a.out)
manifest_path = out / "manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest.setdefault("chronos", {})
torch.manual_seed(0)
for key in a.models:
    repo = f"amazon/chronos-bolt-{key}"
    model = ChronosBoltPipeline.from_pretrained(repo).model.eval()
    cfg = model.chronos_config
    path = out / f"bolt_{key}.onnx"
    wrap = Wrap(model)
    # Reference outputs are taken before tracing: the tracer leaves the HF decoder cache in a different state.
    cases = []
    for t in (64, 256, 512, 2048):
        x = torch.randn(3, t).cumsum(-1) * 3 + 1000
        with torch.no_grad():
            cases.append((x, Ref(model)(x).numpy()))
    with torch.no_grad():
        torch.onnx.export(wrap, (torch.randn(2, 320).cumsum(-1) + 50,), str(path), input_names=["context"], output_names=["quantiles"],
                          dynamic_axes={"context": {0: "b", 1: "t"}, "quantiles": {0: "b"}}, opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for x, ref in cases:
        got = sess.run(None, {"context": x.numpy()})[0]
        worst = max(worst, float(np.max(np.abs(got - ref) / x.std(-1).numpy()[:, None, None])))
    print(key, "worst diff relative to context std", worst)
    assert worst < 1e-3
    manifest["chronos"][key] = {"label": f"Chronos-Bolt-{key}", "params": SIZES[key], "source": repo,
                                "context_length": cfg.context_length, "patch": cfg.input_patch_size,
                                "prediction_length": cfg.prediction_length, "quantiles": list(cfg.quantiles),
                                "model": {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}}
    manifest_path.write_text(json.dumps(manifest, indent=1))
