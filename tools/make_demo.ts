// Precompute everything the landing page shows, with the same engine the app uses, so the page
// needs no model download. Usage: KRONOS_MODELS=<dir> node tools/make_demo.ts > public/demo.json
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-node';
import { runKronos } from '../src/engine/kronos.ts';
import type { Candle, Manifest, OrtLike, OrtSession } from '../src/engine/types.ts';

const models = process.env.KRONOS_MODELS!;
const manifest = JSON.parse(readFileSync(`${models}/manifest.json`, 'utf8')) as Manifest;
const meta = manifest.models.mini!;
const open = (f: string) => ort.InferenceSession.create(`${models}/${f}`) as unknown as Promise<OrtSession>;
const rows = JSON.parse(readFileSync(new URL('../tests/fixtures/klines.json', import.meta.url), 'utf8')) as (string | number)[][];
const candles: Candle[] = rows.map((r) => ({ t: +r[0]!, o: +r[1]!, h: +r[2]!, l: +r[3]!, c: +r[4]!, v: +r[5]!, a: +r[7]! }));
const sess = { enc: await open(`${meta.tokenizer}_enc.onnx`), dec: await open(`${meta.tokenizer}_dec.onnx`), lm: await open(meta.lm.file), s2: await open(meta.s2.file) };
const lib = ort as unknown as OrtLike;
const HOUR = 3600e3;
const r2 = (x: number) => Math.round(x * 100) / 100;
const CONTEXT = 400;

// 1. Hero: 48 hours ahead from the end of the context, 32 sampled paths.
const P = 48;
const ctx = candles.slice(0, CONTEXT);
const times = Array.from({ length: P }, (_, i) => ctx[CONTEXT - 1]!.t + HOUR * (i + 1));
const { paths } = await runKronos(lib, sess, meta, ctx, times, { predLen: P, samples: 32, batch: 8, temperature: 1, topK: 0, topP: 0.9, seed: 7 });

// 2. Tokens of the last context candles, normalised exactly as the engine does.
const FEATS = 6;
const raw = ctx.map((c) => [c.o, c.h, c.l, c.c, c.v, c.a]);
const mean = Array.from({ length: FEATS }, (_, f) => raw.reduce((s, r) => s + r[f]! / CONTEXT, 0));
const std = Array.from({ length: FEATS }, (_, f) => Math.sqrt(raw.reduce((s, r) => s + (r[f]! - mean[f]!) ** 2 / CONTEXT, 0)) + 1e-5);
const x = Float32Array.from(raw.flatMap((r) => r.map((v, f) => Math.max(-5, Math.min(5, (v - mean[f]!) / std[f]!)))));
const z = (await sess.enc.run({ x: new lib.Tensor('float32', x, [1, CONTEXT, FEATS]) })).z!.data as Float32Array;
const TOKENS = 12;
const tokens = Array.from({ length: TOKENS }, (_, k) => {
  const i = CONTEXT - TOKENS + k;
  const bits = Array.from({ length: 20 }, (_, b) => (z[i * 20 + b]! > 0 ? 1 : 0));
  const id = (from: number) => bits.slice(from, from + 10).reduce((s, b, q) => s + (b << q), 0);
  return { t: ctx[i]!.t, s1: id(0), s2: id(10), bits };
});

// 3. Backtest: forecast 24 h ahead every 12 h and compare with what happened.
const H = 24;
const backtest = [];
for (let end = CONTEXT; end + H <= candles.length; end += 12) {
  const c = candles.slice(end - CONTEXT, end);
  const last = c[CONTEXT - 1]!;
  const ft = Array.from({ length: H }, (_, i) => last.t + HOUR * (i + 1));
  const res = await runKronos(lib, sess, meta, c, ft, { predLen: H, samples: 16, batch: 8, temperature: 1, topK: 0, topP: 0.9, seed: end });
  const ends = res.paths.map((p) => p[(H - 1) * FEATS + 3]!).sort((a, b) => a - b);
  const q = (p: number) => ends[Math.round(p * (ends.length - 1))]!;
  const actual = candles[end + H - 1]!.c;
  backtest.push({ t: last.t, last: r2(last.c), median: r2(q(0.5)), p10: r2(q(0.1)), p90: r2(q(0.9)), actual: r2(actual) });
  process.stderr.write('.');
}
process.stderr.write('\n');

process.stdout.write(
  JSON.stringify({
    symbol: 'BTCUSDT',
    interval: '1h',
    model: meta.label,
    context: CONTEXT,
    candles: ctx.slice(-160).map((c) => [c.t, r2(c.o), r2(c.h), r2(c.l), r2(c.c), r2(c.v)]),
    times,
    paths: paths.map((p) => Array.from(p, r2)),
    tokens,
    backtest: { horizon: H, samples: 16, windows: backtest },
  }),
);
