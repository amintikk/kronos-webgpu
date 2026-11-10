// Precompute the forecast shown on the landing page (no model download needed there).
// Usage: KRONOS_MODELS=<dir> node tools/make_demo.ts > public/demo.json
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-node';
import { runKronos } from '../src/engine/kronos.ts';
import type { Manifest, OrtLike, OrtSession } from '../src/engine/types.ts';

const models = process.env.KRONOS_MODELS!;
const manifest = JSON.parse(readFileSync(`${models}/manifest.json`, 'utf8')) as Manifest;
const meta = manifest.models.mini!;
const open = (f: string) => ort.InferenceSession.create(`${models}/${f}`) as unknown as Promise<OrtSession>;
const rows = JSON.parse(readFileSync(new URL('../tests/fixtures/klines.json', import.meta.url), 'utf8')) as (string | number)[][];
const candles = rows.map((r) => ({ t: +r[0]!, o: +r[1]!, h: +r[2]!, l: +r[3]!, c: +r[4]!, v: +r[5]!, a: +r[7]! }));
const T = 400;
const P = 48;
const ctx = candles.slice(0, T);
const times = Array.from({ length: P }, (_, i) => ctx[T - 1]!.t + 3600e3 * (i + 1));
const sess = { enc: await open(`${meta.tokenizer}_enc.onnx`), dec: await open(`${meta.tokenizer}_dec.onnx`), lm: await open(meta.lm.file), s2: await open(meta.s2.file) };
const { paths } = await runKronos(ort as unknown as OrtLike, sess, meta, ctx, times, { predLen: P, samples: 32, batch: 8, temperature: 1, topK: 0, topP: 0.9, seed: 7 });
const r2 = (x: number) => Math.round(x * 100) / 100;
process.stdout.write(
  JSON.stringify({
    symbol: 'BTCUSDT',
    interval: '1h',
    model: meta.label,
    candles: ctx.slice(-160).map((c) => [c.t, r2(c.o), r2(c.h), r2(c.l), r2(c.c), r2(c.v)]),
    times,
    paths: paths.map((p) => Array.from(p, r2)),
  }),
);
