// End-to-end check of the browser engine against the upstream PyTorch predictor (greedy decoding).
// Needs the exported models: KRONOS_MODELS=<dir> npm test (skipped without them).
// The reference in tests/fixtures comes from tools/make_reference.py.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import * as ort from 'onnxruntime-node';
import { runChronos } from '../src/engine/chronos.ts';
import { runKronos } from '../src/engine/kronos.ts';
import { makeRng, sampleToken } from '../src/engine/sampling.ts';
import type { Candle, Manifest, OrtLike, OrtSession } from '../src/engine/types.ts';

const models = process.env.KRONOS_MODELS;
const ref = new URL('./fixtures', import.meta.url).pathname;
const skip = !models;
const lib = ort as unknown as OrtLike;
const open = (file: string) => ort.InferenceSession.create(`${models}/${file}`) as unknown as Promise<OrtSession>;

function candles(): Candle[] {
  const k = JSON.parse(readFileSync(`${ref}/klines.json`, 'utf8')) as (string | number)[][];
  return k.map((r) => ({ t: +r[0]!, o: +r[1]!, h: +r[2]!, l: +r[3]!, c: +r[4]!, v: +r[5]!, a: +r[7]! }));
}

for (const [key, exactSteps] of [['mini', 24], ['small', 12]] as const) {
  test(`kronos-${key} matches upstream greedy forecast`, { skip }, async () => {
    const manifest = JSON.parse(readFileSync(`${models}/manifest.json`, 'utf8')) as Manifest;
    const meta = manifest.models[key]!;
    const expected = JSON.parse(readFileSync(`${ref}/greedy.json`, 'utf8'))[key] as { T: number; P: number; pred: number[][] };
    const all = candles();
    const sess = {
      enc: await open(`${meta.tokenizer}_enc.onnx`),
      dec: await open(`${meta.tokenizer}_dec.onnx`),
      lm: await open(meta.lm.file),
      s2: await open(meta.s2.file),
    };
    const ctx = all.slice(0, expected.T);
    const future = all.slice(expected.T, expected.T + expected.P).map((c) => c.t);
    const params = { predLen: expected.P, samples: 3, batch: 2, temperature: 1, topK: 1, topP: 1, seed: 1 };
    const { paths } = await runKronos(lib, sess, meta, ctx, future, params);
    assert.equal(paths.length, 3);
    for (const path of paths) {
      for (let i = 0; i < exactSteps; i++) {
        for (let f = 0; f < 6; f++) {
          const want = expected.pred[i]![f]!;
          const got = path[i * 6 + f]!;
          assert.ok(Math.abs(got - want) <= 1e-3 * Math.abs(want) + 1e-2, `step ${i} feature ${f}: ${got} vs ${want}`);
        }
      }
      assert.ok(path.every(Number.isFinite));
    }
    // Stochastic sampling still yields finite, distinct paths.
    const sampled = await runKronos(lib, sess, meta, ctx.slice(-128), future.slice(0, 8), { ...params, predLen: 8, topK: 0, topP: 0.9, samples: 4, batch: 4 });
    assert.equal(sampled.paths.length, 4);
    assert.ok(sampled.paths.every((p) => p.every(Number.isFinite)));
    assert.notDeepEqual(Array.from(sampled.paths[0]!), Array.from(sampled.paths[1]!));
  });
}

test('chronos-bolt produces ordered quantiles, with rollout', { skip }, async () => {
  const manifest = JSON.parse(readFileSync(`${models}/manifest.json`, 'utf8')) as Manifest;
  const meta = manifest.chronos.tiny!;
  const closes = candles().map((c) => c.c);
  const res = await runChronos(lib, await open(meta.model.file), meta, closes.slice(0, 500), 100);
  assert.equal(res.quantiles.length, 9);
  const last = closes[499]!;
  for (let i = 0; i < 100; i++) {
    assert.ok(res.quantiles[0]![i]! <= res.quantiles[8]![i]!);
    assert.ok(Math.abs(res.quantiles[4]![i]! / last - 1) < 0.2);
  }
});

test('sampling: greedy, top-k and top-p restrict the support', () => {
  const logits = Float32Array.from([0, 5, 1, 4, -2]);
  const rng = makeRng(7);
  assert.equal(sampleToken(logits, 0, 5, 1, 1, 1, rng), 1);
  const seenK = new Set<number>();
  const seenP = new Set<number>();
  const seenAll = new Set<number>();
  for (let i = 0; i < 3000; i++) {
    seenK.add(sampleToken(logits, 0, 5, 1, 2, 1, rng));
    seenP.add(sampleToken(logits, 0, 5, 1, 0, 0.5, rng));
    seenAll.add(sampleToken(logits, 0, 5, 2, 0, 1, rng));
  }
  assert.deepEqual([...seenK].sort(), [1, 3]);
  assert.deepEqual([...seenP].sort(), [1]);
  assert.equal(seenAll.size, 5);
  assert.equal(makeRng(3)(), makeRng(3)());
});
