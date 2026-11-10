import { makeRng, sampleToken } from './sampling.ts';
import { CancelledError } from './types.ts';
import type { Candle, KronosMeta, KronosParams, OrtLike, OrtSession, OrtTensor, Progress } from './types.ts';

export interface KronosSessions {
  enc: OrtSession;
  dec: OrtSession;
  lm: OrtSession;
  s2: OrtSession;
}

export interface KronosResult {
  /** One entry per sample: predLen rows of [open, high, low, close, volume, amount]. */
  paths: Float32Array[];
}

const FEATS = 6;
const BITS = 10;
/** Fast mode: tokens dropped at once when the window is full, so the cache is rebuilt only every REPRIME steps. */
const REPRIME = 64;
/** Upper bound for the tokenizer decoder's attention buffers per call. */
const DECODE_BUDGET_BYTES = 192e6;

interface LmState {
  logits: Float32Array;
  hLast: OrtTensor;
  cache: Record<string, OrtTensor>;
  batch: number;
  len: number;
}

/** minute, hour, weekday (Monday = 0), day of month, month — in UTC, as the training data. */
export function writeStamp(out: Int32Array, offset: number, timeMs: number): void {
  const d = new Date(timeMs);
  out[offset] = d.getUTCMinutes();
  out[offset + 1] = d.getUTCHours();
  out[offset + 2] = (d.getUTCDay() + 6) % 7;
  out[offset + 3] = d.getUTCDate();
  out[offset + 4] = d.getUTCMonth() + 1;
}

async function cpuData(t: OrtTensor): Promise<Float32Array> {
  if (t.location && t.location !== 'cpu' && t.getData) return (await t.getData()) as Float32Array;
  return t.data as Float32Array;
}

function disposeState(s: LmState): void {
  s.hLast.dispose();
  for (const t of Object.values(s.cache)) t.dispose();
}

export async function runKronos(
  ort: OrtLike,
  sess: KronosSessions,
  meta: KronosMeta,
  candles: Candle[],
  futureTimes: number[],
  params: KronosParams,
  onProgress: Progress = () => {},
  isCancelled: () => boolean = () => false,
): Promise<KronosResult> {
  const T = candles.length;
  const P = params.predLen;
  const clip = params.clip ?? 5;
  const maxCtx = meta.max_context;
  if (T < 2) throw new Error('Se necesitan al menos 2 velas de contexto');
  if (futureTimes.length !== P) throw new Error('futureTimes no coincide con el horizonte');

  // Per-column z-score over the context, clipped, exactly as KronosPredictor.predict.
  const mean = new Float64Array(FEATS);
  const std = new Float64Array(FEATS);
  const raw = new Float64Array(T * FEATS);
  for (let i = 0; i < T; i++) {
    const c = candles[i]!;
    const row = [c.o, c.h, c.l, c.c, c.v, c.a];
    for (let f = 0; f < FEATS; f++) {
      raw[i * FEATS + f] = row[f]!;
      mean[f]! += row[f]! / T;
    }
  }
  for (let i = 0; i < T; i++) for (let f = 0; f < FEATS; f++) std[f]! += (raw[i * FEATS + f]! - mean[f]!) ** 2 / T;
  for (let f = 0; f < FEATS; f++) std[f] = Math.sqrt(std[f]!) + 1e-5;
  const x = new Float32Array(T * FEATS);
  for (let i = 0; i < T; i++) {
    for (let f = 0; f < FEATS; f++) {
      const v = (raw[i * FEATS + f]! - mean[f]!) / std[f]!;
      x[i * FEATS + f] = Math.max(-clip, Math.min(clip, v));
    }
  }

  onProgress('tokenize', 0, 1);
  const encOut = await sess.enc.run({ x: new ort.Tensor('float32', x, [1, T, FEATS]) });
  const z = await cpuData(encOut.z!);
  encOut.z!.dispose();
  const ctx1 = new Int32Array(T);
  const ctx2 = new Int32Array(T);
  for (let i = 0; i < T; i++) {
    let a = 0;
    let b = 0;
    for (let k = 0; k < BITS; k++) {
      if (z[i * 2 * BITS + k]! > 0) a |= 1 << k;
      if (z[i * 2 * BITS + BITS + k]! > 0) b |= 1 << k;
    }
    ctx1[i] = a;
    ctx2[i] = b;
  }

  const stamps = new Int32Array((T + P) * 5);
  for (let i = 0; i < T; i++) writeStamp(stamps, i * 5, candles[i]!.t);
  for (let i = 0; i < P; i++) writeStamp(stamps, (T + i) * 5, futureTimes[i]!);

  const emptyCache = (): Record<string, OrtTensor> => {
    const c: Record<string, OrtTensor> = {
      ck: new ort.Tensor('float32', new Float32Array(0), [1, meta.cross_heads, 0, meta.cross_head_dim]),
      cv: new ort.Tensor('float32', new Float32Array(0), [1, meta.cross_heads, 0, meta.cross_head_dim]),
    };
    for (let l = 0; l < meta.n_layers; l++) {
      c[`pk${l}`] = new ort.Tensor('float32', new Float32Array(0), [1, meta.n_heads, 0, meta.head_dim]);
      c[`pv${l}`] = new ort.Tensor('float32', new Float32Array(0), [1, meta.n_heads, 0, meta.head_dim]);
    }
    return c;
  };

  /** Run the language model on `tn` new tokens per sample on top of `cache` (null = start from scratch). */
  const lmRun = async (
    batch: number,
    tn: number,
    s1: Int32Array,
    s2: Int32Array,
    stamp: Int32Array,
    cache: Record<string, OrtTensor> | null,
    cacheLen: number,
  ): Promise<LmState> => {
    const fresh = cache ? null : emptyCache();
    const out = await sess.lm.run({
      s1: new ort.Tensor('int32', s1, [batch, tn]),
      s2: new ort.Tensor('int32', s2, [batch, tn]),
      stamp: new ort.Tensor('int32', stamp, [batch, tn, 5]),
      ...(cache ?? fresh!),
    });
    const next: Record<string, OrtTensor> = { ck: out.nck!, cv: out.ncv! };
    for (let l = 0; l < meta.n_layers; l++) {
      next[`pk${l}`] = out[`nk${l}`]!;
      next[`pv${l}`] = out[`nv${l}`]!;
    }
    const logits = await cpuData(out.logits!);
    out.logits!.dispose();
    return { logits, hLast: out.h_last!, cache: next, batch, len: cacheLen + tn };
  };

  const start = Math.max(0, T - maxCtx);
  onProgress('context', 0, 1);
  const prefix = await lmRun(1, T - start, ctx1.slice(start), ctx2.slice(start), stamps.slice(start * 5, T * 5), null, 0);

  const S = params.samples;
  const rng = makeRng(params.seed);
  const paths: Float32Array[] = [];
  const window = Math.min(T + P, maxCtx);
  const decodeChunk = Math.max(1, Math.floor(DECODE_BUDGET_BYTES / (16 * window * window)));

  try {
    for (let done = 0; done < S; done += params.batch) {
      const b = Math.min(params.batch, S - done);
      const gen1 = new Int32Array(b * P);
      const gen2 = new Int32Array(b * P);
      let st = prefix;
      try {
        for (let i = 0; i < P; i++) {
          if (isCancelled()) throw new CancelledError();
          const s1new = new Int32Array(b);
          for (let j = 0; j < b; j++) {
            const row = st.batch === 1 ? 0 : j;
            s1new[j] = sampleToken(st.logits, row * meta.s1_vocab, meta.s1_vocab, params.temperature, params.topK, params.topP, rng);
          }
          const s2out = await sess.s2.run({
            h_last: st.hLast,
            s1: new ort.Tensor('int32', s1new, [b]),
            ck: st.cache.ck!,
            cv: st.cache.cv!,
          });
          const l2 = await cpuData(s2out.logits!);
          s2out.logits!.dispose();
          const s2new = new Int32Array(b);
          for (let j = 0; j < b; j++) {
            s2new[j] = sampleToken(l2, j * meta.s2_vocab, meta.s2_vocab, params.temperature, params.topK, params.topP, rng);
            gen1[j * P + i] = s1new[j]!;
            gen2[j * P + i] = s2new[j]!;
          }
          onProgress('predict', done * P + (i + 1) * b, S * P);
          if (i === P - 1) break;

          let next: LmState;
          if (st.len >= maxCtx) {
            // Window is full: rebuild the cache from the most recent tokens of every sample. The exact mode
            // keeps the full window, as the original does; the fast one leaves room for REPRIME steps.
            const w = params.fastWindow ? maxCtx - Math.min(REPRIME, maxCtx >> 2) : maxCtx;
            const end = T + i + 1;
            const w1 = new Int32Array(b * w);
            const w2 = new Int32Array(b * w);
            const ws = new Int32Array(b * w * 5);
            for (let j = 0; j < b; j++) {
              for (let k = 0; k < w; k++) {
                const pos = end - w + k;
                w1[j * w + k] = pos < T ? ctx1[pos]! : gen1[j * P + pos - T]!;
                w2[j * w + k] = pos < T ? ctx2[pos]! : gen2[j * P + pos - T]!;
              }
              ws.set(stamps.subarray((end - w) * 5, end * 5), j * w * 5);
            }
            next = await lmRun(b, w, w1, w2, ws, null, 0);
          } else {
            const stamp = new Int32Array(b * 5);
            for (let j = 0; j < b; j++) stamp.set(stamps.subarray((T + i) * 5, (T + i + 1) * 5), j * 5);
            next = await lmRun(b, 1, s1new, s2new, stamp, st.cache, st.len);
          }
          if (st !== prefix) disposeState(st);
          st = next;
        }
      } finally {
        if (st !== prefix) disposeState(st);
      }

      // Decode the same trailing window upstream uses, a few samples at a time.
      const first = T + P - window;
      for (let c0 = 0; c0 < b; c0 += decodeChunk) {
        const cb = Math.min(decodeChunk, b - c0);
        const bits = new Float32Array(cb * window * 2 * BITS);
        for (let j = 0; j < cb; j++) {
          for (let k = 0; k < window; k++) {
            const pos = first + k;
            const a = pos < T ? ctx1[pos]! : gen1[(c0 + j) * P + pos - T]!;
            const d = pos < T ? ctx2[pos]! : gen2[(c0 + j) * P + pos - T]!;
            const o = (j * window + k) * 2 * BITS;
            for (let q = 0; q < BITS; q++) {
              bits[o + q] = (a >> q) & 1 ? 1 : -1;
              bits[o + BITS + q] = (d >> q) & 1 ? 1 : -1;
            }
          }
        }
        const decOut = await sess.dec.run({ bits: new ort.Tensor('float32', bits, [cb, window, 2 * BITS]) });
        const y = await cpuData(decOut.y!);
        decOut.y!.dispose();
        for (let j = 0; j < cb; j++) {
          const path = new Float32Array(P * FEATS);
          for (let i = 0; i < P; i++) {
            const src = (j * window + window - P + i) * FEATS;
            for (let f = 0; f < FEATS; f++) path[i * FEATS + f] = y[src + f]! * std[f]! + mean[f]!;
          }
          paths.push(path);
        }
      }
      onProgress('decode', paths.length, S);
    }
  } finally {
    disposeState(prefix);
  }
  return { paths };
}
