import type { ChronosResult } from './engine/chronos.ts';
import type { KronosResult } from './engine/kronos.ts';
import type { Candle } from './engine/types.ts';

export interface Band {
  lo: number;
  hi: number;
}

/** A forecast in a model-independent shape: a quantile fan of the close, plus whatever else the model offers. */
export interface Forecast {
  id: number;
  model: string;
  family: 'kronos' | 'chronos';
  label: string;
  /** Open times (ms) of the predicted candles. */
  times: number[];
  /** Last context candle: the forecast starts after it. */
  anchor: Candle;
  levels: number[];
  /** q[levelIndex][step] — close price quantiles. */
  q: Float64Array[];
  /** Index pairs into `levels` for the outer and inner band. */
  outer: Band;
  inner: Band;
  median: number;
  /** Mean predicted candle per step (Kronos only), as upstream averages its samples. */
  mean?: Candle[];
  /** Raw sample paths (Kronos only): rows of [o,h,l,c,v,a]. */
  paths?: Float32Array[];
  samples: number;
  ms: number;
  settings: Record<string, string | number>;
}

export function quantile(sorted: ArrayLike<number>, p: number): number {
  const n = sorted.length;
  if (n === 0) return NaN;
  const pos = (n - 1) * p;
  const lo = Math.floor(pos);
  const hi = Math.min(n - 1, lo + 1);
  return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (pos - lo);
}

const KRONOS_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9];

export function fromKronos(res: KronosResult, times: number[], anchor: Candle): Pick<Forecast, 'levels' | 'q' | 'outer' | 'inner' | 'median' | 'mean' | 'paths' | 'samples'> {
  const P = times.length;
  const S = res.paths.length;
  const q = KRONOS_LEVELS.map(() => new Float64Array(P));
  const mean: Candle[] = [];
  const col = new Float64Array(S);
  for (let i = 0; i < P; i++) {
    const m = [0, 0, 0, 0, 0, 0];
    for (let s = 0; s < S; s++) {
      const p = res.paths[s]!;
      for (let f = 0; f < 6; f++) m[f]! += p[i * 6 + f]! / S;
      col[s] = p[i * 6 + 3]!;
    }
    col.sort();
    KRONOS_LEVELS.forEach((lv, k) => (q[k]![i] = quantile(col, lv)));
    const [o, h, l, c] = [m[0]!, m[1]!, m[2]!, m[3]!];
    mean.push({ t: times[i]!, o, h: Math.max(o, h, l, c), l: Math.min(o, h, l, c), c, v: Math.max(0, m[4]!), a: Math.max(0, m[5]!) });
  }
  void anchor;
  return { levels: KRONOS_LEVELS, q, outer: { lo: 0, hi: 4 }, inner: { lo: 1, hi: 3 }, median: 2, mean, paths: res.paths, samples: S };
}

export function fromChronos(res: ChronosResult): Pick<Forecast, 'levels' | 'q' | 'outer' | 'inner' | 'median' | 'samples'> {
  const lv = res.quantileLevels;
  const at = (p: number) => lv.findIndex((x) => Math.abs(x - p) < 1e-6);
  return {
    levels: lv,
    q: res.quantiles.map((a) => Float64Array.from(a)),
    outer: { lo: at(0.1), hi: at(0.9) },
    inner: { lo: at(0.3), hi: at(0.7) },
    median: at(0.5),
    samples: 0,
  };
}

export interface Summary {
  last: number;
  endMedian: number;
  endLo: number;
  endHi: number;
  loLevel: number;
  hiLevel: number;
  changePct: number;
  /** Probability that the final close is above the last close; null bounds mean "outside the quantile range". */
  pUp: number;
  pUpBound: '' | '<' | '>';
  /** Kronos only. */
  expectedHigh?: number;
  expectedLow?: number;
  endStdPct?: number;
  medianDrawdownPct?: number;
  medianRunupPct?: number;
}

export function summarize(f: Forecast): Summary {
  const end = f.times.length - 1;
  const last = f.anchor.c;
  const endMedian = f.q[f.median]![end]!;
  const s: Summary = {
    last,
    endMedian,
    endLo: f.q[f.outer.lo]![end]!,
    endHi: f.q[f.outer.hi]![end]!,
    loLevel: f.levels[f.outer.lo]!,
    hiLevel: f.levels[f.outer.hi]!,
    changePct: (endMedian / last - 1) * 100,
    pUp: 0.5,
    pUpBound: '',
  };
  if (f.paths?.length) {
    const P = f.times.length;
    const n = f.paths.length;
    let up = 0;
    const highs = new Float64Array(n);
    const lows = new Float64Array(n);
    const dd = new Float64Array(n);
    const ru = new Float64Array(n);
    let m1 = 0;
    let m2 = 0;
    f.paths.forEach((p, k) => {
      const endClose = p[end * 6 + 3]!;
      if (endClose > last) up++;
      const r = endClose / last - 1;
      m1 += r / n;
      m2 += (r * r) / n;
      let hi = -Infinity;
      let lo = Infinity;
      for (let i = 0; i < P; i++) {
        hi = Math.max(hi, p[i * 6 + 1]!, p[i * 6 + 3]!);
        lo = Math.min(lo, p[i * 6 + 2]!, p[i * 6 + 3]!);
      }
      highs[k] = hi;
      lows[k] = lo;
      dd[k] = (lo / last - 1) * 100;
      ru[k] = (hi / last - 1) * 100;
    });
    s.pUp = up / n;
    s.expectedHigh = quantile(highs.sort(), 0.5);
    s.expectedLow = quantile(lows.sort(), 0.5);
    s.endStdPct = Math.sqrt(Math.max(0, m2 - m1 * m1)) * 100;
    s.medianDrawdownPct = quantile(dd.sort(), 0.5);
    s.medianRunupPct = quantile(ru.sort(), 0.5);
  } else {
    const cdf = cdfAt(f, end, last);
    s.pUp = 1 - cdf.p;
    s.pUpBound = cdf.bound === '<' ? '>' : cdf.bound === '>' ? '<' : '';
  }
  return s;
}

/** P(close at `step` <= price), interpolated from the quantile fan. */
function cdfAt(f: Forecast, step: number, price: number): { p: number; bound: '' | '<' | '>' } {
  const qs = f.q.map((a) => a[step]!);
  if (price <= qs[0]!) return { p: f.levels[0]!, bound: '<' };
  const n = qs.length - 1;
  if (price >= qs[n]!) return { p: f.levels[n]!, bound: '>' };
  for (let k = 0; k < n; k++) {
    if (price <= qs[k + 1]!) {
      const w = (price - qs[k]!) / (qs[k + 1]! - qs[k]! || 1);
      return { p: f.levels[k]! + w * (f.levels[k + 1]! - f.levels[k]!), bound: '' };
    }
  }
  return { p: 0.5, bound: '' };
}

export interface TargetOdds {
  /** Probability the price trades at the target at any point of the horizon (Kronos only). */
  touch?: number;
  /** Probability the final close is beyond the target (above it if the target is above the last close). */
  endBeyond: number;
  bound: '' | '<' | '>';
  above: boolean;
}

export function targetOdds(f: Forecast, target: number): TargetOdds {
  const above = target >= f.anchor.c;
  const end = f.times.length - 1;
  if (f.paths?.length) {
    let touch = 0;
    let beyond = 0;
    for (const p of f.paths) {
      let hit = false;
      for (let i = 0; i <= end && !hit; i++) hit = above ? Math.max(p[i * 6 + 1]!, p[i * 6 + 3]!) >= target : Math.min(p[i * 6 + 2]!, p[i * 6 + 3]!) <= target;
      if (hit) touch++;
      if (above ? p[end * 6 + 3]! >= target : p[end * 6 + 3]! <= target) beyond++;
    }
    return { touch: touch / f.paths.length, endBeyond: beyond / f.paths.length, bound: '', above };
  }
  const cdf = cdfAt(f, end, target);
  if (above) return { endBeyond: 1 - cdf.p, bound: cdf.bound === '>' ? '<' : cdf.bound === '<' ? '>' : '', above };
  return { endBeyond: cdf.p, bound: cdf.bound, above };
}

export interface Score {
  steps: number;
  mae: number;
  mapePct: number;
  naiveMae: number;
  /** 1 - MAE / MAE of "price stays where it is". Positive = better than doing nothing. */
  skill: number;
  directionHit: boolean | null;
  /** Share of actual closes inside the outer band. */
  coverage: number;
  endErrorPct: number;
  predictedReturnPct: number;
  actualReturnPct: number;
}

/** Compare a forecast with what actually happened. `actual` holds real candles keyed by open time. */
export function score(f: Forecast, actual: Map<number, Candle>): Score | null {
  let n = 0;
  let abs = 0;
  let ape = 0;
  let naive = 0;
  let inside = 0;
  let lastIdx = -1;
  const last = f.anchor.c;
  f.times.forEach((t, i) => {
    const a = actual.get(t);
    if (!a) return;
    const pred = f.q[f.median]![i]!;
    n++;
    abs += Math.abs(pred - a.c);
    ape += Math.abs(pred - a.c) / a.c;
    naive += Math.abs(last - a.c);
    if (a.c >= f.q[f.outer.lo]![i]! && a.c <= f.q[f.outer.hi]![i]!) inside++;
    lastIdx = i;
  });
  if (!n) return null;
  const realEnd = actual.get(f.times[lastIdx]!)!.c;
  const predEnd = f.q[f.median]![lastIdx]!;
  const pr = predEnd / last - 1;
  const ar = realEnd / last - 1;
  return {
    steps: n,
    mae: abs / n,
    mapePct: (ape / n) * 100,
    naiveMae: naive / n,
    skill: naive > 0 ? 1 - abs / naive : 0,
    directionHit: pr === 0 || ar === 0 ? null : Math.sign(pr) === Math.sign(ar),
    coverage: inside / n,
    endErrorPct: (predEnd / realEnd - 1) * 100,
    predictedReturnPct: pr * 100,
    actualReturnPct: ar * 100,
  };
}
