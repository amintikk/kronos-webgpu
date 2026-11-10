/** Deterministic PRNG (mulberry32) so a seed reproduces a forecast. */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const order = new Map<number, Uint16Array>();
const probs = new Map<number, Float64Array>();

/**
 * Sample one token from a row of logits, following Kronos' `sample_from_logits`:
 * temperature, then top-k if set (>0), otherwise nucleus filtering (top-p < 1).
 */
export function sampleToken(
  logits: Float32Array,
  offset: number,
  vocab: number,
  temperature: number,
  topK: number,
  topP: number,
  rng: () => number,
): number {
  let p = probs.get(vocab);
  let idx = order.get(vocab);
  if (!p || !idx) {
    p = new Float64Array(vocab);
    idx = new Uint16Array(vocab);
    probs.set(vocab, p);
    order.set(vocab, idx);
  }
  const invT = 1 / Math.max(temperature, 1e-3);
  let max = -Infinity;
  let argmax = 0;
  for (let i = 0; i < vocab; i++) {
    const v = logits[offset + i]!;
    if (v > max) {
      max = v;
      argmax = i;
    }
  }
  if (topK === 1) return argmax;
  let sum = 0;
  for (let i = 0; i < vocab; i++) {
    const e = Math.exp((logits[offset + i]! - max) * invT);
    p[i] = e;
    sum += e;
  }
  const filterK = topK > 0;
  const filterP = !filterK && topP < 1;
  if (!filterK && !filterP) {
    let r = rng() * sum;
    for (let i = 0; i < vocab; i++) {
      r -= p[i]!;
      if (r <= 0) return i;
    }
    return argmax;
  }
  for (let i = 0; i < vocab; i++) idx[i] = i;
  const pp = p;
  idx.sort((x, y) => pp[y]! - pp[x]!);
  let keep = vocab;
  let kept = 0;
  if (filterK) {
    keep = Math.min(topK, vocab);
    for (let i = 0; i < keep; i++) kept += p[idx[i]!]!;
  } else {
    // Keep tokens up to and including the first one whose cumulative probability exceeds top-p.
    const limit = topP * sum;
    keep = 0;
    while (keep < vocab) {
      kept += p[idx[keep]!]!;
      keep++;
      if (kept > limit) break;
    }
  }
  let r = rng() * kept;
  for (let i = 0; i < keep; i++) {
    r -= p[idx[i]!]!;
    if (r <= 0) return idx[i]!;
  }
  return idx[0]!;
}
