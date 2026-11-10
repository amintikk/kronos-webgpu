import { t } from '../i18n.ts';
import type { ChronosMeta, OrtLike, OrtSession, Progress } from './types.ts';

export interface ChronosResult {
  quantileLevels: number[];
  /** quantiles[q][step] */
  quantiles: Float32Array[];
}

/**
 * Chronos-Bolt forecasts `prediction_length` steps in one pass. Longer horizons are rolled out by
 * feeding the median back as context, like the reference pipeline does.
 */
export async function runChronos(
  ort: OrtLike,
  session: OrtSession,
  meta: ChronosMeta,
  series: number[],
  predLen: number,
  onProgress: Progress = () => {},
): Promise<ChronosResult> {
  const nq = meta.quantiles.length;
  const mid = meta.quantiles.indexOf(0.5);
  const quantiles = Array.from({ length: nq }, () => new Float32Array(predLen));
  const context = series.slice();
  const rounds = Math.ceil(predLen / meta.prediction_length);
  for (let done = 0, r = 0; done < predLen; done += meta.prediction_length, r++) {
    let n = Math.min(context.length, meta.context_length);
    n -= n % meta.patch;
    if (n < meta.patch) throw new Error(t('Chronos necesita al menos {n} velas de contexto', { n: meta.patch }));
    const input = Float32Array.from(context.slice(context.length - n));
    const out = await session.run({ context: new ort.Tensor('float32', input, [1, n]) });
    const q = out.quantiles!;
    const data = (q.location && q.location !== 'cpu' && q.getData ? await q.getData() : q.data) as Float32Array;
    q.dispose();
    const take = Math.min(meta.prediction_length, predLen - done);
    for (let k = 0; k < nq; k++) {
      for (let i = 0; i < take; i++) quantiles[k]![done + i] = data[k * meta.prediction_length + i]!;
    }
    for (let i = 0; i < take; i++) context.push(data[mid * meta.prediction_length + i]!);
    onProgress('predict', r + 1, rounds);
  }
  return { quantileLevels: meta.quantiles.slice(), quantiles };
}
