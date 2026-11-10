import { t } from '../i18n.ts';
import type { Candle } from '../engine/types.ts';

const ALIASES: Record<string, string[]> = {
  t: ['timestamps', 'timestamp', 'time', 'date', 'datetime', 'open_time', 'opentime', 't'],
  o: ['open', 'o'],
  h: ['high', 'h'],
  l: ['low', 'l'],
  c: ['close', 'c'],
  v: ['volume', 'vol', 'v'],
  a: ['amount', 'quote_volume', 'quote_asset_volume', 'turnover', 'a'],
};

function parseTime(raw: string): number {
  const n = Number(raw);
  if (Number.isFinite(n) && raw.trim() !== '') return n < 1e11 ? n * 1000 : n; // seconds vs milliseconds
  const iso = /[zZ]|[+-]\d\d:?\d\d$/.test(raw) ? raw : `${raw.trim().replace(' ', 'T')}Z`;
  return Date.parse(iso);
}

/** Parse an OHLCV CSV with a header row. Returns candles sorted by time and the inferred bar length. */
export function parseCsv(text: string): { candles: Candle[]; stepMs: number } {
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  if (lines.length < 3) throw new Error(t('El CSV está vacío'));
  const sep = [',', ';', '\t'].sort((a, b) => lines[0]!.split(b).length - lines[0]!.split(a).length)[0]!;
  const header = lines[0]!.split(sep).map((h) => h.trim().toLowerCase().replace(/^"|"$/g, ''));
  const col = (key: string): number => header.findIndex((h) => ALIASES[key]!.includes(h));
  const idx = { t: col('t'), o: col('o'), h: col('h'), l: col('l'), c: col('c'), v: col('v'), a: col('a') };
  for (const k of ['t', 'o', 'h', 'l', 'c'] as const) {
    if (idx[k] < 0) throw new Error(t('Falta la columna «{c}» en el CSV (cabeceras: {h})', { c: ALIASES[k]![0]!, h: header.join(', ') }));
  }
  const candles: Candle[] = [];
  for (const line of lines.slice(1)) {
    const f = line.split(sep).map((s) => s.trim().replace(/^"|"$/g, ''));
    const o = Number(f[idx.o]);
    const h = Number(f[idx.h]);
    const l = Number(f[idx.l]);
    const c = Number(f[idx.c]);
    const t = parseTime(f[idx.t] ?? '');
    if (![t, o, h, l, c].every(Number.isFinite)) continue;
    const v = idx.v >= 0 ? Number(f[idx.v]) || 0 : 0;
    const a = idx.a >= 0 ? Number(f[idx.a]) || 0 : v * ((o + h + l + c) / 4);
    candles.push({ t, o, h, l, c, v, a });
  }
  if (candles.length < 32) throw new Error(t('El CSV necesita al menos 32 filas válidas'));
  candles.sort((x, y) => x.t - y.t);
  const diffs = candles.slice(1).map((c, i) => c.t - candles[i]!.t).sort((x, y) => x - y);
  const stepMs = diffs[diffs.length >> 1]!;
  if (!(stepMs > 0)) throw new Error(t('No se pudo deducir el intervalo del CSV'));
  return { candles, stepMs };
}
