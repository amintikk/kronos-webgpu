export interface Settings {
  source: string;
  symbol: string;
  interval: string;
  lookback: number;
  includeOpen: boolean;
  live: boolean;
  model: string;
  backend: string;
  threads: number;
  horizon: number;
  samples: number;
  batch: number;
  temperature: number;
  topP: number;
  topK: number;
  seed: number;
  randomSeed: boolean;
  offset: number;
  autoPredict: boolean;
  bands: boolean;
  paths: boolean;
  meanCandles: boolean;
  median: boolean;
  volume: boolean;
  log: boolean;
  localTime: boolean;
  theme: string;
  btWindows: number;
  btStride: number;
  btSamples: number;
  target: number;
}

export const DEFAULTS: Settings = {
  source: 'binance',
  symbol: 'BTCUSDT',
  interval: '1h',
  lookback: 400,
  includeOpen: false,
  live: true,
  model: 'kronos:small',
  backend: 'auto',
  threads: 0,
  horizon: 48,
  samples: 16,
  batch: 8,
  temperature: 1,
  topP: 0.9,
  topK: 0,
  seed: 42,
  randomSeed: true,
  offset: 0,
  autoPredict: false,
  bands: true,
  paths: true,
  meanCandles: true,
  median: true,
  volume: true,
  log: false,
  localTime: true,
  theme: 'auto',
  btWindows: 8,
  btStride: 0,
  btSamples: 8,
  target: 0,
};

const KEY = 'kronos-settings';
/** Settings that describe a forecast, and therefore travel in a shared link. */
const SHARED: (keyof Settings)[] = ['source', 'symbol', 'interval', 'lookback', 'includeOpen', 'model', 'horizon', 'samples', 'temperature', 'topP', 'topK', 'seed', 'randomSeed', 'offset'];

function coerce<K extends keyof Settings>(key: K, raw: unknown): Settings[K] | undefined {
  const def = DEFAULTS[key];
  if (typeof def === 'number') {
    const n = Number(raw);
    return (Number.isFinite(n) ? n : undefined) as Settings[K] | undefined;
  }
  if (typeof def === 'boolean') return (raw === true || raw === 'true' || raw === '1') as Settings[K];
  return (typeof raw === 'string' ? raw : undefined) as Settings[K] | undefined;
}

export function loadSettings(): Settings {
  const s: Settings = { ...DEFAULTS };
  const apply = (obj: Record<string, unknown>, keys: (keyof Settings)[]) => {
    for (const k of keys) {
      if (!(k in obj)) continue;
      const v = coerce(k, obj[k]);
      if (v !== undefined) (s as unknown as Record<string, unknown>)[k] = v;
    }
  };
  try {
    apply(JSON.parse(localStorage.getItem(KEY) ?? '{}'), Object.keys(DEFAULTS) as (keyof Settings)[]);
  } catch {
    /* storage blocked or corrupt: defaults */
  }
  if (location.hash.length > 1) {
    apply(Object.fromEntries(new URLSearchParams(location.hash.slice(1))), SHARED);
    history.replaceState(null, '', location.pathname);
  }
  if (s.source === 'csv') s.source = DEFAULTS.source; // an uploaded file does not survive a reload
  return s;
}

export function saveSettings(s: Settings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* storage blocked */
  }
}

export function shareUrl(s: Settings): string {
  const p = new URLSearchParams();
  for (const k of SHARED) if (s[k] !== DEFAULTS[k]) p.set(k, String(s[k]));
  const hash = p.toString();
  return `${location.origin}${location.pathname}${hash ? `#${hash}` : ''}`;
}
