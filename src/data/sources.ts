import type { Candle } from '../engine/types.ts';

export const INTERVAL_MS: Record<string, number> = {
  '1m': 60e3,
  '5m': 300e3,
  '15m': 900e3,
  '30m': 1800e3,
  '1h': 3600e3,
  '2h': 7200e3,
  '4h': 14400e3,
  '6h': 21600e3,
  '12h': 43200e3,
  '1d': 86400e3,
  '1w': 604800e3,
};

export interface SymbolInfo {
  symbol: string;
  /** 24 h quote volume when the exchange reports it; used only for ordering. */
  volume: number;
}

export interface Source {
  id: string;
  name: string;
  intervals: string[];
  defaultSymbol: string;
  /** Candles ending at `endMs` (open time <= endMs), oldest first, at most `pageSize` of them. */
  page(symbol: string, interval: string, endMs: number | null): Promise<Candle[]>;
  symbols(): Promise<SymbolInfo[]>;
  /** Live updates of the candle in progress. Returns an unsubscribe function. */
  subscribe?(symbol: string, interval: string, onCandle: (c: Candle) => void): () => void;
}

async function getJson(url: string): Promise<any> {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = '';
    try {
      detail = (await res.text()).slice(0, 160);
    } catch {
      /* body not readable */
    }
    throw new Error(`${new URL(url).host} respondió ${res.status}${detail ? `: ${detail}` : ''}`);
  }
  return res.json();
}

/** Try each base URL in order; exchanges expose mirrors that are reachable from different regions. */
async function firstOk(bases: string[], path: string): Promise<any> {
  let last: unknown;
  for (const base of bases) {
    try {
      return await getJson(base + path);
    } catch (e) {
      last = e;
    }
  }
  throw last;
}

const num = (v: unknown): number => Number(v);
const typical = (o: number, h: number, l: number, c: number): number => (o + h + l + c) / 4;

const BINANCE = ['https://data-api.binance.vision', 'https://api.binance.com', 'https://api1.binance.com'];

const binance: Source = {
  id: 'binance',
  name: 'Binance',
  intervals: ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w'],
  defaultSymbol: 'BTCUSDT',
  async page(symbol, interval, endMs) {
    const q = `/api/v3/klines?symbol=${encodeURIComponent(symbol)}&interval=${interval}&limit=1000${endMs ? `&endTime=${endMs}` : ''}`;
    const rows = (await firstOk(BINANCE, q)) as unknown[][];
    return rows.map((r) => ({ t: num(r[0]), o: num(r[1]), h: num(r[2]), l: num(r[3]), c: num(r[4]), v: num(r[5]), a: num(r[7]) }));
  },
  async symbols() {
    const rows = (await firstOk(BINANCE, '/api/v3/ticker/24hr?type=MINI')) as { symbol: string; quoteVolume: string }[];
    return rows.map((r) => ({ symbol: r.symbol, volume: usdWeight(r.symbol, num(r.quoteVolume)) }));
  },
  subscribe(symbol, interval, onCandle) {
    let ws: WebSocket | null = null;
    let closed = false;
    let retry = 0;
    let timer = 0;
    const connect = () => {
      ws = new WebSocket(`wss://data-stream.binance.vision/ws/${symbol.toLowerCase()}@kline_${interval}`);
      ws.onopen = () => (retry = 0);
      ws.onmessage = (ev) => {
        const k = JSON.parse(ev.data as string).k;
        if (k) onCandle({ t: num(k.t), o: num(k.o), h: num(k.h), l: num(k.l), c: num(k.c), v: num(k.v), a: num(k.q) });
      };
      ws.onclose = () => {
        if (!closed) timer = window.setTimeout(connect, Math.min(30e3, 1e3 * 2 ** retry++));
      };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(timer);
      ws?.close();
    };
  },
};

/** Rank pairs quoted in dollars first; other quote currencies are not comparable by raw volume. */
function usdWeight(symbol: string, quoteVolume: number): number {
  return /(USDT|USDC|USD|FDUSD)$/.test(symbol.replace('-', '')) ? quoteVolume : quoteVolume * 1e-9;
}

const BYBIT_INTERVAL: Record<string, string> = { '1m': '1', '5m': '5', '15m': '15', '30m': '30', '1h': '60', '2h': '120', '4h': '240', '6h': '360', '12h': '720', '1d': 'D', '1w': 'W' };

const bybit: Source = {
  id: 'bybit',
  name: 'Bybit',
  intervals: Object.keys(BYBIT_INTERVAL),
  defaultSymbol: 'BTCUSDT',
  async page(symbol, interval, endMs) {
    const q = `https://api.bybit.com/v5/market/kline?category=spot&symbol=${encodeURIComponent(symbol)}&interval=${BYBIT_INTERVAL[interval]}&limit=1000${endMs ? `&end=${endMs}` : ''}`;
    const j = await getJson(q);
    if (j.retCode !== 0) throw new Error(`Bybit: ${j.retMsg}`);
    const rows = (j.result.list as string[][]).map((r) => ({ t: num(r[0]), o: num(r[1]), h: num(r[2]), l: num(r[3]), c: num(r[4]), v: num(r[5]), a: num(r[6]) }));
    return rows.reverse();
  },
  async symbols() {
    const j = await getJson('https://api.bybit.com/v5/market/tickers?category=spot');
    return (j.result.list as { symbol: string; turnover24h: string }[]).map((r) => ({ symbol: r.symbol, volume: usdWeight(r.symbol, num(r.turnover24h)) }));
  },
};

const OKX_BAR: Record<string, string> = { '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6Hutc', '12h': '12Hutc', '1d': '1Dutc', '1w': '1Wutc' };

const okx: Source = {
  id: 'okx',
  name: 'OKX',
  intervals: Object.keys(OKX_BAR),
  defaultSymbol: 'BTC-USDT',
  async page(symbol, interval, endMs) {
    // `candles` only reaches back ~1440 bars; older pages come from `history-candles`.
    const recent = !endMs || Date.now() - endMs < 1000 * INTERVAL_MS[interval]!;
    const path = recent ? 'candles' : 'history-candles';
    const q = `https://www.okx.com/api/v5/market/${path}?instId=${encodeURIComponent(symbol)}&bar=${OKX_BAR[interval]}&limit=${recent ? 300 : 100}${endMs ? `&after=${endMs + 1}` : ''}`;
    const j = await getJson(q);
    if (j.code !== '0') throw new Error(`OKX: ${j.msg}`);
    const rows = (j.data as string[][]).map((r) => ({ t: num(r[0]), o: num(r[1]), h: num(r[2]), l: num(r[3]), c: num(r[4]), v: num(r[5]), a: num(r[7]) }));
    return rows.reverse();
  },
  async symbols() {
    const j = await getJson('https://www.okx.com/api/v5/market/tickers?instType=SPOT');
    return (j.data as { instId: string; volCcy24h: string }[]).map((r) => ({ symbol: r.instId, volume: usdWeight(r.instId, num(r.volCcy24h)) }));
  },
};

const COINBASE_STEP: Record<string, number> = { '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '6h': 21600, '1d': 86400 };

const coinbase: Source = {
  id: 'coinbase',
  name: 'Coinbase',
  intervals: Object.keys(COINBASE_STEP),
  defaultSymbol: 'BTC-USD',
  async page(symbol, interval, endMs) {
    const step = COINBASE_STEP[interval]! * 1000;
    const end = endMs ?? Date.now();
    const start = end - 299 * step;
    const q = `https://api.exchange.coinbase.com/products/${encodeURIComponent(symbol)}/candles?granularity=${step / 1000}&start=${new Date(start).toISOString()}&end=${new Date(end).toISOString()}`;
    const rows = (await getJson(q)) as number[][];
    return rows
      .map((r) => ({ t: r[0]! * 1000, o: r[3]!, h: r[2]!, l: r[1]!, c: r[4]!, v: r[5]!, a: r[5]! * typical(r[3]!, r[2]!, r[1]!, r[4]!) }))
      .reverse();
  },
  async symbols() {
    const rows = (await getJson('https://api.exchange.coinbase.com/products')) as { id: string; status: string; quote_currency: string }[];
    const major = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD', 'LINK-USD', 'AVAX-USD', 'LTC-USD'];
    return rows
      .filter((r) => r.status === 'online')
      .map((r) => ({ symbol: r.id, volume: major.includes(r.id) ? 1e12 - major.indexOf(r.id) : r.quote_currency === 'USD' ? 1 : 0 }));
  },
};

export const SOURCES: Source[] = [binance, bybit, okx, coinbase];
export const sourceById = (id: string): Source => SOURCES.find((s) => s.id === id) ?? binance;

/** Fetch the most recent `count` candles (ending at `endMs` if given), paging backwards as needed. */
export async function fetchCandles(source: Source, symbol: string, interval: string, count: number, endMs: number | null = null): Promise<Candle[]> {
  const step = INTERVAL_MS[interval]!;
  const byTime = new Map<number, Candle>();
  let end = endMs;
  for (let guard = 0; guard < 60 && byTime.size < count; guard++) {
    const page = await source.page(symbol, interval, end);
    const before = byTime.size;
    for (const c of page) if (Number.isFinite(c.c) && Number.isFinite(c.t)) byTime.set(c.t, c);
    if (!page.length || byTime.size === before) break;
    end = page[0]!.t - step;
  }
  const all = [...byTime.values()].sort((a, b) => a.t - b.t);
  return all.slice(-count);
}

/** Poll-based live feed for exchanges without a public browser websocket wired in. */
export function subscribe(source: Source, symbol: string, interval: string, onCandle: (c: Candle) => void): () => void {
  if (source.subscribe) return source.subscribe(symbol, interval, onCandle);
  let stopped = false;
  const tick = async () => {
    if (stopped || document.hidden) return;
    try {
      const page = await source.page(symbol, interval, null);
      if (!stopped) for (const c of page.slice(-2)) onCandle(c);
    } catch {
      /* transient network error: next tick retries */
    }
  };
  const timer = window.setInterval(tick, 15e3);
  return () => {
    stopped = true;
    clearInterval(timer);
  };
}
