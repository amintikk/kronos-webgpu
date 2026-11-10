import type { Lang } from '../i18n.ts';
import type { Candle, KronosParams } from './types.ts';

export type Backend = 'webgpu' | 'wasm';
export type Family = 'kronos' | 'chronos';

export type Request =
  | { type: 'init'; backend: Backend; threads: number; lang: Lang }
  | { type: 'load'; id: number; family: Family; key: string }
  | { type: 'kronos'; id: number; key: string; candles: Candle[]; futureTimes: number[]; params: KronosParams }
  | { type: 'chronos'; id: number; key: string; series: number[]; predLen: number }
  | { type: 'cancel' }
  | { type: 'release'; id: number };

export type Response =
  | { type: 'ready'; backend: Backend; threads: number }
  | { type: 'progress'; id: number; stage: string; done: number; total: number }
  | { type: 'download'; id: number; file: string; loaded: number; total: number; cached: boolean }
  | { type: 'done'; id: number; result?: unknown; ms: number }
  | { type: 'error'; id: number; message: string; cancelled: boolean };
