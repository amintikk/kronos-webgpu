import { lang, t } from '../i18n.ts';
import type { ChronosResult } from './chronos.ts';
import type { KronosResult } from './kronos.ts';
import type { Backend, Family, Request, Response } from './protocol.ts';
import type { Candle, KronosParams } from './types.ts';

export interface EngineEvents {
  progress(stage: string, done: number, total: number): void;
  download(file: string, loaded: number, total: number, cached: boolean): void;
}

type Pending = { resolve(v: { result: unknown; ms: number }): void; reject(e: Error): void; events?: Partial<EngineEvents> };
type Job = Request extends infer R ? (R extends { id: number } ? Omit<R, 'id'> : never) : never;

export class CancelError extends Error {}

/** Main-thread handle on the inference worker. One worker per backend; switching backend restarts it. */
export class Engine {
  private worker: Worker | null = null;
  private pending = new Map<number, Pending>();
  private nextId = 1;
  private ready: Promise<{ backend: Backend; threads: number }> | null = null;
  backend: Backend = 'wasm';
  threads = 1;

  start(backend: Backend, threads: number): Promise<{ backend: Backend; threads: number }> {
    this.stop();
    this.backend = backend;
    const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.worker = worker;
    this.ready = new Promise((resolve, reject) => {
      worker.onerror = (e) => {
        reject(new Error(e.message || t('No se pudo iniciar el motor de inferencia')));
        this.failAll(new Error(e.message || t('El motor de inferencia se detuvo')));
      };
      worker.onmessage = (ev: MessageEvent<Response>) => {
        const m = ev.data;
        if (m.type === 'ready') {
          this.threads = m.threads;
          resolve({ backend: m.backend, threads: m.threads });
          return;
        }
        const p = this.pending.get(m.id);
        if (!p) return;
        if (m.type === 'progress') p.events?.progress?.(m.stage, m.done, m.total);
        else if (m.type === 'download') p.events?.download?.(m.file, m.loaded, m.total, m.cached);
        else {
          this.pending.delete(m.id);
          if (m.type === 'done') p.resolve({ result: m.result, ms: m.ms });
          else p.reject(m.cancelled ? new CancelError(m.message) : new Error(m.message));
        }
      };
    });
    this.send({ type: 'init', backend, threads, lang });
    return this.ready;
  }

  stop(): void {
    this.worker?.terminate();
    this.worker = null;
    this.failAll(new CancelError(t('Motor reiniciado')));
  }

  private failAll(e: Error): void {
    for (const p of this.pending.values()) p.reject(e);
    this.pending.clear();
  }

  private send(m: Request): void {
    this.worker?.postMessage(m);
  }

  private async call<T>(msg: Job, events?: Partial<EngineEvents>): Promise<{ result: T; ms: number }> {
    if (!this.ready) throw new Error(t('Motor no iniciado'));
    await this.ready;
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as Pending['resolve'], reject, events });
      this.send({ ...msg, id } as Request);
    });
  }

  load(family: Family, key: string, events?: Partial<EngineEvents>) {
    return this.call<void>({ type: 'load', family, key }, events);
  }
  kronos(key: string, candles: Candle[], futureTimes: number[], params: KronosParams, events?: Partial<EngineEvents>) {
    return this.call<KronosResult>({ type: 'kronos', key, candles, futureTimes, params }, events);
  }
  chronos(key: string, series: number[], predLen: number, events?: Partial<EngineEvents>) {
    return this.call<ChronosResult>({ type: 'chronos', key, series, predLen }, events);
  }
  release() {
    return this.call<void>({ type: 'release' });
  }
  cancel(): void {
    this.send({ type: 'cancel' });
  }
}
