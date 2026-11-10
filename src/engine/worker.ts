/// <reference lib="webworker" />
import { setLang, t } from '../i18n.ts';
import { runChronos } from './chronos.ts';
import { runKronos, type KronosSessions } from './kronos.ts';
import type { Backend, Request, Response } from './protocol.ts';
import { fetchModel, loadManifest } from './store.ts';
import { CancelledError, type FileMeta, type KronosMeta, type OrtLike, type OrtSession } from './types.ts';

type Ort = typeof import('onnxruntime-web');
const ctx = self as unknown as DedicatedWorkerGlobalScope;
const post = (m: Response, transfer: Transferable[] = []) => ctx.postMessage(m, transfer);

let ort: Ort;
let backend: Backend = 'wasm';
let cancelled = false;
let queue: Promise<void> = Promise.resolve();
const kronos = new Map<string, KronosSessions>();
const chronos = new Map<string, OrtSession>();
const tokenizers = new Map<string, { enc: OrtSession; dec: OrtSession }>();

async function init(want: Backend, threads: number): Promise<void> {
  backend = want;
  ort = (want === 'webgpu' ? await import('onnxruntime-web/webgpu') : await import('onnxruntime-web/wasm')) as Ort;
  ort.env.wasm.wasmPaths = new URL('/ort/', ctx.location.origin).href;
  ort.env.wasm.numThreads = ctx.crossOriginIsolated ? threads : 1;
  ort.env.logLevel = 'error';
  post({ type: 'ready', backend, threads: ort.env.wasm.numThreads });
}

async function session(id: number, f: FileMeta, gpuOutputs: string[] = []): Promise<OrtSession> {
  const bytes = await fetchModel(f, (loaded, cached) => post({ type: 'download', id, file: f.file, loaded, total: f.bytes, cached }));
  const opts: Record<string, unknown> = { executionProviders: [backend], graphOptimizationLevel: 'all', logSeverityLevel: 3 };
  if (backend === 'webgpu' && gpuOutputs.length) {
    // Keep the growing key/value cache on the GPU between steps instead of reading it back.
    opts.preferredOutputLocation = Object.fromEntries(gpuOutputs.map((n) => [n, 'gpu-buffer']));
  }
  post({ type: 'progress', id, stage: 'compile', done: 0, total: 1 });
  return (await ort.InferenceSession.create(bytes, opts)) as unknown as OrtSession;
}

function cacheOutputs(meta: KronosMeta): string[] {
  const names = ['h_last', 'nck', 'ncv'];
  for (let l = 0; l < meta.n_layers; l++) names.push(`nk${l}`, `nv${l}`);
  return names;
}

async function load(id: number, family: string, key: string): Promise<void> {
  const manifest = await loadManifest();
  if (family === 'chronos') {
    if (chronos.has(key)) return;
    const meta = manifest.chronos[key];
    if (!meta) throw new Error(t('Modelo desconocido: {k}', { k: key }));
    chronos.set(key, await session(id, meta.model));
    return;
  }
  if (kronos.has(key)) return;
  const meta = manifest.models[key];
  if (!meta) throw new Error(t('Modelo desconocido: {k}', { k: key }));
  let tok = tokenizers.get(meta.tokenizer);
  if (!tok) {
    const t = manifest.tokenizers[meta.tokenizer]!;
    tok = { enc: await session(id, t.enc), dec: await session(id, t.dec) };
    tokenizers.set(meta.tokenizer, tok);
  }
  kronos.set(key, { ...tok, lm: await session(id, meta.lm, cacheOutputs(meta)), s2: await session(id, meta.s2) });
}

async function handle(m: Request): Promise<void> {
  if (m.type === 'init' || m.type === 'cancel') return;
  const started = performance.now();
  try {
    let result: unknown;
    const transfer: Transferable[] = [];
    const progress = (stage: string, done: number, total: number) => post({ type: 'progress', id: m.id, stage, done, total });
    if (m.type === 'load') {
      await load(m.id, m.family, m.key);
    } else if (m.type === 'release') {
      const all = [...kronos.values()].flatMap((k) => [k.lm, k.s2]);
      all.push(...[...tokenizers.values()].flatMap((t) => [t.enc, t.dec]), ...chronos.values());
      kronos.clear();
      tokenizers.clear();
      chronos.clear();
      await Promise.all(all.map((s) => s.release()));
    } else if (m.type === 'kronos') {
      await load(m.id, 'kronos', m.key);
      const meta = (await loadManifest()).models[m.key]!;
      cancelled = false;
      const res = await runKronos(ort as unknown as OrtLike, kronos.get(m.key)!, meta, m.candles, m.futureTimes, m.params, progress, () => cancelled);
      result = res;
      transfer.push(...res.paths.map((p) => p.buffer));
    } else {
      await load(m.id, 'chronos', m.key);
      const meta = (await loadManifest()).chronos[m.key]!;
      const res = await runChronos(ort as unknown as OrtLike, chronos.get(m.key)!, meta, m.series, m.predLen, progress);
      result = res;
      transfer.push(...res.quantiles.map((q) => q.buffer));
    }
    post({ type: 'done', id: m.id, result, ms: performance.now() - started }, transfer);
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    post({ type: 'error', id: m.id, message, cancelled: e instanceof CancelledError });
  }
}

ctx.onmessage = (ev: MessageEvent<Request>) => {
  const m = ev.data;
  if (m.type === 'cancel') {
    cancelled = true;
    return;
  }
  // One job at a time: the sessions share one device and one WASM heap.
  if (m.type === 'init') setLang(m.lang);
  queue = queue.then(() => (m.type === 'init' ? init(m.backend, m.threads) : handle(m)));
};
