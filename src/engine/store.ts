import { t } from '../i18n.ts';
import type { FileMeta, Manifest } from './types.ts';

/** Same-origin copy of the models; always present, used as fallback. */
const LOCAL_BASE = '/models/';
/** Where the weights are downloaded from (a CDN in production, see .env.production). */
const MODEL_BASES = [...new Set([import.meta.env.VITE_MODELS_URL as string | undefined, LOCAL_BASE].filter((b): b is string => !!b))];
const CACHE = 'kronos-models-v1';

const query = (f: FileMeta): string => `${f.file}?v=${f.sha256.slice(0, 12)}`;
/** Cache key: independent of which server the file came from. */
export const modelUrl = (f: FileMeta): string => `${LOCAL_BASE}${query(f)}`;

let manifest: Promise<Manifest> | null = null;
export function loadManifest(): Promise<Manifest> {
  // The catalogue is small and always comes from this site, so it matches the deployed code.
  manifest ??= fetch(`${LOCAL_BASE}manifest.json`, { cache: 'no-cache' }).then((r) => {
    if (!r.ok) throw new Error(t('No se pudo leer el catálogo de modelos ({s})', { s: r.status }));
    return r.json() as Promise<Manifest>;
  });
  return manifest;
}

async function download(f: FileMeta, onBytes: (loaded: number, cached: boolean) => void): Promise<Uint8Array<ArrayBuffer>> {
  let lastError: unknown;
  for (const base of MODEL_BASES) {
    try {
      const res = await fetch(base + query(f), { mode: 'cors' });
      if (!res.ok || !res.body) throw new Error(t('No se pudo descargar {f} ({s})', { f: f.file, s: res.status }));
      const buf = new Uint8Array(f.bytes);
      const reader = res.body.getReader();
      let loaded = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (loaded + value.byteLength > f.bytes) throw new Error(`${f.file}: ${t('tamaño inesperado')}`);
        buf.set(value, loaded);
        loaded += value.byteLength;
        onBytes(loaded, false);
      }
      if (loaded !== f.bytes) throw new Error(`${f.file}: ${t('descarga incompleta ({a}/{b} bytes)', { a: loaded, b: f.bytes })}`);
      return buf;
    } catch (e) {
      lastError = e; // try the next server
    }
  }
  throw lastError;
}

async function openCache(): Promise<Cache | null> {
  try {
    return await caches.open(CACHE);
  } catch {
    return null; // private windows or blocked storage: keep working without persistence
  }
}

export async function isCached(f: FileMeta): Promise<boolean> {
  const cache = await openCache();
  return !!cache && !!(await cache.match(modelUrl(f)));
}

export async function removeCached(files: FileMeta[]): Promise<void> {
  const cache = await openCache();
  if (cache) await Promise.all(files.map((f) => cache.delete(modelUrl(f))));
}

/** Drop cached files that no longer belong to the catalogue (older exports). */
export async function pruneCache(m: Manifest): Promise<void> {
  const cache = await openCache();
  if (!cache) return;
  const keep = new Set(allFiles(m).map((f) => new URL(modelUrl(f), location.origin).href));
  for (const req of await cache.keys()) if (!keep.has(req.url)) await cache.delete(req);
}

export function allFiles(m: Manifest): FileMeta[] {
  return [
    ...Object.values(m.tokenizers).flatMap((t) => [t.enc, t.dec]),
    ...Object.values(m.models).flatMap((k) => [k.lm, k.s2]),
    ...Object.values(m.chronos).map((c) => c.model),
  ];
}

/** Download a model file (or read it from the browser cache), reporting bytes as they arrive. */
export async function fetchModel(f: FileMeta, onBytes: (loaded: number, cached: boolean) => void): Promise<Uint8Array> {
  const url = modelUrl(f);
  const cache = await openCache();
  const hit = await cache?.match(url);
  if (hit) {
    const buf = new Uint8Array(await hit.arrayBuffer());
    if (buf.byteLength === f.bytes) {
      onBytes(f.bytes, true);
      return buf;
    }
    await cache!.delete(url);
  }
  const buf = await download(f, onBytes);
  try {
    await cache?.put(url, new Response(buf, { headers: { 'content-type': 'application/octet-stream', 'content-length': String(f.bytes) } }));
  } catch {
    // Quota exceeded: the model still runs this session, it just will not persist.
  }
  return buf;
}
