import type { FileMeta, Manifest } from './types.ts';

export const MODELS_BASE = '/models/';
const CACHE = 'kronos-models-v1';

export const modelUrl = (f: FileMeta): string => `${MODELS_BASE}${f.file}?v=${f.sha256.slice(0, 12)}`;

let manifest: Promise<Manifest> | null = null;
export function loadManifest(): Promise<Manifest> {
  manifest ??= fetch(`${MODELS_BASE}manifest.json`, { cache: 'no-cache' }).then((r) => {
    if (!r.ok) throw new Error(`No se pudo leer el catálogo de modelos (${r.status})`);
    return r.json() as Promise<Manifest>;
  });
  return manifest;
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
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`No se pudo descargar ${f.file} (${res.status})`);
  const buf = new Uint8Array(f.bytes);
  const reader = res.body.getReader();
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.byteLength > f.bytes) throw new Error(`${f.file}: tamaño inesperado`);
    buf.set(value, loaded);
    loaded += value.byteLength;
    onBytes(loaded, false);
  }
  if (loaded !== f.bytes) throw new Error(`${f.file}: descarga incompleta (${loaded}/${f.bytes} bytes)`);
  try {
    await cache?.put(url, new Response(buf, { headers: { 'content-type': 'application/octet-stream', 'content-length': String(f.bytes) } }));
  } catch {
    // Quota exceeded: the model still runs this session, it just will not persist.
  }
  return buf;
}
