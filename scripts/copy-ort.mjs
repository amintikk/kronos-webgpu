// Copy the ONNX Runtime WebAssembly binaries next to the app so nothing is loaded from a CDN.
import { copyFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const from = join(root, 'node_modules/onnxruntime-web/dist');
const to = join(root, 'public/ort');
mkdirSync(to, { recursive: true });
for (const base of ['ort-wasm-simd-threaded', 'ort-wasm-simd-threaded.asyncify']) {
  for (const ext of ['mjs', 'wasm']) copyFileSync(join(from, `${base}.${ext}`), join(to, `${base}.${ext}`));
}
