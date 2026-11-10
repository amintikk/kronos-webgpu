import { createReadStream, existsSync, readFileSync, statSync } from 'node:fs';
import { join, normalize, resolve } from 'node:path';
import { defineConfig, type Plugin } from 'vite';

// SharedArrayBuffer (multi-threaded WASM) needs a cross-origin isolated page.
const isolation = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' };

/** Serve the exported models from KRONOS_MODELS in dev/preview; in production nginx serves /models/. */
function localModels(): Plugin {
  const dir = process.env.KRONOS_MODELS;
  const handler = (req: { url?: string }, res: import('node:http').ServerResponse, next: () => void) => {
    if (!dir || !req.url?.startsWith('/models/')) return next();
    const file = join(dir, normalize(req.url.slice('/models/'.length).split('?')[0]!).replace(/^(\.\.[/\\])+/, ''));
    if (!existsSync(file) || !statSync(file).isFile()) return next();
    res.setHeader('Content-Length', statSync(file).size);
    res.setHeader('Content-Type', file.endsWith('.json') ? 'application/json' : 'application/octet-stream');
    createReadStream(file).pipe(res);
  };
  return {
    name: 'local-models',
    configureServer: (s) => void s.middlewares.use(handler),
    configurePreviewServer: (s) => void s.middlewares.use(handler),
  };
}

/** Static pages share a header and a footer: `<!--#header-->` and `<!--#footer-->` are replaced at build time. */
function partials(): Plugin {
  const dir = resolve(__dirname, 'src/site/partials');
  return {
    name: 'partials',
    transformIndexHtml: (html) => html.replace(/<!--#(\w+)-->/g, (_, name: string) => readFileSync(join(dir, `${name}.html`), 'utf8')),
  };
}

const pages = ['index.html', 'app/index.html', 'aviso-legal/index.html', 'privacidad/index.html', 'terminos/index.html', 'licencias/index.html'];

export default defineConfig({
  plugins: [localModels(), partials()],
  build: { target: 'es2022', chunkSizeWarningLimit: 900, rollupOptions: { input: pages.map((p) => resolve(__dirname, p)) } },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: { headers: isolation },
  preview: { headers: isolation },
});
