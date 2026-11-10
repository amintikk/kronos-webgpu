import './styles.css';
import { ForecastChart, type ChartColors, type Pinned, type ViewOptions } from './chart.ts';
import { parseCsv } from './data/csv.ts';
import { INTERVAL_MS, SOURCES, fetchCandles, sourceById, subscribe } from './data/sources.ts';
import { CancelError, Engine, type EngineEvents } from './engine/client.ts';
import type { Backend } from './engine/protocol.ts';
import { allFiles, fetchModel, isCached, loadManifest, pruneCache, removeCached } from './engine/store.ts';
import type { Candle, FileMeta, Manifest } from './engine/types.ts';
import { fromChronos, fromKronos, score, summarize, targetOdds, type Forecast, type Score } from './forecast.ts';
import { DEFAULTS, loadSettings, saveSettings, shareUrl, type Settings } from './settings.ts';
import '@fontsource-variable/inter';
import { $, download, fmtBytes, fmtCompact, fmtDuration, fmtNum, fmtPct, fmtPrice, fmtTime, h, toast } from './ui/dom.ts';
import { hydrateIcons, icon, type IconName } from './ui/icons.ts';
import { enhanceCombo, enhanceSelect } from './ui/select.ts';

const S = loadSettings();
const engine = new Engine();
let manifest: Manifest;
let chart: ForecastChart;

/** Every candle we hold, oldest first; the last one may still be forming. */
let all: Candle[] = [];
let stepMs = INTERVAL_MS[S.interval]!;
let csv: { name: string; candles: Candle[]; stepMs: number } | null = null;
let unsubscribe: (() => void) | null = null;
let dataToken = 0;

let current: Forecast | null = null;
let pinned: Pinned[] = [];
let busy = false;
let booting = false;
let nextForecastId = 1;
let gpuLabel = '';
let backtest: { rows: { forecast: Forecast; score: Score }[]; model: string; running: boolean } | null = null;
const inMemory = new Set<string>();

// ---------------------------------------------------------------- helpers

function log(msg: string): void {
  const el = $('log');
  el.textContent += `${new Date().toLocaleTimeString('es-ES')}  ${msg}\n`;
  el.scrollTop = el.scrollHeight;
}

function notice(msg: string | null): void {
  const el = $('notice');
  el.hidden = !msg;
  $('notice-text').textContent = msg ?? '';
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function chartColors(): ChartColors {
  return {
    surface: cssVar('--surface'),
    text: cssVar('--text'),
    grid: cssVar('--grid'),
    border: cssVar('--border'),
    up: cssVar('--up'),
    down: cssVar('--down'),
    forecast: cssVar('--accent'),
    muted: cssVar('--muted'),
  };
}

const viewOptions = (): ViewOptions => ({ bands: S.bands, paths: S.paths, meanCandles: S.meanCandles, median: S.median, volume: S.volume, log: S.log, localTime: S.localTime });

const modelParts = (): { family: 'kronos' | 'chronos'; key: string } => {
  const [family, key] = S.model.split(':');
  return { family: family === 'chronos' ? 'chronos' : 'kronos', key: key ?? 'small' };
};

function modelInfo(id = S.model): { label: string; maxContext: number; files: FileMeta[]; params: string; source: string } {
  const [family, key] = id.split(':');
  if (family === 'chronos') {
    const m = manifest.chronos[key!]!;
    return { label: m.label, maxContext: m.context_length, files: [m.model], params: m.params, source: m.source };
  }
  const m = manifest.models[key!]!;
  const t = manifest.tokenizers[m.tokenizer]!;
  return { label: m.label, maxContext: m.max_context, files: [t.enc, t.dec, m.lm, m.s2], params: m.params, source: m.source };
}

/** Candles usable as model context: closed ones, plus the forming one if the user asked for it. */
function contextBase(): Candle[] {
  const last = all[all.length - 1];
  if (!last || csv || S.includeOpen) return all;
  return last.t + stepMs > Date.now() ? all.slice(0, -1) : all;
}

const pairLabel = (): string => (csv ? csv.name : `${S.symbol} · ${S.interval} · ${sourceById(S.source).name}`);

// ---------------------------------------------------------------- controls

interface Field {
  key: keyof Settings;
  label: string;
  kind: 'select' | 'number' | 'text' | 'check';
  options?: () => [string, string][];
  min?: number;
  max?: number;
  step?: number;
  hint?: string;
}
const inputs = new Map<keyof Settings, HTMLInputElement | HTMLSelectElement>();
/** Custom dropdowns mirror a hidden native select; call these after changing one from code. */
const selectSync = new Map<keyof Settings, () => void>();
let symbolList: string[] = [];

function setSelect(key: keyof Settings, value: string): void {
  (inputs.get(key) as HTMLSelectElement).value = value;
  selectSync.get(key)?.();
}
const hints = new Map<string, HTMLElement>();

const SECTIONS: { title: string; icon: IconName; open: boolean; fields: Field[]; extra?: () => HTMLElement[] }[] = [
  {
    title: 'Datos',
    icon: 'database',
    open: true,
    fields: [
      { key: 'source', label: 'Fuente', kind: 'select', options: () => [...SOURCES.map((s) => [s.id, s.name] as [string, string]), ['csv', 'Archivo CSV']] },
      { key: 'symbol', label: 'Par', kind: 'text' },
      { key: 'interval', label: 'Intervalo', kind: 'select', options: () => (csv ? [[S.interval, 'del archivo']] : sourceById(S.source).intervals.map((i) => [i, i])) },
      { key: 'lookback', label: 'Velas de contexto', kind: 'number', min: 32, max: 2048, step: 16, hint: 'lookback' },
      { key: 'includeOpen', label: 'Usar la vela en curso', kind: 'check' },
      { key: 'live', label: 'Actualizar en directo', kind: 'check' },
    ],
    extra: () => [
      h(
        'div',
        { class: 'field wide' },
        h('label', { for: 'csv-file' }, 'Cargar CSV propio'),
        h('input', { id: 'csv-file', type: 'file', accept: '.csv,text/csv', hidden: true, onchange: onCsv }),
        h(
          'div',
          { class: 'filepick' },
          h('button', { type: 'button', class: 'btn small', onclick: () => $('csv-file').click() }, icon('upload', 14), 'Elegir archivo'),
          h('span', { id: 'csv-name' }, 'Ninguno'),
        ),
        h('div', { class: 'hint' }, 'Columnas: timestamps, open, high, low, close y, si las hay, volume y amount.'),
      ),
    ],
  },
  {
    title: 'Modelo',
    icon: 'cpu',
    open: true,
    fields: [
      { key: 'model', label: 'Modelo', kind: 'select', options: modelOptions, hint: 'model' },
      { key: 'backend', label: 'Motor', kind: 'select', options: () => [['auto', 'Automático'], ['webgpu', 'WebGPU'], ['wasm', 'CPU (WASM)']] },
      { key: 'threads', label: 'Hilos de CPU', kind: 'number', min: 0, max: 32, step: 1, hint: 'threads' },
    ],
  },
  {
    title: 'Predicción',
    icon: 'activity',
    open: true,
    fields: [
      { key: 'horizon', label: 'Horizonte (velas)', kind: 'number', min: 1, max: 512, step: 1, hint: 'horizon' },
      { key: 'samples', label: 'Trayectorias', kind: 'number', min: 1, max: 256, step: 1, hint: 'samples' },
      { key: 'batch', label: 'En paralelo', kind: 'number', min: 1, max: 64, step: 1 },
      { key: 'temperature', label: 'Temperatura', kind: 'number', min: 0.1, max: 2, step: 0.05 },
      { key: 'topP', label: 'Top-p', kind: 'number', min: 0.05, max: 1, step: 0.01 },
      { key: 'topK', label: 'Top-k (0 = sin límite)', kind: 'number', min: 0, max: 1024, step: 1 },
      { key: 'randomSeed', label: 'Semilla aleatoria', kind: 'check' },
      { key: 'seed', label: 'Semilla', kind: 'number', min: 0, max: 4294967295, step: 1 },
      { key: 'fastWindow', label: 'Modo rápido con contexto largo', kind: 'check', hint: 'fastWindow' },
      { key: 'offset', label: 'Velas atrás (0 = ahora)', kind: 'number', min: 0, max: 5000, step: 1, hint: 'offset' },
      { key: 'autoPredict', label: 'Predecir al cerrar cada vela', kind: 'check' },
    ],
  },
  {
    title: 'Vista',
    icon: 'eye',
    open: false,
    fields: [
      { key: 'median', label: 'Mediana', kind: 'check' },
      { key: 'bands', label: 'Bandas de percentiles', kind: 'check' },
      { key: 'paths', label: 'Trayectorias', kind: 'check' },
      { key: 'meanCandles', label: 'Velas medias', kind: 'check' },
      { key: 'volume', label: 'Volumen', kind: 'check' },
      { key: 'log', label: 'Escala logarítmica', kind: 'check' },
      { key: 'localTime', label: 'Hora local (si no, UTC)', kind: 'check' },
    ],
  },
  {
    title: 'Backtest',
    icon: 'history',
    open: false,
    fields: [
      { key: 'btWindows', label: 'Ventanas', kind: 'number', min: 1, max: 100, step: 1 },
      { key: 'btStride', label: 'Separación (0 = horizonte)', kind: 'number', min: 0, max: 2000, step: 1 },
      { key: 'btSamples', label: 'Trayectorias por ventana', kind: 'number', min: 1, max: 64, step: 1 },
    ],
    extra: () => [
      h('button', { id: 'btn-backtest', class: 'btn', style: 'justify-self:start', onclick: () => void runBacktest() }, icon('play', 12), 'Ejecutar backtest'),
      h('div', { class: 'hint' }, 'Repite la predicción en puntos del pasado y la compara con lo que ocurrió. Usa el modelo, el contexto y el horizonte actuales.'),
    ],
  },
];

function modelOptions(): [string, string][] {
  if (!manifest) return [[S.model, S.model]];
  return [
    ...Object.entries(manifest.models).map(([k, m]) => [`kronos:${k}`, `${m.label} (${m.params})`] as [string, string]),
    ...Object.entries(manifest.chronos).map(([k, m]) => [`chronos:${k}`, `${m.label} (${m.params})`] as [string, string]),
  ];
}

function fillSelect(key: keyof Settings): void {
  const el = inputs.get(key) as HTMLSelectElement;
  const field = SECTIONS.flatMap((s) => s.fields).find((f) => f.key === key)!;
  const opts = field.options!();
  el.replaceChildren(...opts.map(([v, t]) => h('option', { value: v }, t)));
  if (!opts.some(([v]) => v === S[key])) (S as unknown as Record<string, unknown>)[key] = opts[0]![0];
  el.value = String(S[key]);
  selectSync.get(key)?.();
}

function stepper(input: HTMLInputElement): HTMLElement {
  const bump = (dir: 1 | -1) => () => {
    if (input.disabled) return;
    if (dir > 0) input.stepUp();
    else input.stepDown();
    input.dispatchEvent(new Event('change'));
  };
  return h(
    'div',
    { class: 'num' },
    h('button', { type: 'button', tabIndex: -1, 'aria-label': 'Menos', onclick: bump(-1) }, icon('minus', 14)),
    input,
    h('button', { type: 'button', tabIndex: -1, 'aria-label': 'Más', onclick: bump(1) }, icon('plus', 14)),
  );
}

function buildControls(): void {
  const root = $('controls');
  let openState: Record<string, boolean> = {};
  try {
    openState = JSON.parse(localStorage.getItem('kronos-sections') ?? '{}');
  } catch {
    /* storage blocked */
  }
  for (const sec of SECTIONS) {
    const fields = h('div', { class: 'fields' });
    for (const f of sec.fields) {
      const id = `f-${f.key}`;
      let input: HTMLInputElement | HTMLSelectElement;
      let control: HTMLElement;
      if (f.kind === 'select') control = input = h('select', { id });
      else if (f.kind === 'check') control = input = h('input', { id, type: 'checkbox', class: 'switch', role: 'switch', checked: S[f.key] as boolean });
      else if (f.kind === 'number') {
        input = h('input', { id, type: 'number', min: f.min, max: f.max, step: f.step, value: String(S[f.key]), inputMode: 'decimal' });
        control = stepper(input);
      } else control = input = h('input', { id, type: 'text', class: 'input', value: String(S[f.key]), spellcheck: false, autocomplete: 'off' });
      inputs.set(f.key, input);
      input.addEventListener('change', () => onInput(f, input));
      const row = h(f.kind === 'check' ? 'label' : 'div', { class: `field${f.kind === 'check' ? ' check' : f.key === 'model' ? ' wide' : ''}` }, f.kind === 'check' ? h('span', {}, f.label) : h('label', { for: id }, f.label), control);
      if (f.hint) {
        const hint = h('div', { class: 'hint' });
        hints.set(f.hint, hint);
        row.append(hint);
      }
      fields.append(row);
    }
    for (const el of sec.extra?.() ?? []) fields.append(el);
    const acc = h('section', { class: `acc${(openState[sec.title] ?? sec.open) ? ' open' : ''}` });
    const head = h(
      'button',
      { type: 'button', class: 'acc-head', 'aria-expanded': String(acc.classList.contains('open')) },
      h('span', { class: 'acc-ico' }, icon(sec.icon, 15)),
      h('span', { class: 'title' }, sec.title),
      (() => {
        const c = icon('chevronDown', 16);
        c.classList.add('chev');
        return c;
      })(),
    );
    head.addEventListener('click', () => {
      const open = acc.classList.toggle('open');
      head.setAttribute('aria-expanded', String(open));
      openState[sec.title] = open;
      try {
        localStorage.setItem('kronos-sections', JSON.stringify(openState));
      } catch {
        /* storage blocked */
      }
    });
    acc.append(head, h('div', { class: 'acc-body' }, h('div', { class: 'acc-inner' }, fields)));
    root.append(acc);
  }
  for (const f of SECTIONS.flatMap((s) => s.fields)) {
    if (f.kind !== 'select') continue;
    selectSync.set(f.key, enhanceSelect(inputs.get(f.key) as HTMLSelectElement));
    fillSelect(f.key);
  }
  const symbol = inputs.get('symbol') as HTMLInputElement;
  enhanceCombo(symbol, () => symbolList);
  symbol.addEventListener('focus', () => void loadSymbols());
}

function onInput(f: Field, input: HTMLInputElement | HTMLSelectElement): void {
  let value: string | number | boolean;
  if (f.kind === 'check') value = (input as HTMLInputElement).checked;
  else if (f.kind === 'number') {
    const n = Number(input.value);
    const clamped = Math.min(f.max ?? Infinity, Math.max(f.min ?? -Infinity, Number.isFinite(n) ? n : (DEFAULTS[f.key] as number)));
    value = f.step && Number.isInteger(f.step) && f.step >= 1 ? Math.round(clamped) : clamped;
    input.value = String(value);
  } else value = f.key === 'symbol' ? input.value.trim().toUpperCase() : input.value;
  if (f.key === 'symbol') input.value = String(value);
  (S as unknown as Record<string, unknown>)[f.key] = value;
  saveSettings(S);
  void onChange(f.key);
}

async function onChange(key: keyof Settings): Promise<void> {
  switch (key) {
    case 'source':
      if (S.source === 'csv' && !csv) {
        toast('Elige un archivo CSV más abajo');
        S.source = DEFAULTS.source;
        setSelect('source', S.source);
        return;
      }
      if (S.source !== 'csv') {
        csv = null;
        S.symbol = sourceById(S.source).defaultSymbol;
        (inputs.get('symbol') as HTMLInputElement).value = S.symbol;
        symbolsFor = '';
        symbolList = [];
        $('csv-name').textContent = 'Ninguno';
      }
      fillSelect('interval');
      saveSettings(S);
      await loadData();
      break;
    case 'symbol':
    case 'interval':
      await loadData();
      break;
    case 'live':
      wireLive();
      break;
    case 'model':
      updateHints();
      renderModels();
      if (inMemory.size) {
        inMemory.clear();
        void engine.release().catch(() => {});
      }
      break;
    case 'backend':
    case 'threads':
      await startEngine();
      break;
    case 'bands':
    case 'paths':
    case 'meanCandles':
    case 'median':
    case 'volume':
    case 'log':
    case 'localTime':
      chart.setView(viewOptions());
      renderLegend();
      renderTable();
      break;
    default:
      updateHints();
  }
}

function updateHints(): void {
  if (!manifest) return;
  const info = modelInfo();
  const { family } = modelParts();
  const size = info.files.reduce((a, f) => a + f.bytes, 0);
  hints.get('model')!.textContent = `${fmtBytes(size)} de descarga la primera vez. Contexto máximo: ${info.maxContext} velas.${family === 'chronos' ? ' Solo predice el cierre.' : ''}`;
  hints.get('lookback')!.textContent = S.lookback > info.maxContext ? `Este modelo usará las últimas ${info.maxContext}.` : '';
  hints.get('threads')!.textContent = engine.backend === 'wasm' ? `0 = automático. En uso: ${engine.threads}.` : '0 = automático. Solo se aplica al motor de CPU.';
  hints.get('samples')!.textContent = family === 'chronos' ? 'Chronos-Bolt no muestrea: devuelve cuantiles directamente.' : 'Más trayectorias dan bandas más estables y tardan más.';
  const chronosMax = family === 'chronos' ? manifest.chronos[modelParts().key]!.prediction_length : 0;
  hints.get('horizon')!.textContent =
    chronosMax && S.horizon > chronosMax ? `Más de ${chronosMax} velas se generan encadenando la mediana: las bandas lejanas son menos fiables.` : '';
  hints.get('offset')!.textContent = S.offset > 0 ? 'Predice desde el pasado y compara con lo ocurrido.' : '';
  const overflow = family === 'kronos' && Math.min(S.lookback, info.maxContext) + S.horizon > info.maxContext;
  hints.get('fastWindow')!.textContent = !overflow
    ? `Sin efecto ahora: contexto y horizonte caben en las ${info.maxContext} velas del modelo.`
    : S.fastWindow
      ? 'Más rápido, pero el modelo ve hasta 64 velas menos de historia y el resultado se aparta del Kronos original.'
      : 'Exacto como el Kronos original, pero cada vela pasado el límite recalcula todo el contexto y tarda más.';
  for (const k of ['samples', 'batch', 'temperature', 'topP', 'topK', 'seed', 'randomSeed', 'fastWindow'] as const) inputs.get(k)!.disabled = family === 'chronos' || (k === 'seed' && S.randomSeed);
  for (const k of ['symbol', 'interval', 'live', 'includeOpen'] as const) inputs.get(k)!.disabled = !!csv;
  for (const sync of selectSync.values()) sync();
}

let symbolsFor = '';
async function loadSymbols(): Promise<void> {
  if (csv || symbolsFor === S.source) return;
  symbolsFor = S.source;
  try {
    const list = await sourceById(S.source).symbols();
    list.sort((a, b) => b.volume - a.volume);
    symbolList = list.map((s) => s.symbol);
    (inputs.get('symbol') as HTMLInputElement & { refreshSuggestions?: () => void }).refreshSuggestions?.();
  } catch (e) {
    symbolsFor = '';
    log(`No se pudo cargar la lista de pares: ${(e as Error).message}`);
  }
}

// ---------------------------------------------------------------- data

async function loadData(minCount = 0): Promise<void> {
  const token = ++dataToken;
  unsubscribe?.();
  unsubscribe = null;
  notice(null);
  if (csv) {
    all = csv.candles;
    stepMs = csv.stepMs;
  } else {
    stepMs = INTERVAL_MS[S.interval]!;
    const want = Math.max(1000, minCount, S.lookback + S.offset + S.horizon + 50);
    $('pair').textContent = `${S.symbol} · cargando`;
    try {
      const data = await fetchCandles(sourceById(S.source), S.symbol, S.interval, want);
      if (token !== dataToken) return;
      if (data.length < 34) throw new Error(`Solo hay ${data.length} velas para ${S.symbol}`);
      all = data;
      log(`${data.length} velas de ${S.symbol} ${S.interval} (${sourceById(S.source).name})`);
    } catch (e) {
      if (token !== dataToken) return;
      $('pair').textContent = pairLabel();
      notice(`No se pudieron cargar las velas de ${S.symbol}: ${(e as Error).message}`);
      log(`Error de datos: ${(e as Error).message}`);
      return;
    }
  }
  if (minCount === 0) {
    current = null;
    pinned = [];
    backtest = null;
  }
  $('pair').textContent = pairLabel();
  chart.setCandles(all, minCount === 0);
  chart.setForecast(current, pinned);
  renderPrice();
  renderAll();
  wireLive();
}

function wireLive(): void {
  unsubscribe?.();
  unsubscribe = null;
  const on = S.live && !csv && all.length > 0;
  $('live-dot').hidden = !on;
  if (!on) return;
  const token = dataToken;
  unsubscribe = subscribe(sourceById(S.source), S.symbol, S.interval, (c) => {
    if (token !== dataToken) return;
    const last = all[all.length - 1]!;
    if (c.t < last.t) return;
    const isNew = c.t > last.t;
    if (isNew) all.push(c);
    else all[all.length - 1] = c;
    chart.updateLast(c);
    renderPrice();
    if (isNew && S.autoPredict && !busy && S.offset === 0) void predict();
  });
}

function renderPrice(): void {
  const last = all[all.length - 1];
  const prev = all[all.length - 2];
  if (!last) return;
  $('price').textContent = fmtPrice(last.c);
  const up = !prev || last.c >= prev.c;
  const trend = $('trend');
  if (trend.hidden || trend.classList.contains('up') !== up) {
    trend.hidden = false;
    trend.className = `trend ${up ? 'up' : 'down'}`;
    trend.replaceChildren(icon(up ? 'arrowUp' : 'arrowDown', 15));
  }
}

async function onCsv(ev: Event): Promise<void> {
  const file = (ev.target as HTMLInputElement).files?.[0];
  if (!file) return;
  try {
    const parsed = parseCsv(await file.text());
    csv = { name: file.name, ...parsed };
    S.source = 'csv';
    setSelect('source', 'csv');
    $('csv-name').textContent = file.name;
    fillSelect('interval');
    updateHints();
    log(`CSV «${file.name}»: ${parsed.candles.length} velas, intervalo ${fmtDuration(parsed.stepMs)}`);
    await loadData();
  } catch (e) {
    notice((e as Error).message);
  }
}

// ---------------------------------------------------------------- engine

/** Pick the backend and (re)start the worker. Predicting is blocked meanwhile: a restart drops queued jobs. */
async function startEngine(): Promise<void> {
  booting = true;
  syncButtons();
  try {
    await chooseAndBoot();
  } finally {
    booting = false;
    syncButtons();
  }
}

function syncButtons(): void {
  ($('btn-predict') as HTMLButtonElement).disabled = busy || booting;
  const bt = document.getElementById('btn-backtest') as HTMLButtonElement | null;
  if (bt) bt.disabled = busy || booting;
}

async function chooseAndBoot(): Promise<void> {
  let backend: Backend = 'wasm';
  gpuLabel = '';
  if (S.backend !== 'wasm') {
    try {
      // The first request can come back empty while the GPU process is still starting.
      const adapter = (await navigator.gpu?.requestAdapter()) ?? (navigator.gpu ? await new Promise<GPUAdapter | null>((r) => setTimeout(() => r(navigator.gpu.requestAdapter()), 400)) : null);
      if (adapter) {
        const info = adapter.info;
        gpuLabel = [info?.vendor, info?.architecture].filter(Boolean).join(' ');
        // A software rasteriser is slower than multi-threaded WASM: only use it when asked for explicitly.
        const software = (info as { isFallbackAdapter?: boolean } | undefined)?.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test(`${gpuLabel} ${info?.description ?? ''}`);
        if (!software || S.backend === 'webgpu') backend = 'webgpu';
        else log('WebGPU solo está disponible por software en este equipo; se usa la CPU.');
      }
    } catch {
      /* no WebGPU */
    }
    if (backend === 'wasm' && S.backend === 'webgpu') notice('Este navegador no ofrece WebGPU. Se usa la CPU (WASM), que es más lenta.');
  }
  await bootEngine(backend);
}

async function bootEngine(backend: Backend): Promise<void> {
  const cores = navigator.hardwareConcurrency || 4;
  const threads = S.threads > 0 ? S.threads : Math.min(8, Math.max(1, Math.floor(cores * 0.75)));
  inMemory.clear();
  const chip = $('backend-chip');
  chip.textContent = 'Iniciando';
  try {
    const ready = await engine.start(backend, threads);
    chip.dataset.state = ready.backend === 'webgpu' ? 'gpu' : 'cpu';
    chip.textContent = ready.backend === 'webgpu' ? `WebGPU${gpuLabel ? ` · ${gpuLabel}` : ''}` : `CPU · ${ready.threads} ${ready.threads === 1 ? 'hilo' : 'hilos'}`;
    log(`Motor: ${chip.textContent}${ready.backend === 'wasm' && !crossOriginIsolated ? ' (sin aislamiento de origen: un solo hilo)' : ''}`);
  } catch (e) {
    chip.textContent = 'Motor no disponible';
    notice(`No se pudo iniciar el motor de inferencia: ${(e as Error).message}`);
  }
  updateHints();
  renderModels();
}

function progressEvents(prefix = ''): Partial<EngineEvents> {
  const fill = $('progress-fill');
  const text = $('progress-text');
  let t0 = 0;
  let first = 0;
  return {
    download(file, loaded, total, cached) {
      fill.style.width = `${(loaded / total) * 100}%`;
      text.textContent = cached ? `${prefix}Leyendo ${file} de la caché…` : `${prefix}Descargando ${file}: ${fmtBytes(loaded)} de ${fmtBytes(total)}`;
    },
    progress(stage, done, total) {
      if (stage === 'compile') text.textContent = `${prefix}Preparando el modelo…`;
      else if (stage === 'tokenize' || stage === 'context') {
        fill.style.width = '0%';
        text.textContent = `${prefix}Leyendo el contexto…`;
      } else if (stage === 'decode') text.textContent = `${prefix}Convirtiendo a precios…`;
      else {
        const now = performance.now();
        if (!t0) {
          t0 = now;
          first = done;
        }
        fill.style.width = `${(done / total) * 100}%`;
        const rate = (done - first) / Math.max(1, now - t0);
        const eta = rate > 0 ? (total - done) / rate : 0;
        text.textContent = `${prefix}${fmtNum((done / total) * 100)} %${eta > 1500 ? ` · quedan ${fmtDuration(eta)}` : ''}`;
      }
    },
  };
}

function setBusy(on: boolean): void {
  busy = on;
  $('progress').hidden = !on;
  syncButtons();
  $('btn-cancel').hidden = !on;
  if (on) {
    $('progress-fill').style.width = '0%';
    $('progress-text').textContent = 'Preparando';
  }
}

/** Run the selected model with the context ending just before `endIdx` of the context base. */
async function runForecast(base: Candle[], endIdx: number, samples: number, events: Partial<EngineEvents>): Promise<Forecast> {
  const { family, key } = modelParts();
  const info = modelInfo();
  const look = Math.min(S.lookback, info.maxContext, endIdx);
  if (look < 32) throw new Error('No hay suficientes velas de contexto (mínimo 32)');
  const ctx = base.slice(endIdx - look, endIdx);
  const anchor = ctx[ctx.length - 1]!;
  const times = Array.from({ length: S.horizon }, (_, i) => anchor.t + stepMs * (i + 1));
  const common = { id: nextForecastId++, model: S.model, family, label: info.label, times, anchor };
  if (family === 'kronos') {
    const seed = S.randomSeed ? Math.floor(Math.random() * 2 ** 32) : S.seed;
    const params = { predLen: S.horizon, samples, batch: Math.min(S.batch, samples), temperature: S.temperature, topK: S.topK, topP: S.topP, seed, fastWindow: S.fastWindow };
    const { result, ms } = await engine.kronos(key, ctx, times, params, events);
    inMemory.add(S.model);
    return { ...common, ...fromKronos(result, times, anchor), ms, settings: { contexto: look, horizonte: S.horizon, trayectorias: samples, temperatura: S.temperature, top_p: S.topP, top_k: S.topK, semilla: seed, ...(look + S.horizon > info.maxContext ? { ventana: S.fastWindow ? 'rápida' : 'exacta' } : {}) } };
  }
  const { result, ms } = await engine.chronos(key, ctx.map((c) => c.c), S.horizon, events);
  inMemory.add(S.model);
  return { ...common, ...fromChronos(result), ms, settings: { contexto: look - (look % 16), horizonte: S.horizon } };
}

/** WebGPU can fail on drivers we cannot test for; in automatic mode retry once on the CPU. */
async function withFallback<T>(job: () => Promise<T>): Promise<T> {
  try {
    return await job();
  } catch (e) {
    if (e instanceof CancelError || engine.backend !== 'webgpu' || S.backend !== 'auto') throw e;
    log(`WebGPU falló (${(e as Error).message}). Se reintenta en CPU.`);
    notice('WebGPU dio un error en este equipo; se ha cambiado al motor de CPU, más lento.');
    await bootEngine('wasm');
    return job();
  }
}

async function predict(): Promise<void> {
  if (busy || booting) return;
  const base = contextBase();
  const endIdx = base.length - S.offset;
  if (endIdx < 32) {
    if (!csv && S.offset > 0) {
      await loadData(S.lookback + S.offset + S.horizon + 50);
      if (contextBase().length - S.offset >= 32) return predict();
    }
    notice('No hay suficientes velas para esa combinación de contexto y «velas atrás».');
    return;
  }
  setBusy(true);
  notice(null);
  try {
    const f = await withFallback(() => runForecast(base, endIdx, S.samples, progressEvents()));
    current = f;
    chart.setForecast(current, pinned);
    chart.focus();
    log(`${f.label}: ${f.times.length} velas${f.samples ? ` × ${f.samples} trayectorias` : ''} en ${fmtDuration(f.ms)}`);
    renderAll();
  } catch (e) {
    if (e instanceof CancelError) log('Predicción cancelada');
    else {
      notice(`La predicción falló: ${(e as Error).message}`);
      log(`Error: ${(e as Error).message}`);
    }
  } finally {
    setBusy(false);
    renderModels();
  }
}

async function runBacktest(): Promise<void> {
  if (busy || booting) return;
  const stride = S.btStride || S.horizon;
  const need = Math.min(S.lookback, modelInfo().maxContext) + S.horizon + (S.btWindows - 1) * stride + 2;
  if (!csv && all.length < need) await loadData(need + 50);
  const base = contextBase();
  const windows = Math.min(S.btWindows, Math.floor((base.length - S.horizon - 32) / stride) + 1);
  if (windows < 1) {
    notice('No hay suficiente histórico para el backtest.');
    return;
  }
  if (windows < S.btWindows) log(`Histórico insuficiente: backtest con ${windows} ventanas en lugar de ${S.btWindows}`);
  selectTab('backtest');
  setBusy(true);
  notice(null);
  const run = { rows: [] as { forecast: Forecast; score: Score }[], model: modelInfo().label, running: true };
  backtest = run;
  const actual = new Map(base.map((c) => [c.t, c]));
  try {
    for (let w = 0; w < windows; w++) {
      const endIdx = base.length - S.horizon - (windows - 1 - w) * stride;
      const f = await withFallback(() => runForecast(base, endIdx, S.btSamples, progressEvents(`Ventana ${w + 1} de ${windows} · `)));
      const sc = score(f, actual);
      if (sc) run.rows.push({ forecast: f, score: sc });
      renderBacktest();
    }
    log(`Backtest de ${run.model}: ${run.rows.length} ventanas`);
  } catch (e) {
    if (e instanceof CancelError) log('Backtest cancelado');
    else {
      notice(`El backtest falló: ${(e as Error).message}`);
      log(`Error: ${(e as Error).message}`);
    }
  } finally {
    run.running = false;
    setBusy(false);
    renderBacktest();
    renderModels();
  }
}

// ---------------------------------------------------------------- rendering

function renderAll(): void {
  renderLegend();
  renderSummary();
  renderTable();
  renderBacktest();
  ($('btn-pin') as HTMLButtonElement).disabled = !current || pinned.length >= 2;
  ($('btn-clear') as HTMLButtonElement).disabled = !current && !pinned.length;
  ($('btn-export') as HTMLButtonElement).disabled = !current;
}

function pct(level: number): string {
  return `p${Math.round(level * 100)}`;
}

function renderLegend(): void {
  const el = $('legend');
  const items: HTMLElement[] = [];
  const f = current;
  if (f) {
    if (S.median) items.push(h('span', {}, h('i', { class: 'line', style: 'border-color: var(--accent)' }), 'Mediana'));
    if (S.bands) {
      items.push(h('span', {}, h('i', { style: 'background: var(--accent); opacity: .22' }), `${pct(f.levels[f.outer.lo]!)}–${pct(f.levels[f.outer.hi]!)}`));
      items.push(h('span', {}, h('i', { style: 'background: var(--accent); opacity: .45' }), `${pct(f.levels[f.inner.lo]!)}–${pct(f.levels[f.inner.hi]!)}`));
    }
    if (S.meanCandles && f.mean) items.push(h('span', {}, h('i', { class: 'hollow', style: 'border-color: var(--up)' }), 'Velas previstas'));
    if (S.paths && f.paths) items.push(h('span', {}, h('i', { class: 'line', style: 'border-color: var(--accent); opacity: .5; border-top-width: 1px' }), 'Trayectorias'));
  }
  for (const p of pinned) items.push(h('span', {}, h('i', { class: 'dash', style: `border-color: ${p.color}` }), `Fijada · ${p.forecast.label}`));
  if (!items.length) items.push(h('span', { class: 'muted' }, 'Sin predicción'));
  el.replaceChildren(...items);
}

function tile(k: string, v: string | HTMLElement, s?: string | HTMLElement): HTMLElement {
  return h('div', { class: 'tile' }, h('div', { class: 'k' }, k), h('div', { class: 'v' }, v), s ? h('div', { class: 's' }, s) : null);
}

const delta = (pctValue: number, text = fmtPct(pctValue)): HTMLElement =>
  h('span', { class: `delta ${pctValue >= 0 ? 'up' : 'down'}` }, icon(pctValue >= 0 ? 'arrowUp' : 'arrowDown', 13), text.replace(/^[+-]/, ''));

const mark = (ok: boolean | null, text: string): HTMLElement =>
  ok === null ? h('span', {}, '—') : h('span', { class: `mark ${ok ? 'ok' : 'ko'}` }, h('span', { class: 'dot' }, icon(ok ? 'check' : 'x', 12)), text);

const empty = (name: IconName, text: string): HTMLElement => h('div', { class: 'empty' }, h('div', { class: 'bubble' }, icon(name, 22)), h('div', {}, text));

function probText(p: number, bound: string): string {
  return `${bound === '<' ? 'menos del ' : bound === '>' ? 'más del ' : ''}${fmtNum(p * 100)} %`;
}

function renderSummary(): void {
  const el = $('tab-summary');
  const f = current;
  if (!f) {
    el.replaceChildren(
      empty('chart', 'Pulsa «Predecir» para generar una predicción. La primera vez se descarga el modelo y se guarda en la caché del navegador; después funciona sin volver a descargarlo.'),
    );
    return;
  }
  const s = summarize(f);
  const end = f.times[f.times.length - 1]!;
  const tiles: HTMLElement[] = [];
  if (s.expectedHigh !== undefined) {
    tiles.push(
      tile('Máximo esperado', fmtPrice(s.expectedHigh), fmtPct(s.medianRunupPct!)),
      tile('Mínimo esperado', fmtPrice(s.expectedLow!), fmtPct(s.medianDrawdownPct!)),
      tile('Dispersión final', `± ${fmtNum(s.endStdPct!, 2)} %`, 'entre trayectorias'),
    );
  }
  tiles.push(tile('Cálculo', fmtDuration(f.ms), `${f.label} · ${engine.backend === 'webgpu' ? 'WebGPU' : 'CPU'}`));
  const parts: HTMLElement[] = [hero(f, s, end), h('div', { class: 'tiles compact animate' }, ...tiles)];

  const sc = score(f, new Map(all.map((c) => [c.t, c])));
  // Only closed candles count as "what happened".
  const closedScore = sc && f.times[0]! + stepMs <= Date.now() ? sc : null;
  if (closedScore) {
    parts.push(h('div', { class: 'section-title' }, `Comparación con lo ocurrido (${closedScore.steps} de ${f.times.length} velas)`));
    parts.push(h('div', { class: 'tiles animate' }, ...scoreTiles(closedScore)));
  }

  const input = h('input', { type: 'number', class: 'input', step: 'any', value: S.target > 0 ? String(S.target) : '', placeholder: fmtPrice(s.endHi), id: 'target' });
  const out = h('span', {});
  const update = () => {
    const t = Number(input.value);
    S.target = t > 0 ? t : 0;
    saveSettings(S);
    if (!(t > 0)) {
      out.textContent = '';
      return;
    }
    const o = targetOdds(f, t);
    const dir = o.above ? 'por encima' : 'por debajo';
    out.textContent = `${o.touch !== undefined ? `Lo toca en algún momento: ${fmtNum(o.touch * 100)} %. ` : ''}Cierra ${dir} al final: ${probText(o.endBeyond, o.bound)}.`;
  };
  input.addEventListener('input', update);
  parts.push(h('div', { class: 'inline-form card' }, h('label', { for: 'target' }, 'Precio objetivo'), input, out));
  update();
  parts.push(
    h('div', { class: 'meta-line' }, Object.entries(f.settings).map(([k, v]) => `${k}: ${v}`).join(' · ')),
  );
  el.replaceChildren(...parts);
}

/** Headline card: where the median ends, the odds of going up and the spread of outcomes. */
function hero(f: Forecast, s: ReturnType<typeof summarize>, end: number): HTMLElement {
  const up = Math.round(s.pUp * 100);
  const bound = s.pUpBound === '<' ? 'menos del ' : s.pUpBound === '>' ? 'más del ' : '';
  const lo = Math.min(s.endLo, s.last);
  const hi = Math.max(s.endHi, s.last);
  const pad = (hi - lo) * 0.08 || hi * 0.001;
  const at = (p: number) => `${((p - lo + pad) / (hi - lo + 2 * pad)) * 100}%`;
  const fromNow = S.offset === 0 && !csv;
  return h(
    'div',
    { class: 'hero card' },
    h(
      'div',
      { class: 'hero-main' },
      h('div', { class: 'k' }, `Precio previsto · ${fmtTime(end, S.localTime)}`),
      h('div', { class: 'hero-v' }, fmtPrice(s.endMedian)),
      h('div', { class: 's' }, delta(s.changePct), h('span', {}, `desde ${fmtPrice(s.last)} · ${fmtTime(f.anchor.t, S.localTime)}`)),
    ),
    h(
      'div',
      { class: 'hero-side' },
      h(
        'div',
        { class: 'meter' },
        h('div', { class: 'meter-head' }, h('span', {}, '¿Acaba más arriba que al inicio?')),
        h('div', { class: 'odds' }, h('span', { class: 'odds-up', style: `width:${up}%` }), h('span', { class: 'odds-down' })),
        h(
          'div',
          { class: 'meter-foot' },
          h('span', { class: 'up' }, icon('arrowUp', 13), `Sube ${bound}${up} %`),
          h('span', { class: 'down' }, `Baja ${100 - up} %`, icon('arrowDown', 13)),
        ),
      ),
      h(
        'div',
        { class: 'meter' },
        h('div', { class: 'meter-head' }, h('span', {}, `Rango ${pct(s.loLevel)}–${pct(s.hiLevel)} al final`), h('b', {}, `${fmtPct((s.endLo / s.last - 1) * 100, 1)} / ${fmtPct((s.endHi / s.last - 1) * 100, 1)}`)),
        h(
          'div',
          { class: 'range' },
          h('span', { class: 'range-band', style: `left:${at(s.endLo)};right:calc(100% - ${at(s.endHi)})` }),
          h('span', { class: 'range-now', style: `left:${at(s.last)}`, title: fromNow ? 'Precio actual' : 'Inicio de la predicción' }),
          h('span', { class: 'range-mid', style: `left:${at(s.endMedian)}`, title: 'Mediana' }),
        ),
        h(
          'div',
          { class: 'meter-foot' },
          h('span', {}, fmtPrice(s.endLo)),
          h('span', { class: 'lg now' }, fromNow ? 'ahora' : 'inicio'),
          h('span', { class: 'lg mid' }, 'mediana'),
          h('span', {}, fmtPrice(s.endHi)),
        ),
      ),
    ),
  );
}

function scoreTiles(sc: Score): HTMLElement[] {
  return [
    tile('Error medio (MAPE)', `${fmtNum(sc.mapePct, 2)} %`, `MAE ${fmtPrice(sc.mae)}`),
    tile('Habilidad frente a «no se mueve»', fmtPct(sc.skill * 100, 0), sc.skill > 0 ? 'mejor que el precio constante' : 'peor que el precio constante'),
    tile('Dirección', mark(sc.directionHit, sc.directionHit ? 'Acertada' : 'Fallada'), `previsto ${fmtPct(sc.predictedReturnPct)} · real ${fmtPct(sc.actualReturnPct)}`),
    tile('Cierres dentro de la banda', `${fmtNum(sc.coverage * 100)} %`, 'lo esperable es un 80 %'),
  ];
}

function renderBacktest(): void {
  const el = $('tab-backtest');
  const bt = backtest;
  if (!bt || !bt.rows.length) {
    el.replaceChildren(
      empty('history', bt?.running ? 'Ejecutando la primera ventana' : 'Aún no hay backtest. Configúralo en el panel «Backtest» y pulsa «Ejecutar backtest».'),
    );
    return;
  }
  const n = bt.rows.length;
  const sum = (fn: (s: Score) => number) => bt.rows.reduce((a, r) => a + fn(r.score), 0);
  const judged = bt.rows.filter((r) => r.score.directionHit !== null);
  const hits = judged.filter((r) => r.score.directionHit).length;
  const skill = 1 - sum((s) => s.mae) / (sum((s) => s.naiveMae) || 1);
  // Toy strategy: take the side the forecast points to on each window, no costs.
  const follow = bt.rows.map((r) => Math.sign(r.score.predictedReturnPct) * r.score.actualReturnPct);
  const compounded = (follow.reduce((a, r) => a * (1 + r / 100), 1) - 1) * 100;
  const tiles = h(
    'div',
    { class: 'tiles' },
    tile('Ventanas', bt.running ? `${n} en curso` : `${n}`, bt.model),
    tile('Error medio (MAPE)', `${fmtNum(sum((s) => s.mapePct) / n, 2)} %`),
    tile('Habilidad frente a «no se mueve»', fmtPct(skill * 100, 0), skill > 0 ? 'mejor que el precio constante' : 'no mejora al precio constante'),
    tile('Dirección acertada', judged.length ? `${hits} de ${judged.length}` : '—', judged.length ? `${fmtNum((hits / judged.length) * 100)} %; al azar sería el 50 %` : undefined),
    tile('Cierres dentro de la banda', `${fmtNum((sum((s) => s.coverage) / n) * 100)} %`, 'lo esperable es un 80 %'),
    tile('Seguir la señal', delta(compounded), 'rentabilidad compuesta, sin comisiones'),
  );
  const rows = bt.rows.map((r) => {
    const s = r.score;
    return h(
      'tr',
      { class: 'clickable', title: 'Ver esta ventana en el gráfico', onclick: () => showForecast(r.forecast) },
      h('td', {}, fmtTime(r.forecast.anchor.t, S.localTime, true)),
      h('td', {}, fmtPct(s.predictedReturnPct)),
      h('td', {}, fmtPct(s.actualReturnPct)),
      h('td', {}, mark(s.directionHit, s.directionHit ? 'Sí' : 'No')),
      h('td', {}, `${fmtNum(s.mapePct, 2)} %`),
      h('td', {}, fmtPct(s.skill * 100, 0)),
      h('td', {}, `${fmtNum(s.coverage * 100)} %`),
    );
  });
  const head = ['Inicio', 'Previsto', 'Real', 'Dirección', 'MAPE', 'Habilidad', 'En banda'].map((t) => h('th', {}, t));
  el.replaceChildren(
    tiles,
    h('div', { class: 'section-title' }, 'Por ventana (pulsa una fila para verla en el gráfico)'),
    h('div', { class: 'table-wrap card' }, h('table', {}, h('thead', {}, h('tr', {}, ...head)), h('tbody', {}, ...rows))),
    h(
      'div',
      { class: 'hint', style: 'margin-top:8px' },
      'Pocas ventanas dicen poco: con menos de 30 las diferencias suelen ser ruido. «Previsto» y «Real» son la variación entre el último cierre del contexto y el final del horizonte.',
    ),
  );
}

function showForecast(f: Forecast): void {
  current = f;
  chart.setForecast(current, pinned);
  chart.focus();
  renderAll();
  $('chart').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function tableRows(f: Forecast): { head: string[]; rows: (string | number)[][] } {
  const actual = new Map(all.map((c) => [c.t, c]));
  const head = ['time', ...f.levels.map(pct), ...(f.mean ? ['mean_open', 'mean_high', 'mean_low', 'mean_close', 'mean_volume'] : []), 'actual_close'];
  const rows = f.times.map((t, i) => [
    new Date(t).toISOString(),
    ...f.q.map((q) => q[i]!),
    ...(f.mean ? [f.mean[i]!.o, f.mean[i]!.h, f.mean[i]!.l, f.mean[i]!.c, f.mean[i]!.v] : []),
    actual.get(t)?.c ?? '',
  ]);
  return { head, rows };
}

function renderTable(): void {
  const el = $('tab-table');
  const f = current;
  if (!f) {
    el.replaceChildren(empty('chart', 'Sin predicción todavía.'));
    return;
  }
  const actual = new Map(all.map((c) => [c.t, c]));
  const head = ['Vela', pct(f.levels[f.outer.lo]!), 'Mediana', pct(f.levels[f.outer.hi]!), 'Variación', ...(f.mean ? ['Volumen medio'] : []), 'Real'];
  const body = f.times.map((t, i) => {
    const med = f.q[f.median]![i]!;
    const real = actual.get(t);
    return h(
      'tr',
      {},
      h('td', {}, fmtTime(t, S.localTime)),
      h('td', {}, fmtPrice(f.q[f.outer.lo]![i]!)),
      h('td', {}, fmtPrice(med)),
      h('td', {}, fmtPrice(f.q[f.outer.hi]![i]!)),
      h('td', {}, fmtPct((med / f.anchor.c - 1) * 100)),
      f.mean ? h('td', {}, fmtCompact(f.mean[i]!.v)) : null,
      h('td', {}, real ? fmtPrice(real.c) : '—'),
    );
  });
  el.replaceChildren(h('div', { class: 'table-wrap card' }, h('table', {}, h('thead', {}, h('tr', {}, ...head.map((t) => h('th', {}, t)))), h('tbody', {}, ...body))));
}

async function renderModels(): Promise<void> {
  if (!manifest) return;
  const el = $('tab-models');
  const ids = modelOptions().map(([id]) => id);
  const rows = await Promise.all(
    ids.map(async (id) => {
      const info = modelInfo(id);
      const cached = (await Promise.all(info.files.map(isCached))).every(Boolean);
      const size = info.files.reduce((a, f) => a + f.bytes, 0);
      const state = inMemory.has(id) ? 'mem' : cached ? 'cached' : 'none';
      const status = h('span', { class: `pill ${state}` }, state === 'mem' ? 'En memoria' : state === 'cached' ? 'Descargado' : 'Sin descargar');
      const get = h('button', { class: 'btn small', hidden: cached }, icon('download', 14), 'Descargar');
      get.addEventListener('click', async () => {
        get.disabled = true;
        try {
          let before = 0;
          for (const f of info.files) {
            await fetchModel(f, (loaded) => (status.textContent = `Descargando ${fmtBytes(before + loaded)} de ${fmtBytes(size)}`));
            before += f.bytes;
          }
        } catch (e) {
          notice((e as Error).message);
        }
        void renderModels();
      });
      const del = h('button', { class: 'btn small ghost icon-sm', hidden: !cached, title: 'Borrar de la caché', 'aria-label': 'Borrar' }, icon('trash', 14));
      del.addEventListener('click', async () => {
        // Tokenizers are shared between Kronos sizes; only drop files no other cached model needs.
        const others = (await Promise.all(ids.filter((o) => o !== id).map(async (o) => ((await Promise.all(modelInfo(o).files.slice(-2).map(isCached))).every(Boolean) ? modelInfo(o).files : [])))).flat();
        await removeCached(info.files.filter((f) => !others.some((o) => o.file === f.file)));
        void renderModels();
      });
      const use = h('button', { class: 'btn small', disabled: S.model === id }, S.model === id ? 'En uso' : 'Usar');
      use.addEventListener('click', () => {
        S.model = id;
        setSelect('model', id);
        saveSettings(S);
        void onChange('model');
      });
      return h(
        'div',
        { class: `model-row card${S.model === id ? ' current' : ''}` },
        h('div', { class: 'avatar' }, icon(id.startsWith('chronos') ? 'activity' : 'cpu', 18)),
        h('div', { class: 'model-info' }, h('div', { class: 'name' }, info.label), h('div', { class: 'meta' }, `${info.params} parámetros · ${fmtBytes(size)} · contexto ${info.maxContext}`), status),
        h('div', { class: 'btns' }, use, get, del),
      );
    }),
  );
  let usage = '';
  try {
    const est = await navigator.storage?.estimate();
    if (est?.usage !== undefined && est.quota) usage = `Almacenamiento del sitio: ${fmtBytes(est.usage)} usados de ${fmtBytes(est.quota)} disponibles.`;
  } catch {
    /* not available */
  }
  el.replaceChildren(h('div', { class: 'models' }, ...rows), h('div', { class: 'hint', style: 'margin-top:8px' }, `${usage} Los modelos se guardan en este navegador y no se vuelven a descargar.`));
}

// ---------------------------------------------------------------- actions

function updateTabFade(): void {
  const bar = document.querySelector<HTMLElement>('.tabs')!;
  bar.classList.toggle('fade-l', bar.scrollLeft > 4);
  bar.classList.toggle('fade-r', bar.scrollLeft + bar.clientWidth < bar.scrollWidth - 4);
}

function selectTab(name: string): void {
  for (const b of document.querySelectorAll<HTMLButtonElement>('.tabs button')) {
    const on = b.dataset.tab === name;
    b.setAttribute('aria-selected', String(on));
    if (on) {
      const ind = $('tab-ind');
      ind.style.width = `${b.offsetWidth}px`;
      ind.style.transform = `translateX(${b.offsetLeft}px)`;
      const bar = b.parentElement!;
      if (b.offsetLeft < bar.scrollLeft || b.offsetLeft + b.offsetWidth > bar.scrollLeft + bar.clientWidth) bar.scrollTo({ left: b.offsetLeft - 24, behavior: 'smooth' });
    }
  }
  for (const t of document.querySelectorAll<HTMLElement>('.tab')) t.hidden = t.id !== `tab-${name}`;
}

function exportAs(kind: string): void {
  const f = current;
  if (!f) return;
  const stamp = new Date(f.anchor.t).toISOString().slice(0, 16).replace(/[:T]/g, '-');
  const name = `${csv ? 'csv' : S.symbol}-${f.model.replace(':', '-')}-${stamp}`;
  if (kind === 'png') {
    chart.screenshot().toBlob((b) => b && download(`${name}.png`, b));
    return;
  }
  const { head, rows } = tableRows(f);
  if (kind === 'csv') {
    download(`${name}.csv`, [head.join(','), ...rows.map((r) => r.join(','))].join('\n'), 'text/csv');
    return;
  }
  const payload = {
    model: f.label,
    source: f.model,
    pair: pairLabel(),
    anchor: { time: new Date(f.anchor.t).toISOString(), close: f.anchor.c },
    settings: f.settings,
    columns: head,
    rows,
    paths: f.paths?.map((p) => Array.from({ length: f.times.length }, (_, i) => p[i * 6 + 3]!)),
  };
  download(`${name}.json`, JSON.stringify(payload, null, 1), 'application/json');
}

function applyTheme(): void {
  if (S.theme === 'dark' || S.theme === 'light') document.documentElement.dataset.theme = S.theme;
  else delete document.documentElement.dataset.theme;
  const dark = S.theme === 'dark' || (S.theme === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches);
  $('btn-theme').replaceChildren(icon(dark ? 'sun' : 'moon', 18));
  chart?.setColors(chartColors());
}

function wireActions(): void {
  $('btn-predict').addEventListener('click', () => void predict());
  $('btn-cancel').addEventListener('click', () => engine.cancel());
  $('btn-pin').addEventListener('click', () => {
    if (!current || pinned.length >= 2) return;
    const used = new Set(pinned.map((p) => p.color));
    const color = [cssVar('--pin-1'), cssVar('--pin-2')].find((c) => !used.has(c))!;
    pinned = [...pinned, { forecast: current, color }];
    current = null;
    chart.setForecast(null, pinned);
    renderAll();
    toast('Predicción fijada. Lanza otra para compararlas.');
  });
  $('btn-clear').addEventListener('click', () => {
    current = null;
    pinned = [];
    chart.setForecast(null, pinned);
    renderAll();
  });
  const menu = $('export-menu');
  const exportBtn = $('btn-export');
  const toggleMenu = (open: boolean) => {
    menu.classList.toggle('open', open);
    exportBtn.setAttribute('aria-expanded', String(open));
  };
  exportBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleMenu(!menu.classList.contains('open'));
  });
  document.addEventListener('click', () => toggleMenu(false));
  const drawer = (open: boolean) => document.body.classList.toggle('drawer-open', open);
  $('btn-settings').addEventListener('click', () => drawer(true));
  $('btn-close-settings').addEventListener('click', () => drawer(false));
  $('backdrop').addEventListener('click', () => drawer(false));
  addEventListener('resize', () => selectTab(document.querySelector<HTMLElement>('.tabs [aria-selected="true"]')?.dataset.tab ?? 'summary'));
  menu.addEventListener('click', (e) => {
    const kind = (e.target as HTMLElement).closest<HTMLElement>('[data-export]')?.dataset.export;
    if (kind) exportAs(kind);
  });
  $('btn-share').addEventListener('click', async () => {
    const url = shareUrl(S);
    try {
      await navigator.clipboard.writeText(url);
      toast('Enlace copiado');
    } catch {
      prompt('Copia este enlace', url);
    }
  });
  $('btn-theme').addEventListener('click', () => {
    const dark = S.theme === 'dark' || (S.theme === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches);
    S.theme = dark ? 'light' : 'dark';
    saveSettings(S);
    applyTheme();
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
  for (const b of document.querySelectorAll<HTMLButtonElement>('.tabs button')) b.addEventListener('click', () => selectTab(b.dataset.tab!));
  document.querySelector('.tabs')!.addEventListener('scroll', updateTabFade, { passive: true });
  addEventListener('resize', updateTabFade);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) void predict();
    if (e.key === 'Escape') {
      if (document.body.classList.contains('drawer-open')) drawer(false);
      else if (busy) engine.cancel();
    }
  });
}

async function main(): Promise<void> {
  hydrateIcons();
  buildControls();
  chart = new ForecastChart($('chart'), chartColors(), viewOptions());
  applyTheme();
  wireActions();
  renderAll();
  selectTab('summary');
  updateTabFade();
  void document.fonts?.ready.then(() => {
    selectTab('summary');
    updateTabFade();
  });
  log(`Kronos WebGPU · aislamiento de origen: ${crossOriginIsolated ? 'sí' : 'no'} · núcleos: ${navigator.hardwareConcurrency ?? '?'}`);
  const data = loadData();
  try {
    manifest = await loadManifest();
    fillSelect('model');
    saveSettings(S);
    void pruneCache(manifest);
    void navigator.storage?.persist?.().catch(() => {});
    log(`Catálogo: ${allFiles(manifest).length} archivos de modelo`);
    await startEngine();
  } catch (e) {
    notice((e as Error).message);
  }
  await data;
}

void main();
