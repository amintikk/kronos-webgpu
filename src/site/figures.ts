// Figures for the landing page. All of them draw precomputed results from public/demo.json
// (produced by tools/make_demo.ts with the same engine as the app). Colours come from CSS
// classes, so the figures follow the theme without re-rendering.
import { locale, t } from '../i18n.ts';
import { fmtPct, fmtPrice } from '../ui/dom.ts';

export interface DemoData {
  model: string;
  context: number;
  candles: number[][];
  times: number[];
  paths: number[][];
  tokens: { t: number; s1: number; s2: number; bits: number[] }[];
  backtest: { horizon: number; samples: number; windows: { t: number; last: number; median: number; p10: number; p90: number; actual: number }[] };
}

const NS = 'http://www.w3.org/2000/svg';
type Attrs = Record<string, string | number>;

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Attrs = {}, text?: string): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (text !== undefined) e.textContent = text;
  return e;
}

function svg(width: number, height: number, label: string): SVGSVGElement {
  return el('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': label, class: 'fig-svg' });
}

const fmtDate = (t: number) => new Date(t).toLocaleString(locale, { timeZone: 'UTC', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });

/** Redraw on width changes; figures are cheap to rebuild. */
function responsive(host: HTMLElement, draw: (width: number) => void): void {
  let last = 0;
  const run = () => {
    const w = Math.round(host.clientWidth);
    if (w && w !== last) {
      last = w;
      draw(w);
    }
  };
  new ResizeObserver(run).observe(host);
  run();
}

/** A tooltip shared by the figures, positioned inside the figure's frame. */
function tooltip(frame: HTMLElement) {
  const tip = document.createElement('div');
  tip.className = 'fig-tip';
  frame.append(tip);
  return {
    show(x: number, y: number, rows: [string, string][], title: string) {
      tip.replaceChildren();
      const h = document.createElement('b');
      h.textContent = title;
      tip.append(h);
      for (const [k, v] of rows) {
        const r = document.createElement('div');
        const a = document.createElement('span');
        const b = document.createElement('span');
        a.textContent = k;
        b.textContent = v;
        r.append(a, b);
        tip.append(r);
      }
      const fw = frame.clientWidth;
      tip.classList.add('show');
      const tw = tip.offsetWidth;
      const left = x + 14 + tw > fw ? x - 14 - tw : x + 14;
      tip.style.transform = `translate(${Math.max(0, left)}px, ${Math.max(0, y - tip.offsetHeight / 2)}px)`;
    },
    hide() {
      tip.classList.remove('show');
    },
  };
}

// ------------------------------------------------------------ 1. candles to tokens

export function tokensFigure(host: HTMLElement, d: DemoData): void {
  const frame = host;
  const tip = tooltip(frame);
  const byTime = new Map(d.candles.map((c) => [c[0]!, c]));
  const items = d.tokens.map((t) => ({ ...t, c: byTime.get(t.t)! }));
  responsive(frame, (W) => {
    frame.querySelector('svg')?.remove();
    const n = items.length;
    const labelW = W >= 560 ? 132 : 0;
    const colW = (W - labelW) / n;
    const cell = Math.max(4, Math.min(11, Math.floor(colW * 0.42)));
    const gap = cell >= 8 ? 2 : 1;
    const stackH = 10 * cell + 9 * gap;
    const candleH = 96;
    const top = 8;
    const s1Top = top + candleH + 26;
    const s2Top = s1Top + stackH + 22;
    const idsY = s2Top + stackH + 18;
    const H = idsY + (colW >= 34 ? 8 : -10);
    const s = svg(W, H, t('Las últimas 12 velas del ejemplo y las dos fichas de 10 bits en que las convierte el tokenizador de Kronos'));
    const lo = Math.min(...items.map((i) => i.c[3]!));
    const hi = Math.max(...items.map((i) => i.c[2]!));
    const y = (p: number) => top + ((hi - p) / (hi - lo || 1)) * candleH;
    if (labelW) {
      for (const [text, yy] of [[t('Vela'), top + candleH / 2], [t('Ficha gruesa'), s1Top + stackH / 2], [t('Ficha fina'), s2Top + stackH / 2]] as const) {
        s.append(el('text', { x: 0, y: yy, class: 'fig-label', 'dominant-baseline': 'middle' }, text));
      }
      s.append(el('text', { x: 0, y: s1Top + stackH / 2 + 16, class: 'fig-note', 'dominant-baseline': 'middle' }, '10 bits'));
      s.append(el('text', { x: 0, y: s2Top + stackH / 2 + 16, class: 'fig-note', 'dominant-baseline': 'middle' }, '10 bits'));
    } else {
      // Narrow screens: no label column, so name each row just above it.
      s.append(el('text', { x: 0, y: s1Top - 9, class: 'fig-note' }, t('Ficha gruesa · 10 bits')));
      s.append(el('text', { x: 0, y: s2Top - 8, class: 'fig-note' }, t('Ficha fina · 10 bits')));
    }
    items.forEach((it, k) => {
      const cx = labelW + colW * (k + 0.5);
      const [, o, h, l, c] = it.c as [number, number, number, number, number];
      const up = c >= o;
      const g = el('g', { class: 'fig-col', style: `--i:${k}` });
      g.append(el('rect', { x: labelW + colW * k, y: 0, width: colW, height: H, class: 'fig-hit' }));
      g.append(el('line', { x1: cx, x2: cx, y1: y(h), y2: y(l), class: up ? 'stroke-up' : 'stroke-down' }));
      const bw = Math.max(3, Math.min(14, colW * 0.36));
      g.append(el('rect', { x: cx - bw / 2, y: Math.min(y(o), y(c)), width: bw, height: Math.max(1.5, Math.abs(y(o) - y(c))), rx: 1.5, class: up ? 'fill-up' : 'fill-down' }));
      // Bits, least significant first from the top, as the tokenizer builds the index.
      for (let b = 0; b < 20; b++) {
        const coarse = b < 10;
        const row = coarse ? b : b - 10;
        const yy = (coarse ? s1Top : s2Top) + row * (cell + gap);
        const on = it.bits[b] === 1;
        g.append(el('rect', { x: cx - cell / 2, y: yy, width: cell, height: cell, rx: Math.min(2, cell / 4), class: on ? (coarse ? 'bit-on' : 'bit-on-fine') : 'bit-off' }));
      }
      if (colW >= 34) {
        g.append(el('text', { x: cx, y: idsY, class: 'fig-id', 'text-anchor': 'middle' }, `${it.s1}`));
        g.append(el('text', { x: cx, y: idsY + 15, class: 'fig-id muted', 'text-anchor': 'middle' }, `${it.s2}`));
      }
      g.addEventListener('pointerenter', () => {
        const r = (g.firstChild as SVGRectElement).getBoundingClientRect();
        const f = frame.getBoundingClientRect();
        tip.show(r.left - f.left + r.width / 2, top + candleH / 2, [
          [t('Apertura'), fmtPrice(o)],
          [t('Cierre'), fmtPrice(c)],
          [t('Máx. / mín.'), `${fmtPrice(h)} / ${fmtPrice(l)}`],
          [t('Ficha gruesa'), String(it.s1)],
          [t('Ficha fina'), String(it.s2)],
        ], fmtDate(it.t));
      });
      g.addEventListener('pointerleave', () => tip.hide());
      s.append(g);
    });
    if (colW >= 34) s.setAttribute('height', String(idsY + 22));
    if (colW >= 34) s.setAttribute('viewBox', `0 0 ${W} ${idsY + 22}`);
    frame.prepend(s);
  });
}

// ------------------------------------------------------------ 2. from paths to a band

const quantile = (sorted: number[], p: number) => {
  const pos = (sorted.length - 1) * p;
  const lo = Math.floor(pos);
  const hi = Math.min(sorted.length - 1, lo + 1);
  return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (pos - lo);
};

export function pathsFigure(host: HTMLElement, controls: HTMLElement, d: DemoData): void {
  const frame = host;
  const counts = [1, 4, 8, 32];
  let shown = 4;
  const history = d.candles.slice(-36);
  const anchor = history[history.length - 1]!;
  const P = d.times.length;
  const closes = d.paths.map((p) => Array.from({ length: P }, (_, i) => p[i * 6 + 3]!));
  const all = [...history.map((c) => c[4]!), ...closes.flat()];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  let width = 0;

  const draw = () => {
    const W = width;
    frame.querySelector('svg')?.remove();
    const H = W >= 560 ? 300 : 230;
    const pad = { l: 0, r: 64, t: 12, b: 12 };
    const steps = history.length + P;
    const x = (i: number) => pad.l + (i / (steps - 1)) * (W - pad.l - pad.r);
    const y = (p: number) => pad.t + ((hi - p) / (hi - lo)) * (H - pad.t - pad.b);
    const s = svg(W, H, t(shown === 1 ? 'Precio de cierre y 1 trayectoria prevista' : 'Precio de cierre y {n} trayectorias previstas, con su mediana y la banda del 80 %', { n: shown }));
    const a = history.length - 1;
    const line = (vals: number[], from: number) => vals.map((v, i) => `${i ? 'L' : 'M'}${x(from + i).toFixed(1)},${y(v).toFixed(1)}`).join('');
    s.append(el('line', { x1: x(a), x2: x(a), y1: 0, y2: H, class: 'fig-divider' }));
    s.append(el('path', { d: line(history.map((c) => c[4]!), 0), class: 'fig-history', pathLength: 1 }));
    const subset = closes.slice(0, shown).map((p) => [anchor[4]!, ...p]);
    const pts = (k: number) => subset.map((p) => p[k]!).sort((m, n) => m - n);
    if (shown >= 4) {
      const upper = Array.from({ length: P + 1 }, (_, k) => quantile(pts(k), 0.9));
      const lower = Array.from({ length: P + 1 }, (_, k) => quantile(pts(k), 0.1));
      s.append(el('path', { d: `${line(upper, a)}L${lower.map((_, i) => `${x(a + P - i).toFixed(1)},${y(lower[P - i]!).toFixed(1)}`).join('L')}Z`, class: 'fig-band' }));
    }
    subset.forEach((p, k) => s.append(el('path', { d: line(p, a), class: shown === 1 ? 'fig-path solo' : 'fig-path', pathLength: 1, style: `--i:${k % 16}` })));
    if (shown > 1) {
      const median = Array.from({ length: P + 1 }, (_, k) => quantile(pts(k), 0.5));
      s.append(el('path', { d: line(median, a), class: 'fig-median', pathLength: 1 }));
      s.append(el('text', { x: x(a + P) + 8, y: y(median[P]!), class: 'fig-label accent', 'dominant-baseline': 'middle' }, t('mediana')));
    }
    s.append(el('text', { x: x(a) - 8, y: H - 4, class: 'fig-note', 'text-anchor': 'end' }, t('pasado')));
    s.append(el('text', { x: x(a) + 8, y: H - 4, class: 'fig-note' }, t('previsto')));
    frame.prepend(s);
  };

  controls.replaceChildren(
    ...counts.map((n) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = n === 1 ? t('1 trayectoria') : `${n}`;
      b.setAttribute('aria-pressed', String(n === shown));
      b.addEventListener('click', () => {
        shown = n;
        for (const o of controls.querySelectorAll('button')) o.setAttribute('aria-pressed', String(o === b));
        draw();
      });
      return b;
    }),
  );
  responsive(frame, (w) => {
    width = w;
    draw();
  });
}

// ------------------------------------------------------------ 3. work done per forecast

export function workFigure(host: HTMLElement, d: DemoData): void {
  const T = d.context;
  const P = d.times.length;
  const S = d.paths.length;
  // Upstream re-runs the transformer over the whole window for every new candle, for every sample.
  let upstream = 0;
  for (let i = 0; i < P; i++) upstream += (T + i) * S;
  // Here: the context once, shared by all samples, then one token per sample and step.
  const ours = T + (P - 1) * S;
  const rows = [
    { label: t('Kronos original'), value: upstream, cls: 'bar-muted' },
    { label: 'Kronos WebGPU', value: ours, cls: 'bar-accent' },
  ];
  responsive(host, (W) => {
    host.querySelector('svg')?.remove();
    const labelW = W >= 560 ? 150 : 0;
    const rowH = labelW ? 44 : 62;
    const H = rows.length * rowH;
    const s = svg(W, H, t('Fichas procesadas para una predicción: {a} en el original frente a {b}', { a: upstream.toLocaleString(locale), b: ours.toLocaleString(locale) }));
    const valueW = 110;
    const max = W - labelW - valueW;
    rows.forEach((r, k) => {
      const y0 = k * rowH + (labelW ? 8 : 26);
      if (labelW) s.append(el('text', { x: 0, y: y0 + 12, class: 'fig-label', 'dominant-baseline': 'middle' }, r.label));
      else s.append(el('text', { x: 0, y: y0 - 10, class: 'fig-label' }, r.label));
      const w = Math.max(3, (r.value / upstream) * max);
      s.append(el('rect', { x: labelW, y: y0, width: w, height: 24, rx: 4, class: `${r.cls} fig-bar`, style: `--i:${k}` }));
      s.append(el('text', { x: labelW + w + 10, y: y0 + 12, class: 'fig-value', 'dominant-baseline': 'middle' }, r.value.toLocaleString(locale)));
    });
    host.prepend(s);
  });
  const ratio = document.querySelector<HTMLElement>('[data-fig="work-ratio"]');
  if (ratio) ratio.textContent = t('{n} veces menos cálculo', { n: Math.round(upstream / ours) });
}

// ------------------------------------------------------------ 4. backtest

export function backtestFigure(host: HTMLElement, stats: HTMLElement, d: DemoData): void {
  const w = d.backtest.windows.map((r) => ({ ...r, pred: (r.median / r.last - 1) * 100, real: (r.actual / r.last - 1) * 100, inside: r.actual >= r.p10 && r.actual <= r.p90 }));
  const hits = w.filter((r) => r.pred * r.real > 0).length;
  const inside = w.filter((r) => r.inside).length;
  const mae = w.reduce((s, r) => s + Math.abs(r.median - r.actual), 0);
  const naive = w.reduce((s, r) => s + Math.abs(r.last - r.actual), 0);
  const skill = (1 - mae / naive) * 100;
  const fmtDay = (t: number) => new Date(t).toLocaleDateString(locale, { timeZone: 'UTC', day: 'numeric', month: 'long' });
  stats.replaceChildren(
    ...([
      [t('Dirección acertada'), t('{a} de {b}', { a: hits, b: w.length })],
      [t('Dentro de la banda del 80 %'), t('{a} de {b}', { a: inside, b: w.length })],
      [t('Error frente a «no se mueve»'), skill >= 0 ? t('{n} % menor', { n: Math.round(skill) }) : t('{n} % mayor', { n: Math.round(-skill) })],
    ] as const).map(([k, v]) => {
      const div = document.createElement('div');
      const a = document.createElement('span');
      const b = document.createElement('b');
      a.textContent = k;
      b.textContent = v;
      div.append(a, b);
      return div;
    }),
  );
  const range = document.querySelector<HTMLElement>('[data-fig="bt-range"]');
  if (range) range.textContent = t('entre el {a} y el {b}', { a: fmtDay(w[0]!.t), b: fmtDay(w[w.length - 1]!.t + d.backtest.horizon * 3600e3) });

  const tip = tooltip(host);
  responsive(host, (W) => {
    host.querySelector('svg')?.remove();
    const H = Math.min(380, Math.max(260, W * 0.55));
    const pad = { l: 44, r: 12, t: 26, b: 40 };
    const ext = Math.ceil(Math.max(...w.flatMap((r) => [Math.abs(r.pred), Math.abs(r.real)])) * 1.1 * 2) / 2;
    const x = (v: number) => pad.l + ((v + ext) / (2 * ext)) * (W - pad.l - pad.r);
    const y = (v: number) => pad.t + ((ext - v) / (2 * ext)) * (H - pad.t - pad.b);
    const s = svg(W, H, t('Variación prevista frente a variación real en {n} predicciones a 24 horas', { n: w.length }));
    // Quadrants where the predicted direction was right.
    s.append(el('rect', { x: x(0), y: y(ext), width: x(ext) - x(0), height: y(0) - y(ext), class: 'quad-right' }));
    s.append(el('rect', { x: x(-ext), y: y(0), width: x(0) - x(-ext), height: y(-ext) - y(0), class: 'quad-right' }));
    const step = ext > 4 ? 2 : ext > 2 ? 1 : 0.5;
    for (let v = -Math.floor(ext / step) * step; v <= ext + 1e-9; v += step) {
      s.append(el('line', { x1: x(v), x2: x(v), y1: pad.t, y2: H - pad.b, class: v === 0 ? 'fig-axis' : 'fig-grid' }));
      s.append(el('line', { x1: pad.l, x2: W - pad.r, y1: y(v), y2: y(v), class: v === 0 ? 'fig-axis' : 'fig-grid' }));
      const lab = `${v > 0 ? '+' : ''}${v.toLocaleString(locale)} %`;
      s.append(el('text', { x: x(v), y: H - pad.b + 16, class: 'fig-note', 'text-anchor': 'middle' }, lab));
      s.append(el('text', { x: pad.l - 8, y: y(v), class: 'fig-note', 'text-anchor': 'end', 'dominant-baseline': 'middle' }, lab));
    }
    s.append(el('line', { x1: x(-ext), y1: y(-ext), x2: x(ext), y2: y(ext), class: 'fig-diagonal' }));
    s.append(el('text', { x: (pad.l + W - pad.r) / 2, y: H - 4, class: 'fig-label', 'text-anchor': 'middle' }, t('Variación prevista (mediana)')));
    const yl = el('text', { x: 0, y: 12, class: 'fig-label' }, t('Variación real'));
    s.append(yl);
    for (const r of w) {
      const g = el('g', { class: 'fig-dot', style: `--i:${w.indexOf(r)}` });
      g.append(el('circle', { cx: x(r.pred), cy: y(r.real), r: 14, class: 'fig-hit' }));
      g.append(el('circle', { cx: x(r.pred), cy: y(r.real), r: 5.5, class: r.inside ? 'dot-in' : 'dot-out' }));
      g.addEventListener('pointerenter', () => {
        tip.show(x(r.pred), y(r.real), [
          [t('Último cierre'), fmtPrice(r.last)],
          [t('Previsto'), `${fmtPrice(r.median)} (${fmtPct(r.pred)})`],
          [t('Real'), `${fmtPrice(r.actual)} (${fmtPct(r.real)})`],
          [t('Banda del 80 %'), `${fmtPrice(r.p10)} – ${fmtPrice(r.p90)}`],
        ], t('Desde {d}', { d: fmtDate(r.t) }));
      });
      g.addEventListener('pointerleave', () => tip.hide());
      s.append(g);
    }
    host.prepend(s);
  });
}
