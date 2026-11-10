import { locale, t } from './i18n.ts';
import type { Candle } from './engine/types.ts';
import type { Forecast } from './forecast.ts';
import { fmtCompact, fmtPct, fmtPrice, h } from './ui/dom.ts';

export interface ChartColors {
  surface: string;
  text: string;
  grid: string;
  border: string;
  up: string;
  down: string;
  forecast: string;
  muted: string;
}

export interface ViewOptions {
  bands: boolean;
  paths: boolean;
  meanCandles: boolean;
  median: boolean;
  volume: boolean;
  log: boolean;
  localTime: boolean;
}

export interface Pinned {
  forecast: Forecast;
  color: string;
}

const alpha = (hex: string, a: number): string => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
};

const FONT = "'Inter Variable', Inter, system-ui, sans-serif";
const AXIS_H = 30;
const PAD_TOP = 14;
const VOLUME_SHARE = 0.15;
const REVEAL_MS = 900;
const MAX_PATHS = 60;
const TIME_STEPS = [60e3, 300e3, 900e3, 1800e3, 3600e3, 7200e3, 14400e3, 21600e3, 43200e3, 86400e3, 2 * 86400e3, 7 * 86400e3, 14 * 86400e3, 30 * 86400e3, 91 * 86400e3, 365 * 86400e3];
const easeOut = (t: number): number => 1 - (1 - t) ** 3;

interface Plot {
  left: number;
  right: number;
  top: number;
  /** Bottom of the price area. */
  bottom: number;
  volTop: number;
  volBottom: number;
  barW: number;
}

/**
 * Canvas chart built around the forecast: real candles, the quantile fan, sample paths and the
 * predicted candles share one time axis that keeps going past the last real candle.
 * The x axis is the candle index (so gaps in the data do not leave holes), fractional while panning.
 */
export class ForecastChart {
  private root: HTMLElement;
  private canvas = h('canvas', { class: 'chart-canvas', tabIndex: 0, 'aria-label': t('Gráfico de velas y predicción') });
  private overlay = h('canvas', { class: 'chart-overlay' });
  private tip = h('div', { class: 'chart-tip' });
  private g: CanvasRenderingContext2D;
  private og: CanvasRenderingContext2D;
  private w = 0;
  private hgt = 0;
  private dpr = 1;

  private colors: ChartColors;
  private view: ViewOptions;
  private data: Candle[] = [];
  private index = new Map<number, number>();
  private stepMs = 3600e3;
  private forecast: Forecast | null = null;
  private pinned: Pinned[] = [];
  private shiftMs = 0;

  /** Visible range in candle-index units, and where it is heading when animating. */
  private from = 0;
  private to = 120;
  private goal: { from: number; to: number } | null = null;
  private yLo = 0;
  private yHi = 1;
  private ySnap = true;
  private revealAt = 0;
  private velocity = 0;
  private pointers = new Map<number, { x: number; y: number }>();
  private drag: { x: number; t: number; pinch: number } | null = null;
  private hover: { x: number; y: number } | null = null;
  private plot: Plot = { left: 0, right: 0, top: 0, bottom: 0, volTop: 0, volBottom: 0, barW: 1 };
  private frame = 0;

  /** wheel: zoom with the mouse wheel (off on pages where the wheel must scroll the page). */
  private wheelZoom: boolean;

  constructor(el: HTMLElement, colors: ChartColors, view: ViewOptions, opts: { wheel?: boolean } = {}) {
    this.wheelZoom = opts.wheel ?? true;
    this.root = el;
    this.colors = colors;
    this.view = view;
    el.classList.add('chart-root');
    el.append(this.canvas, this.overlay, this.tip);
    this.g = this.canvas.getContext('2d')!;
    this.og = this.overlay.getContext('2d')!;
    this.applyView();
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    this.wire();
    void document.fonts?.ready.then(() => this.invalidate());
  }

  // ------------------------------------------------------------ public API

  setColors(colors: ChartColors): void {
    this.colors = colors;
    this.invalidate();
  }

  setView(view: ViewOptions): void {
    this.view = view;
    this.applyView();
    this.ySnap = true;
    this.invalidate();
  }

  private applyView(): void {
    this.shiftMs = this.view.localTime ? -new Date().getTimezoneOffset() * 60e3 : 0;
  }

  setCandles(candles: Candle[], fit = false): void {
    const prevLast = this.data[this.data.length - 1];
    const prevIdx = this.data.length - 1;
    this.data = candles.slice();
    this.index = new Map(candles.map((c, i) => [c.t, i]));
    const diffs = candles.slice(1).map((c, i) => c.t - candles[i]!.t).sort((a, b) => a - b);
    this.stepMs = diffs[diffs.length >> 1] || this.stepMs;
    if (fit) {
      const n = candles.length;
      const bars = this.idealBars();
      this.from = Math.max(-2, n - bars);
      this.to = n + Math.max(4, bars * 0.05);
      this.goal = null;
      this.ySnap = true;
    } else if (prevLast) {
      // More history was loaded: keep looking at the same candles.
      const moved = (this.index.get(prevLast.t) ?? prevIdx) - prevIdx;
      this.from += moved;
      this.to += moved;
      if (this.goal) this.goal = { from: this.goal.from + moved, to: this.goal.to + moved };
    }
    this.invalidate();
  }

  /** Update (or append) the newest real candle. */
  updateLast(c: Candle): void {
    const n = this.data.length;
    const last = this.data[n - 1];
    if (last && c.t < last.t) return;
    if (last && c.t === last.t) this.data[n - 1] = c;
    else {
      this.data.push(c);
      this.index.set(c.t, n);
      // Keep following the right edge if the user is looking at it.
      if (this.to >= n - 1 && !this.drag) {
        this.from += 1;
        this.to += 1;
      }
    }
    this.invalidate();
  }

  setForecast(f: Forecast | null, pinned: Pinned[] = this.pinned): void {
    if (f && f !== this.forecast) this.revealAt = performance.now();
    this.forecast = f;
    this.pinned = pinned;
    this.invalidate();
  }

  /** Glide to the end of the context plus the whole forecast. */
  focus(): void {
    const n = this.data.length;
    if (!n) return;
    const f = this.forecast;
    const anchor = f ? this.indexOf(f.anchor.t) : n - 1;
    const horizon = f ? f.times.length : 0;
    const end = f ? this.indexOf(f.times[horizon - 1]!) : n - 1;
    const history = Math.max(Math.min(this.idealBars(), horizon * 2.5 + 24), horizon * 1.2, 24);
    this.goal = { from: anchor - history, to: Math.max(end, f ? anchor : n - 1) + Math.max(3, horizon * 0.1) + 1 };
    this.velocity = 0;
    this.invalidate();
  }

  /** About 4 px per candle, between 48 and 160 candles. */
  private idealBars(): number {
    return Math.round(Math.min(160, Math.max(48, (this.w || 800) / 4.2)));
  }

  fitAll(): void {
    this.goal = { from: -2, to: this.total() + 4 };
    this.invalidate();
  }

  screenshot(): HTMLCanvasElement {
    this.draw();
    return this.canvas;
  }

  // ------------------------------------------------------------ geometry

  private indexOf(t: number): number {
    const hit = this.index.get(t);
    if (hit !== undefined) return hit;
    const n = this.data.length;
    if (!n) return 0;
    const last = this.data[n - 1]!.t;
    if (t > last) return n - 1 + (t - last) / this.stepMs;
    const first = this.data[0]!.t;
    if (t < first) return (t - first) / this.stepMs;
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (this.data[mid]!.t <= t) lo = mid;
      else hi = mid;
    }
    return lo;
  }

  private timeAt(i: number): number {
    const n = this.data.length;
    if (!n) return 0;
    if (i >= n) return this.data[n - 1]!.t + (i - (n - 1)) * this.stepMs;
    if (i < 0) return this.data[0]!.t + i * this.stepMs;
    return this.data[i]!.t;
  }

  /** One past the last index that holds anything (real or predicted). */
  private total(): number {
    let end = this.data.length;
    for (const f of [this.forecast, ...this.pinned.map((p) => p.forecast)]) if (f) end = Math.max(end, this.indexOf(f.times[f.times.length - 1]!) + 1);
    return end;
  }

  private clampView(): void {
    const total = this.total();
    let span = Math.min(Math.max(this.to - this.from, 12), total + 80);
    if (this.from < -span * 0.6) this.from = -span * 0.6;
    if (this.from > total - span * 0.3) this.from = total - span * 0.3;
    span = Math.max(12, span);
    this.to = this.from + span;
  }

  private x = (i: number): number => this.plot.left + (i - this.from + 0.5) * this.plot.barW;
  private scale = (p: number): number => (this.view.log ? Math.log(Math.max(p, 1e-12)) : p);
  private y = (p: number): number => this.plot.top + ((this.yHi - this.scale(p)) / (this.yHi - this.yLo || 1)) * (this.plot.bottom - this.plot.top);
  private priceAt = (py: number): number => {
    const s = this.yHi - ((py - this.plot.top) / (this.plot.bottom - this.plot.top)) * (this.yHi - this.yLo);
    return this.view.log ? Math.exp(s) : s;
  };

  private resize(): void {
    const r = this.root.getBoundingClientRect();
    if (!r.width || !r.height) return;
    this.w = r.width;
    this.hgt = r.height;
    this.dpr = Math.min(3, devicePixelRatio || 1);
    for (const c of [this.canvas, this.overlay]) {
      c.width = Math.round(this.w * this.dpr);
      c.height = Math.round(this.hgt * this.dpr);
    }
    this.ySnap = true;
    this.draw();
  }

  private invalidate(): void {
    if (!this.frame) this.frame = requestAnimationFrame(() => this.tick());
  }

  private tick(): void {
    this.frame = 0;
    let moving = false;
    if (this.goal) {
      const df = this.goal.from - this.from;
      const dt = this.goal.to - this.to;
      if (Math.abs(df) < 0.05 && Math.abs(dt) < 0.05) {
        this.from = this.goal.from;
        this.to = this.goal.to;
        this.goal = null;
      } else {
        this.from += df * 0.18;
        this.to += dt * 0.18;
        moving = true;
      }
    } else if (Math.abs(this.velocity) > 0.002 && !this.drag) {
      this.from -= this.velocity * 16;
      this.to -= this.velocity * 16;
      this.velocity *= 0.92;
      this.clampView();
      moving = true;
    }
    moving = this.draw() || moving;
    if (performance.now() - this.revealAt < REVEAL_MS) moving = true;
    if (moving) this.invalidate();
  }

  // ------------------------------------------------------------ drawing

  /** Price range of everything visible, in scale space. */
  private range(i0: number, i1: number): [number, number] {
    let lo = Infinity;
    let hi = -Infinity;
    const take = (v: number) => {
      if (!Number.isFinite(v) || (this.view.log && v <= 0)) return;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    };
    for (let i = Math.max(0, i0); i <= Math.min(this.data.length - 1, i1); i++) {
      take(this.data[i]!.l);
      take(this.data[i]!.h);
    }
    const v = this.view;
    const fan = (f: Forecast, own: boolean) => {
      f.times.forEach((t, k) => {
        const i = this.indexOf(t);
        if (i < i0 || i > i1) return;
        take(f.q[f.median]![k]!);
        if (!own) return;
        if (v.bands) {
          take(f.q[f.outer.lo]![k]!);
          take(f.q[f.outer.hi]![k]!);
        }
        if (v.meanCandles && f.mean) {
          take(f.mean[k]!.l);
          take(f.mean[k]!.h);
        }
        if (v.paths && f.paths) for (const p of f.paths.slice(0, MAX_PATHS)) take(p[k * 6 + 3]!);
      });
    };
    if (this.forecast) fan(this.forecast, true);
    for (const p of this.pinned) fan(p.forecast, false);
    if (lo === Infinity) return [0, 1];
    const a = this.scale(lo);
    const b = this.scale(hi);
    const pad = (b - a || Math.abs(b) * 0.01 || 1) * 0.08;
    return [a - pad, b + pad];
  }

  private tickStep = 1;

  private fmtTick(p: number): string {
    const d = this.tickStep >= 1 ? 0 : Math.min(8, Math.ceil(-Math.log10(this.tickStep) - 1e-9));
    return p.toLocaleString(locale, { minimumFractionDigits: d, maximumFractionDigits: d });
  }

  private priceTicks(): number[] {
    const { top, bottom } = this.plot;
    const count = Math.max(2, Math.floor((bottom - top) / 58));
    const lo = this.priceAt(bottom);
    const hi = this.priceAt(top);
    const raw = (hi - lo) / count;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
    this.tickStep = step;
    const ticks: number[] = [];
    let prevY = Infinity;
    for (let p = Math.ceil(lo / step) * step; p <= hi; p += step) {
      const py = this.y(p);
      if (py < top + 8 || py > bottom - 4 || Math.abs(prevY - py) < 26) continue;
      ticks.push(p);
      prevY = py;
    }
    return ticks;
  }

  private timeLabel(t: number, step: number): { text: string; strong: boolean } {
    const d = new Date(t + this.shiftMs);
    const midnight = d.getUTCHours() === 0 && d.getUTCMinutes() === 0;
    if (step >= 86400e3 || midnight) {
      const first = d.getUTCDate() === 1 && d.getUTCMonth() === 0;
      return { text: d.toLocaleDateString(locale, { timeZone: 'UTC', day: 'numeric', month: 'short', year: first || step >= 91 * 86400e3 ? 'numeric' : undefined }), strong: true };
    }
    return { text: d.toLocaleTimeString(locale, { timeZone: 'UTC', hour: '2-digit', minute: '2-digit' }), strong: false };
  }

  private tag(g: CanvasRenderingContext2D, py: number, text: string, bg: string, ink = '#ffffff'): void {
    const { right, top, bottom } = this.plot;
    if (py < top - 4 || py > bottom + 4) return;
    g.font = `600 11px ${FONT}`;
    const tw = g.measureText(text).width;
    g.fillStyle = bg;
    g.beginPath();
    g.roundRect(right + 5, py - 10, tw + 12, 20, 6);
    g.fill();
    g.fillStyle = ink;
    g.textAlign = 'left';
    g.textBaseline = 'middle';
    g.fillText(text, right + 11, py + 0.5);
  }

  /** Paint everything. Returns true while the vertical scale is still easing. */
  private draw(): boolean {
    const g = this.g;
    const c = this.colors;
    const v = this.view;
    const f = this.forecast;
    const n = this.data.length;
    if (!this.w) return false;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.fillStyle = c.surface;
    g.fillRect(0, 0, this.w, this.hgt);
    if (!n) return false;

    // Layout: the axis width follows the widest price label.
    g.font = `500 11px ${FONT}`;
    const axisW = Math.max(56, g.measureText(fmtPrice(this.data[n - 1]!.h * 1.1)).width + 26);
    const priceBottom = this.hgt - AXIS_H;
    const volH = v.volume ? Math.round((priceBottom - PAD_TOP) * VOLUME_SHARE) : 0;
    const plot: Plot = { left: 0, right: this.w - axisW, top: PAD_TOP, bottom: priceBottom - volH - (volH ? 10 : 0), volTop: priceBottom - volH, volBottom: priceBottom, barW: 1 };
    plot.barW = (plot.right - plot.left) / (this.to - this.from);
    this.plot = plot;
    const i0 = Math.floor(this.from) - 1;
    const i1 = Math.ceil(this.to) + 1;

    // Vertical scale eases towards what is visible.
    const [lo, hi] = this.range(i0, i1);
    let easing = false;
    if (this.ySnap) {
      this.yLo = lo;
      this.yHi = hi;
      this.ySnap = false;
    } else {
      const span = hi - lo || 1;
      if (Math.abs(lo - this.yLo) > span * 0.002 || Math.abs(hi - this.yHi) > span * 0.002) {
        this.yLo += (lo - this.yLo) * 0.22;
        this.yHi += (hi - this.yHi) * 0.22;
        easing = true;
      } else {
        this.yLo = lo;
        this.yHi = hi;
      }
    }

    const reveal = f ? easeOut(Math.min(1, (performance.now() - this.revealAt) / REVEAL_MS)) : 1;
    const anchorX = f ? this.x(this.indexOf(f.anchor.t)) : 0;
    const endX = f ? this.x(this.indexOf(f.times[f.times.length - 1]!)) + plot.barW / 2 : 0;
    const revealX = anchorX + (endX - anchorX) * reveal;

    // Horizontal grid and price labels.
    g.lineWidth = 1;
    g.textBaseline = 'middle';
    g.textAlign = 'left';
    // Price tags on the axis win over tick labels that would sit underneath them.
    const tagYs = [this.y(this.data[n - 1]!.c)];
    if (f && v.median && reveal > 0.98) tagYs.push(this.y(f.q[f.median]![f.times.length - 1]!));
    for (const p of this.priceTicks()) {
      const py = Math.round(this.y(p)) + 0.5;
      const hidden = tagYs.some((ty) => Math.abs(ty - py) < 16);
      g.strokeStyle = c.grid;
      g.beginPath();
      g.moveTo(plot.left, py);
      g.lineTo(plot.right, py);
      g.stroke();
      if (hidden) continue;
      g.fillStyle = c.muted;
      g.font = `500 11px ${FONT}`;
      g.fillText(this.fmtTick(p), plot.right + 11, py);
    }

    g.save();
    g.beginPath();
    g.rect(plot.left, 0, plot.right - plot.left, priceBottom);
    g.clip();

    // Forecast region: a soft tint from the last context candle onwards.
    if (f) {
      const grad = g.createLinearGradient(anchorX, 0, revealX, 0);
      grad.addColorStop(0, alpha(c.forecast, 0.07));
      grad.addColorStop(1, alpha(c.forecast, 0.015));
      g.fillStyle = grad;
      g.fillRect(anchorX, 0, revealX - anchorX, priceBottom);
    }

    // Volume strip.
    const bodyW = Math.max(1, Math.min(plot.barW * 0.72, 22));
    if (volH) {
      let vmax = 0;
      for (let i = Math.max(0, i0); i <= Math.min(n - 1, i1); i++) vmax = Math.max(vmax, this.data[i]!.v);
      const meanVol = f?.mean && v.meanCandles ? f.mean : [];
      for (const k of meanVol) vmax = Math.max(vmax, k.v);
      const bar = (i: number, vol: number, color: string) => {
        const bh = vmax ? (vol / vmax) * volH : 0;
        g.fillStyle = color;
        g.beginPath();
        g.roundRect(this.x(i) - bodyW / 2, plot.volBottom - bh, bodyW, bh, Math.min(2, bodyW / 3));
        g.fill();
      };
      for (let i = Math.max(0, i0); i <= Math.min(n - 1, i1); i++) {
        const k = this.data[i]!;
        bar(i, k.v, alpha(k.c >= k.o ? c.up : c.down, 0.38));
      }
      for (const k of meanVol) {
        const i = this.indexOf(k.t);
        if (i >= n && this.x(i) <= revealX) bar(i, k.v, alpha(c.forecast, 0.35));
      }
    }

    g.beginPath();
    g.rect(plot.left, 0, plot.right - plot.left, plot.bottom + 6);
    g.clip();

    const trace = (fc: Forecast, value: (k: number) => number, reverse = false, start = true) => {
      const pts: [number, number][] = [[this.x(this.indexOf(fc.anchor.t)), this.y(fc.anchor.c)]];
      fc.times.forEach((t, k) => pts.push([this.x(this.indexOf(t)), this.y(value(k))]));
      if (reverse) pts.reverse();
      pts.forEach(([px, py], k) => (k === 0 && start ? g.moveTo(px, py) : g.lineTo(px, py)));
    };

    if (f) {
      g.save();
      g.beginPath();
      g.rect(plot.left, 0, revealX - plot.left, priceBottom);
      g.clip();
      g.lineJoin = 'round';
      if (v.bands) {
        for (const [band, a] of [[f.outer, 0.13], [f.inner, 0.2]] as const) {
          g.beginPath();
          trace(f, (k) => f.q[band.hi]![k]!);
          trace(f, (k) => f.q[band.lo]![k]!, true, false);
          g.closePath();
          g.fillStyle = alpha(c.forecast, a);
          g.fill();
        }
        g.lineWidth = 1;
        g.strokeStyle = alpha(c.forecast, 0.45);
        for (const idx of [f.outer.lo, f.outer.hi]) {
          g.beginPath();
          trace(f, (k) => f.q[idx]![k]!);
          g.stroke();
        }
      }
      if (v.paths && f.paths) {
        const shown = f.paths.slice(0, MAX_PATHS);
        g.lineWidth = 1;
        g.strokeStyle = alpha(c.forecast, Math.max(0.1, Math.min(0.4, 3.5 / shown.length)));
        for (const p of shown) {
          g.beginPath();
          trace(f, (k) => p[k * 6 + 3]!);
          g.stroke();
        }
      }
      g.restore();
    }

    // Real candles.
    const candle = (i: number, k: Candle, hollow: boolean) => {
      const px = this.x(i);
      const up = k.c >= k.o;
      const color = up ? c.up : c.down;
      const yo = this.y(k.o);
      const yc = this.y(k.c);
      const top = Math.min(yo, yc);
      const bh = Math.max(1, Math.abs(yc - yo));
      g.strokeStyle = color;
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(Math.round(px) + 0.5, this.y(k.h));
      g.lineTo(Math.round(px) + 0.5, hollow ? top : this.y(k.l));
      if (hollow) {
        g.moveTo(Math.round(px) + 0.5, top + bh);
        g.lineTo(Math.round(px) + 0.5, this.y(k.l));
      }
      g.stroke();
      g.beginPath();
      g.roundRect(px - bodyW / 2, top, bodyW, bh, Math.min(2, bodyW / 4, bh / 2));
      if (hollow) {
        g.fillStyle = alpha(color, 0.1);
        g.fill();
        if (bodyW >= 3) g.stroke();
      } else {
        g.fillStyle = color;
        g.fill();
      }
    };
    for (let i = Math.max(0, i0); i <= Math.min(n - 1, i1); i++) candle(i, this.data[i]!, false);

    for (const p of this.pinned) {
      g.beginPath();
      trace(p.forecast, (k) => p.forecast.q[p.forecast.median]![k]!);
      g.setLineDash([6, 5]);
      g.lineWidth = 2;
      g.strokeStyle = p.color;
      g.stroke();
      g.setLineDash([]);
    }

    if (f) {
      g.save();
      g.beginPath();
      g.rect(plot.left, 0, revealX - plot.left, priceBottom);
      g.clip();
      if (v.meanCandles && f.mean) for (const k of f.mean) candle(this.indexOf(k.t), k, true);
      if (v.median) {
        g.beginPath();
        trace(f, (k) => f.q[f.median]![k]!);
        g.lineWidth = 2;
        g.lineJoin = 'round';
        g.lineCap = 'round';
        g.strokeStyle = c.forecast;
        g.stroke();
      }
      g.restore();
      if (v.median && reveal > 0.98) {
        const last = f.times.length - 1;
        const px = this.x(this.indexOf(f.times[last]!));
        const py = this.y(f.q[f.median]![last]!);
        g.fillStyle = c.surface;
        g.beginPath();
        g.arc(px, py, 5.5, 0, Math.PI * 2);
        g.fill();
        g.fillStyle = c.forecast;
        g.beginPath();
        g.arc(px, py, 3.5, 0, Math.PI * 2);
        g.fill();
      }
      // Divider between what happened and what is predicted.
      g.setLineDash([3, 4]);
      g.strokeStyle = c.muted;
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(Math.round(anchorX) + 0.5, 0);
      g.lineTo(Math.round(anchorX) + 0.5, priceBottom);
      g.stroke();
      g.setLineDash([]);
      g.font = `500 11px ${FONT}`;
      g.fillStyle = alpha(c.forecast, 0.95);
      g.textAlign = 'left';
      g.textBaseline = 'top';
      const label = `${t('Predicción')} · ${f.label}`;
      const room = Math.min(endX, plot.right) - anchorX - 20;
      if (room > g.measureText(label).width) g.fillText(label, anchorX + 10, 10);
      else if (room > g.measureText('Predicción').width) g.fillText('Predicción', anchorX + 10, 10);
    }

    // Last traded price.
    const lastC = this.data[n - 1]!;
    const lastY = Math.round(this.y(lastC.c)) + 0.5;
    const lastColor = lastC.c >= lastC.o ? c.up : c.down;
    g.setLineDash([2, 4]);
    g.strokeStyle = alpha(lastColor, 0.7);
    g.beginPath();
    g.moveTo(plot.left, lastY);
    g.lineTo(plot.right, lastY);
    g.stroke();
    g.setLineDash([]);
    g.restore();

    // Time axis.
    g.strokeStyle = c.border;
    g.beginPath();
    g.moveTo(0, priceBottom + 0.5);
    g.lineTo(this.w, priceBottom + 0.5);
    g.stroke();
    const need = (76 / plot.barW) * this.stepMs;
    const tstep = TIME_STEPS.find((s) => s >= need && s >= this.stepMs) ?? TIME_STEPS[TIME_STEPS.length - 1]!;
    g.textBaseline = 'middle';
    g.textAlign = 'center';
    const label = (i: number, lastX: number): number => {
      const px = this.x(i);
      if (px < 24 || px > plot.right - 20 || px - lastX < 56) return lastX;
      const lab = this.timeLabel(this.timeAt(i), tstep);
      g.font = `${lab.strong ? 600 : 500} 11px ${FONT}`;
      g.fillStyle = lab.strong ? c.text : c.muted;
      g.fillText(lab.text, px, priceBottom + AXIS_H / 2 + 1);
      return px;
    };
    let lastLabelX = -Infinity;
    // Intraday bars: label round clock times. Daily and longer (or irregular data): every k-th bar.
    if (this.stepMs < 86400e3) {
      for (let i = Math.max(i0, -500); i <= i1; i++) {
        const t = this.timeAt(i) + this.shiftMs;
        if (((t % tstep) + tstep) % tstep === 0) lastLabelX = label(i, lastLabelX);
      }
    }
    if (lastLabelX === -Infinity) {
      const every = Math.max(1, Math.ceil(76 / plot.barW));
      for (let i = Math.ceil(Math.max(i0, -500) / every) * every; i <= i1; i += every) lastLabelX = label(i, lastLabelX);
    }

    // Axis tags: last price and where the median ends.
    this.tag(g, lastY, fmtPrice(lastC.c), lastColor);
    if (f && v.median && reveal > 0.98 && endX > plot.left && endX - plot.barW < plot.right) {
      this.tag(g, this.y(f.q[f.median]![f.times.length - 1]!), fmtPrice(f.q[f.median]![f.times.length - 1]!), c.forecast);
    }

    this.drawHover();
    return easing;
  }

  // ------------------------------------------------------------ crosshair and tooltip

  private drawHover(): void {
    const og = this.og;
    og.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    og.clearRect(0, 0, this.w, this.hgt);
    const hv = this.hover;
    const { left, right, top, bottom, volBottom, barW } = this.plot;
    const n = this.data.length;
    if (!hv || !n || this.drag || hv.x < left || hv.x > right || hv.y > volBottom) {
      this.tip.classList.remove('show');
      return;
    }
    const c = this.colors;
    const i = Math.floor(this.from + (hv.x - left) / barW);
    const px = Math.round(this.x(i)) + 0.5;
    const t = this.timeAt(i);
    const f = this.forecast;
    const step = f ? f.times.indexOf(t) : -1;
    const candle = this.data[i];

    og.strokeStyle = alpha(c.muted, 0.7);
    og.lineWidth = 1;
    og.setLineDash([3, 3]);
    og.beginPath();
    og.moveTo(px, 0);
    og.lineTo(px, volBottom);
    if (hv.y >= top && hv.y <= bottom) {
      og.moveTo(left, Math.round(hv.y) + 0.5);
      og.lineTo(right, Math.round(hv.y) + 0.5);
    }
    og.stroke();
    og.setLineDash([]);
    if (hv.y >= top && hv.y <= bottom) this.tag(og, hv.y, fmtPrice(this.priceAt(hv.y)), c.text, c.surface);

    // Markers on the fan at this step.
    if (f && step >= 0) {
      const dot = (p: number, r: number, color: string) => {
        og.fillStyle = c.surface;
        og.beginPath();
        og.arc(px, this.y(p), r + 2, 0, Math.PI * 2);
        og.fill();
        og.fillStyle = color;
        og.beginPath();
        og.arc(px, this.y(p), r, 0, Math.PI * 2);
        og.fill();
      };
      if (this.view.bands) {
        dot(f.q[f.outer.lo]![step]!, 2.5, alpha(c.forecast, 0.7));
        dot(f.q[f.outer.hi]![step]!, 2.5, alpha(c.forecast, 0.7));
      }
      if (this.view.median) dot(f.q[f.median]![step]!, 4, c.forecast);
    }

    // Time label on the axis.
    const when = new Date(t + this.shiftMs).toLocaleString(locale, { timeZone: 'UTC', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
    og.font = `600 11px ${FONT}`;
    const tw = og.measureText(when).width + 14;
    const tx = Math.max(2, Math.min(right - tw - 2, px - tw / 2));
    og.fillStyle = c.text;
    og.beginPath();
    og.roundRect(tx, volBottom + 5, tw, 20, 6);
    og.fill();
    og.fillStyle = c.surface;
    og.textAlign = 'center';
    og.textBaseline = 'middle';
    og.fillText(when, tx + tw / 2, volBottom + 15.5);

    this.fillTip(when, candle, f, step);
    const tipW = this.tip.offsetWidth;
    const tipH = this.tip.offsetHeight;
    const side = px + 16 + tipW > right ? px - 16 - tipW : px + 16;
    this.tip.style.transform = `translate(${Math.max(6, side)}px, ${Math.max(6, Math.min(bottom - tipH, hv.y - tipH / 2))}px)`;
    this.tip.classList.add('show');
  }

  private tipKey = '';
  private fillTip(when: string, candle: Candle | undefined, f: Forecast | null, step: number): void {
    const key = `${when}|${candle?.c}|${candle?.v}|${f?.id}|${step}`;
    if (key === this.tipKey) return;
    this.tipKey = key;
    const row = (label: string, value: string, cls = '') => h('div', { class: `tip-row ${cls}` }, h('span', {}, label), h('b', {}, value));
    const parts: HTMLElement[] = [h('div', { class: 'tip-head' }, when)];
    if (candle) {
      const ch = (candle.c / candle.o - 1) * 100;
      parts.push(
        h(
          'div',
          { class: 'tip-grid' },
          row(t('Apertura'), fmtPrice(candle.o)),
          row(t('Máximo'), fmtPrice(candle.h)),
          row(t('Mínimo'), fmtPrice(candle.l)),
          row(t('Cierre'), fmtPrice(candle.c)),
          row(t('Variación'), fmtPct(ch), ch >= 0 ? 'up' : 'down'),
          row(t('Volumen'), fmtCompact(candle.v)),
        ),
      );
    }
    if (f && step >= 0) {
      const pc = (lv: number) => `p${Math.round(lv * 100)}`;
      const med = f.q[f.median]![step]!;
      parts.push(
        h('div', { class: 'tip-sub' }, h('i', {}), t('Predicción · vela {a} de {b}', { a: step + 1, b: f.times.length })),
        h(
          'div',
          { class: 'tip-grid' },
          row(t('Mediana'), fmtPrice(med)),
          row(t('Desde el inicio'), fmtPct((med / f.anchor.c - 1) * 100), med >= f.anchor.c ? 'up' : 'down'),
          row(`${pc(f.levels[f.inner.lo]!)}–${pc(f.levels[f.inner.hi]!)}`, `${fmtPrice(f.q[f.inner.lo]![step]!)} – ${fmtPrice(f.q[f.inner.hi]![step]!)}`, 'wide'),
          row(`${pc(f.levels[f.outer.lo]!)}–${pc(f.levels[f.outer.hi]!)}`, `${fmtPrice(f.q[f.outer.lo]![step]!)} – ${fmtPrice(f.q[f.outer.hi]![step]!)}`, 'wide'),
          candle ? row(t('Error de la mediana'), fmtPct((med / candle.c - 1) * 100), 'wide') : null,
        ),
      );
    }
    this.tip.replaceChildren(...parts);
  }

  // ------------------------------------------------------------ interaction

  private zoom(factor: number, atX: number): void {
    const pivot = this.from + (atX - this.plot.left) / this.plot.barW;
    const span = Math.min(Math.max((this.to - this.from) * factor, 12), this.total() + 80);
    const share = (pivot - this.from) / (this.to - this.from);
    this.from = pivot - span * share;
    this.to = this.from + span;
    this.goal = null;
    this.clampView();
    this.invalidate();
  }

  private pan(bars: number): void {
    this.from -= bars;
    this.to -= bars;
    this.goal = null;
    this.clampView();
    this.invalidate();
  }

  private wire(): void {
    const el = this.canvas;
    const pos = (e: PointerEvent | WheelEvent) => {
      const r = el.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    el.addEventListener(
      'wheel',
      (e) => {
        if (!this.wheelZoom && !e.ctrlKey) return;
        e.preventDefault();
        if (Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.shiftKey) this.pan(-(e.deltaX || e.deltaY) / this.plot.barW);
        else this.zoom(Math.exp(e.deltaY * (e.ctrlKey ? 0.01 : 0.0016)), pos(e).x);
      },
      { passive: false },
    );
    el.addEventListener('pointerdown', (e) => {
      try {
        el.setPointerCapture(e.pointerId);
      } catch {
        /* pointer already gone */
      }
      this.pointers.set(e.pointerId, pos(e));
      const pts = [...this.pointers.values()];
      this.drag = { x: pts[0]!.x, t: performance.now(), pinch: pts.length > 1 ? Math.abs(pts[0]!.x - pts[1]!.x) : 0 };
      this.velocity = 0;
      this.goal = null;
      el.classList.add('grabbing');
    });
    el.addEventListener('pointermove', (e) => {
      const p = pos(e);
      if (!this.pointers.has(e.pointerId)) {
        this.hover = p;
        this.drawHover();
        return;
      }
      this.pointers.set(e.pointerId, p);
      const pts = [...this.pointers.values()];
      const d = this.drag!;
      if (pts.length > 1) {
        const dist = Math.abs(pts[0]!.x - pts[1]!.x);
        if (d.pinch > 10 && dist > 10) this.zoom(d.pinch / dist, (pts[0]!.x + pts[1]!.x) / 2);
        d.pinch = dist;
        return;
      }
      const now = performance.now();
      const dx = p.x - d.x;
      this.velocity = dx / this.plot.barW / Math.max(1, now - d.t);
      d.x = p.x;
      d.t = now;
      this.pan(dx / this.plot.barW);
    });
    const release = (e: PointerEvent) => {
      this.pointers.delete(e.pointerId);
      if (this.pointers.size) {
        const rest = [...this.pointers.values()][0]!;
        this.drag = { x: rest.x, t: performance.now(), pinch: 0 };
        this.velocity = 0;
        return;
      }
      if (this.drag && performance.now() - this.drag.t > 80) this.velocity = 0;
      this.drag = null;
      el.classList.remove('grabbing');
      if (e.pointerType !== 'mouse') this.hover = null;
      this.invalidate();
    };
    el.addEventListener('pointerup', release);
    el.addEventListener('pointercancel', release);
    el.addEventListener('pointerleave', () => {
      if (this.pointers.size) return;
      this.hover = null;
      this.drawHover();
    });
    el.addEventListener('dblclick', () => this.focus());
    el.addEventListener('keydown', (e) => {
      const span = this.to - this.from;
      if (e.key === 'ArrowLeft') this.pan(span * 0.12);
      else if (e.key === 'ArrowRight') this.pan(-span * 0.12);
      else if (e.key === '+' || e.key === '=') this.zoom(0.8, (this.plot.left + this.plot.right) / 2);
      else if (e.key === '-') this.zoom(1.25, (this.plot.left + this.plot.right) / 2);
      else if (e.key === 'Home') this.focus();
      else return;
      e.preventDefault();
    });
  }
}
