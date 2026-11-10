import '@fontsource-variable/inter';
import './site.css';
import { ForecastChart } from '../chart.ts';
import type { Candle } from '../engine/types.ts';
import { fromKronos, summarize, type Forecast } from '../forecast.ts';
import { fmtPct, fmtPrice } from '../ui/dom.ts';
import { hydrateIcons, icon } from '../ui/icons.ts';
import { backtestFigure, pathsFigure, tokensFigure, workFigure, type DemoData } from './figures.ts';

const KEY = 'kronos-settings';

function readTheme(): string {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? '{}').theme ?? 'auto';
  } catch {
    return 'auto';
  }
}

function isDark(theme: string): boolean {
  return theme === 'dark' || (theme !== 'light' && matchMedia('(prefers-color-scheme: dark)').matches);
}

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const colors = () => ({
  surface: css('--bg'), // the chart sits on the page, not on a card
  text: css('--text'),
  grid: css('--grid'),
  border: css('--border'),
  up: css('--up'),
  down: css('--down'),
  forecast: css('--accent'),
  muted: css('--muted'),
});

let chart: ForecastChart | null = null;

function applyTheme(theme: string): void {
  if (theme === 'dark' || theme === 'light') document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  document.getElementById('btn-theme')?.replaceChildren(icon(isDark(theme) ? 'sun' : 'moon', 18));
  chart?.setColors(colors());
}

function wireTheme(): void {
  let theme = readTheme();
  applyTheme(theme);
  document.getElementById('btn-theme')?.addEventListener('click', () => {
    theme = isDark(theme) ? 'light' : 'dark';
    try {
      // Shared with the app, so the choice carries over.
      const s = JSON.parse(localStorage.getItem(KEY) ?? '{}');
      localStorage.setItem(KEY, JSON.stringify({ ...s, theme }));
    } catch {
      /* storage blocked: still switch for this page */
    }
    applyTheme(theme);
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => applyTheme(theme));
}

function wireNav(): void {
  const toggle = document.getElementById('nav-toggle');
  const nav = document.getElementById('site-nav');
  if (!toggle || !nav) return;
  const set = (open: boolean) => {
    document.body.classList.toggle('nav-open', open);
    toggle.setAttribute('aria-expanded', String(open));
  };
  toggle.addEventListener('click', () => set(!document.body.classList.contains('nav-open')));
  nav.addEventListener('click', (e) => {
    if ((e.target as HTMLElement).closest('a')) set(false);
  });
}

async function demo(): Promise<void> {
  const el = document.getElementById('demo-chart');
  if (!el) return;
  const d = (await (await fetch('/demo.json')).json()) as DemoData;
  figures(d);
  const candles: Candle[] = d.candles.map(([t, o, h, l, c, v]) => ({ t: t!, o: o!, h: h!, l: l!, c: c!, v: v!, a: 0 }));
  const anchor = candles[candles.length - 1]!;
  const stats = fromKronos({ paths: d.paths.map((p) => Float32Array.from(p)) }, d.times, anchor);
  const f: Forecast = { id: 1, model: 'kronos:mini', family: 'kronos', label: d.model, times: d.times, anchor, ms: 0, settings: {}, ...stats };
  chart = new ForecastChart(el, colors(), { bands: true, paths: true, meanCandles: false, median: true, volume: false, log: false, localTime: false }, { wheel: false });
  chart.setCandles(candles, true);
  // Start the reveal when the chart is actually on screen.
  const show = () => {
    chart!.setForecast(f, []);
    chart!.focus();
  };
  new IntersectionObserver((entries, io) => {
    if (entries.some((e) => e.isIntersecting)) {
      io.disconnect();
      show();
    }
  }).observe(el);
  const s = summarize(f);
  const up = Math.round(s.pUp * 100);
  document.getElementById('demo-foot')?.replaceChildren(
    ...[
      ['Mediana a 48 h', fmtPrice(s.endMedian)],
      ['Variación prevista', fmtPct(s.changePct)],
      ['80 % de los escenarios', `${fmtPct((s.endLo / s.last - 1) * 100, 1)} a ${fmtPct((s.endHi / s.last - 1) * 100, 1)}`],
      ['Escenarios al alza', `${up} %`],
    ].map(([k, v]) => {
      const div = document.createElement('div');
      const a = document.createElement('span');
      const b = document.createElement('b');
      a.textContent = k!;
      b.textContent = v!;
      div.append(a, b);
      return div;
    }),
  );
}

function figures(d: DemoData): void {
  const run = (id: string, f: (host: HTMLElement) => void) => {
    const host = document.getElementById(id);
    if (!host) return;
    try {
      f(host);
    } catch {
      host.closest('figure')?.remove(); // a broken figure is worse than none
    }
  };
  run('fig-tokens', (h) => tokensFigure(h, d));
  run('fig-paths', (h) => pathsFigure(h, document.getElementById('fig-paths-controls')!, d));
  run('fig-work', (h) => workFigure(h, d));
  run('fig-backtest', (h) => backtestFigure(h, document.getElementById('fig-backtest-stats')!, d));
}

hydrateIcons();
wireTheme();
wireNav();
void demo().catch(() => document.querySelector('.figure')?.remove());
