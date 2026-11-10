import { locale } from '../i18n.ts';

type Child = Node | string | number | null | undefined | false;

/** Tiny DOM builder. Text always goes through text nodes, never innerHTML. */
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, unknown> = {}, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v as EventListener);
    else if (k === 'class') el.className = String(v);
    else if (k in el && k !== 'list') (el as unknown as Record<string, unknown>)[k] = v;
    else el.setAttribute(k, String(v));
  }
  for (const c of children) if (c !== null && c !== undefined && c !== false) el.append(c instanceof Node ? c : String(c));
  return el;
}

export const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;

export function fmtPrice(p: number): string {
  if (!Number.isFinite(p)) return '—';
  const a = Math.abs(p);
  const d = a >= 1000 ? 2 : a >= 10 ? 3 : a >= 1 ? 4 : a >= 0.01 ? 6 : 8;
  return p.toLocaleString(locale, { minimumFractionDigits: Math.min(d, 2), maximumFractionDigits: d });
}

export function fmtPct(p: number, digits = 2): string {
  if (!Number.isFinite(p)) return '—';
  if (Math.abs(p) < 0.5 * 10 ** -digits) p = 0; // no "-0,00 %"
  return `${p > 0 ? '+' : ''}${p.toLocaleString(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits })} %`;
}

export function fmtNum(n: number, digits = 0): string {
  return Number.isFinite(n) ? n.toLocaleString(locale, { maximumFractionDigits: digits }) : '—';
}

export function fmtCompact(n: number): string {
  return Number.isFinite(n) ? n.toLocaleString(locale, { notation: 'compact', maximumFractionDigits: 2 }) : '—';
}

export function fmtBytes(b: number): string {
  return b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(b >= 1e8 ? 0 : 1)} MB`;
}

export function fmtTime(ms: number, local: boolean, withYear = false): string {
  return new Date(ms).toLocaleString(locale, {
    timeZone: local ? undefined : 'UTC',
    year: withYear ? 'numeric' : undefined,
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function fmtDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const s = ms / 1000;
  return s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

export function download(name: string, data: Blob | string, type = 'text/plain'): void {
  const blob = typeof data === 'string' ? new Blob([data], { type }) : data;
  const url = URL.createObjectURL(blob);
  const a = h('a', { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

let toastTimer = 0;
export function toast(text: string): void {
  const el = $('toast');
  el.textContent = text;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => el.classList.remove('show'), 2600);
}
