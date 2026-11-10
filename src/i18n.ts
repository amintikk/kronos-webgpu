// Two languages: Spanish (source text, served at /) and English (served under /en/).
// t() takes the Spanish text itself as the key, so the code stays readable and a missing
// translation falls back to Spanish. tests/i18n.test.ts fails if any t('…') has no English entry.
import { EN } from './i18n.en.ts';

export type Lang = 'es' | 'en';

function detect(): Lang {
  if (typeof document !== 'undefined') return document.documentElement.lang.startsWith('en') ? 'en' : 'es';
  return 'es';
}

export let lang: Lang = detect();
export let locale = lang === 'en' ? 'en-US' : 'es-ES';

/** Workers have no document: the page tells them its language. */
export function setLang(l: Lang): void {
  lang = l;
  locale = l === 'en' ? 'en-US' : 'es-ES';
}

export function t(es: string, vars?: Record<string, string | number>): string {
  let s = lang === 'en' ? (EN[es] ?? es) : es;
  if (vars) s = s.replace(/\{(\w+)\}/g, (_: string, k: string) => String(vars[k] ?? ''));
  return s;
}

/** Path of the same page in the current language. */
export const localPath = (p: string): string => (lang === 'en' ? `/en${p}` : p);
