// Every text passed to t() must have an English translation, and placeholders must match.
import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';
import { EN } from '../src/i18n.en.ts';

/** Keys built at run time (forecast settings shown under the summary, sentence variants). */
const DYNAMIC = ['contexto', 'horizonte', 'trayectorias', 'temperatura', 'top_p', 'top_k', 'semilla', 'ventana', 'Sube {p} %', 'Sube menos del {p} %', 'Sube más del {p} %'];

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? files(p) : p.endsWith('.ts') && !p.endsWith('i18n.en.ts') ? [p] : [];
  });
}

export function keys(): Set<string> {
  const out = new Set<string>(DYNAMIC);
  const lit = /'((?:[^'\\]|\\.)*)'/g;
  for (const f of files(new URL('../src', import.meta.url).pathname)) {
    const src = readFileSync(f, 'utf8');
    // t('…') and t(cond ? '…' : '…'): take the string literals of the first argument.
    for (const m of src.matchAll(/\bt\(((?:'(?:[^'\\]|\\.)*'|[^,()'])+)[,)]/g)) {
      for (const s of m[1]!.matchAll(lit)) out.add(s[1]!.replace(/\\'/g, "'"));
    }
  }
  return out;
}

test('every t() text has an English translation with the same placeholders', () => {
  const missing: string[] = [];
  for (const k of keys()) {
    if (!/\p{L}/u.test(k.replace(/\{\w+\}/g, ''))) continue; // operators and placeholders only
    const en = EN[k];
    if (en === undefined) {
      missing.push(k);
      continue;
    }
    const ph = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort().join(',');
    assert.equal(ph(en), ph(k), `placeholders differ for: ${k}`);
  }
  assert.deepEqual(missing, [], `missing English for ${missing.length} texts`);
});
