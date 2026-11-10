import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseCsv } from '../src/data/csv.ts';

test('parseCsv reads headers in any order, infers the bar length and derives amount', () => {
  const rows = Array.from({ length: 40 }, (_, i) => `2026-01-01 ${String(Math.floor(i / 4)).padStart(2, '0')}:${String((i % 4) * 15).padStart(2, '0')}:00,${100 + i},${99 + i},${101 + i},${100.5 + i},${10 + i}`);
  const { candles, stepMs } = parseCsv(['timestamps,open,low,high,close,volume', ...rows.reverse()].join('\n'));
  assert.equal(candles.length, 40);
  assert.equal(stepMs, 15 * 60e3);
  assert.equal(candles[0]!.t, Date.UTC(2026, 0, 1, 0, 0));
  assert.deepEqual([candles[0]!.o, candles[0]!.h, candles[0]!.l, candles[0]!.c, candles[0]!.v], [100, 101, 99, 100.5, 10]);
  assert.equal(candles[0]!.a, 10 * ((100 + 101 + 99 + 100.5) / 4));
});

test('parseCsv accepts epoch seconds and semicolons, and names the missing column', () => {
  const rows = Array.from({ length: 33 }, (_, i) => `${1700000000 + i * 3600};1;2;0.5;1.5`);
  const { candles, stepMs } = parseCsv(['time;open;high;low;close', ...rows].join('\n'));
  assert.equal(stepMs, 3600e3);
  assert.equal(candles[0]!.t, 1700000000000);
  assert.equal(candles[0]!.v, 0);
  assert.throws(() => parseCsv(['time,open,high,low', ...rows].join('\n')), /close/);
});
