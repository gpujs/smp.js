// The promise smp.js makes: the annotated source runs as plain JavaScript, and
// the compiled wasm produces the SAME numbers. If that ever stops being true the
// single-threaded fallback is a lie, so it is the first thing tested.

import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { existsSync, rmSync } from "node:fs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const OUT = join(ROOT, "test", ".build");

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeInputs(T) {
  const rng = mulberry32(12345);
  const prices = new Float64Array(T);
  let p = 100;
  for (let i = 0; i < T; i++) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    p *= Math.exp(-0.00005 + 0.01 * g);
    prices[i] = p;
  }
  const fastV = [5, 10, 15, 20], slowV = [40, 60, 80], slV = [0.02, 0.05], tpV = [0.04, 0.1];
  const rows = [];
  for (const f of fastV) for (const s of slowV) for (const sl of slV) for (const tp of tpV) {
    if (f < s) rows.push(f, s, sl, tp);
  }
  return { prices, params: Float64Array.from(rows), nRuns: rows.length / 4 };
}

test("compiled wasm matches the source run as plain JavaScript", async (t) => {
  rmSync(OUT, { recursive: true, force: true });
  execFileSync(process.execPath, [
    join(ROOT, "bin", "smp.js"), "build", join(ROOT, "examples", "backtest.js"),
    "--out", OUT,
  ], { encoding: "utf8" });

  assert.ok(existsSync(join(OUT, "backtest.wasm")), "wasm was emitted");
  assert.ok(existsSync(join(OUT, "backtest.js")), "loader was emitted");

  const T = 20000;
  const { prices, params, nRuns } = makeInputs(T);

  // --- arm 1: the annotated source, executed by node with no compiler ---
  const js = await import(join(ROOT, "examples", "backtest.js"));
  const expected = new Float64Array(nRuns);
  js.runSweep(prices, T, params, nRuns, expected);
  assert.ok(expected.every((v) => Number.isFinite(v) && v > 0), "JS produced sane equities");
  assert.ok(new Set(expected).size > 1, "JS output is not degenerate");

  // --- arm 2: the compiled wasm ---
  const { load } = await import(join(OUT, "backtest.js"));
  const mod = await load();

  const pricesBuf = mod.alloc.f64(T);
  const paramsBuf = mod.alloc.f64(params.length);
  const outBuf = mod.alloc.f64(nRuns);
  pricesBuf.view.set(prices);
  paramsBuf.view.set(params);

  mod.kernels.runSweep(pricesBuf.ptr, T, paramsBuf.ptr, nRuns, outBuf.ptr);

  let differing = 0;
  for (let i = 0; i < nRuns; i++) if (!Object.is(expected[i], outBuf.view[i])) differing++;
  assert.equal(differing, 0, `wasm differs from JS in ${differing}/${nRuns} runs`);

  await mod.destroy();
});

test("threaded run matches single-threaded, bit for bit", async () => {
  const T = 20000;
  const { prices, params, nRuns } = makeInputs(T);

  const js = await import(join(ROOT, "examples", "backtest.js"));
  const expected = new Float64Array(nRuns);
  js.runSweep(prices, T, params, nRuns, expected);

  const { load } = await import(join(OUT, "backtest.js"));

  for (const threads of [1, 2, 4]) {
    const mod = await load({ threads });
    const pricesBuf = mod.alloc.f64(T);
    const paramsBuf = mod.alloc.f64(params.length);
    const outBuf = mod.alloc.f64(nRuns);
    pricesBuf.view.set(prices);
    paramsBuf.view.set(params);
    outBuf.view.fill(0);

    mod.parallel.runSweep(nRuns, pricesBuf.ptr, T, paramsBuf.ptr, nRuns, outBuf.ptr);

    let differing = 0;
    for (let i = 0; i < nRuns; i++) if (!Object.is(expected[i], outBuf.view[i])) differing++;
    assert.equal(differing, 0, `${threads} threads: ${differing}/${nRuns} runs differ from serial`);

    // Every index must be written exactly once: a scheduling gap would leave a
    // zero here and would otherwise masquerade as a speedup.
    assert.equal(outBuf.view.some((v) => v === 0), false, `${threads} threads: some runs never executed`);

    await mod.destroy();
  }
});
