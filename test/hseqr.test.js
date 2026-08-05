// HSEQR is the flagship example and the source of the README benchmark, so it
// gets correctness coverage rather than only timing.
//
// Eigenvalues are checked against properties that hold independently of the
// algorithm -- an upper-triangular matrix must return its own diagonal, and the
// trace must equal the sum of the real parts -- so a plausible-but-wrong solver
// cannot pass by agreeing with itself.

import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { rmSync } from "node:fs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const OUT = join(ROOT, "test", ".build-hseqr");

const N = 32;
const K = 24;

function makeBatch(n, K, seed = 5150) {
  const mats = new Float64Array(K * n * n);
  for (let b = 0; b < K; b++) {
    let a = (seed + b * 7919) >>> 0;
    const rng = () => {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const base = b * n * n;
    for (let i = 0; i < n; i++) {
      for (let j = i === 0 ? 0 : i - 1; j < n; j++) mats[base + i * n + j] = rng() * 2 - 1;
    }
  }
  return mats;
}

test("HSEQR: analytic spectra and the trace identity", async () => {
  const { runHSEQR } = await import(join(ROOT, "examples", "hseqr.js"));

  // Upper triangular: the eigenvalues are exactly the diagonal.
  for (const n of [5, 12]) {
    const h = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = i; j < n; j++) h[i * n + j] = i === j ? (i + 1) * 1.5 : 0.3 * (j - i);
    const wr = new Float64Array(n), wi = new Float64Array(n);
    runHSEQR(h, 0, n, wr, 0, wi, 0, 60);
    const got = [...wr].sort((a, b) => a - b);
    for (let i = 0; i < n; i++) {
      assert.ok(Math.abs(got[i] - (i + 1) * 1.5) < 1e-12, `n=${n} eigenvalue ${i}`);
      assert.equal(wi[i], 0, "upper triangular has no complex eigenvalues");
    }
  }

  // Known complex conjugate pairs from 2x2 rotation blocks.
  const blocks = [[1.0, 2.0], [-3.0, 0.5]];
  const n = blocks.length * 2;
  const h = new Float64Array(n * n);
  blocks.forEach(([re, im], b) => {
    const o = b * 2;
    h[o * n + o] = re; h[o * n + o + 1] = im;
    h[(o + 1) * n + o] = -im; h[(o + 1) * n + o + 1] = re;
  });
  const wr = new Float64Array(n), wi = new Float64Array(n);
  runHSEQR(h, 0, n, wr, 0, wi, 0, 60);
  const pairs = [...Array(n).keys()].map((i) => [wr[i], Math.abs(wi[i])]).sort((a, b) => a[0] - b[0]);
  assert.ok(Math.abs(pairs[0][0] + 3.0) < 1e-12 && Math.abs(pairs[0][1] - 0.5) < 1e-12, "complex pair -3 +- 0.5i");
  assert.ok(Math.abs(pairs[3][0] - 1.0) < 1e-12 && Math.abs(pairs[3][1] - 2.0) < 1e-12, "complex pair 1 +- 2i");
});

test("HSEQR: compiled wasm is bit-identical to the source as plain JS", async () => {
  rmSync(OUT, { recursive: true, force: true });
  execFileSync(process.execPath, [
    join(ROOT, "bin", "smp.js"), "build", join(ROOT, "examples", "hseqr.js"), "--out", OUT,
  ], { encoding: "utf8" });

  const mats = makeBatch(N, K);
  const js = await import(join(ROOT, "examples", "hseqr.js"));

  const scratch = new Float64Array(K * N * N);
  const wrJs = new Float64Array(K * N);
  const wiJs = new Float64Array(K * N);
  js.runBatch(mats, N, wrJs, wiJs, scratch, K);

  // Trace identity: sum of eigenvalue real parts equals the trace, per matrix.
  for (let b = 0; b < K; b++) {
    let tr = 0, sum = 0;
    for (let i = 0; i < N; i++) { tr += mats[b * N * N + i * N + i]; sum += wrJs[b * N + i]; }
    assert.ok(Math.abs(tr - sum) / Math.abs(tr) < 1e-10, `matrix ${b}: trace ${tr} vs ${sum}`);
  }
  assert.ok([...wiJs].some((v) => v !== 0), "the batch exercises the complex path");

  const { load } = await import(join(OUT, "hseqr.js"));

  for (const threads of [0, 2, 4]) {
    const mod = await load(threads > 1 ? { threads } : {});
    const m = mod.alloc.f64(K * N * N);
    const sc = mod.alloc.f64(K * N * N);
    const wr = mod.alloc.f64(K * N);
    const wi = mod.alloc.f64(K * N);
    m.view.set(mats);
    wr.view.fill(NaN); wi.view.fill(NaN);

    if (threads > 1) mod.parallel.runBatch(K, m.ptr, N, wr.ptr, wi.ptr, sc.ptr, K);
    else mod.kernels.runBatch(m.ptr, N, wr.ptr, wi.ptr, sc.ptr, K);

    let bad = 0;
    for (let i = 0; i < K * N; i++) {
      if (!Object.is(wr.view[i], wrJs[i])) bad++;
      if (!Object.is(wi.view[i], wiJs[i])) bad++;
    }
    const label = threads > 1 ? `${threads} threads` : "1 thread";
    assert.equal(bad, 0, `${label}: ${bad} eigenvalues differ from the JS run`);

    // Pre-filled with NaN, so an unwritten slot is a scheduling gap rather than
    // a coincidentally-correct zero.
    assert.ok(![...wr.view].some(Number.isNaN), `${label}: some matrices never ran`);

    await mod.destroy();
  }
});
