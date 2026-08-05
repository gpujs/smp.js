// Benchmark: batched HSEQR, the LAPACK phase a GPU cannot take.
//
//   node examples/bench-hseqr.js              # JS vs smp.js wasm
//   node examples/bench-hseqr.js --lapack     # also vs LAPACK compiled to wasm
//                                             #   (needs: npm i -D pyodide)
//
// Build first:
//   node bin/smp.js build examples/hseqr.js --out examples/build --memory 600

import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import os from "node:os";

import { runHSEQR, runBatch } from "./hseqr.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const N = Number(process.env.N || 64);
const K = Number(process.env.K || 512);
const WANT_LAPACK = process.argv.includes("--lapack");

const med = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
const time = (fn, warm = 2, iters = 7) => {
  for (let i = 0; i < warm; i++) fn();
  const s = [];
  for (let i = 0; i < iters; i++) { const t = performance.now(); fn(); s.push(performance.now() - t); }
  return med(s);
};

/** Deterministic upper Hessenberg batch: full upper triangle plus one subdiagonal. */
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

async function main() {
  console.log(`=== batched HSEQR: ${K} nonsymmetric matrices, n=${N} ===`);
  console.log(`${os.cpus()[0].model}, node ${process.version}\n`);

  const mats = makeBatch(N, K);

  // ---- arm 1: this file, run as plain JavaScript ----
  const scratchJs = new Float64Array(K * N * N);
  const wrJs = new Float64Array(K * N);
  const wiJs = new Float64Array(K * N);
  runBatch(mats, N, wrJs, wiJs, scratchJs, K);

  const nComplex = wiJs.reduce((a, v) => a + (v !== 0 ? 1 : 0), 0);
  console.log(`${nComplex}/${K * N} eigenvalues complex -- genuinely nonsymmetric, not a symmetric problem in disguise\n`);

  const tJs = time(() => runBatch(mats, N, wrJs, wiJs, scratchJs, K));

  // ---- arm 2: the same source, compiled by smp.js ----
  const { load } = await import(join(HERE, "build", "hseqr.js"));

  const results = [];
  for (const threads of [0, 2, 4, 8]) {
    const mod = await load(threads > 1 ? { threads } : {});
    const m = mod.alloc.f64(K * N * N);
    const sc = mod.alloc.f64(K * N * N);
    const wr = mod.alloc.f64(K * N);
    const wi = mod.alloc.f64(K * N);
    m.view.set(mats);
    wr.view.fill(0); wi.view.fill(0);

    const call = threads > 1
      ? () => mod.parallel.runBatch(K, m.ptr, N, wr.ptr, wi.ptr, sc.ptr, K)
      : () => mod.kernels.runBatch(m.ptr, N, wr.ptr, wi.ptr, sc.ptr, K);

    call();
    let bad = 0;
    for (let i = 0; i < K * N; i++) {
      if (!Object.is(wr.view[i], wrJs[i])) bad++;
      if (!Object.is(wi.view[i], wiJs[i])) bad++;
    }

    const t = time(call);
    results.push({ threads: threads || 1, label: threads > 1 ? `${threads}t` : "1t", ms: t, bad });
    await mod.destroy();
  }

  console.log(`  source as plain JS   ${tJs.toFixed(1)} ms`);
  for (const r of results) {
    console.log(
      `  smp.js wasm ${r.label.padEnd(4)}     ${r.ms.toFixed(1).padStart(6)} ms   ` +
      `${(tJs / r.ms).toFixed(2)}x vs JS   ${r.bad === 0 ? "bit-identical" : `MISMATCH ${r.bad}`}`
    );
  }

  if (!WANT_LAPACK) {
    console.log(`\n(run with --lapack to compare against LAPACK compiled to wasm)`);
    return;
  }

  // ---- arm 3: reference LAPACK, Fortran -> Emscripten -> wasm, via Pyodide ----
  let loadPyodide;
  try { ({ loadPyodide } = await import("pyodide")); }
  catch { console.error("\n--lapack needs pyodide:  npm i -D pyodide"); return; }

  console.log(`\nloading Pyodide + numpy + scipy...`);
  const py = await loadPyodide({ stdout: () => {}, stderr: () => {} });
  await py.loadPackage(["numpy", "scipy"]);
  py.FS.writeFile("/mats.bin", new Uint8Array(mats.buffer.slice(0)));

  const out = JSON.parse(await py.runPythonAsync(`
import numpy as np, os, time, json, scipy
from scipy.linalg import lapack

K, n = ${K}, ${N}
mats = np.fromfile('/mats.bin', dtype=np.float64).reshape(K, n, n)
# Fortran order once, outside every timed region: LAPACK is column-major and we
# are measuring the solver, not a layout conversion.
fmats = [np.asfortranarray(mats[k]) for k in range(K)]
tiny  = [np.asfortranarray(np.eye(4) + 0.1) for _ in range(K)]

# scipy's default lwork for dgehrd is a WORKSPACE QUERY that returns instantly
# and measures nothing. Request the real workspace.
lw = int(np.real(lapack.dgehrd_lwork(n)[0]))

def med(xs): return sorted(xs)[len(xs)//2]
def bench(fn, warm=2, reps=5):
    for _ in range(warm): fn()
    o=[]
    for _ in range(reps):
        t=time.perf_counter(); fn(); o.append((time.perf_counter()-t)*1000)
    return med(o)

def geev():    [lapack.dgeev(a, compute_vl=0, compute_vr=0, overwrite_a=0) for a in fmats]
def gehrd():   [lapack.dgehrd(a, lwork=lw, overwrite_a=0) for a in fmats]
def overhead():[lapack.dgeev(a, compute_vl=0, compute_vr=0, overwrite_a=0) for a in tiny]

wr, wi, _, _, info = lapack.dgeev(fmats[0], compute_vl=0, compute_vr=0)
json.dumps({"scipy": scipy.__version__, "cpu_count": os.cpu_count(),
            "geev": bench(geev), "gehrd": bench(gehrd), "overhead": bench(overhead),
            "wr0": wr.tolist(), "wi0": wi.tolist()})
`));

  // Our inputs are ALREADY Hessenberg, so dgehrd's reflectors are all trivially
  // zero and the reduction collapses to O(n^2). dgeev's cost here really is the
  // QR iteration, which is what makes this comparable.
  const geevNet = out.geev - out.overhead;
  const gehrdNet = out.gehrd - out.overhead;
  const qrOnly = geevNet - gehrdNet;

  const ours = [];
  for (let i = 0; i < N; i++) ours.push([wrJs[i], Math.abs(wiJs[i])]);
  const theirs = out.wr0.map((re, i) => [re, Math.abs(out.wi0[i])]);
  const key = (z) => z[0] * 1e6 + z[1];
  ours.sort((a, b) => key(a) - key(b));
  theirs.sort((a, b) => key(a) - key(b));
  let maxDiff = 0;
  for (let i = 0; i < N; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(ours[i][0] - theirs[i][0]), Math.abs(ours[i][1] - theirs[i][1]));
  }

  const best = results.reduce((a, b) => (a.ms < b.ms ? a : b));
  const one = results.find((r) => r.label === "1t");

  console.log(`\nLAPACK via Pyodide (scipy ${out.scipy}, cpu_count ${out.cpu_count} -- no pthreads)`);
  console.log(`  spectrum agreement vs ours: max abs diff ${maxDiff.toExponential(3)}`);
  console.log(`  python/scipy wrapper, ${K} calls : ${out.overhead.toFixed(1)} ms (subtracted)`);
  console.log(`  dgeev  (balance + reduce + QR)   : ${out.geev.toFixed(1)} ms`);
  console.log(`  dgehrd (reduce only)             : ${out.gehrd.toFixed(1)} ms`);
  console.log(`  => dhseqr (QR) estimate          : ${qrOnly.toFixed(1)} ms`);
  console.log(`\n  smp.js 1 thread vs LAPACK : ${(qrOnly / one.ms).toFixed(2)}x`);
  console.log(`  smp.js ${best.label} vs LAPACK    : ${(qrOnly / best.ms).toFixed(2)}x`);
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
