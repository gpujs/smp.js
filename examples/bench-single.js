// Unbatched: ONE matrix at a time. The counterpart to bench-hseqr.js.
//
//   node examples/bench-single.js [--lapack]
//
// Why this matters: HSEQR has no parallelism inside a single matrix -- the bulge
// chase at step k consumes what step k-1 produced. So on one problem the pool has
// nothing to do and the only thing left is AOT codegen. This benchmark isolates
// that, so the batched 11.9x is not mistaken for something a single call gets.

import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import os from "node:os";
import { runHSEQR } from "./hseqr.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const WANT_LAPACK = process.argv.includes("--lapack");
const SIZES = (process.env.SIZES || "64,128,256,512").split(",").map(Number);

const med = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];

// runHSEQR destroys its input, so a pristine copy is restored before every
// iteration -- outside the timed region, or the memcpy would be charged to the solve.
function timeWithSetup(setup, fn, warm, iters) {
  for (let i = 0; i < warm; i++) { setup(); fn(); }
  const s = [];
  for (let i = 0; i < iters; i++) {
    setup();
    const t = performance.now();
    fn();
    s.push(performance.now() - t);
  }
  return med(s);
}

function makeHessenberg(n, seed = 5150) {
  let a = seed >>> 0;
  const rng = () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const h = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i === 0 ? 0 : i - 1; j < n; j++) h[i * n + j] = rng() * 2 - 1;
  }
  return h;
}

async function main() {
  console.log("=== UNBATCHED HSEQR: one matrix per call ===");
  console.log(`${os.cpus()[0].model}, node ${process.version}\n`);

  const { load } = await import(join(HERE, "build", "hseqr.js"));
  const mod = await load();
  const rows = [];

  for (const n of SIZES) {
    const src = makeHessenberg(n);

    const hJs = new Float64Array(n * n);
    const wrJs = new Float64Array(n);
    const wiJs = new Float64Array(n);
    const tJs = timeWithSetup(
      () => hJs.set(src),
      () => runHSEQR(hJs, 0, n, wrJs, 0, wiJs, 0, 60),
      3, 15
    );

    const h = mod.alloc.f64(n * n);
    const wr = mod.alloc.f64(n);
    const wi = mod.alloc.f64(n);
    const tWasm = timeWithSetup(
      () => h.view.set(src),
      () => mod.kernels.runHSEQR(h.ptr, 0, n, wr.ptr, 0, wi.ptr, 0, 60),
      3, 15
    );

    let bad = 0;
    for (let i = 0; i < n; i++) {
      if (!Object.is(wr.view[i], wrJs[i])) bad++;
      if (!Object.is(wi.view[i], wiJs[i])) bad++;
    }
    rows.push({ n, tJs, tWasm, bad });
  }

  // Cost of crossing into wasm at all: a call that returns immediately.
  const dummy = mod.alloc.f64(4);
  const dwr = mod.alloc.f64(2);
  const dwi = mod.alloc.f64(2);
  const t0 = performance.now();
  const REPS = 200000;
  for (let i = 0; i < REPS; i++) mod.kernels.runHSEQR(dummy.ptr, 0, 1, dwr.ptr, 0, dwi.ptr, 0, 60);
  const boundaryUs = ((performance.now() - t0) * 1000) / REPS;

  console.log("  n     plain JS     smp.js wasm    speedup   identical");
  console.log("  ----  -----------  -------------  --------  ---------");
  for (const r of rows) {
    console.log(
      `  ${String(r.n).padEnd(4)}  ${r.tJs.toFixed(3).padStart(8)} ms  ` +
      `${r.tWasm.toFixed(3).padStart(10)} ms  ${(r.tJs / r.tWasm).toFixed(2).padStart(6)}x  ` +
      `${r.bad === 0 ? "yes" : "NO"}`
    );
  }
  console.log(`\n  JS->wasm boundary: ${boundaryUs.toFixed(2)} us/call`);
  console.log(`  threads on one matrix: no effect -- HSEQR is strictly sequential within a`);
  console.log(`  single problem, so the pool has nothing to distribute.`);

  await mod.destroy();

  if (!WANT_LAPACK) return;

  let loadPyodide;
  try { ({ loadPyodide } = await import("pyodide")); }
  catch { console.error("\n--lapack needs pyodide:  npm i -D pyodide"); return; }

  console.log(`\nloading Pyodide + numpy + scipy...`);
  const py = await loadPyodide({ stdout: () => {}, stderr: () => {} });
  await py.loadPackage(["numpy", "scipy"]);

  const out = JSON.parse(await py.runPythonAsync(`
import numpy as np, time, json
from scipy.linalg import lapack

def med(xs): return sorted(xs)[len(xs)//2]
res = {}
for n in ${JSON.stringify(SIZES)}:
    a = np.zeros((n, n))
    rng = np.random.default_rng(5150)
    for i in range(n):
        lo = 0 if i == 0 else i - 1
        a[i, lo:] = rng.random(n - lo) * 2 - 1
    f = np.asfortranarray(a)
    def one(): lapack.dgeev(f, compute_vl=0, compute_vr=0, overwrite_a=0)
    for _ in range(3): one()
    s = []
    for _ in range(15):
        t = time.perf_counter(); one(); s.append((time.perf_counter() - t) * 1000)
    res[str(n)] = med(s)
json.dumps(res)
`));

  console.log("\n  n     smp.js wasm    LAPACK dgeev   smp.js is");
  console.log("  ----  -------------  -------------  ---------");
  for (const r of rows) {
    const lap = out[String(r.n)];
    console.log(
      `  ${String(r.n).padEnd(4)}  ${r.tWasm.toFixed(3).padStart(10)} ms  ` +
      `${lap.toFixed(3).padStart(10)} ms  ${(lap / r.tWasm).toFixed(2).padStart(6)}x`
    );
  }
  console.log(`\n  note: dgeev on a DENSE matrix also pays the Hessenberg reduction, which the`);
  console.log(`  batched benchmark avoids by feeding it already-Hessenberg input.`);
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
