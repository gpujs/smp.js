// Benchmark: Vincenty inverse geodesic distance over a set of coordinate pairs.
//
//   node examples/bench-geodesic.js
//
// Build first:
//   node bin/smp.js build examples/geodesic.js --out examples/build --memory 800
//
// The reference arm is geographiclib-geodesic, Karney's implementation and the
// standard against which geodesic code is checked. It is a different (more
// robust) algorithm than Vincenty, so it serves as an accuracy oracle as well as
// a speed comparison -- agreement to sub-millimetre is the correctness check.

import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import os from "node:os";
import { geodesicDistance, distanceBatch } from "./geodesic.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const N = Number(process.env.N || 1_000_000);

const med = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
const time = (fn, warm = 1, iters = 5) => {
  for (let i = 0; i < warm; i++) fn();
  const s = [];
  for (let i = 0; i < iters; i++) { const t = performance.now(); fn(); s.push(performance.now() - t); }
  return med(s);
};

/** Origin-destination pairs: origins worldwide, destinations within ~800 km.
 *  That is the shape of a delivery or nearest-facility workload, and it avoids
 *  the near-antipodal cases where Vincenty is documented not to converge. */
function makePairs(n, seed = 20260805) {
  let a = seed >>> 0;
  const rng = () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const lat1 = new Float64Array(n), lon1 = new Float64Array(n);
  const lat2 = new Float64Array(n), lon2 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    lat1[i] = (rng() * 120) - 60;
    lon1[i] = (rng() * 360) - 180;
    lat2[i] = Math.max(-89, Math.min(89, lat1[i] + (rng() * 14 - 7)));
    lon2[i] = lon1[i] + (rng() * 14 - 7);
  }
  return { lat1, lon1, lat2, lon2 };
}

async function main() {
  console.log(`=== Vincenty inverse geodesic distance: ${N.toLocaleString()} coordinate pairs ===`);
  console.log(`${os.cpus()[0].model}, node ${process.version}\n`);

  const { lat1, lon1, lat2, lon2 } = makePairs(N);
  const outJs = new Float64Array(N);

  distanceBatch(lat1, lon1, lat2, lon2, outJs, N);
  const failed = outJs.reduce((c, v) => c + (v < 0 ? 1 : 0), 0);
  console.log(`${failed} of ${N} pairs failed to converge (Vincenty's documented antipodal limitation)`);

  const tJs = time(() => distanceBatch(lat1, lon1, lat2, lon2, outJs, N));

  const { load } = await import(join(HERE, "build", "geodesic.js"));
  const rows = [];
  for (const threads of [0, 2, 4, 8]) {
    const mod = await load(threads > 1 ? { threads } : {});
    const a1 = mod.alloc.f64(N), o1 = mod.alloc.f64(N);
    const a2 = mod.alloc.f64(N), o2 = mod.alloc.f64(N);
    const out = mod.alloc.f64(N);
    a1.view.set(lat1); o1.view.set(lon1); a2.view.set(lat2); o2.view.set(lon2);
    out.view.fill(NaN);

    const call = threads > 1
      ? () => mod.parallel.distanceBatch(N, a1.ptr, o1.ptr, a2.ptr, o2.ptr, out.ptr, N)
      : () => mod.kernels.distanceBatch(a1.ptr, o1.ptr, a2.ptr, o2.ptr, out.ptr, N);

    call();
    let bad = 0;
    for (let i = 0; i < N; i++) if (!Object.is(out.view[i], outJs[i])) bad++;

    rows.push({ label: threads > 1 ? `${threads}t` : "1t", ms: time(call), bad });
    await mod.destroy();
  }

  // ---- reference: Karney, via geographiclib-geodesic ----
  let geod = null;
  try {
    const m = await import("geographiclib-geodesic");
    geod = (m.Geodesic ?? m.default?.Geodesic).WGS84;
  } catch { /* optional */ }

  let tRef = null, maxDiff = 0;
  if (geod) {
    // Accuracy oracle on a sample: Vincenty and Karney are different algorithms,
    // so they agree to their shared accuracy rather than bit for bit.
    for (let i = 0; i < 20000; i++) {
      if (outJs[i] < 0) continue;
      maxDiff = Math.max(maxDiff, Math.abs(geod.Inverse(lat1[i], lon1[i], lat2[i], lon2[i]).s12 - outJs[i]));
    }
    // Timed on a subset; the full million takes minutes in a per-call JS API.
    const M = Math.min(N, 50_000);
    const t = time(() => { for (let i = 0; i < M; i++) geod.Inverse(lat1[i], lon1[i], lat2[i], lon2[i]); }, 1, 3);
    tRef = (t / M) * N; // scaled to the full set
    console.log(`agreement with geographiclib (Karney): max ${(maxDiff * 1000).toFixed(3)} mm over 20k pairs\n`);
  }

  console.log(`  source as plain JS        ${tJs.toFixed(0).padStart(6)} ms`);
  for (const r of rows) {
    console.log(
      `  smp.js wasm ${r.label.padEnd(3)}           ${r.ms.toFixed(0).padStart(6)} ms   ` +
      `${(tJs / r.ms).toFixed(2)}x vs JS   ${r.bad === 0 ? "bit-identical" : `MISMATCH ${r.bad}`}`
    );
  }
  if (tRef) {
    const best = rows.reduce((a, b) => (a.ms < b.ms ? a : b));
    console.log(`  geographiclib (Karney)    ${tRef.toFixed(0).padStart(6)} ms   (scaled from 50k pairs)`);
    console.log(`\n  smp.js 1 thread vs geographiclib : ${(tRef / rows[0].ms).toFixed(2)}x`);
    console.log(`  smp.js ${best.label} vs geographiclib      : ${(tRef / best.ms).toFixed(2)}x`);
  }
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
