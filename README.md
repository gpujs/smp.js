# smp.js

Compile annotated JavaScript to threaded WebAssembly. OpenMP-style directives,
written as JSDoc comments — so the annotated source still runs as ordinary
JavaScript with no compiler involved.

```js
/**
 * Assign every point to the zone containing it. One directive is the entire
 * concurrency surface.
 *
 * @kernel
 * @parallel for schedule(dynamic)
 * @shared verts, polyOff, polyLen, bbox
 * @param {Float64Array} px
 * @param {Float64Array} py
 * @param {i32} n
 * @param {Float64Array} verts
 * @param {Int32Array} polyOff
 * @param {Int32Array} polyLen
 * @param {Float64Array} bbox
 * @param {i32} nPoly
 * @param {Int32Array} out
 */
export function assignZones(px, py, n, verts, polyOff, polyLen, bbox, nPoly, out) {
  for (let i = 0; i < n; i++) {
    const x = px[i], y = py[i];
    let found = -1;
    for (let p = 0; p < nPoly; p++) {
      if (x < bbox[p*4] || x > bbox[p*4+1] || y < bbox[p*4+2] || y > bbox[p*4+3]) continue;
      if (pointInPolygon(x, y, verts, polyOff[p], polyLen[p]) === 1) { found = p; break; }
    }
    out[i] = found;
  }
}
```

```console
$ npx smp.js build examples/spatial-join.js --out build --memory 400
examples/spatial-join.js -> build/spatial-join.wasm  [pointInPolygon, assignZones]
  loader: build/spatial-join.js
```

```js
import { load } from "./build/spatial-join.js";

const smp = await load({ threads: navigator.hardwareConcurrency });
const px  = smp.alloc.f64(nPoints);
const py  = smp.alloc.f64(nPoints);
const out = smp.alloc.i32(nPoints);
px.view.set(lons); py.view.set(lats);

smp.parallel.assignZones(nPoints, px.ptr, py.ptr, nPoints,
                         verts.ptr, off.ptr, len.ptr, bbox.ptr, nPoly, out.ptr);
```

## Benchmark

**Spatial join** — a million GPS points against 200 delivery zones. Geofencing,
catchment analysis, "which service area is this address in": one of the most-run
computations in anything location-shaped, and the batch is inherent rather than
staged for a demo.

Apple M2 Max, Node 24. Reproduce with `npm run bench`:

| | time | vs plain JS |
|---|---:|---:|
| source run as plain JavaScript | 597 ms | 1.00× |
| **smp.js, 1 thread** | **247 ms** | **2.42×** |
| smp.js, 4 threads | 68 ms | 8.82× |
| **smp.js, 8 threads** | **34 ms** | **17.39×** |

Every result is **bit-identical** to the same source run as plain JavaScript.

**Why a GPU is a poor fit here.** Ray casting is SIMT-hostile in three ways, all
of them visible in the snippet above: per-point early exit (one point finds its
zone and stops while its neighbour keeps scanning), polygon vertex counts that
vary by orders of magnitude in real boundary data, and irregular access into a
ragged vertex array. Coordinates are f64 because f32 gives roughly metre-level
error on lat/lon — exactly the scale at which "inside or outside" is being decided.

### Other examples

| example | what it shows |
|---|---|
| `examples/spatial-join.js` | the benchmark above — 2.4× codegen, 17.4× threaded |
| `examples/hseqr.js` | LAPACK's nonsymmetric eigensolver, and a comparison against **LAPACK compiled to wasm**: 1.55× single-threaded, 11.35× at 8 threads, 2.4 KB against Pyodide's ~29 MB. `npm run bench:lapack` |
| `examples/geodesic.js` | Vincenty distance — the case a GPU genuinely *cannot* run (WGSL has no f64 and the 1e-12 convergence floor is unreachable in f32). Also the case where AOT wins nothing: see [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) |
| `examples/backtest.js` | a shorter parameter sweep, if you want something small to read |

## Why comments

Because `SharedArrayBuffer` requires cross-origin isolation
(`COOP: same-origin`, `COEP: require-corp`), and that breaks third-party embeds.
Any design needs a working single-threaded fallback — and with directives-as-
comments the fallback isn't a code path you maintain, **it's the source file**.
Delete the compiler and it still runs.

That property is enforced by the test suite, not merely claimed: the example is
executed as plain JS and compared bit-for-bit against the compiled wasm at every
thread count.

## Install

```console
npm i -D smp.js assemblyscript
```

`assemblyscript` is a peer dependency: smp.js emits AssemblyScript and hands off
to `asc`.

## CLI

```console
smp.js build <file...> [--out dir]   compile to wasm + an ES module loader
smp.js emit  <file>                  print the generated AssemblyScript
smp.js check <file...>               validate directives, emit nothing
```

| option | |
|---|---|
| `--out <dir>` | output directory (default `<input>/smp-build`) |
| `--no-threads` | omit shared memory and atomics |
| `--no-simd` | omit SIMD |
| `--optimize <0-3>` | asc optimize level (default 3) |
| `--memory <pages>` | initial = maximum pages (default 256 = 16 MB) |
| `--verbose` | show intermediate paths |

`smp.js emit` is the debugging tool worth knowing: it prints exactly what gets
handed to `asc`.

## Bundler plugin

One plugin for **Rolldown, Rollup and Vite** — they share the hook shape.

```js
// vite.config.js
import smp from "smp.js/plugin";

export default {
  plugins: [smp()],
};
```

```js
import { load } from "./kernels.smp.js";
```

By default it matches `*.smp.js`; pass `include` to change that. Any file may be
opted in explicitly with `?smp`.

**The Vite dev server gets `COOP`/`COEP` headers automatically.** Forgetting them
is the most common way a threaded build fails — silently, with `SharedArrayBuffer`
simply `undefined`. Disable with `crossOriginIsolation: false`.

| option | default |
|---|---|
| `include` | `[/\.smp\.m?js$/]` |
| `exclude` | `[/node_modules/]` |
| `threads` | `true` |
| `simd` | `true` |
| `memory` | `256` pages |
| `crossOriginIsolation` | `true` |

## Directives

All directives are JSDoc tags. Full reference in [`docs/DIRECTIVES.md`](docs/DIRECTIVES.md).

| directive | meaning |
|---|---|
| `@kernel` | compile this function to wasm |
| `@param {type} name` | parameter type — **required** on every parameter |
| `@returns {type}` | return type (default `void`) |
| `@parallel for schedule(dynamic\|static)` | run the function's `for` loop across a worker pool |
| `@shared x` | one copy, read by all threads |
| `@private x` | one copy per thread *(parsed, not yet lowered — see Roadmap)* |
| `@simd`, `@simdlen N` | vectorise an inner loop *(checked, not yet lowered — see Roadmap)* |

Types: `f64` `f32` `i32` `i64` `u32` `bool`, and `Float64Array` `Float32Array`
`Int32Array` (passed as base pointers into wasm memory).

### The supported subset

Kernels are deliberately restricted, because AssemblyScript has **no closures**
and **no thread-safe GC** — so `--runtime stub` and raw linear memory are the only
safe cross-thread configuration.

Supported: scalar arithmetic, typed-array indexing, `for` / `while` / `do` / `if`
/ `else` / `break` / `continue` / `return`, `Math.*`, and calls to other kernels
in the same file.

Not supported: closures, objects, classes, strings, allocation inside a kernel,
and calls to non-kernel functions. Each of these produces a diagnostic pointing at
**your JavaScript**, never at generated code:

```
error: kernels.js:41:18 'helper' is not a @kernel in this module
  help: mark it:  @kernel  — kernels may only call Math.* and other kernels in the same file
```

### One divergence to know about

`@param {i32}` narrows. JavaScript will happily pass `5.5`; wasm truncates it to
`5`. For integral values — which is what `{i32}` asserts — JS and wasm agree
exactly. Pass a non-integral value and the fallback and the compiled build will
disagree.

## Why this design

It came out of a measurement study rather than a hunch. The findings that shaped
it:

- **Threading needs no compiler.** Plain JS across `worker_threads` over a
  `SharedArrayBuffer` reached ~90% parallel efficiency at 8 performance cores.
  So `@parallel for` lowers to a scheduler, and the value is ergonomics and
  correctness, not throughput.
- **AOT codegen is where the speed is.** Hand-written AssemblyScript ran
  **1.75×** faster than JIT'd JS on a branchy f64 kernel, and the gap held at
  **101%** of its single-thread value under 8-way threading.
- **The residual matters most.** Batched HSEQR — the LAPACK phase a GPU cannot
  take — is where the compiler earns its keep, and where the benchmark above
  comes from. (The research measured a larger gap, 2.1×, in Chromium with
  hand-written AssemblyScript; Node is faster for both sides, so the ratio
  reported above is the smaller and more conservative one.)
- **SIMD is real but modest.** Vectorising to `f64x2` gave **1.21×**. wasm f64
  SIMD is only 2 lanes wide, and there is no gather/scatter, so strided loops
  cannot vectorise at all. Hence directives with diagnostics rather than
  auto-vectorisation that silently declines.

## Architecture

```
annotated JS  →  smp.js  →  AssemblyScript  →  asc  →  Binaryen  →  wasm
```

smp.js is **not** a fork of AssemblyScript and not a compiler from scratch. It is
a directive-lowering pass that emits AssemblyScript source and shells out to stock
`asc`. Everything it needs to emit — raw linear memory, atomic barriers, `f64x2`,
`v128.bitselect` — is already expressible in unmodified AssemblyScript.

## Roadmap

**0.1.0 (now)** — `@kernel`, `@param`/`@returns`, `@parallel for schedule(dynamic)`,
worker pool with dynamic stealing, CLI, bundler plugin, single-threaded fallback,
bit-identity tests.

**Next**
- `@simd` lowering to `v128` (currently parsed and checked, emits scalar)
- `@private` lowering to per-thread storage (currently parsed, not lowered)
- Stride and dependence diagnostics for `@simd`
- `schedule(static)` distinct from dynamic
- `@reduction` with a deterministic blocked form, so bit-identity survives
- Monomorphisation: `{f32,f64} × {row-major,col-major}` variants at build time
- Source maps from generated AssemblyScript back to the original JS
- A separate `lda` parameter for kernels, so callers can pad the leading
  dimension away from a power of two (see `docs/PERFORMANCE.md`)

## Development

```console
npm test                 # bit-identity: source-as-JS vs compiled wasm
npm run bench            # the HSEQR benchmark above
npm run bench:lapack     # ...including LAPACK (needs: npm i -D pyodide)
node bin/smp.js emit examples/hseqr.js
```

| example | |
|---|---|
| `examples/hseqr.js` | HSEQR eigensolver — the flagship, and the benchmark above |
| `examples/backtest.js` | a smaller parameter sweep, if you want something shorter to read |
| `examples/bench-single.js` | unbatched timings, and the power-of-two cliff |

## License

MIT — see [LICENSE](LICENSE).
