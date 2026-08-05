# smp.js

Compile annotated JavaScript to threaded WebAssembly. OpenMP-style directives,
written as JSDoc comments — so the annotated source still runs as ordinary
JavaScript with no compiler involved.

```js
/**
 * Solve a batch of eigenproblems. One directive is the entire concurrency
 * surface -- `runHSEQR` below it is 150 lines of branchy f64 numerics that never
 * mention threads, memory or the compiler.
 *
 * @kernel
 * @parallel for schedule(dynamic)
 * @shared mats
 * @param {Float64Array} mats
 * @param {i32} n
 * @param {Float64Array} wrOut
 * @param {Float64Array} wiOut
 * @param {Float64Array} scratch
 * @param {i32} K
 */
export function runBatch(mats, n, wrOut, wiOut, scratch, K) {
  for (let b = 0; b < K; b++) {
    const base = b * n * n;
    for (let i = 0; i < n * n; i++) scratch[base + i] = mats[base + i];
    runHSEQR(scratch, base, n, wrOut, b * n, wiOut, b * n, 60);
  }
}
```

```console
$ npx smp.js build examples/hseqr.js --out build --memory 600
examples/hseqr.js -> build/hseqr.wasm  [runHSEQR, runBatch]
  loader: build/hseqr.js
```

```js
import { load } from "./build/hseqr.js";

const smp = await load({ threads: navigator.hardwareConcurrency });

const mats    = smp.alloc.f64(K * n * n);
const scratch = smp.alloc.f64(K * n * n);  // each iteration works in its own slice
const wr      = smp.alloc.f64(K * n);
const wi      = smp.alloc.f64(K * n);
mats.view.set(batch);

smp.parallel.runBatch(K, mats.ptr, n, wr.ptr, wi.ptr, scratch.ptr, K);
// wr.view / wi.view now hold K * n eigenvalues
```

## Benchmark

`examples/hseqr.js` is **HSEQR** — the Francis double-shift QR iteration for
nonsymmetric eigenvalues. It is the LAPACK phase a GPU cannot take: f64-critical
(at f32 epsilon, clustered eigenvalues never separate, and WGSL has no f64),
branch-divergent, strictly sequential within one matrix, and small enough to be
latency-bound. So the parallelism has to come from batching across matrices.

512 matrices at n=64, Apple M2 Max, Node 24. Reproduce with `npm run bench`:

| | time | vs plain JS | vs LAPACK |
|---|---:|---:|---:|
| source run as plain JavaScript | 317.1 ms | 1.00× | 0.95× |
| **smp.js, 1 thread** | **194.9 ms** | **1.63×** | **1.55×** |
| smp.js, 4 threads | 52.2 ms | 6.08× | 5.79× |
| **smp.js, 8 threads** | **26.6 ms** | **11.90×** | **11.35×** |
| LAPACK `dhseqr` (Fortran → Emscripten → wasm) | 302.3 ms | — | 1.00× |

Every smp.js result is **bit-identical** to the same source run as plain
JavaScript. The spectra agree with LAPACK to `1.5e-14`, and 21,930 of 32,768
eigenvalues come out complex — this is genuinely exercising the nonsymmetric
path, not a symmetric problem in disguise.

**Batching is doing a lot of that work.** HSEQR has no parallelism *inside* one
matrix, so a single call gets codegen only — about **1.5–1.6×**, at any thread
count. The 11.9× needs independent problems to spread across cores. There is also
a sharp cache cliff at power-of-two leading dimensions (n=512 runs 2.8× slower
than n=511). Both are measured in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

Payload matters as much as the timing:

| | |
|---|---|
| `hseqr.wasm` | **2.4 KB** |
| Pyodide runtime + numpy + scipy wheels | ~29 MB, ~2 s startup |

**How the LAPACK arm is measured.** scipy exposes `dgeev`, not `dhseqr`, and
`dgeev` = balance + `dgehrd` + `dhseqr`. Our inputs are *already* Hessenberg, so
every Householder reflector in `dgehrd` is trivially zero and the reduction
collapses to O(n²) — measured at 2.4 ms against 74 ms on dense controls. So
`dgeev`'s cost here really is the QR iteration. Python wrapper overhead (1.2 ms
over 512 calls) is measured separately and subtracted. At n=64, LAPACK's `dhseqr`
delegates to `dlahqr`, the same classic double-shift this kernel implements;
above LAPACK's `NMIN` (~75) it switches to multishift with aggressive early
deflation and would win on algorithm rather than codegen.

Pyodide's scipy is a non-pthreads wasm32 build (`os.cpu_count() == 1`), so the
multi-threaded rows compare against a single-threaded LAPACK. That is the only
wasm LAPACK available; the comparison is noted rather than hidden.

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
