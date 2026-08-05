# smp.js

Compile annotated JavaScript to threaded WebAssembly. OpenMP-style directives,
written as JSDoc comments — so the annotated source still runs as ordinary
JavaScript with no compiler involved.

```js
/**
 * @kernel
 * @parallel for schedule(dynamic)
 * @param {Float64Array} prices
 * @param {i32} T
 * @param {Float64Array} params
 * @param {i32} nRuns
 * @param {Float64Array} out
 */
export function runSweep(prices, T, params, nRuns, out) {
  for (let r = 0; r < nRuns; r++) {
    const b = r * 4;
    out[r] = runBacktest(prices, T, params[b], params[b + 1], params[b + 2], params[b + 3]);
  }
}
```

```console
$ npx smp.js build src/kernels.js --out build
src/kernels.js -> build/kernels.wasm  [runBacktest, runSweep]
  loader: build/kernels.js
```

```js
import { load } from "./build/kernels.js";

const smp = await load({ threads: navigator.hardwareConcurrency });
const prices = smp.alloc.f64(200_000);
const out    = smp.alloc.f64(nRuns);
prices.view.set(series);

smp.parallel.runSweep(nRuns, prices.ptr, 200_000, params.ptr, nRuns, out.ptr);
```

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
| `@private x` | one copy per thread |
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
- **The residual matters most.** On batched HSEQR — the LAPACK phase a GPU cannot
  take (f64-critical, branch-divergent, sequential per instance) — compiled
  output was **2.1× faster than LAPACK compiled to wasm** via Emscripten, at
  2 KB instead of ~29 MB.
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
- Stride and dependence diagnostics for `@simd`
- `schedule(static)` distinct from dynamic
- `@reduction` with a deterministic blocked form, so bit-identity survives
- Monomorphisation: `{f32,f64} × {row-major,col-major}` variants at build time
- Source maps from generated AssemblyScript back to the original JS

## Development

```console
npm test                 # bit-identity: source-as-JS vs compiled wasm
npm run example          # build examples/backtest.js
node bin/smp.js emit examples/backtest.js
```

## License

MIT — see [LICENSE](LICENSE).
