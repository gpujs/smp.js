# smp.js directive reference

Every directive is a **JSDoc comment**, so an annotated file is still valid
JavaScript. Delete the compiler and it runs single-threaded, scalar, producing
bit-identical results. That property is verified in this repo, not assumed:
`examples/backtest.js` is executed as plain JS by the test suite and compared
bit-for-bit against the compiled wasm at every thread count (`npm test`).

Three granularities, three families.

## `@parallel for` — across problem instances

```js
/**
 * @parallel for schedule(dynamic)
 * @shared mats
 * @private scratch
 */
for (let k = 0; k < K; k++) { ... }
```

| clause | meaning |
|---|---|
| `schedule(static)` | contiguous split. Uniform per-iteration cost. |
| `schedule(dynamic[, chunk])` | shared atomic cursor. Use when per-iteration cost varies. |
| `@shared x` | one copy, read by all threads. Lives in the SharedArrayBuffer. |
| `@private x` | one copy **per thread**, hoisted out of the loop. |

`@private` is the clause that carries real weight. In the HSEQR batch it turns
`new Float64Array(n*n)` from 512 allocations into one per worker. Get it wrong by
hand and you either allocate per iteration or share a scratch buffer across
threads and silently corrupt results.

**Measured:** ~90% parallel efficiency at 8 performance cores (7.37× on W4,
7.15–7.38× on W1/W2), bit-identical to serial at every thread count.

## `@simd` — within one inner loop

```js
/**
 * @simd
 * @simdlen 2
 */
for (let j = k; j <= nn; j++) {
  p = h[k * n + j] + q * h[(k + 1) * n + j];
  h[k * n + j] -= p * x;
}
```

| clause | meaning |
|---|---|
| `@simd` | vectorise this loop. Compiler verifies, then lowers to `v128`. |
| `@simdlen N` | lanes per vector. Default 2 for f64 (128-bit wasm vectors). |
| `@reduction(+:acc)` | permit vectorising a reduction. **Changes summation order** — see below. |
| `@aligned p:16` | promise base alignment; enables aligned loads. |

**This is a directive, not auto-vectorisation.** AssemblyScript lowers through
Binaryen, which has no loop vectoriser, so auto-vectorisation would mean building
dependence and alias analysis from scratch — a major component for a measured
1.21×. The directive reduces it to verify-and-lower.

### What the compiler must check

1. **Unit stride** in the induction variable. `h[k*n + j]` with induction `j` is
   stride 1: fine. `h[i*n + k]` with induction `i` is stride *n*: rejected, because
   wasm SIMD has no gather/scatter.
2. **No cross-iteration dependence** — no iteration reads what another writes.
3. **Loop-invariant branches must be resolved** before vectorising — either by
   unswitching or by masking. A scalar per-element branch cannot be vectorised as
   written, so the compiler has to pick one. See the measured comparison below.
4. **Tail generated automatically** for counts not divisible by `simdlen`.

### Unswitching vs bitselect masking — measured

The `k != nn - 1` test in the HSEQR row update is loop-invariant in `j`. Two ways
to handle it, both implemented and both bit-identical:

| approach | time | vs scalar | module size |
|---|---:|---:|---:|
| scalar wasm (no SIMD) | 174.5 ms | — | 2140 B |
| **unswitch** — hoist the test, emit two specialised loops | **143.8 ms** | 1.21× | 2542 B |
| **bitselect** — one loop, always 3-row, `v128.bitselect` discards the masked row | 147.3 ms | 1.18× | 2529 B |

**Unswitching is only 2.4% faster.** Closer than expected, for two reasons: the
masked-off case occurs on just one iteration per bulge chase, so the unswitched
version takes the same 3-row path almost always anyway; and bitselect adds only
two `v128` ops to roughly a dozen in the loop body.

Code size did *not* separate them here — 13 bytes apart, so at this scale the
usual "unswitching bloats code" argument is theoretical. It becomes real with
multiple invariant branches: unswitching is 2ⁿ loop copies for n conditions,
masking stays linear.

**Masking carries a hazard unswitching does not.** When `k == nn-1` the third row
is row `nn+1`, off the end of the matrix — and a masked lane still has to *load*
it. The address must be clamped to a zeroed dummy row, and the store bitselected
back to the original value rather than skipped. Get that wrong and you read past
the matrix, or alias the clamped row against a row you are actively updating.

**Guidance:** unswitch for one or two invariant conditions; switch to masking
beyond that, and pay the 2.4%. Reserve bitselect proper for genuinely
*data-dependent* per-element divergence, which is what it is actually for — the
HSEQR inner loops have none.

### Diagnostics beat silence

```
error: @simd requires unit stride in the induction variable
  --> hseqr.js:214
   |  for (i = l; i <= mmin; i++) {
   |       ^ `h[i*n+k]` has stride n (= 64), not 1
   = wasm SIMD has no gather/scatter; only contiguous accesses vectorise
   = help: interchange the loops, or store h column-major
```

This is the main argument for directives over auto-vectorisation. An
auto-vectoriser silently declines and the author never learns why the kernel runs
at half speed.

### On `@reduction`

A vectorised reduction accumulates per-lane and combines at the end, which
**changes summation order and therefore the result**. Every other directive here
preserves bit-identity; this one cannot. It should be opt-in, and the compiler
should say so when it is used, because bit-identity against a scalar reference is
the cheapest correctness gate a numerics author has.

*(The deterministic alternative, used for W3's reductions in this repo: block over
a **fixed** count independent of lane and thread count, then combine partials in
block order. Same answer every time, bit-comparable to serial.)*

### Measured, browser, 512 HSEQR matrices at n=64

| | time | gain |
|---|---:|---:|
| our JS | 332.7 ms | — |
| our wasm, scalar | 175.1 ms | 1.90× over JS |
| our wasm + `@simd` | **144.9 ms** | **1.21× over scalar** |
| LAPACK `dhseqr` (Pyodide/SciPy, wasm) | 373.8 ms | — |

Bit-identical throughout. 1.21× is close to the structural ceiling: f64 gives only
2 lanes, and the companion column loop cannot vectorise at all, so Amdahl binds
before any instruction executes.

## Why not `@simd` on the outer loop too

OpenMP allows `parallel for simd`. It is not useful here: the outer loop is over
whole matrices, whose bodies are branchy and divergent, so the lanes would
immediately need masking. Threads handle across-instance parallelism at ~90%
efficiency already. Keep the two granularities separate — threads across
instances, vectors within inner loops.
