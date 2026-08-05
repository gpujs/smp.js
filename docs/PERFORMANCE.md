# Performance notes

Measured on Apple M2 Max (8 performance + 4 efficiency cores), Node 24.
Reproduce with `npm run bench` and `node examples/bench-single.js`.

## Batched vs unbatched

The README's headline — 11.9× — is a *batched* number, and the distinction
matters more than it might look.

HSEQR has **no parallelism inside a single matrix**: the bulge chase at step k
consumes what step k−1 produced. So on one problem the worker pool has nothing to
distribute and `@parallel for` contributes exactly nothing. All that remains is
AOT codegen.

| | speedup over the same source as plain JS |
|---|---:|
| one matrix, any thread count | **~1.5–1.6×** |
| 512 matrices, 1 thread | 1.63× |
| 512 matrices, 8 threads | **11.9×** |

Unbatched, one matrix per call:

| n | plain JS | smp.js wasm | speedup |
|---:|---:|---:|---:|
| 128 | 3.99 ms | 2.43 ms | 1.64× |
| 192 | 12.68 ms | 7.99 ms | 1.59× |
| 256 | 26.97 ms | 19.90 ms | 1.36× |
| 320 | 52.98 ms | 33.91 ms | 1.56× |
| 384 | 91.91 ms | 61.51 ms | 1.49× |
| 448 | 139.88 ms | 92.36 ms | 1.51× |

All bit-identical to the JS run. The JS→wasm boundary costs ~0.02 µs/call, so it
is irrelevant at any size worth compiling.

**So: batch if you can.** Threads are worth ~7× on top of codegen's ~1.6×, but
only when there are independent problems to spread. One big matrix gets the 1.6×
and nothing else.

### A caveat on very small n

At n=64 the isolated-call benchmark reports **5.04×**, which is not a number to
quote. The same kernel inside the batched loop shows 1.63×. The difference is on
the JavaScript side: 1.80 ms per matrix called in isolation versus 0.62 ms per
matrix inside a tight loop, because V8 optimises the batch far better. The wasm
time is essentially identical either way (0.358 ms vs 0.38 ms). Treat the batched
figure as the honest one at small n.

## The power-of-two cliff

**At n = 512, wasm is 2.8× slower than it should be — slower than plain JS.**

| n | plain JS | smp.js wasm | speedup |
|---:|---:|---:|---:|
| 504 | 197.8 ms | 125.0 ms | 1.58× |
| 511 | 198.3 ms | 126.0 ms | 1.57× |
| **512** | 207.0 ms | **354.6 ms** | **0.58×** |
| 513 | 198.8 ms | 126.2 ms | 1.58× |
| 520 | 225.2 ms | 143.1 ms | 1.57× |

A cliff, not a trend: ±1 removes it entirely.

**Cause.** The matrix is stored with leading dimension `n`, so the row stride is
`n * 8` bytes. At n=512 that is exactly 4096 bytes, and rows land in the same
cache set — the classic power-of-two leading dimension problem that LAPACK avoids
by carrying `lda` separately from `n`. The column-walking loop in the bulge chase
(`h[i*n + k]` for consecutive `i`) then conflict-misses on every access.

Shifting the base address does **not** help, and that is diagnostic rather than
disappointing: padding the base moves every row together and preserves the
stride. Verified — offsets from 8 to 512 bytes all measure ~353 ms.

Plain JS shows only a ~4% bump at the same size. Why V8 is resilient here is not
diagnosed; likely differences in prefetch or access ordering.

**Workaround today:** avoid a power-of-two leading dimension. `n=513` with the
last row and column ignored costs 0.4% more arithmetic and runs 2.8× faster.

**Fix planned:** give kernels a separate `lda` parameter, exactly as LAPACK does,
so callers can pad the leading dimension independently of the matrix dimension.
This is the strongest argument yet for `lda` in the API rather than deriving the
stride from `n`.

## What this says about where smp.js helps

| workload shape | expect |
|---|---|
| many independent problems (batch, sweep, ensemble) | **~1.6× codegen × ~7× threads ≈ 11×** |
| one large problem with internal parallelism | codegen only, until `@parallel for` can express intra-problem work |
| one small problem | codegen only, ~1.6× |
| any of the above with a power-of-two leading dimension | pad it first |
