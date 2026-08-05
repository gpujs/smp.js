# Survey: how much of BLAS, LAPACK and Eigen can smp.js improve?

Reference LAPACK 3.12.1, Reference BLAS, Eigen 3.4.0. Classification is derived
from the source, not from recall, using two checkable criteria.

## Method

**Does it belong on a GPU?** Proxy: *does the routine reach level-3 BLAS
(`DGEMM`, `DTRSM`, `DSYRK`, `DSYR2K`, `DTRMM`, `DSYMM`) through the call graph?*
Computed as a **transitive closure**, which matters — `DORGQR` contains no
`CALL DGEMM` of its own but reaches it through `DLARFB`, and direct-call
detection undercounts badly.

**Does it iterate?** Proxy: *does the `INFO` parameter block document a positive
return code with convergence-failure semantics?* That is LAPACK's own statement
that the algorithm can fail to terminate, scoped to the `INFO` docs rather than
grepping whole files (which flagged `DISNAN`, a NaN test, as iterative).

Counted over **double-precision real** routines only. S/C/Z are the same
algorithms in other precisions; counting all four inflates every number ~4×.
LAPACK's `SRC` holds 2,038 `.f` files, of which 505 are `d*`.

**Calibration.** Against 11 routines known to be iterative, the classifier
catches 8. It misses `DSTEMR`, `DTGSJA` and `DHGEQZ`, so the iterative counts
below are **under**-estimates by roughly 3–5.

## LAPACK: 330 double-precision routines (excluding 175 `DLA*` auxiliary)

| | count | share |
|---|---:|---:|
| **A** — iterative, never reaches BLAS3 | 21 | 6.4% |
| **B** — iterative *and* reaches BLAS3 | 33 | 10.0% |
| **C** — BLAS3-bound, not iterative | 126 | 38.2% |
| **D** — neither: scalar, BLAS-1/2, memory-bound | 150 | 45.5% |

**A + B = 54 routines, ~16.4%** (≈57, ~17%, correcting for the known misses).

**A** — the core: `DBDSQR`, `DBDSVDX`, `DBBCSD`, `DGESVJ`, `DHSEIN`, `DPTEQR`,
`DSTEBZ`, `DSTEIN`, `DSTEQR`, `DSTERF`, `DSTEV`, `DSTEVX`, `DSBEV`, `DSBEVX`,
`DSBGV`, `DSPEV`, `DSPEVX`, `DSPGV`, `DSPGVX`, and the 2-stage variants.

**B** — hybrid: `DGEEV`, `DGEES`, `DGEEVX`, `DGEESX`, `DHSEQR`, `DSYEV`,
`DSYEVD`, `DSYEVX`, `DSYGV`, `DGESVD`, `DGESDD`, `DGESVDX`, `DGESVDQ`, `DGEJSV`,
`DBDSDC`, `DSTEDC`, `DGELSD`, `DGELSS`, `DORCSD`. These reduce first (BLAS3, GPU
territory) then iterate (smp.js territory) — the split the original plan
identified.

**Look at what A ∪ B actually is: it is the entire eigenvalue and SVD surface of
LAPACK.** LU, Cholesky and QR — `DGETRF`, `DPOTRF`, `DGEQRF` and their families —
land in C, exactly as expected.

## BLAS: 35 double-precision routines

No good fits, on the merits. Level 1 (`DAXPY`, `DDOT`, `DSCAL`) and level 2
(`DGEMV`, `DTRSV`) are memory-bandwidth-bound, where neither AOT codegen nor
threads buy much. Level 3 is `DGEMM` and friends — a blocking, packing and
microkernel problem, and a different engineering discipline from what smp.js
does.

## Eigen 3.4.0: 65 dense and sparse solver headers

13 carry `ComputationInfo::NoConvergence` or an explicit iteration bound — Eigen's
direct analogue of LAPACK's `INFO > 0`:

| module | iterative classes |
|---|---|
| `Eigenvalues` | `RealSchur`, `ComplexSchur`, `SelfAdjointEigenSolver`, `ComplexEigenSolver`, `RealQZ` |
| `SVD` | `BDCSVD` |
| `IterativeLinearSolvers` | `ConjugateGradient`, `BiCGSTAB`, `LeastSquaresConjugateGradient` |

Not iterative — direct factorisations, all BLAS3-shaped: `LLT`, `LDLT`,
`PartialPivLU`, `FullPivLU`, `HouseholderQR`, `ColPivHouseholderQR`,
`FullPivHouseholderQR`, `CompleteOrthogonalDecomposition`.

Same ~20% ratio as LAPACK, and the same split: **eigenvalues, SVD and iterative
solvers on one side; LU/Cholesky/QR on the other.** `JacobiSVD` is iterative but
carries no bound, so the classifier misses it — the same undercount as `DSTEMR`.

## The number changes completely on the web

Everything above assumes the GEMM-bound 38% has somewhere better to go. **On the
web it does not.** WGSL has no `f64` — not "slow", absent, and not on the WebGPU
roadmap. So for f64 work in a browser:

| | native (CUDA, f64 GPUs) | web (WGSL, f32 only) |
|---|---|---|
| A + B — iterative | smp.js | smp.js |
| C — BLAS3-bound | GPU | **nowhere else to go** |
| D — memory-bound | CPU either way | CPU either way |

So the honest answer is two numbers:

- **~17% is smp.js's *differentiated* territory** — iterative kernels where AOT
  plus threading is the right tool on any platform, and where no GPU port would
  help even if f64 existed, because the iteration is branch-divergent and
  sequential within a problem.
- **~55% (A + B + C) is *addressable* on the web**, because f64 GEMM has no GPU
  path there. But addressable is not the same as differentiated: a tuned wasm
  GEMM is a blocking-and-packing problem, and someone will eventually ship one.

## What this implies for smp.js

Target the **symmetric and nonsymmetric eigenproblem, the SVD, and iterative
refinement** — categories A and B. That is where the measured profile fits:
branch-divergent, f64-critical, sequential within one problem, parallel across
problems.

The measurements back this up. From `docs/PERFORMANCE.md`:

| kernel | dominated by | codegen | threaded |
|---|---|---:|---:|
| spatial join | arithmetic + branches | 2.42× | 17.39× |
| HSEQR (category B's iteration phase) | f64 algebra + branches | 1.63× | 11.90× |
| geodesic | `sin`/`cos`/`atan2` | 0.95× | 6.98× |

Category A and B routines look like the middle row. Category C looks like a
problem smp.js is not built for, and category D is memory-bound.

There is also a real precedent for the gap: `@stdlib` ships **no symmetric
eigensolver at all** — not published, and not present on the unreleased
`develop` branch. The 17% is not merely addressable; for the web it is largely
unserved.

## Caveats

- The iteration classifier **undercounts** by ~3–5 in LAPACK and at least 1 in
  Eigen (`JacobiSVD`). Treat 17% as a floor.
- "Reaches BLAS3" answers *is it GEMM-shaped*, not *is a GPU faster*. At small
  batched sizes a GEMM is too small to amortise GPU dispatch, which moves some of
  category C toward smp.js — measured in `docs/PERFORMANCE.md` as the batched-vs-
  unbatched distinction.
- Double precision only. Multiply by ~4 for a raw S/D/C/Z routine count; the
  count of distinct *algorithms* to implement does not change.
- Category D being 45% is not a failure of the analysis. Much of LAPACK is
  scaling, equilibration, norms and other O(n²) helpers that are memory-bound in
  any language.

## Reproducing

The classification scripts are throwaway; the inputs are not:

```
Reference LAPACK 3.12.1  https://github.com/Reference-LAPACK/lapack (v3.12.1)
Eigen 3.4.0              https://gitlab.com/libeigen/eigen (3.4.0)
```

Criteria, restated so they can be re-run: transitive call-graph closure from each
`d*.f` to `{DGEMM, DTRSM, DSYRK, DSYR2K, DTRMM, DSYMM}`; and a regex for
convergence-failure semantics inside the `\param[out] INFO` block.
