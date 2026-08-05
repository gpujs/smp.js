// HSEQR -- eigenvalues of an upper Hessenberg matrix by Francis double-shift QR.
//
// This is the LAPACK routine a GPU cannot take, for four independent reasons
// that are all visible in the code below:
//
//   1. f64 or nothing. The deflation test compares a subdiagonal against the
//      machine epsilon of its neighbours. At f32 epsilon (1.2e-7) tight
//      eigenvalue clusters never separate. WGSL has no f64 and it is not
//      scheduled.
//   2. Branch-divergent. Every sweep searches for a deflation point, tests two
//      block sizes, and fires an exceptional shift on a fixed iteration
//      schedule. Neighbouring matrices in a batch take wildly different paths --
//      the worst case for SIMT lockstep.
//   3. Sequential within one matrix. The bulge chase at step k consumes what
//      step k-1 produced. There is no parallelism inside a single problem.
//   4. Latency-bound. A 64x64 problem is ~32 KB and lives in L1; there is
//      nothing for memory bandwidth to hide.
//
// So the parallelism has to come from batching ACROSS matrices, which is exactly
// what `@parallel for` expresses in five lines at the bottom of this file.
//
// This file is ordinary JavaScript. Run it under node with no compiler and it
// works, single-threaded, producing identical results.

/**
 * Eigenvalues of upper Hessenberg `h` (n x n, row-major). `h` is DESTROYED.
 * Real parts land in wr, imaginary parts in wi.
 *
 * Offsets rather than slices: a kernel receives base pointers into wasm memory,
 * and `.subarray()` has no meaning there. The same signature works unchanged
 * when this file runs as plain JavaScript.
 *
 * @kernel
 * @param {Float64Array} h
 * @param {i32} hOff
 * @param {i32} n
 * @param {Float64Array} wr
 * @param {i32} wrOff
 * @param {Float64Array} wi
 * @param {i32} wiOff
 * @param {i32} maxIter
 * @returns {i32} total QR iterations, or -1 if a block failed to converge
 */
export function runHSEQR(h, hOff, n, wr, wrOff, wi, wiOff, maxIter) {
  const EPS = 2.220446049250313e-16;

  let nn = 0, m = 0, l = 0, k = 0, j = 0, its = 0, i = 0, mmin = 0;
  let z = 0.0, y = 0.0, x = 0.0, w = 0.0, v = 0.0, u = 0.0;
  let t = 0.0, s = 0.0, r = 0.0, q = 0.0, p = 0.0;
  let iters = 0;
  let anorm = 0.0;

  for (i = 0; i < n; i++) {
    for (j = i - 1 < 0 ? 0 : i - 1; j < n; j++) anorm += Math.abs(h[hOff + i * n + j]);
  }

  nn = n - 1;
  t = 0.0;

  while (nn >= 0) {
    its = 0;
    do {
      // Deflation search. Wholly data-dependent -- this is where SIMT lanes
      // diverge immediately.
      for (l = nn; l >= 1; l--) {
        s = Math.abs(h[hOff + (l - 1) * n + l - 1]) + Math.abs(h[hOff + l * n + l]);
        if (s === 0.0) s = anorm;
        if (Math.abs(h[hOff + l * n + l - 1]) <= EPS * s) {
          h[hOff + l * n + l - 1] = 0.0;
          break;
        }
      }
      if (l < 0) l = 0;

      x = h[hOff + nn * n + nn];

      if (l === nn) {
        wr[wrOff + nn] = x + t;
        wi[wiOff + nn] = 0.0;
        nn = nn - 1;
      } else {
        y = h[hOff + (nn - 1) * n + nn - 1];
        w = h[hOff + nn * n + nn - 1] * h[hOff + (nn - 1) * n + nn];

        if (l === nn - 1) {
          // A 2x2 block: a real pair, or a complex conjugate pair.
          p = 0.5 * (y - x);
          q = p * p + w;
          z = Math.sqrt(Math.abs(q));
          x = x + t;
          if (q >= 0.0) {
            z = p + (p >= 0.0 ? Math.abs(z) : -Math.abs(z));
            wr[wrOff + nn - 1] = x + z;
            wr[wrOff + nn] = x + z;
            if (z !== 0.0) wr[wrOff + nn] = x - w / z;
            wi[wiOff + nn - 1] = 0.0;
            wi[wiOff + nn] = 0.0;
          } else {
            wr[wrOff + nn - 1] = x + p;
            wr[wrOff + nn] = x + p;
            wi[wiOff + nn] = z;
            wi[wiOff + nn - 1] = -z;
          }
          nn = nn - 2;
        } else {
          if (its === maxIter) return -1;

          // Exceptional shift on a fixed schedule: taken by roughly one matrix
          // in a batch and not its neighbours.
          if (its === 10 || its === 20) {
            t = t + x;
            for (i = 0; i <= nn; i++) h[hOff + i * n + i] = h[hOff + i * n + i] - x;
            s = Math.abs(h[hOff + nn * n + nn - 1]) + Math.abs(h[hOff + (nn - 1) * n + nn - 2]);
            x = 0.75 * s;
            y = x;
            w = -0.4375 * s * s;
          }
          its = its + 1;
          iters = iters + 1;

          for (m = nn - 2; m >= l; m--) {
            z = h[hOff + m * n + m];
            r = x - z;
            s = y - z;
            p = (r * s - w) / h[hOff + (m + 1) * n + m] + h[hOff + m * n + m + 1];
            q = h[hOff + (m + 1) * n + m + 1] - z - r - s;
            r = h[hOff + (m + 2) * n + m + 1];
            s = Math.abs(p) + Math.abs(q) + Math.abs(r);
            p = p / s;
            q = q / s;
            r = r / s;
            if (m === l) break;
            u = Math.abs(h[hOff + m * n + m - 1]) * (Math.abs(q) + Math.abs(r));
            v = Math.abs(p) * (Math.abs(h[hOff + (m - 1) * n + m - 1]) + Math.abs(z) + Math.abs(h[hOff + (m + 1) * n + m + 1]));
            if (u <= EPS * v) break;
          }

          for (i = m + 2; i <= nn; i++) {
            h[hOff + i * n + i - 2] = 0.0;
            if (i !== m + 2) h[hOff + i * n + i - 3] = 0.0;
          }

          // Bulge chase. Step k strictly depends on step k-1.
          for (k = m; k <= nn - 1; k++) {
            if (k !== m) {
              p = h[hOff + k * n + k - 1];
              q = h[hOff + (k + 1) * n + k - 1];
              r = 0.0;
              if (k !== nn - 1) r = h[hOff + (k + 2) * n + k - 1];
              x = Math.abs(p) + Math.abs(q) + Math.abs(r);
              if (x !== 0.0) {
                p = p / x;
                q = q / x;
                r = r / x;
              }
            }
            s = Math.sqrt(p * p + q * q + r * r);
            s = p >= 0.0 ? Math.abs(s) : -Math.abs(s);
            if (s !== 0.0) {
              if (k === m) {
                if (l !== m) h[hOff + k * n + k - 1] = -h[hOff + k * n + k - 1];
              } else {
                h[hOff + k * n + k - 1] = -s * x;
              }
              p = p + s;
              x = p / s;
              y = q / s;
              z = r / s;
              q = q / p;
              r = r / p;

              for (j = k; j <= nn; j++) {
                p = h[hOff + k * n + j] + q * h[hOff + (k + 1) * n + j];
                if (k !== nn - 1) {
                  p = p + r * h[hOff + (k + 2) * n + j];
                  h[hOff + (k + 2) * n + j] = h[hOff + (k + 2) * n + j] - p * z;
                }
                h[hOff + (k + 1) * n + j] = h[hOff + (k + 1) * n + j] - p * y;
                h[hOff + k * n + j] = h[hOff + k * n + j] - p * x;
              }

              mmin = nn < k + 3 ? nn : k + 3;
              for (i = l; i <= mmin; i++) {
                p = x * h[hOff + i * n + k] + y * h[hOff + i * n + k + 1];
                if (k !== nn - 1) {
                  p = p + z * h[hOff + i * n + k + 2];
                  h[hOff + i * n + k + 2] = h[hOff + i * n + k + 2] - p * r;
                }
                h[hOff + i * n + k + 1] = h[hOff + i * n + k + 1] - p * q;
                h[hOff + i * n + k] = h[hOff + i * n + k] - p;
              }
            }
          }
        }
      }
    } while (l < nn - 1 && nn >= 0);
  }

  return iters;
}

/**
 * Solve a batch. One directive is the entire concurrency surface.
 *
 * `dynamic` is doing real work here: QR iteration count is data-dependent, so
 * matrices cost different amounts and a static split would strand threads
 * behind whichever chunk drew the slow ones.
 *
 * Every iteration writes to its own slice of `scratch`, keyed off the loop
 * variable. That is race-free by construction and needs no per-thread storage --
 * worth preferring over `@private` even once `@private` is lowered.
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
 * @returns {void}
 */
export function runBatch(mats, n, wrOut, wiOut, scratch, K) {
  for (let b = 0; b < K; b++) {
    const base = b * n * n;
    for (let i = 0; i < n * n; i++) scratch[base + i] = mats[base + i];
    runHSEQR(scratch, base, n, wrOut, b * n, wiOut, b * n, 60);
  }
}
