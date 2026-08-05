// A parameter sweep: N independent backtests over one shared f64 price series.
//
// This file is ORDINARY JAVASCRIPT. Run it under node with no compiler and it
// works, single-threaded. Run it through smp.js and the same source becomes
// threaded WebAssembly. The directives are comments; that is the whole trick.

/**
 * One backtest: a dual moving-average crossover with stop-loss and take-profit.
 * A branchy sequential state machine -- no parallelism inside a single run, which
 * is exactly why the parallelism has to come from sweeping many of them.
 *
 * @kernel
 * @param {Float64Array} prices
 * @param {i32} T
 * @param {i32} fast
 * @param {i32} slow
 * @param {f64} slPct
 * @param {f64} tpPct
 * @returns {f64}
 */
export function runBacktest(prices, T, fast, slow, slPct, tpPct) {
  if (fast >= slow || slow > T || fast < 1) return 1.0;

  // Rolling sums: O(T) per backtest. Re-summing each window would be O(T*window)
  // and would make every speedup downstream look better than it is.
  let fastSum = 0.0;
  let slowSum = 0.0;
  for (let i = 0; i < slow; i++) {
    const p = prices[i];
    slowSum += p;
    if (i >= slow - fast) fastSum += p;
  }

  const invFast = 1.0 / fast;
  const invSlow = 1.0 / slow;

  let position = 0;
  let entryPrice = 0.0;
  let equity = 1.0;
  let slLevel = 0.0;
  let tpLevel = 0.0;
  let prevAbove = 0;

  for (let i = slow - 1; i < T; i++) {
    if (i > slow - 1) {
      const p = prices[i];
      slowSum += p - prices[i - slow];
      fastSum += p - prices[i - fast];
    }

    const fastMA = fastSum * invFast;
    const slowMA = slowSum * invSlow;
    const price = prices[i];
    const above = fastMA > slowMA ? 1 : 0;

    if (position === 0) {
      if (above === 1 && prevAbove === 0) {
        position = 1;
        entryPrice = price;
        slLevel = price * (1.0 - slPct);
        tpLevel = price * (1.0 + tpPct);
      }
    } else if (price <= slLevel || price >= tpLevel || fastMA < slowMA) {
      equity *= price / entryPrice;
      position = 0;
    }

    prevAbove = above;
  }

  if (position === 1) equity *= prices[T - 1] / entryPrice;
  return equity;
}

/**
 * Sweep the parameter grid. One directive is the entire concurrency surface.
 *
 * `dynamic` because per-run cost varies with the parameters -- a static split
 * strands threads behind whichever chunk drew the slow tuples.
 *
 * @kernel
 * @parallel for schedule(dynamic)
 * @shared prices, params
 * @param {Float64Array} prices
 * @param {i32} T
 * @param {Float64Array} params
 * @param {i32} nRuns
 * @param {Float64Array} out
 * @returns {void}
 */
export function runSweep(prices, T, params, nRuns, out) {
  for (let r = 0; r < nRuns; r++) {
    const b = r * 4;
    out[r] = runBacktest(prices, T, params[b], params[b + 1], params[b + 2], params[b + 3]);
  }
}
