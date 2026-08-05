// Vincenty inverse: accurate distance between two points on the WGS84 ellipsoid.
//
// WHY THIS AND NOT A GPU
//
// WGSL has no f64. Not "f64 is slow on the GPU" -- the type does not exist, and
// it is not on the WebGPU roadmap. Vincenty iterates until successive values of
// lambda agree to 1e-12; f32 epsilon is 1.2e-7, so the loop below cannot converge
// in single precision at all. This is not a precision preference, it is a
// correctness floor.
//
// The iteration count is also data-dependent: 4-6 for ordinary pairs, ~20 for
// near-antipodal ones, and Vincenty genuinely fails to converge for some. Lanes
// in a SIMT group would diverge on every pair.
//
// And the parallelism is not staged for the benchmark. Anyone computing travel
// distances has many pairs by construction -- addresses to depots, an
// origin-destination matrix, a nearest-facility query. Batching is what the
// workload already looks like.
//
// This file is ordinary JavaScript. Run it under node with no compiler and it
// works, single-threaded, producing identical results.

/**
 * Distance in metres between two WGS84 coordinates, along the geodesic.
 * Returns -1 when the iteration fails to converge (near-antipodal points),
 * which is the documented limitation of Vincenty's inverse method.
 *
 * @kernel
 * @param {f64} lat1
 * @param {f64} lon1
 * @param {f64} lat2
 * @param {f64} lon2
 * @returns {f64}
 */
export function geodesicDistance(lat1, lon1, lat2, lon2) {
  // WGS84. `b` is written out rather than derived so the JS and wasm paths
  // cannot disagree on how (1-f)*a was folded.
  const a = 6378137.0;
  const f = 1.0 / 298.257223563;
  const b = 6356752.314245179;
  const DEG = 0.017453292519943295;

  const L = (lon2 - lon1) * DEG;
  const U1 = Math.atan((1.0 - f) * Math.tan(lat1 * DEG));
  const U2 = Math.atan((1.0 - f) * Math.tan(lat2 * DEG));
  const sinU1 = Math.sin(U1);
  const cosU1 = Math.cos(U1);
  const sinU2 = Math.sin(U2);
  const cosU2 = Math.cos(U2);

  let lambda = L;
  let lambdaP = 0.0;
  let iter = 0;
  let sinSigma = 0.0;
  let cosSigma = 0.0;
  let sigma = 0.0;
  let cos2Alpha = 0.0;
  let cos2SigmaM = 0.0;

  do {
    const sinLambda = Math.sin(lambda);
    const cosLambda = Math.cos(lambda);

    const t1 = cosU2 * sinLambda;
    const t2 = cosU1 * sinU2 - sinU1 * cosU2 * cosLambda;
    sinSigma = Math.sqrt(t1 * t1 + t2 * t2);

    if (sinSigma === 0.0) return 0.0; // coincident points

    cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda;
    sigma = Math.atan2(sinSigma, cosSigma);

    const sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma;
    cos2Alpha = 1.0 - sinAlpha * sinAlpha;

    // Equatorial lines give cos2Alpha == 0; the term is defined to be zero there.
    cos2SigmaM = cos2Alpha === 0.0 ? 0.0 : cosSigma - 2.0 * sinU1 * sinU2 / cos2Alpha;

    const C = f / 16.0 * cos2Alpha * (4.0 + f * (4.0 - 3.0 * cos2Alpha));
    lambdaP = lambda;
    lambda = L + (1.0 - C) * f * sinAlpha *
      (sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1.0 + 2.0 * cos2SigmaM * cos2SigmaM)));

    iter = iter + 1;
    // 1e-12 is the convergence floor, and it is why this cannot be f32:
    // single-precision epsilon is 1.2e-7, five orders of magnitude too coarse.
  } while (Math.abs(lambda - lambdaP) > 1e-12 && iter < 200);

  if (iter >= 200) return -1.0;

  const u2 = cos2Alpha * (a * a - b * b) / (b * b);
  const A = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)));
  const B = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)));

  const d1 = cosSigma * (-1.0 + 2.0 * cos2SigmaM * cos2SigmaM);
  const d2 = B / 6.0 * cos2SigmaM * (-3.0 + 4.0 * sinSigma * sinSigma) *
    (-3.0 + 4.0 * cos2SigmaM * cos2SigmaM);
  const deltaSigma = B * sinSigma * (cos2SigmaM + B / 4.0 * (d1 - d2));

  return b * A * (sigma - deltaSigma);
}

/**
 * Distances for a whole set of coordinate pairs. One directive is the entire
 * concurrency surface.
 *
 * `dynamic` because the iteration count above is data-dependent: a chunk that
 * happens to draw near-antipodal pairs costs several times its neighbours, and a
 * static split would leave threads waiting on it.
 *
 * @kernel
 * @parallel for schedule(dynamic)
 * @shared lat1, lon1, lat2, lon2
 * @param {Float64Array} lat1
 * @param {Float64Array} lon1
 * @param {Float64Array} lat2
 * @param {Float64Array} lon2
 * @param {Float64Array} out
 * @param {i32} n
 * @returns {void}
 */
export function distanceBatch(lat1, lon1, lat2, lon2, out, n) {
  for (let i = 0; i < n; i++) {
    out[i] = geodesicDistance(lat1[i], lon1[i], lat2[i], lon2[i]);
  }
}
