// Spatial join: which zone does each point fall in?
//
// A million GPS pings against a few hundred delivery zones. Geofencing,
// catchment analysis, "which census tract / postcode / service area is this" --
// this is one of the most-run computations in anything location-shaped, and the
// batch is inherent: you always have many points.
//
// WHY NOT A GPU
//
// Ray casting is SIMT-hostile in three independent ways, all visible below:
//   1. Per-point early exit. The moment one point finds its zone it stops; its
//      neighbour in the same warp keeps scanning. Divergence on every point.
//   2. Wildly variable polygon sizes. Real administrative boundaries run from
//      4 vertices to tens of thousands, so the inner loop trip count is
//      data-dependent per polygon AND per point.
//   3. Irregular, pointer-chasing memory access into a ragged vertex array.
//
// Coordinates are f64 throughout. f32 gives roughly metre-level error on lat/lon,
// which is exactly the scale at which "inside or outside this boundary" is being
// decided, and error compounds through any reprojection done first.
//
// This file is ordinary JavaScript. Run it under node with no compiler and it
// works, single-threaded, producing identical results.

/**
 * Ray-casting point-in-polygon. Returns 1 if inside, 0 if outside.
 *
 * Vertices are read from a shared ragged array at `vOff`, so all polygons live
 * in one flat allocation rather than an array of arrays -- there is no such thing
 * as an array of arrays in wasm linear memory.
 *
 * @kernel
 * @param {f64} px
 * @param {f64} py
 * @param {Float64Array} verts
 * @param {i32} vOff
 * @param {i32} nVerts
 * @returns {i32}
 */
export function pointInPolygon(px, py, verts, vOff, nVerts) {
  let inside = 0;
  let j = nVerts - 1;
  for (let i = 0; i < nVerts; i++) {
    const xi = verts[vOff + i * 2];
    const yi = verts[vOff + i * 2 + 1];
    const xj = verts[vOff + j * 2];
    const yj = verts[vOff + j * 2 + 1];
    // Crossing test: does the upward ray from (px,py) cross edge j->i?
    if (((yi > py) !== (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
      inside = 1 - inside;
    }
    j = i;
  }
  return inside;
}

/**
 * Assign every point to the first zone containing it, or -1 for none.
 * One directive is the entire concurrency surface.
 *
 * `dynamic` earns its place here: a point inside the first zone tested exits
 * immediately, while a point outside every zone scans all of them. Cost per
 * iteration varies by orders of magnitude, so a static split would leave threads
 * waiting on whichever chunk drew the misses.
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
 * @returns {void}
 */
export function assignZones(px, py, n, verts, polyOff, polyLen, bbox, nPoly, out) {
  for (let i = 0; i < n; i++) {
    const x = px[i];
    const y = py[i];
    let found = -1;
    for (let p = 0; p < nPoly; p++) {
      // Bounding-box reject first: cheap, and it is what makes the per-point cost
      // so uneven.
      if (x < bbox[p * 4] || x > bbox[p * 4 + 1] || y < bbox[p * 4 + 2] || y > bbox[p * 4 + 3]) continue;
      if (pointInPolygon(x, y, verts, polyOff[p], polyLen[p]) === 1) {
        found = p;
        break;
      }
    }
    out[i] = found;
  }
}
