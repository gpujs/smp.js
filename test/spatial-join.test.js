// The flagship example, and the regression test for argument-slot truncation.
//
// assignZones takes NINE parameters. The pool control block used to hold eight
// and silently drop the rest, so the output pointer arrived as 0 and every
// threaded write landed at address 0 -- with entirely plausible timings. Any
// kernel here with more than eight parameters guards that.

import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { rmSync } from "node:fs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const OUT = join(ROOT, "test", ".build-spatial");

function fixture(nPoints = 20000, nPoly = 40, nv = 24) {
  let a = 424242 >>> 0;
  const rng = () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const verts = [], polyOff = new Int32Array(nPoly), polyLen = new Int32Array(nPoly);
  const bbox = new Float64Array(nPoly * 4);
  for (let p = 0; p < nPoly; p++) {
    const cx = rng() * 360 - 180, cy = rng() * 140 - 70, r = 0.5 + rng() * 3;
    polyOff[p] = verts.length; polyLen[p] = nv;
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (let v = 0; v < nv; v++) {
      const th = 2 * Math.PI * v / nv, rr = r * (0.6 + 0.8 * rng());
      const x = cx + rr * Math.cos(th), y = cy + rr * Math.sin(th);
      verts.push(x, y);
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    bbox[p * 4] = x0; bbox[p * 4 + 1] = x1; bbox[p * 4 + 2] = y0; bbox[p * 4 + 3] = y1;
  }
  const px = new Float64Array(nPoints), py = new Float64Array(nPoints);
  for (let i = 0; i < nPoints; i++) {
    if (rng() < 0.6) {
      const p = (rng() * nPoly) | 0;
      px[i] = bbox[p * 4] + rng() * (bbox[p * 4 + 1] - bbox[p * 4]);
      py[i] = bbox[p * 4 + 2] + rng() * (bbox[p * 4 + 3] - bbox[p * 4 + 2]);
    } else { px[i] = rng() * 360 - 180; py[i] = rng() * 140 - 70; }
  }
  return { px, py, verts: Float64Array.from(verts), polyOff, polyLen, bbox, nPoly, nPoints };
}

test("point-in-polygon: a square, checked by hand", async () => {
  const { pointInPolygon } = await import(join(ROOT, "examples", "spatial-join.js"));
  const sq = Float64Array.from([0, 0, 10, 0, 10, 10, 0, 10]);
  assert.equal(pointInPolygon(5, 5, sq, 0, 4), 1, "centre is inside");
  assert.equal(pointInPolygon(-1, 5, sq, 0, 4), 0, "left of it is outside");
  assert.equal(pointInPolygon(11, 5, sq, 0, 4), 0, "right of it is outside");
  assert.equal(pointInPolygon(5, 11, sq, 0, 4), 0, "above it is outside");
  assert.equal(pointInPolygon(9.999, 9.999, sq, 0, 4), 1, "just inside the corner");
});

test("spatial join: 9-argument kernel is correct at every thread count", async () => {
  rmSync(OUT, { recursive: true, force: true });
  execFileSync(process.execPath, [
    join(ROOT, "bin", "smp.js"), "build", join(ROOT, "examples", "spatial-join.js"),
    "--out", OUT, "--memory", "64",
  ], { encoding: "utf8" });

  const f = fixture();
  const js = await import(join(ROOT, "examples", "spatial-join.js"));
  const expected = new Int32Array(f.nPoints);
  js.assignZones(f.px, f.py, f.nPoints, f.verts, f.polyOff, f.polyLen, f.bbox, f.nPoly, expected);

  const hits = expected.reduce((c, v) => c + (v >= 0 ? 1 : 0), 0);
  assert.ok(hits > f.nPoints * 0.1, `fixture is degenerate: only ${hits} hits`);
  assert.ok(hits < f.nPoints * 0.9, `fixture is degenerate: ${hits} hits of ${f.nPoints}`);

  const { load } = await import(join(OUT, "spatial-join.js"));

  for (const threads of [0, 2, 4, 8]) {
    const mod = await load(threads > 1 ? { threads } : {});
    const px = mod.alloc.f64(f.nPoints), py = mod.alloc.f64(f.nPoints);
    const verts = mod.alloc.f64(f.verts.length);
    const off = mod.alloc.i32(f.nPoly), len = mod.alloc.i32(f.nPoly);
    const bbox = mod.alloc.f64(f.nPoly * 4), out = mod.alloc.i32(f.nPoints);
    px.view.set(f.px); py.view.set(f.py); verts.view.set(f.verts);
    off.view.set(f.polyOff); len.view.set(f.polyLen); bbox.view.set(f.bbox);
    // -999 is not a value the kernel can produce, so an untouched slot is visible.
    out.view.fill(-999);

    const args = [px.ptr, py.ptr, f.nPoints, verts.ptr, off.ptr, len.ptr, bbox.ptr, f.nPoly, out.ptr];
    if (threads > 1) mod.parallel.assignZones(f.nPoints, ...args);
    else mod.kernels.assignZones(...args);

    const label = threads > 1 ? `${threads} threads` : "1 thread";
    let bad = 0;
    for (let i = 0; i < f.nPoints; i++) if (out.view[i] !== expected[i]) bad++;
    assert.equal(bad, 0, `${label}: ${bad}/${f.nPoints} points assigned differently than the JS run`);
    assert.ok(![...out.view].includes(-999), `${label}: some points were never processed`);

    await mod.destroy();
  }
});
