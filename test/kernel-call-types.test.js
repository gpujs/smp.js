// A kernel calling another kernel must inherit the callee's DECLARED return
// type. Assuming f64 mistypes every local bound to such a call, and the error
// surfaces far away -- as an AssemblyScript cast failure on the line where the
// local is finally used, which is a line with nothing wrong with it. Compiling
// a real symmetric eigensolver is what found this: `const it = tqli(...)` in a
// kernel declared `@returns {i32}`.
//
// Also covers the arena: allocation is a bump pointer, so without reset() a
// caller that runs many batches exhausts wasm memory and the failure reads as a
// capacity problem rather than a leak.

import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { rmSync, writeFileSync, mkdirSync } from "node:fs";
import { createModule } from "../src/runtime/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const OUT = join(ROOT, "test", ".build-calltypes");
const SRC = join(OUT, "calltypes.js");

const SOURCE = `
/**
 * Returns an i32. The caller must not assume f64.
 * @kernel
 * @param {Float64Array} a
 * @param {i32} n
 * @returns {i32}
 */
export function countPositive(a, n) {
  let c = 0;
  for (let i = 0; i < n; i++) if (a[i] > 0.0) c = c + 1;
  return c;
}

/**
 * Binds the i32 result to a local, then returns it. If the local were typed
 * f64 this would not compile.
 * @kernel
 * @param {Float64Array} a
 * @param {i32} n
 * @param {Int32Array} out
 * @returns {i32}
 */
export function useCount(a, n, out) {
  const c = countPositive(a, n);
  if (c < 0) return -1;
  out[0] = c;
  return c;
}
`;

test("a local bound to a kernel call takes the callee's return type", async () => {
  mkdirSync(OUT, { recursive: true });
  writeFileSync(SRC, SOURCE);
  // Would fail with "Conversion from type 'f64' to 'i32' requires an explicit
  // cast" before the fix.
  execFileSync("node", [join(ROOT, "bin", "smp.js"), "build", SRC,
    "--out", OUT, "--no-threads", "--memory", "16"], { stdio: "pipe" });

  const meta = {
    kernels: ["countPositive", "useCount"], parallel: [],
    memory: { initial: 16, maximum: 16, shared: false },
  };
  const mod = await createModule(new URL(`file://${join(OUT, "calltypes.wasm")}`), meta, {});
  const a = mod.alloc.f64(8);
  a.view.set([1, -1, 2, -2, 3, 0, 4, -4]);
  const out = mod.alloc.i32(1);

  assert.equal(mod.kernels.countPositive(a.ptr, 8), 4);
  assert.equal(mod.kernels.useCount(a.ptr, 8, out.ptr), 4);
  assert.equal(out.view[0], 4, "i32 store must round-trip exactly");
  rmSync(OUT, { recursive: true, force: true });
});

test("arena reset makes repeated allocation bounded", async () => {
  const meta = { kernels: [], parallel: [], memory: { initial: 2, maximum: 2, shared: false } };
  mkdirSync(OUT, { recursive: true });
  writeFileSync(SRC, SOURCE);
  execFileSync("node", [join(ROOT, "bin", "smp.js"), "build", SRC,
    "--out", OUT, "--no-threads", "--memory", "2"], { stdio: "pipe" });
  const mod = await createModule(new URL(`file://${join(OUT, "calltypes.wasm")}`), meta, {});

  // 2 pages = 128 KB. Without reset this exhausts after a handful of rounds.
  for (let i = 0; i < 500; i++) {
    mod.alloc.reset();
    const buf = mod.alloc.f64(1024);
    buf.view[0] = i;
    assert.equal(buf.view[0], i);
  }
  rmSync(OUT, { recursive: true, force: true });
});
