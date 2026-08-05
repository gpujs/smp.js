// Pool worker. Runs under Node worker_threads and browser Workers unchanged:
// the only difference is how the bootstrap payload arrives.

import {
  C_GEN, C_DONE, C_CURSOR, C_CHUNK, C_TOTAL, C_FN, C_NARGS, C_QUIT, C_ARGS,
} from "./index.js";

const IS_NODE = typeof process !== "undefined" && !!process.versions?.node;

// Spin before parking. Parking immediately makes `notify` wake k-1 futex-parked
// threads serially, which measured ~27us per barrier at 8 threads against ~0.23us
// for spin-then-park. The park path is still kept for oversubscription, where a
// pure spin would burn a core against a thread that is not scheduled.
const SPIN = 20000;

async function boot({ ctrlBuf, memory, wasmUrl, tid, threads, fnNames }) {
  const ctrl = new Int32Array(ctrlBuf);
  const args = new Float64Array(ctrlBuf, C_ARGS * 4, 8);

  let bytes;
  if (IS_NODE) {
    const { readFile } = await import("node:fs/promises");
    bytes = await readFile(new URL(wasmUrl));
  } else {
    bytes = await (await fetch(wasmUrl)).arrayBuffer();
  }

  const { instance } = await WebAssembly.instantiate(bytes, {
    env: { memory, abort: () => { throw new Error("smp.js: wasm abort in worker"); } },
  });

  // Resolve the range entry points once; a lookup per chunk would land in the
  // measured region.
  const fns = fnNames.map((n) => instance.exports[`${n}__range`]);

  // Signal readiness only after instantiation, so module setup can never be
  // attributed to a dispatch.
  if (IS_NODE) (await import("node:worker_threads")).parentPort.postMessage({ ready: true });
  else self.postMessage({ ready: true });

  let gen = 0;
  for (;;) {
    if (IS_NODE) {
      Atomics.wait(ctrl, C_GEN, gen);
    } else {
      // Browser workers may block on Atomics.wait; spin briefly first so short
      // dispatches do not pay a futex round trip.
      let spun = 0;
      while (Atomics.load(ctrl, C_GEN) === gen) {
        if (++spun > SPIN) { Atomics.wait(ctrl, C_GEN, gen); spun = 0; }
      }
    }
    gen = Atomics.load(ctrl, C_GEN);
    if (Atomics.load(ctrl, C_QUIT) === 1) break;

    const fn = fns[Atomics.load(ctrl, C_FN)];
    const total = Atomics.load(ctrl, C_TOTAL);
    const chunk = Atomics.load(ctrl, C_CHUNK);
    const nargs = Atomics.load(ctrl, C_NARGS);
    const a = [];
    for (let i = 0; i < nargs; i++) a.push(args[i]);

    // Dynamic work stealing. The cursor is touched once per chunk, not once per
    // item, so atomic traffic scales with chunk count.
    if (fn) {
      for (;;) {
        const start = Atomics.add(ctrl, C_CURSOR, chunk);
        if (start >= total) break;
        const end = start + chunk < total ? start + chunk : total;
        fn(...a, start, end);
      }
    }

    Atomics.add(ctrl, C_DONE, 1);
    Atomics.notify(ctrl, C_DONE);
  }
}

if (IS_NODE) {
  const { workerData } = await import("node:worker_threads");
  boot(workerData);
} else {
  self.addEventListener("message", (e) => { boot(e.data); }, { once: true });
}
