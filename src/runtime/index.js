// smp.js runtime: memory, allocation, and the worker pool that `@parallel for`
// lowers to.
//
// The pool design is the one measured in the research that preceded this project:
// dynamic work stealing on a shared atomic cursor, workers spawned and warmed
// ONCE outside any measured region, and a spin-then-park barrier. That
// combination reached ~90% parallel efficiency at 8 performance cores on batched
// f64 kernels, with output bit-identical to serial at every thread count.

export const IS_NODE = typeof process !== "undefined" && !!process.versions?.node;

// Control block. Indices are spaced onto separate 64-byte lines wherever two
// threads write them: the steal cursor is hit by every worker once per chunk, and
// sharing a line with the barrier counter turns each claim into a coherence
// round-trip that shows up as scaling loss.
export const C_GEN = 0;
export const C_DONE = 16;
export const C_CURSOR = 32;
export const C_CHUNK = 48;
export const C_TOTAL = 49;
export const C_FN = 50;
export const C_NARGS = 51;
export const C_QUIT = 52;
export const C_ARGS = 64; // f64 args live in a parallel Float64Array view
// 32 slots, not 8. A kernel with more parameters than the control block holds
// used to lose the surplus silently: the dropped pointers arrived as 0 and every
// threaded write landed at address 0 while the timings still looked plausible.
export const MAX_ARGS = 32;
export const C_INTS = 192;

/** ~32 chunks per thread: enough to balance a skewed workload, few enough that
 *  atomic traffic stays proportional to chunk count rather than item count. */
export function chunkFor(total, threads) {
  return Math.max(1, Math.floor(total / (threads * 32)) || 1);
}

async function loadBytes(url) {
  if (IS_NODE) {
    const { readFile } = await import("node:fs/promises");
    return readFile(url);
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`smp.js: failed to fetch ${url}: ${r.status}`);
  return r.arrayBuffer();
}

/** Bump allocator over the wasm linear memory. Kernels never allocate; the host
 *  places every buffer up front, which is what keeps allocation out of any hot
 *  path. */
class Arena {
  constructor(memory, base) {
    this.memory = memory;
    this.offset = base;
  }
  #bump(bytes) {
    const p = (this.offset + 7) & ~7;
    this.offset = p + bytes;
    const cap = this.memory.buffer.byteLength;
    if (this.offset > cap) {
      throw new Error(
        `smp.js: out of wasm memory (need ${this.offset} bytes, have ${cap}). ` +
        `Rebuild with a larger --memory, e.g. npx smp.js build ... --memory ${Math.ceil(this.offset / 65536) + 8}`
      );
    }
    return p;
  }
  f64(n) { const ptr = this.#bump(n * 8); return { ptr, view: new Float64Array(this.memory.buffer, ptr, n) }; }
  f32(n) { const ptr = this.#bump(n * 4); return { ptr, view: new Float32Array(this.memory.buffer, ptr, n) }; }
  i32(n) { const ptr = this.#bump(n * 4); return { ptr, view: new Int32Array(this.memory.buffer, ptr, n) }; }
}

class Pool {
  constructor({ wasmUrl, memory, threads, fnNames, NodeWorker }) {
    this.threads = threads;
    this.NodeWorker = NodeWorker;
    this.ctrlBuf = new SharedArrayBuffer(C_INTS * 4);
    this.ctrl = new Int32Array(this.ctrlBuf);
    this.args = new Float64Array(this.ctrlBuf, C_ARGS * 4, MAX_ARGS);
    this.fnNames = fnNames;
    this.workers = [];

    const workerUrl = new URL("./worker.js", import.meta.url);
    const ready = [];
    for (let tid = 0; tid < threads; tid++) {
      const data = { ctrlBuf: this.ctrlBuf, memory, wasmUrl: String(wasmUrl), tid, threads, fnNames };
      let w;
      if (IS_NODE) {
        // Constructed lazily so the browser build never touches worker_threads.
        w = new this.NodeWorker(workerUrl, { workerData: data });
        ready.push(new Promise((res, rej) => { w.once("message", res); w.once("error", rej); }));
      } else {
        w = new Worker(workerUrl, { type: "module" });
        ready.push(new Promise((res, rej) => {
          w.addEventListener("message", function once(e) { w.removeEventListener("message", once); res(e.data); }, { once: true });
          w.addEventListener("error", rej, { once: true });
        }));
        w.postMessage(data);
      }
      this.workers.push(w);
    }
    this.readyP = Promise.all(ready);
  }

  async ready() { await this.readyP; }

  /** Dispatch and join. Everything bulk lives in shared memory; only small
   *  control words cross here. */
  run(fnIndex, total, args) {
    if (args.length > MAX_ARGS) {
      throw new Error(
        `smp.js: kernel takes ${args.length} arguments, but the pool control block holds ${MAX_ARGS}. ` +
        `Raise MAX_ARGS in src/runtime/index.js and src/runtime/worker.js together.`
      );
    }
    const c = this.ctrl;
    for (let i = 0; i < args.length; i++) this.args[i] = args[i];
    Atomics.store(c, C_NARGS, args.length);
    Atomics.store(c, C_FN, fnIndex);
    Atomics.store(c, C_TOTAL, total);
    Atomics.store(c, C_CHUNK, chunkFor(total, this.threads));
    Atomics.store(c, C_CURSOR, 0);
    Atomics.store(c, C_DONE, 0);
    Atomics.add(c, C_GEN, 1);
    Atomics.notify(c, C_GEN);

    let v = Atomics.load(c, C_DONE);
    while (v < this.threads) {
      // Node's main thread may block on a futex; a browser main thread may not,
      // so it spins. Either way the join never busy-waits against the workers
      // for long, because the dispatcher holds no work of its own.
      if (IS_NODE) Atomics.wait(c, C_DONE, v);
      v = Atomics.load(c, C_DONE);
    }
  }

  async destroy() {
    Atomics.store(this.ctrl, C_QUIT, 1);
    Atomics.add(this.ctrl, C_GEN, 1);
    Atomics.notify(this.ctrl, C_GEN);
    for (const w of this.workers) {
      if (IS_NODE) await w.terminate();
      else w.terminate();
    }
    this.workers.length = 0;
  }
}

/**
 * Instantiate a compiled smp.js module.
 *
 * @param {URL|string} wasmUrl
 * @param {{kernels: string[], parallel: string[], memory: {initial: number, maximum: number, shared: boolean}}} meta
 * @param {{threads?: number}} [options]
 */
export async function createModule(wasmUrl, meta, options = {}) {
  const wantThreads = options.threads ?? 0;
  const shared = !!meta.memory.shared;

  if (wantThreads > 1 && !shared) {
    throw new Error("smp.js: this module was built with --no-threads; rebuild without it to use a pool");
  }
  if (wantThreads > 1 && !IS_NODE && typeof SharedArrayBuffer === "undefined") {
    throw new Error(
      "smp.js: SharedArrayBuffer is unavailable. A browser needs cross-origin isolation:\n" +
      "  Cross-Origin-Opener-Policy: same-origin\n" +
      "  Cross-Origin-Embedder-Policy: require-corp\n" +
      "Check `crossOriginIsolated === true`. Single-threaded calls work without it."
    );
  }

  const memory = new WebAssembly.Memory({
    initial: meta.memory.initial,
    maximum: meta.memory.maximum,
    ...(shared ? { shared: true } : {}),
  });

  const bytes = await loadBytes(wasmUrl);
  const { instance } = await WebAssembly.instantiate(bytes, {
    env: { memory, abort: (m, f, l, c) => { throw new Error(`smp.js: wasm abort at ${l}:${c}`); } },
  });

  // __heap_base is 1024 for a --runtime stub module; everything below is reserved.
  const arena = new Arena(memory, 1024);

  const mod = {
    memory,
    exports: instance.exports,
    alloc: arena,
    kernels: {},
    parallel: {},
    threads: 0,
    async destroy() { if (mod._pool) await mod._pool.destroy(); },
  };

  for (const name of meta.kernels) {
    const f = instance.exports[name];
    if (f) mod.kernels[name] = f;
  }

  if (wantThreads > 1 && meta.parallel.length) {
    // Resolved before construction: the Pool spawns in its constructor, and the
    // browser build must never statically reference node:worker_threads.
    let NodeWorker = null;
    if (IS_NODE) ({ Worker: NodeWorker } = await import("node:worker_threads"));

    const pool = new Pool({
      wasmUrl, memory, threads: wantThreads, fnNames: meta.parallel, NodeWorker,
    });
    mod._pool = pool;
    await pool.ready();
    mod.threads = wantThreads;

    meta.parallel.forEach((name, i) => {
      mod.parallel[name] = (total, ...args) => pool.run(i, total, args);
    });
  } else {
    // Single-threaded fallback: run the whole range in-process. Same results,
    // no SharedArrayBuffer, no cross-origin isolation required.
    for (const name of meta.parallel) {
      const f = instance.exports[`${name}__range`];
      if (f) mod.parallel[name] = (total, ...args) => f(...args, 0, total);
    }
  }

  return mod;
}
