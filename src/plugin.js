// Bundler plugin for Rolldown, Rollup and Vite.
//
// These three share the same plugin object shape, so one implementation covers
// all of them. Vite-only hooks (`config`, `configureServer`) are simply ignored
// by Rollup and Rolldown.
//
//   // vite.config.js / rolldown.config.js / rollup.config.js
//   import smp from "smp.js/plugin";
//   export default { plugins: [smp()] };
//
// Then import the annotated source directly:
//
//   import { load } from "./kernels.js?smp";
//   const mod = await load({ threads: navigator.hardwareConcurrency });

import { readFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, basename } from "node:path";
import { toAssemblyScript, runAsc, DEFAULTS } from "./compile.js";

const QUERY = "smp";

function matches(id, include, exclude) {
  const clean = id.split("?")[0];
  if (exclude?.some((re) => re.test(clean))) return false;
  if (include?.length) return include.some((re) => re.test(clean));
  return false;
}

/**
 * @param {object} [options]
 * @param {RegExp[]} [options.include]  default: files ending in .smp.js / .smp.mjs
 * @param {RegExp[]} [options.exclude]  default: node_modules
 * @param {boolean}  [options.threads]  emit shared memory + atomics (default true)
 * @param {boolean}  [options.simd]     enable SIMD (default true)
 * @param {number}   [options.memory]   initial = maximum pages (default 256)
 * @param {boolean}  [options.crossOriginIsolation]  Vite dev server: send COOP/COEP (default true)
 */
export default function smp(options = {}) {
  const include = options.include ?? [/\.smp\.m?js$/];
  const exclude = options.exclude ?? [/node_modules/];
  const opts = {
    ...DEFAULTS,
    threads: options.threads ?? true,
    simd: options.simd ?? true,
    initialMemory: options.memory ?? DEFAULTS.initialMemory,
    maximumMemory: options.memory ?? DEFAULTS.maximumMemory,
  };
  const isolate = options.crossOriginIsolation ?? true;

  let tmp;

  return {
    name: "smp.js",
    enforce: "pre",

    buildStart() {
      tmp = mkdtempSync(join(tmpdir(), "smp-"));
    },

    // Vite dev server: SharedArrayBuffer requires cross-origin isolation, and
    // forgetting the headers is the single most common way a threaded build
    // fails -- silently, with SharedArrayBuffer simply undefined. Set them here
    // so `crossOriginIsolated` is true out of the box.
    configureServer(server) {
      if (!isolate || !opts.threads) return;
      server.middlewares.use((_req, res, next) => {
        res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
        res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
        next();
      });
    },
    configurePreviewServer(server) {
      if (!isolate || !opts.threads) return;
      server.middlewares.use((_req, res, next) => {
        res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
        res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
        next();
      });
    },

    async transform(code, id) {
      const [path, query] = id.split("?");
      const explicit = query?.split("&").includes(QUERY);
      if (!explicit && !matches(id, include, exclude)) return null;
      if (!/@kernel\b/.test(code)) return null;

      const filename = basename(path);
      const result = toAssemblyScript(code, filename);
      if (!result) return null;

      for (const w of result.warnings) this.warn(w);

      const stem = filename.replace(/\.[cm]?js$/, "");
      const asPath = join(tmp, `${stem}.ts`);
      const wasmPath = join(tmp, `${stem}.wasm`);
      const { writeFileSync } = await import("node:fs");
      writeFileSync(asPath, result.code);

      const asc = runAsc(asPath, wasmPath, opts);
      if (!asc.ok) {
        this.error(`smp.js: asc failed for ${filename}\n${asc.stderr || asc.stdout}`);
        return null;
      }

      // Emit the wasm as an asset so the bundler owns hashing and the final URL.
      const ref = this.emitFile({
        type: "asset",
        name: `${stem}.wasm`,
        source: readFileSync(wasmPath),
      });

      const names = result.kernels.map((k) => k.fn.id.name);
      const parallel = result.kernels.filter((k) => k.directives.parallel).map((k) => k.fn.id.name);

      return {
        code: `import { createModule } from "smp.js/runtime";

export const wasmUrl = new URL(import.meta.ROLLUP_FILE_URL_${ref}, import.meta.url);

export const meta = {
  kernels: ${JSON.stringify(names)},
  parallel: ${JSON.stringify(parallel)},
  memory: { initial: ${opts.initialMemory}, maximum: ${opts.maximumMemory}, shared: ${!!opts.threads} },
};

export function load(options = {}) { return createModule(wasmUrl, meta, options); }
export default load;
`,
        map: null,
      };
    },
  };
}

export { smp };
