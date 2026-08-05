// smp.js command line.
//
//   npx smp.js build src/kernels.js --out build
//   npx smp.js emit  src/kernels.js          # show the generated AssemblyScript
//   npx smp.js check src/kernels.js          # directives only, no asc

import { readFileSync } from "node:fs";
import { relative } from "node:path";
import { compileFile, toAssemblyScript, analyze, DEFAULTS } from "./compile.js";
import { Diag } from "./emit-as.js";

const USAGE = `smp.js -- compile annotated JavaScript to threaded, SIMD WebAssembly

USAGE
  smp.js build <file...> [options]     compile to wasm + an ES module loader
  smp.js emit  <file>                  print the generated AssemblyScript
  smp.js check <file...>               validate directives, emit nothing

OPTIONS
  --out <dir>            output directory        (default: <input>/smp-build)
  --no-threads           omit shared memory and atomics
  --no-simd              omit SIMD
  --optimize <0-3>       asc optimize level      (default: ${DEFAULTS.optimizeLevel})
  --memory <pages>       initial = maximum pages (default: ${DEFAULTS.initialMemory})
  --verbose              show the asc invocation
  -h, --help             this text
  -v, --version

Directives are JSDoc comments, so an annotated file still runs as plain
JavaScript with no compiler involved. See README.md.`;

function parseArgs(argv) {
  const o = { cmd: null, files: [], out: null, verbose: false, opts: {} };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!o.cmd && !a.startsWith("-")) { o.cmd = a; continue; }
    switch (a) {
      case "--out": o.out = argv[++i]; break;
      case "--no-threads": o.opts.threads = false; break;
      case "--no-simd": o.opts.simd = false; break;
      case "--optimize": o.opts.optimizeLevel = Number(argv[++i]); break;
      case "--memory": {
        const p = Number(argv[++i]);
        o.opts.initialMemory = p; o.opts.maximumMemory = p;
        break;
      }
      case "--verbose": o.verbose = true; break;
      case "-h": case "--help": o.help = true; break;
      case "-v": case "--version": o.version = true; break;
      default:
        if (a.startsWith("-")) { o.bad = a; }
        else o.files.push(a);
    }
  }
  return o;
}

const rel = (p) => relative(process.cwd(), p) || p;

export async function main(argv) {
  const a = parseArgs(argv);

  if (a.version) {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
    console.log(pkg.version);
    return 0;
  }
  if (a.help || !a.cmd) { console.log(USAGE); return a.cmd ? 0 : 1; }
  if (a.bad) { console.error(`unknown option ${a.bad}\n`); console.log(USAGE); return 1; }
  if (!a.files.length) { console.error(`${a.cmd}: no input files\n`); console.log(USAGE); return 1; }

  try {
    switch (a.cmd) {
      case "emit": {
        const src = readFileSync(a.files[0], "utf8");
        const r = toAssemblyScript(src, rel(a.files[0]));
        if (!r) { console.error(`no @kernel functions in ${a.files[0]}`); return 1; }
        for (const w of r.warnings) console.error(`warning: ${w}`);
        process.stdout.write(r.code);
        return 0;
      }

      case "check": {
        let bad = 0;
        for (const f of a.files) {
          const { kernels } = analyze(readFileSync(f, "utf8"), rel(f));
          if (!kernels.length) { console.log(`${rel(f)}: no kernels`); continue; }
          for (const k of kernels) {
            const d = k.directives;
            const tags = [
              d.parallel ? `@parallel for schedule(${d.parallel.schedule})` : null,
              d.simd ? `@simd${d.simd.simdlen ? ` simdlen(${d.simd.simdlen})` : ""}` : null,
              d.shared.length ? `@shared ${d.shared.join(",")}` : null,
              d.private.length ? `@private ${d.private.join(",")}` : null,
            ].filter(Boolean);
            console.log(`${rel(f)}: kernel ${k.fn.id.name}(${d.params.map((p) => `${p.name}: ${p.type}`).join(", ")}) -> ${d.returns ?? "void"}`);
            for (const t of tags) console.log(`    ${t}`);
            for (const u of d.unknown) { console.error(`    warning: unknown directive '@${u}'`); bad++; }
          }
          // Surface emission errors during `check` too -- that is the point of it.
          toAssemblyScript(readFileSync(f, "utf8"), rel(f));
        }
        return bad ? 1 : 0;
      }

      case "build": {
        let failed = 0;
        for (const f of a.files) {
          const r = compileFile(f, { outDir: a.out, ...a.opts });
          for (const w of r.warnings ?? []) console.error(`warning: ${w}`);
          if (r.skipped) { console.log(`${rel(f)}: skipped (no kernels)`); continue; }
          if (!r.ok) {
            console.error(`${rel(f)}: asc failed\n${r.stderr}`);
            console.error(`generated AssemblyScript kept at ${rel(r.as)} for inspection`);
            failed++;
            continue;
          }
          console.log(`${rel(f)} -> ${rel(r.wasm)}  [${r.kernels.join(", ")}]`);
          console.log(`  loader: ${rel(r.loader)}`);
          if (a.verbose) console.log(`  as:     ${rel(r.as)}`);
        }
        return failed ? 1 : 0;
      }

      default:
        console.error(`unknown command '${a.cmd}'\n`);
        console.log(USAGE);
        return 1;
    }
  } catch (e) {
    if (e instanceof Diag) {
      console.error(`error: ${e.message}`);
      if (e.help) console.error(`  help: ${e.help}`);
      return 1;
    }
    console.error(e.stack || String(e));
    return 1;
  }
}
