# Architecture: why smp.js is not a fork of AssemblyScript

**Neither. It is a source-to-source transpiler that *emits* AssemblyScript and
hands off to the stock `asc` toolchain.** AssemblyScript is a backend you depend
on, not a codebase you fork.

This matches the handoff's own Stage 4 scope — *"Restricted JS subset →
AssemblyScript → Binaryen → wasm with shared memory"* — and this study produced
the evidence that it is sufficient.

```
  annotated JS  (mockup/variant-a-*.js)   <- your users write this
        │
        │  YOUR COMPILER: parse, check directives, lower
        ▼
  AssemblyScript source                    <- stage3/assembly/*.ts
        │
        │  stock `asc` (unmodified)
        ▼
  Binaryen  ->  wasm + shared memory
```

## The evidence

Every kernel in this repo was written in **stock AssemblyScript 0.28.20 and
compiled with an unmodified `npx asc`**. No fork, no patch, no plugin:

| kernel | stock AS features used |
|---|---|
| `w1.ts` | `load<f64>`, `store<f64>` |
| `w3.ts` (threaded barrier) | `atomic.add/load/store/wait/notify` |
| `w4.ts` (HSEQR) | `load<f64>`, `store<f64>` |
| `w4simd.ts` | `v128.load/store`, `f64x2.add/mul/sub/splat` |
| `w4mask.ts` | `v128.bitselect`, `i64x2.splat` |

That covers everything the compiler has to emit: raw linear memory, an in-wasm
sense-reversing barrier, f64x2 vectorisation, and bitselect masking. **The target
language already expresses all of it.** There is no observed reason to fork.

## The three options, and why the middle one wins

### A. Transpile to AssemblyScript source ✅

- **You own:** a parser for the JS subset, JSDoc directive extraction, the
  `@simd` stride/dependence checks, unswitch-or-mask selection, `@parallel for`
  lowering to a pool runtime, and AS emission.
- **You get free:** type checking of your generated code, Binaryen's optimiser,
  wasm ABI and memory layout, and a *readable* intermediate you can eyeball when
  codegen misbehaves.
- **Cost:** diagnostics must be mapped back from generated AS to the user's JS.
  Real work, but well-understood — keep source positions on every emitted node.

### B. Fork or extend AssemblyScript ❌

Buys IR-level access and lets errors surface natively. Costs a permanent merge
burden against upstream. Justified only if you need to change AS's *type system*
or *syntax* — and Variant A deliberately needs neither, because the directives
are comments that never reach `asc` at all. They are consumed by your pass and
lowered into ordinary AS constructs.

### C. Ground up ❌ (for a first cut)

You would reimplement a type checker and Binaryen bindings for no measured gain.

## The honest caveat: how much is AS actually buying?

Worth being clear-eyed. The constraints this study established mean you use
almost none of AssemblyScript's value-add:

| AS feature | usable here? |
|---|---|
| managed objects / GC | **No** — no thread-safe GC (0 atomics in `rt/tlsf`, `rt/itcms`, `rt/tcms`, `rt/stub`); `--runtime stub` only |
| closures | **No** — `ERROR AS100: Not implemented: Closures` |
| stdlib | **No** — kernels live in raw linear memory |
| classes, generics | Not needed — kernels are scalar loops |
| **type checker** | **Yes** — catches codegen bugs |
| **Binaryen wiring** | **Yes** — `--optimizeLevel 3`, wasm emission |

So AS is buying a type checker, a readable intermediate, and Binaryen plumbing.
That is genuinely useful for developing codegen, but it is not load-bearing
architecturally. A later version could emit **Binaryen IR directly** via the
`binaryen` npm package (which is what `asc` itself depends on) and lose nothing
but the readable intermediate.

Recommendation: **emit AS source for v1** — fastest path, and when your generated
code is wrong you can read it. Treat direct-to-Binaryen as an optimisation to
take only if AS becomes a constraint. Nothing in this study suggests it will.

## What you are actually building

Not a language implementation. A **directive-lowering pass** over a restricted JS
subset:

1. Parse the JS subset; read JSDoc types and directives.
2. Monomorphise — emit `{f32,f64} × {row-major,col-major}` variants at build
   time rather than dispatching on layout at access time.
3. `@parallel for` → a work-stealing pool over SharedArrayBuffer.
4. `@simd` → verify unit stride and absence of cross-iteration dependence, choose
   unswitch (≤2 invariant branches) or bitselect masking, emit `v128`, generate
   the tail.
5. `@shared` / `@private` → SAB layout and per-thread scratch hoisting.
6. Emit AssemblyScript; invoke `asc` with
   `--enable threads --enable simd --importMemory --sharedMemory --runtime stub`.
7. Map diagnostics back to the original JS positions.

Steps 3–5 are the whole product. Steps 1, 2, 6 are plumbing. Step 7 is what makes
it pleasant to use.
