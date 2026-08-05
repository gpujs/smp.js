// smp.js public API.
//
//   import { compileFile, toAssemblyScript } from "smp.js";        // compiler
//   import smp from "smp.js/plugin";                               // bundler plugin
//   import { createModule } from "smp.js/runtime";                 // runtime

export { compileFile, toAssemblyScript, analyze, runAsc, DEFAULTS } from "./compile.js";
export { Diag } from "./emit-as.js";
export { parseDirectiveComment, TYPES, ARRAY_ELEM } from "./directives.js";
export { default as plugin } from "./plugin.js";
