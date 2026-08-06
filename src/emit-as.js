// JS subset -> AssemblyScript.
//
// Diagnostics carry ORIGINAL JavaScript positions, never positions in the
// generated AssemblyScript. Users must never be shown a line number in code they
// did not write; every check below runs against the JS AST for that reason.

import { ARRAY_ELEM, isArrayType } from "./directives.js";

export class Diag extends Error {
  constructor(message, node, help) {
    super(message);
    this.name = "SmpError";
    this.node = node;
    this.help = help;
  }
}

const AS_BIN = { "===": "==", "!==": "!=" };

/** Numeric literal with a decimal point or exponent is f64; a bare integer is i32. */
function literalType(raw) {
  return /[.eE]/.test(String(raw)) ? "f64" : "i32";
}

/** Does this expression mutate anything? Used to guard doubly-emitted addresses. */
function hasSideEffect(node) {
  if (!node || typeof node !== "object") return false;
  if (node.type === "UpdateExpression" || node.type === "AssignmentExpression") return true;
  if (node.type === "CallExpression") return true;
  for (const k of Object.keys(node)) {
    if (k === "type" || k === "start" || k === "end" || k === "loc") continue;
    const v = node[k];
    if (Array.isArray(v)) { if (v.some(hasSideEffect)) return true; }
    else if (v && typeof v === "object" && v.type) { if (hasSideEffect(v)) return true; }
  }
  return false;
}

class Emitter {
  constructor({ source, filename, fnName, params, returns, simdChecks, knownFns, fnSigs }) {
    this.source = source;
    this.knownFns = knownFns ?? new Set();
    this.fnSigs = fnSigs ?? new Map();
    this.filename = filename;
    this.fnName = fnName;
    this.returns = returns || "void";
    this.simdChecks = simdChecks || [];
    this.warnings = [];

    // name -> { kind: 'scalar'|'array', type }
    this.scope = new Map();
    for (const p of params) {
      this.scope.set(p.name, isArrayType(p.type)
        ? { kind: "array", type: p.type }
        : { kind: "scalar", type: p.type });
    }
    this.params = params;
  }

  loc(node) {
    const upto = this.source.slice(0, node.start);
    const line = upto.split("\n").length;
    const col = node.start - upto.lastIndexOf("\n");
    return `${this.filename}:${line}:${col}`;
  }

  fail(msg, node, help) {
    throw new Diag(`${this.loc(node)} ${msg}`, node, help);
  }

  /** Best-effort static type, enough to pick i32 vs f64 for declarations. */
  typeOf(node) {
    switch (node.type) {
      case "Literal": return literalType(node.raw);
      case "Identifier": return this.scope.get(node.name)?.type ?? "f64";
      case "MemberExpression": {
        const base = node.object.name && this.scope.get(node.object.name);
        if (base?.kind === "array") return ARRAY_ELEM[base.type].as;
        return "f64";
      }
      case "BinaryExpression": {
        if (["==", "!=", "===", "!==", "<", ">", "<=", ">="].includes(node.operator)) return "bool";
        const l = this.typeOf(node.left);
        const r = this.typeOf(node.right);
        if (node.operator === "/") return "f64"; // JS `/` is always float division
        return l === "f64" || r === "f64" ? "f64" : "i32";
      }
      case "LogicalExpression": return "bool";
      case "UnaryExpression":
        return node.operator === "!" ? "bool" : this.typeOf(node.argument);
      case "ConditionalExpression": {
        const c = this.typeOf(node.consequent);
        return c === "bool" ? this.typeOf(node.alternate) : c;
      }
      case "CallExpression": {
        // A call to a sibling kernel has a declared return type -- use it.
        // Assuming f64 here silently mistypes every local bound to a kernel
        // call, and the error only surfaces much later as an AssemblyScript
        // cast failure at the point the local is *used*, pointing at a line
        // with nothing wrong with it. `const it = tqli(...)` in a kernel
        // returning i32 is the case that found this.
        if (node.callee.type === "Identifier") {
          const sig = this.fnSigs.get(node.callee.name);
          if (sig?.returns && sig.returns !== "void") return sig.returns;
        }
        return "f64"; // Math.* is f64 in AssemblyScript, including floor/round
      }
      case "AssignmentExpression": return this.typeOf(node.right);
      case "UpdateExpression": return this.typeOf(node.argument);
      default: return "f64";
    }
  }

  /** Address expression for `arr[index]`, without enclosing parens. */
  addr(member) {
    const name = member.object.name;
    const entry = this.scope.get(name);
    const { shift } = ARRAY_ELEM[entry.type];
    // Compound assignment emits the address twice, so a side-effecting index
    // would be applied twice. Reject it rather than miscompile it silently.
    if (hasSideEffect(member.property)) {
      this.fail("index expression may not have side effects", member.property,
        "hoist it:  const idx = i++;  arr[idx] += x;");
    }
    return `${name} + (<usize>(${this.expr(member.property)}) << ${shift})`;
  }

  arrayEntry(node) {
    if (node.type !== "MemberExpression" || !node.computed) return null;
    if (node.object.type !== "Identifier") return null;
    const e = this.scope.get(node.object.name);
    return e && e.kind === "array" ? e : null;
  }

  expr(node) {
    switch (node.type) {
      case "Literal":
        if (typeof node.value === "boolean") return String(node.value);
        if (typeof node.value !== "number") this.fail(`unsupported literal ${node.raw}`, node);
        return literalType(node.raw) === "f64" ? String(node.raw) : String(node.raw);

      case "Identifier":
        if (!this.scope.has(node.name)) {
          this.fail(`unknown identifier '${node.name}'`, node,
            "kernels may only reference their parameters and their own locals; closures are not supported");
        }
        return node.name;

      case "MemberExpression": {
        const e = this.arrayEntry(node);
        if (e) return `load<${ARRAY_ELEM[e.type].as}>(${this.addr(node)})`;
        // Math.<fn>
        if (node.object.type === "Identifier" && node.object.name === "Math") {
          return `Math.${node.property.name}`;
        }
        this.fail("unsupported member access", node,
          "only typed-array indexing and Math.* are supported inside a kernel");
        break;
      }

      case "CallExpression": {
        const args = () => node.arguments.map((a) => this.expr(a)).join(", ");

        // Math.* maps straight through: AssemblyScript's Math is f64.
        if (node.callee.type === "MemberExpression" &&
            node.callee.object.type === "Identifier" &&
            node.callee.object.name === "Math") {
          return `Math.${node.callee.property.name}(${args()})`;
        }

        // A call to another @kernel in the same module. AssemblyScript compiles
        // these natively, so no inlining is required on our side -- but it will
        // not implicitly narrow, so arguments are cast to the declared parameter
        // types. JS does this coercion silently; wasm requires it in the open.
        if (node.callee.type === "Identifier" && this.knownFns.has(node.callee.name)) {
          const sig = this.fnSigs.get(node.callee.name);
          const cast = node.arguments.map((a, i) => {
            const want = sig?.params?.[i]?.type;
            const emitted = this.expr(a);
            if (!want || isArrayType(want)) return emitted;
            const got = this.typeOf(a);
            if (got === want) return emitted;
            // Narrowing f64 -> i32 truncates. That matches JS only when the
            // value is already integral, which is what @param {i32} asserts.
            return `<${want}>(${emitted})`;
          });
          return `${node.callee.name}(${cast.join(", ")})`;
        }

        if (node.callee.type === "Identifier") {
          this.fail(`'${node.callee.name}' is not a @kernel in this module`, node,
            `mark it:  @kernel  — kernels may only call Math.* and other kernels in the same file`);
        }
        this.fail("unsupported call", node, "only Math.* and other kernels in this file may be called");
        break;
      }

      case "BinaryExpression": {
        const op = AS_BIN[node.operator] ?? node.operator;
        const lt = this.typeOf(node.left);
        const rt = this.typeOf(node.right);
        let l = this.expr(node.left);
        let r = this.expr(node.right);

        // AssemblyScript will not implicitly mix i32 and f64 the way JS does, so
        // the narrower side is promoted explicitly. Widening i32 -> f64 is
        // lossless, so this cannot change a result.
        if (lt === "f64" && rt === "i32") r = `<f64>(${r})`;
        else if (rt === "f64" && lt === "i32") l = `<f64>(${l})`;

        // JS `/` is always floating-point division; i32/i32 in AS would truncate.
        if (node.operator === "/" && lt !== "f64" && rt !== "f64") {
          l = `<f64>(${l})`;
          r = `<f64>(${r})`;
        }
        return `(${l} ${op} ${r})`;
      }

      case "LogicalExpression":
        return `(${this.expr(node.left)} ${node.operator} ${this.expr(node.right)})`;

      case "UnaryExpression":
        return `(${node.operator}${this.expr(node.argument)})`;

      case "ConditionalExpression":
        return `(${this.expr(node.test)} ? ${this.expr(node.consequent)} : ${this.expr(node.alternate)})`;

      case "AssignmentExpression":
        return this.assign(node);

      case "UpdateExpression": {
        const a = this.expr(node.argument);
        return node.prefix ? `${node.operator}${a}` : `${a}${node.operator}`;
      }

      default:
        this.fail(`unsupported expression '${node.type}'`, node);
    }
  }

  assign(node) {
    const { left, right, operator } = node;
    const e = this.arrayEntry(left);

    if (e) {
      const as = ARRAY_ELEM[e.type].as;
      const a = this.addr(left);
      if (operator === "=") return `store<${as}>(${a}, ${this.expr(right)})`;
      // Compound assignment re-materialises the address rather than spilling it
      // to a temp. Index expressions in a kernel are pure arithmetic over
      // locals, so recomputing is side-effect free -- but it is why an index
      // expression containing `i++` is rejected below.
      const bin = operator.slice(0, -1);
      return `store<${as}>(${a}, load<${as}>(${a}) ${bin} ${this.expr(right)})`;
    }

    if (left.type !== "Identifier") this.fail("unsupported assignment target", left);
    return `${left.name} ${operator} ${this.expr(right)}`;
  }

  declare(decl, kindKeyword) {
    // One keyword for the whole statement, however many declarators follow:
    // `let a: i32 = 0, b: f64 = 1.0`. Emitting the keyword per declarator is a
    // parse error, and it is invisible until a kernel declares two things at once.
    const parts = [];
    for (const d of decl.declarations) {
      if (d.id.type !== "Identifier") this.fail("destructuring is not supported in a kernel", d.id);
      if (!d.init) {
        this.scope.set(d.id.name, { kind: "scalar", type: "f64" });
        parts.push(`${d.id.name}: f64 = 0.0`);
        continue;
      }
      const t = this.typeOf(d.init) === "bool" ? "bool" : this.typeOf(d.init);
      const init = this.expr(d.init);
      this.scope.set(d.id.name, { kind: "scalar", type: t });
      parts.push(`${d.id.name}: ${t} = ${init}`);
    }
    return `${kindKeyword} ${parts.join(", ")}`;
  }

  stmt(node, indent = "  ") {
    const I = indent;
    switch (node.type) {
      case "BlockStatement":
        return `{\n${node.body.map((s) => this.stmt(s, I + "  ")).join("\n")}\n${I}}`;

      case "VariableDeclaration":
        return `${I}${this.declare(node, node.kind === "const" ? "const" : "let")};`;

      case "ExpressionStatement":
        return `${I}${this.expr(node.expression)};`;

      case "ReturnStatement":
        return `${I}return${node.argument ? " " + this.expr(node.argument) : ""};`;

      case "IfStatement": {
        const cons = this.stmt(node.consequent, I).replace(/^\s+/, "");
        let s = `${I}if (${this.expr(node.test)}) ${cons}`;
        if (node.alternate) {
          const alt = this.stmt(node.alternate, I).replace(/^\s+/, "");
          s += ` else ${alt}`;
        }
        return s;
      }

      case "ForStatement": {
        const init = node.init
          ? (node.init.type === "VariableDeclaration"
              ? this.declare(node.init, "let")
              : this.expr(node.init))
          : "";
        const test = node.test ? this.expr(node.test) : "";
        const update = node.update ? this.expr(node.update) : "";
        const body = this.stmt(node.body, I).replace(/^\s+/, "");
        return `${I}for (${init}; ${test}; ${update}) ${body}`;
      }

      case "WhileStatement":
        return `${I}while (${this.expr(node.test)}) ${this.stmt(node.body, I).replace(/^\s+/, "")}`;

      case "DoWhileStatement":
        return `${I}do ${this.stmt(node.body, I).replace(/^\s+/, "")} while (${this.expr(node.test)});`;

      case "BreakStatement": return `${I}break;`;
      case "ContinueStatement": return `${I}continue;`;
      case "EmptyStatement": return `${I};`;

      default:
        this.fail(`unsupported statement '${node.type}'`, node);
    }
  }

  signature(name, extra = []) {
    const ps = this.params.map((p) => `${p.name}: ${isArrayType(p.type) ? "usize" : p.type}`);
    return `export function ${name}(${[...ps, ...extra].join(", ")}): ${this.returns}`;
  }
}

/**
 * Emit AssemblyScript for one @kernel function.
 * Returns { code, warnings }.
 */
export function emitKernel({ fn, directives, source, filename, knownFns, fnSigs }) {
  const declared = new Map(directives.params.map((p) => [p.name, p.type]));
  const params = fn.params.map((p) => {
    if (p.type !== "Identifier") {
      throw new Diag(`${filename}: kernel parameters must be plain identifiers`, p);
    }
    const type = declared.get(p.name);
    if (!type) {
      throw new Diag(
        `${filename}: parameter '${p.name}' of kernel '${fn.id.name}' has no @param type`,
        p,
        `add:  @param {f64} ${p.name}`
      );
    }
    return { name: p.name, type };
  });

  const em = new Emitter({
    source, filename, fnName: fn.id.name, params, knownFns, fnSigs,
    returns: directives.returns ?? "void",
  });

  const body = fn.body.body.map((s) => em.stmt(s, "  ")).join("\n");
  let code = `${em.signature(fn.id.name)} {\n${body}\n}\n`;

  // @parallel for -> an additional range-limited export the pool dispatches to.
  if (directives.parallel) {
    const loop = fn.body.body.find((s) => s.type === "ForStatement");
    if (!loop) {
      throw new Diag(
        `${filename}: '${fn.id.name}' is marked @parallel for but its body contains no for loop`,
        fn
      );
    }
    if (loop.init?.type !== "VariableDeclaration" || loop.init.declarations.length !== 1) {
      throw new Diag(
        `${filename}: @parallel for requires a single loop variable, e.g. for (let i = 0; i < n; i++)`,
        loop
      );
    }
    const iv = loop.init.declarations[0].id.name;

    const em2 = new Emitter({
      source, filename, fnName: fn.id.name, params, knownFns, fnSigs, returns: "void",
    });
    em2.scope.set(iv, { kind: "scalar", type: "i32" });
    const inner = em2.stmt(loop.body, "  ").replace(/^\s+/, "");
    code += `\n${em2.signature(fn.id.name + "__range", ["__lo: i32", "__hi: i32"])} {\n` +
            `  for (let ${iv}: i32 = __lo; ${iv} < __hi; ${iv}++) ${inner}\n}\n`;
  }

  return { code, warnings: em.warnings };
}
