// Directive extraction.
//
// Every smp.js directive is a JSDoc comment. That is the whole reason an
// annotated file still runs as plain JavaScript: the compiler reads the comments,
// `node` ignores them. The single-threaded fallback is not a code path anyone has
// to maintain -- it is the source file.

/** Supported types in @param / @returns. */
export const TYPES = new Set(["f64", "f32", "i32", "i64", "u32", "bool", "Float64Array", "Float32Array", "Int32Array"]);

/** Typed-array element widths, used for pointer arithmetic during emission. */
export const ARRAY_ELEM = {
  Float64Array: { as: "f64", shift: 3 },
  Float32Array: { as: "f32", shift: 2 },
  Int32Array: { as: "i32", shift: 2 },
};

export const isArrayType = (t) => Object.prototype.hasOwnProperty.call(ARRAY_ELEM, t);

/**
 * Parse the body of one block comment into a directive record.
 * Unknown tags are collected rather than dropped, so the caller can warn about
 * `@paralel for` instead of silently compiling it single-threaded.
 */
export function parseDirectiveComment(raw) {
  const d = {
    kernel: false,
    params: [],
    returns: null,
    parallel: null, // { schedule: 'dynamic' | 'static', chunk: number | null }
    shared: [],
    private: [],
    simd: null, // { simdlen: number | null }
    unknown: [],
  };

  // Strip the leading `*` of each JSDoc line before matching.
  const lines = raw.split("\n").map((l) => l.replace(/^\s*\*?\s?/, "").trimEnd());

  for (const line of lines) {
    const m = /^@(\w+)\b(.*)$/.exec(line.trim());
    if (!m) continue;
    const [, tag, restRaw] = m;
    const rest = restRaw.trim();

    switch (tag) {
      case "kernel":
        d.kernel = true;
        break;

      case "param": {
        // @param {f64} name  -- description
        const pm = /^\{([^}]+)\}\s+(\[?[\w$]+\]?)/.exec(rest);
        if (pm) d.params.push({ type: pm[1].trim(), name: pm[2].replace(/[[\]]/g, "") });
        break;
      }

      case "returns":
      case "return": {
        const rm = /^\{([^}]+)\}/.exec(rest);
        if (rm) d.returns = rm[1].trim();
        break;
      }

      case "parallel": {
        // @parallel for schedule(dynamic) | @parallel for schedule(static)
        const sm = /schedule\(\s*(\w+)\s*(?:,\s*(\d+)\s*)?\)/.exec(rest);
        d.parallel = {
          schedule: sm ? sm[1] : "dynamic",
          chunk: sm && sm[2] ? Number(sm[2]) : null,
        };
        break;
      }

      case "shared":
        d.shared.push(...rest.split(/[,\s]+/).filter(Boolean));
        break;

      case "private":
        d.private.push(...rest.split(/[,\s]+/).filter(Boolean));
        break;

      case "simd":
        d.simd = d.simd || { simdlen: null };
        break;

      case "simdlen":
        d.simd = d.simd || { simdlen: null };
        d.simd.simdlen = Number(rest) || null;
        break;

      case "reduction":
        d.simd = d.simd || { simdlen: null };
        d.reduction = rest;
        break;

      // JSDoc tags that are none of our business.
      case "type": case "typedef": case "see": case "example":
      case "description": case "file": case "author": case "license":
        break;

      default:
        d.unknown.push(tag);
    }
  }

  return d;
}

/**
 * Associate block comments with the node that follows them.
 *
 * Returns a Map from node.start to the directive record. Position-keyed rather
 * than node-keyed so the emitter can look up by the node it is walking without
 * threading extra state through.
 */
export function attachDirectives(comments, nodes) {
  const byStart = new Map();
  const sorted = [...comments].sort((a, b) => a.start - b.start);

  for (const node of nodes) {
    // The nearest block comment that ends before this node starts, with nothing
    // but whitespace between. Anything else belongs to some other construct.
    let best = null;
    for (const c of sorted) {
      if (c.type !== "Block") continue;
      if (c.end > node.start) break;
      best = c;
    }
    if (!best) continue;
    byStart.set(node.start, { comment: best, directives: parseDirectiveComment(best.value) });
  }

  return byStart;
}

/** Line-comment directives, e.g. a bare `// @simd` above a loop. */
export function lineDirectivesBefore(comments, node, source) {
  const collected = [];
  for (const c of comments) {
    if (c.type !== "Line") continue;
    if (c.end > node.start) continue;
    const between = source.slice(c.end, node.start);
    // Only whitespace and other line comments may intervene.
    if (!/^[\s]*(\/\/[^\n]*\n[\s]*)*$/.test(between)) continue;
    collected.push(c.value);
  }
  if (!collected.length) return null;
  return parseDirectiveComment(collected.join("\n"));
}
