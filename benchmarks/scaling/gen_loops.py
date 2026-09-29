#!/usr/bin/env python3
"""Loop nesting depth x loop body size.

    python3 gen_loops.py        # writes src/loops_n{N}_b{B}.rs

  N = loop nesting depth  -> how many loop heads (labels carrying Prusti's
                             inferred permission invariant) the CFG nests
  B = statements per body -> how much the innermost body does per iteration

Each loop runs a counter up to a bound read from a struct field, so the loop
reads the heap it frames; the innermost body also writes through the `&mut`.
"""

from genlib import CELL, header, indent, write

NESTING = [1, 2, 3, 4]
BODY = [5, 20]


def body(b):
    out = []
    for i in range(b):
        if i % 3 == 0:
            out.append(f"acc = acc + {i + 1};")
        elif i % 3 == 1:
            out.append("acc = acc * 2 - c.v;")
        else:
            out.append("c.w = c.w + acc;")
    return out


def nest(level, n, b):
    if level == n:
        return body(b)
    var = f"i{level}"
    return [
        f"let mut {var} = 0;",
        f"while {var} < c.v {{",
        indent(nest(level + 1, n, b), 4),
        f"    {var} += 1;",
        "}",
    ]


def emit(n, b):
    return (
        header("gen_loops.py", f"Loop nesting depth {n}, {b} statements in the innermost body.")
        + CELL
        + f"""
pub fn run(c: &mut Cell) -> i32 {{
    let mut acc = 0;
{indent(nest(0, n, b), 4)}
    acc
}}
"""
    )


if __name__ == "__main__":
    for n in NESTING:
        for b in BODY:
            write(f"loops_n{n}_b{b}", emit(n, b))
