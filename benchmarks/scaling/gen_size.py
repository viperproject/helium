#!/usr/bin/env python3
"""Program size: units per file, and statements per method.

    python3 gen_size.py         # writes src/units_k{K}.rs and src/straight_k{K}.rs

  K = independent functions, each over its own struct type (`units`). Every
      unit is the same small body, so per-unit work that is constant gives a
      linear curve, and any per-unit work proportional to the whole program
      (its types, axioms, other units) gives a quadratic one.
  K = statements in one straight-line body (`straight`): field updates
      through a `&mut Cell`, with a call to a helper taking the same `&mut`
      every fourth statement. No branches, so the obligation count is linear
      in K and any growth in per-statement cost is the cost of a larger
      accumulated state.

Two families (`units`, `straight` in suite.json).
"""

from genlib import CELL, header, write

UNITS = [4, 8, 16, 32, 48, 64]
STATEMENTS = [16, 32, 64, 96, 128]


def emit_units(k):
    units = [
        f"pub struct T{i} {{\n    pub v: i32,\n    pub w: i32,\n}}\n\n"
        f"pub fn f{i}(t: &mut T{i}, x: i32) -> i32 {{\n    t.v = t.v + x;\n    t.w - x\n}}\n"
        for i in range(k)
    ]
    return header("gen_size.py", f"{k} independent functions, one struct type each.") + "\n" + "\n".join(units)


def emit_straight(k):
    stmts = []
    for i in range(k):
        if i % 4 == 3:
            stmts.append("    bump(c);")
        elif i % 2 == 0:
            stmts.append(f"    c.v = c.v + {i};")
        else:
            stmts.append(f"    c.w = c.v - c.w;")
    body = "\n".join(stmts)
    return (
        header("gen_size.py", f"One straight-line body of {k} statements.")
        + CELL
        + f"""
pub fn bump(c: &mut Cell) {{
    c.w = c.w + 1;
}}

pub fn run(c: &mut Cell) -> i32 {{
{body}
    c.v
}}
"""
    )


if __name__ == "__main__":
    for k in UNITS:
        write(f"units_k{k}", emit_units(k))
    for k in STATEMENTS:
        write(f"straight_k{k}", emit_straight(k))
