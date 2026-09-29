#!/usr/bin/env python3
"""Writes through `&mut` x place path length.

    python3 gen_mut.py          # writes src/mut_w{W}_p{P}.rs

  W = writes through one `&mut` parameter -> heap updates the permission
      model applies and frames around
  P = place path length: each write goes `g.a...x`, P fields deep
      -> how many predicate levels each write unfolds and folds back
"""

from genlib import header, write

WRITES = [1, 2, 4, 8, 16, 32]
PATHS = [1, 2, 3, 4]


def types(p):
    out = ["pub struct L0 {\n    pub x: i32,\n    pub y: i32,\n}\n"]
    for i in range(1, p):
        out.append(f"pub struct L{i} {{\n    pub a: L{i - 1},\n    pub b: L{i - 1},\n}}\n")
    return "\n".join(out)


def emit(w, p):
    prefix = "g" + ".a" * (p - 1)
    writes = []
    for i in range(w):
        field = "x" if i % 2 == 0 else "y"
        writes.append(f"    {prefix}.{field} = {prefix}.{field} + {i + 1};")
    body = "\n".join(writes)
    return (
        header("gen_mut.py", f"{w} writes through `&mut`, each {p} fields deep.")
        + "\n"
        + types(p)
        + f"""
pub fn run(g: &mut L{p - 1}) -> i32 {{
{body}
    {prefix}.x
}}
"""
    )


if __name__ == "__main__":
    for w in WRITES:
        for p in PATHS:
            write(f"mut_w{w}_p{p}", emit(w, p))
