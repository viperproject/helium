#!/usr/bin/env python3
"""Struct nesting depth: fold/unfold load.

    python3 gen_struct.py       # writes src/struct_d{D}.rs

  D = nesting depth of the argument's type. `S{i}` holds two `S{i-1}` and an
      int; `S0` is two ints. Writing a leaf unfolds D predicate levels, and
      handing a sub-struct to a helper as `&mut` splits the footprint at every
      level on the way down.
"""

from genlib import header, write

DEPTHS = [1, 2, 3, 4, 5, 6]


def emit(d):
    types = ["pub struct S0 {\n    pub x: i32,\n    pub y: i32,\n}\n"]
    helpers = ["pub fn touch0(s: &mut S0) -> i32 {\n    s.x = s.x + 1;\n    s.y\n}\n"]
    for i in range(1, d + 1):
        types.append(f"pub struct S{i} {{\n    pub l: S{i - 1},\n    pub r: S{i - 1},\n    pub n: i32,\n}}\n")
        right_leaf = "s.r." + "l." * (i - 1) + "x"
        helpers.append(
            f"pub fn touch{i}(s: &mut S{i}) -> i32 {{\n"
            f"    s.n = s.n + 1;\n"
            f"    touch{i - 1}(&mut s.l) + {right_leaf}\n"
            f"}}\n"
        )
    leaf = "s." + "l." * d + "x"
    run = f"pub fn run(s: &mut S{d}) -> i32 {{\n    {leaf} = {leaf} + 1;\n    touch{d}(s)\n}}\n"
    return header("gen_struct.py", f"Struct nesting depth {d}.") + "\n" + "\n".join(types + helpers + [run])


if __name__ == "__main__":
    for d in DEPTHS:
        write(f"struct_d{d}", emit(d))
