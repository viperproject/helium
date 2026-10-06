#!/usr/bin/env python3
"""Aggregate width: struct fields, tuple elements.

    python3 gen_width.py        # writes src/struct_w{W}.rs and src/tuple_w{W}.rs

  W = fields of one struct (`struct_width`): `read_all` reads each field of
      an owned value, `write_all` writes each through `&mut`. One predicate with
      W slots, unfolded and folded once per statement.
  W = elements of one tuple (`tuple_width`): built from scalars, projected
      element by element, and rebuilt. Tuple snapshots are W-ary
      constructor applications, so every projection and rebuild touches a
      W-wide value.

Depth is `gen_struct.py`; argument count is `gen_args.py`.
Two families (`struct_width`, `tuple_width` in suite.json).
"""

from genlib import header, write

STRUCT_WIDTHS = [4, 8, 16, 32, 64, 96]
TUPLE_WIDTHS = [2, 4, 8, 12, 16, 20]


def emit_struct(w):
    fields = "\n".join(f"    pub f{i}: i32," for i in range(w))
    reads = "\n".join(f"    t = t + s.f{i};" for i in range(w))
    writes = "\n".join(f"    s.f{i} = {i + 1};" for i in range(w))
    return (
        header("gen_width.py", f"One struct with {w} fields.")
        + f"""
pub struct S {{
{fields}
}}

pub fn read_all(s: S) -> i32 {{
    let mut t = 0;
{reads}
    t
}}

pub fn write_all(s: &mut S) {{
{writes}
}}
"""
    )


def emit_tuple(w):
    ty = "(" + ", ".join(["i32"] * w) + ")"
    build = "(" + ", ".join(f"x + {i}" for i in range(w)) + ")"
    reads = "\n".join(f"    let v{i} = t.{i};" for i in range(w))
    total = " + ".join(f"v{i}" for i in range(w))
    rebuild = "(" + ", ".join(f"v{(i + 1) % w}" for i in range(w)) + ")"
    return (
        header("gen_width.py", f"One {w}-tuple built, projected and rebuilt.")
        + f"""
pub fn make(x: i32) -> {ty} {{
    {build}
}}

pub fn rotate(t: {ty}) -> {ty} {{
{reads}
    {rebuild}
}}

pub fn sum(t: {ty}) -> i32 {{
{reads}
    {total}
}}
"""
    )


if __name__ == "__main__":
    for w in STRUCT_WIDTHS:
        write(f"struct_w{w}", emit_struct(w))
    for w in TUPLE_WIDTHS:
        write(f"tuple_w{w}", emit_tuple(w))
