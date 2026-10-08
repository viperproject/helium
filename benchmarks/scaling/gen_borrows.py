#!/usr/bin/env python3
"""Reborrows in sequence, and conditional moves.

    python3 gen_borrows.py      # writes src/reborrow_n{N}.rs and src/condmove_k{K}.rs

  N = sequential reborrow-write pairs on one `&mut Cell` (`reborrow`): each
      `&mut c.v` / `&mut c.w` is created, written through, and expires
      before the next, so N borrow lifetimes start and end in one body.
  K = sequential conditional moves of one local (`condmove`): each step
      moves `cur` out on one branch only and then re-initialises it, so the
      local's ownership after every join depends on the branch taken, K
      times over (2^K paths through the body).

Call-chain reborrows are `gen_calls.py`.
Two families (`reborrow`, `condmove` in suite.json).
"""

from genlib import CELL, header, write

REBORROWS = [4, 8, 16, 32, 64]
CONDMOVES = [1, 2, 4, 8, 16, 32]


def emit_reborrow(n):
    steps = []
    for i in range(n):
        field = "v" if i % 2 == 0 else "w"
        steps.append(f"    let r{i} = &mut c.{field};\n    *r{i} = *r{i} + {i + 1};")
    body = "\n".join(steps)
    return (
        header("gen_borrows.py", f"{n} sequential reborrow-write pairs.")
        + CELL
        + f"""
pub fn run(c: &mut Cell) -> i32 {{
{body}
    c.v + c.w
}}
"""
    )


def emit_condmove(k):
    steps = []
    for i in range(k):
        steps.append(f"    let t{i} = if x > {i} {{ cur }} else {{ Cell {{ v: {i}, w: x }} }};\n    cur = t{i};")
    body = "\n".join(steps)
    return (
        header("gen_borrows.py", f"{k} sequential conditional moves of one local ({2 ** k} paths).")
        + CELL
        + f"""
pub fn run(c: Cell, x: i32) -> i32 {{
    let mut cur = c;
{body}
    cur.v + cur.w
}}
"""
    )


if __name__ == "__main__":
    for n in REBORROWS:
        write(f"reborrow_n{n}", emit_reborrow(n))
    for k in CONDMOVES:
        write(f"condmove_k{k}", emit_condmove(k))
