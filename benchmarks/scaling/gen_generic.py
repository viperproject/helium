#!/usr/bin/env python3
"""Generic round-trips, and trait impls.

    python3 gen_generic.py      # writes src/generic_k{K}.rs and src/impls_k{K}.rs

  K = round-trips of a `Cell` through a generic identity (`generic`): every
      call converts the concrete value to the generic representation and
      back, then reads a field of the result.
  K = impls of one trait, one per struct type (`impls`). `run` uses only the
      first type, so K-1 impls (and whatever the encoding states about them)
      are irrelevant to it; each impl's method is also a unit of its own.

Two families (`generic`, `impls` in suite.json).
"""

from genlib import CELL, header, write

ROUND_TRIPS = [2, 4, 8, 16, 32, 64]
IMPLS = [2, 4, 8, 16, 32, 64]


def emit_generic(k):
    steps = "\n".join(f"    let c = id(c);\n    t = t + c.v;" for _ in range(k))
    return (
        header("gen_generic.py", f"{k} round-trips through a generic identity.")
        + CELL
        + f"""
pub fn id<T>(x: T) -> T {{
    x
}}

pub fn run(c: Cell) -> i32 {{
    let mut t = 0;
{steps}
    t + c.w
}}
"""
    )


def emit_impls(k):
    impls = []
    for i in range(k):
        impls.append(
            f"pub struct T{i} {{\n    pub v: i32,\n}}\n\n"
            f"impl Get for T{i} {{\n    fn get(&mut self) -> i32 {{\n        self.v + {i}\n    }}\n}}\n"
        )
    return (
        header("gen_generic.py", f"One trait with {k} impls; `run` uses one of them.")
        + """
pub trait Get {
    fn get(&mut self) -> i32;
}

"""
        + "\n".join(impls)
        + """
pub fn run(a: &mut T0) -> i32 {
    a.get() + a.get()
}
"""
    )


if __name__ == "__main__":
    for k in ROUND_TRIPS:
        write(f"generic_k{k}", emit_generic(k))
    for k in IMPLS:
        write(f"impls_k{k}", emit_impls(k))
