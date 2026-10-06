#!/usr/bin/env python3
"""Match width, and sequential decisions.

    python3 gen_match.py        # writes src/nway_k{K}.rs and src/seqmatch_k{K}.rs

  K = arms of one integer `match` (`nway`). Each arm writes a different
      constant through a `&mut`, and all arms rejoin at one block: one join
      with K predecessors, a K-deep `else if` cascade of guards, and K-1
      disproven equalities on the scrutinee.
  K = sequential decisions (`seqmatch`): K pairs of an `if` that builds an
      enum and a `match` that takes it apart again. Unlike `seq_if`'s
      independent branches, each match is decided by the `if` before it, so
      a precise verifier must carry the correlation through the join.

Two families (`nway`, `seqmatch` in suite.json).
"""

from genlib import CELL, header, write

NWAY = [4, 8, 16, 32, 64, 128]
SEQMATCH = [2, 4, 8, 16, 32]


def emit_nway(k):
    arms = "\n".join(f"        {i} => c.w = {i + 1}," for i in range(k - 1))
    return (
        header("gen_match.py", f"One integer match with {k} arms.")
        + CELL
        + f"""
pub fn run(c: &mut Cell, x: i32) -> i32 {{
    match x {{
{arms}
        _ => c.w = 0,
    }}
    c.v + c.w
}}
"""
    )


def emit_seqmatch(k):
    pairs = []
    for i in range(k):
        pairs.append(f"    let e{i} = if v > {i} {{ E::B(v) }} else {{ E::A }};")
        pairs.append(f"    t = t + match e{i} {{ E::A => 0, E::B(x) => x }};")
    body = "\n".join(pairs)
    return (
        header("gen_match.py", f"{k} sequential (if, match) decision pairs.")
        + f"""
pub enum E {{
    A,
    B(i32),
}}

pub fn churn(v: i32) -> i32 {{
    let mut t = 0;
{body}
    t
}}
"""
    )


if __name__ == "__main__":
    for k in NWAY:
        write(f"nway_k{k}", emit_nway(k))
    for k in SEQMATCH:
        write(f"seqmatch_k{k}", emit_seqmatch(k))
