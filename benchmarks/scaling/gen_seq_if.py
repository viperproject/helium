#!/usr/bin/env python3
"""Sequential if-blocks: path explosion.

    python3 gen_seq_if.py       # writes src/seqif_k{K}.rs

  K = two-way branches one after another in one block -> 2^K paths through
      the function, where nesting K deep (`../rust/gen_depth.py`) gives K+1.

Each branch writes through a `&mut`, so both arms carry heap state into the
join.
"""

from genlib import CELL, header, write

BRANCHES = [1, 2, 4, 8, 12, 16]


def emit(k):
    blocks = []
    for i in range(k):
        blocks.append(
            f"""    if c.v > {i} * k {{
        c.w = c.w + {i + 1};
    }} else {{
        c.w = c.w - {i + 1};
    }}"""
        )
    body = "\n".join(blocks)
    return (
        header("gen_seq_if.py", f"{k} sequential two-way branches ({2 ** k} paths).")
        + CELL
        + f"""
pub fn run(c: &mut Cell, k: i32) -> i32 {{
{body}
    c.v + c.w
}}
"""
    )


if __name__ == "__main__":
    for k in BRANCHES:
        write(f"seqif_k{k}", emit(k))
