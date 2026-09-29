#!/usr/bin/env python3
"""Checked operations per block.

    python3 gen_arith.py        # writes src/arith_k{K}.rs

  K = arithmetic operations in one straight-line block. Half are divisions
      or remainders by a parameter, each a divide-by-zero check Prusti always
      emits; the rest are `+ - *`, which add overflow checks only when the
      encoding is made with PRUSTI_CHECK_OVERFLOWS=true (off by default in
      tools/prusti_encode.sh, as for ../rust/).

The divisor is guarded once at the top, so every check is provable from the
same path condition: what is measured is the number of obligations, not their
difficulty.
"""

from genlib import CELL, header, write

OPS = [5, 10, 20, 40, 80]


def emit(k):
    ops = []
    for i in range(k):
        r = i % 4
        if r == 0:
            ops.append(f"    acc = acc + c.v - {i};")
        elif r == 1:
            ops.append("    acc = acc / d + c.w;")
        elif r == 2:
            ops.append(f"    acc = acc * 2 + {i};")
        else:
            ops.append(f"    acc = acc % d + {i};")
    body = "\n".join(ops)
    return (
        header("gen_arith.py", f"{k} arithmetic operations in one block.")
        + CELL
        + f"""
pub fn run(c: &Cell, d: i32) -> i32 {{
    if d <= 0 {{
        return 0;
    }}
    let mut acc = c.v;
{body}
    acc
}}
"""
    )


if __name__ == "__main__":
    for k in OPS:
        write(f"arith_k{k}", emit(k))
