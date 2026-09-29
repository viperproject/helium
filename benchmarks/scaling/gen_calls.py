#!/usr/bin/env python3
"""Call-chain depth, and call fan-out.

    python3 gen_calls.py        # writes src/calls_depth{D}.rs and src/calls_fanout{F}.rs

  D = chain depth: `f0` calls `f1` calls ... `f{D}`, each passing its `&mut`
      on -> how many reborrows and contract exhale/inhale pairs stack up
  F = fan-out: one function calls F distinct helpers in sequence, each with
      the same `&mut` -> how many call sites one body frames around

Two families (`call_depth`, `call_fanout` in suite.json).
"""

from genlib import CELL, header, write

DEPTHS = [1, 2, 4, 8, 16]
FANOUTS = [1, 2, 4, 8, 16]


def emit_depth(d):
    fns = []
    for i in range(d + 1):
        if i == d:
            body = "    c.w = c.w + x;\n    c.v + x"
        else:
            body = f"    c.v = c.v + {i + 1};\n    f{i + 1}(c, x + 1) + 1"
        fns.append(f"pub fn f{i}(c: &mut Cell, x: i32) -> i32 {{\n{body}\n}}\n")
    return header("gen_calls.py", f"A call chain {d} deep.") + CELL + "\n" + "\n".join(fns)


def emit_fanout(f):
    helpers = [
        f"pub fn h{i}(c: &mut Cell) -> i32 {{\n    c.w = c.w + {i + 1};\n    c.v\n}}\n" for i in range(f)
    ]
    calls = "\n".join(f"    acc = acc + h{i}(c);" for i in range(f))
    run = f"pub fn run(c: &mut Cell) -> i32 {{\n    let mut acc = 0;\n{calls}\n    acc\n}}\n"
    return (
        header("gen_calls.py", f"One function calling {f} helpers.") + CELL + "\n" + "\n".join(helpers + [run])
    )


if __name__ == "__main__":
    for d in DEPTHS:
        write(f"calls_depth{d}", emit_depth(d))
    for f in FANOUTS:
        write(f"calls_fanout{f}", emit_fanout(f))
