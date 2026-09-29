#!/usr/bin/env python3
"""Argument count x argument kind.

    python3 gen_args.py         # writes src/args_{val,ref,mut}_n{N}.rs

  N    = arguments -> how many type predicates the contract inhales and
         exhales
  kind = by value (`Cell`), shared (`&Cell`) or mutable (`&mut Cell`) borrow
         -> an owned predicate, a read-only borrow, or a borrow written
            through and handed back

The kind is categorical, so each kind is its own family (`args_value`,
`args_ref`, `args_mut` in suite.json) with N as its knob.
"""

from genlib import CELL, header, write

COUNTS = [1, 2, 4, 8, 16]
KINDS = {"val": "Cell", "ref": "&Cell", "mut": "&mut Cell"}


def emit(kind, n):
    ty = KINDS[kind]
    params = ", ".join(f"a{i}: {ty}" for i in range(n))
    reads = " + ".join(f"a{i}.v" for i in range(n))
    writes = "".join(f"    a{i}.w = a{i}.w + s;\n" for i in range(n)) if kind == "mut" else ""
    return (
        header("gen_args.py", f"{n} arguments of type `{ty}`.")
        + CELL
        + f"""
pub fn run({params}) -> i32 {{
    let s = {reads};
{writes}    s
}}
"""
    )


if __name__ == "__main__":
    for kind in KINDS:
        for n in COUNTS:
            write(f"args_{kind}_n{n}", emit(kind, n))
