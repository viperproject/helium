#!/usr/bin/env python3
"""Enum shapes: variant count, enum-typed fields, Option nesting.

    python3 gen_enums.py        # writes src/enum_tag_v{N}.rs, src/enum_fields_n{N}.rs,
                                #        src/option_d{D}.rs

  N = variants of one fieldless enum (`enum_tag`). `tag` matches every variant
      and `from` builds one from an integer, so the snapshot function's
      discriminant ladder, the type's axioms and two N-arm joins all grow
      with N. The payload-carrying version is `../rust/gen_enum.py`.
  N = enum-typed fields of one struct (`enum_fields`), each read by its own
      two-arm match: N independent discriminant decisions in one body, each
      under a field projection.
  D = nesting depth of `Option<Option<...>>` (`option_depth`), peeled by D
      nested matches: discriminant facts stack up D snapshot levels deep.

Three families (`enum_tag`, `enum_fields`, `option_depth` in suite.json).
"""

from genlib import header, write

TAG_VARIANTS = [8, 16, 32, 64, 128]
FIELDS = [2, 4, 8, 16, 32]
OPTION_DEPTHS = [2, 4, 8, 12, 16]


def emit_tag(n):
    variants = "\n".join(f"    V{i}," for i in range(n))
    tag_arms = "\n".join(f"        E::V{i} => {i}," for i in range(n))
    from_arms = "\n".join(f"        {i} => E::V{i}," for i in range(1, n))
    return (
        header("gen_enums.py", f"One fieldless enum with {n} variants.")
        + f"""
pub enum E {{
{variants}
}}

pub fn tag(e: E) -> i32 {{
    match e {{
{tag_arms}
    }}
}}

pub fn from(x: i32) -> E {{
    match x {{
{from_arms}
        _ => E::V0,
    }}
}}
"""
    )


def emit_fields(n):
    fields = "\n".join(f"    pub f{i}: E," for i in range(n))
    reads = "\n".join(f"    t = t + match s.f{i} {{ E::A => {i}, E::B(v) => v }};" for i in range(n))
    return (
        header("gen_enums.py", f"A struct of {n} enum-typed fields, each read by a match.")
        + f"""
pub enum E {{
    A,
    B(i32),
}}

pub struct S {{
{fields}
}}

pub fn read_all(s: S) -> i32 {{
    let mut t = 0;
{reads}
    t
}}
"""
    )


def peel(i, d, pad):
    if i == d:
        return f"x{d}"
    inner = peel(i + 1, d, pad + "    ")
    return f"match x{i} {{\n{pad}    Some(x{i + 1}) => {inner},\n{pad}    None => 0,\n{pad}}}"


def emit_option(d):
    ty = "i32"
    for _ in range(d):
        ty = f"Option<{ty}>"
    return (
        header("gen_enums.py", f"`Option` nested {d} deep, peeled by {d} nested matches.")
        + f"\npub fn peel(x0: {ty}) -> i32 {{\n    {peel(0, d, '    ')}\n}}\n"
    )


if __name__ == "__main__":
    for n in TAG_VARIANTS:
        write(f"enum_tag_v{n}", emit_tag(n))
    for n in FIELDS:
        write(f"enum_fields_n{n}", emit_fields(n))
    for d in OPTION_DEPTHS:
        write(f"option_d{d}", emit_option(d))
