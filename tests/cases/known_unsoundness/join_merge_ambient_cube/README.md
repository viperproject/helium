# A join's heap merge states location axioms on every path

Design class **A** (untyped path conditions and ambient phase state), census rows 3 and 5
of `plans/verifier-soundness-interfaces.md`. Source: #49 bug 3; row 5 (#53, #54-2) is the
same misreading in `heap_union` at a loop head.

`assume_location_axioms` gates its facts by `ctx.current_cube()`, read as "the path this
heap exists on". That is true in a block body, but a join is walked with the cube cleared
(`end_block`), and its `HeapInst::Merge` carries the join's cube as its own pc. Every
merged heap's axioms (non-aliasing from summed permission) are stated with no path
condition and can make a feasible arm infeasible.

```viper
method m(a: Ref, b: Ref, c: Bool, d: Bool)
  requires acc(a.f, 1/2) && acc(b.f, 1/2)
{
  if (d) {
    inhale acc(a.f, 1/2)
    if (c) {}
  } else {
    assume a == b
    assert false   // accepted: the merged heap of the d-arm derived a != b everywhere
  }
}
```

**What a structural fix looks like.** The heap algebra reads no ambient cube. A heap
carries the region it exists on, set when the phase creates it, and every assumption the
algebra makes is gated by that region. A path condition is one typed, absolute `Region`,
never a raw literal slice that may be a delta or empty. See §4.1, §4.2, §4.6.
