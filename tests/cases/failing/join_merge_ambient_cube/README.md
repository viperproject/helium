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

**How it was fixed.** The heap algebra reads no ambient cube. Every `Heap` carries a
`heap::Region`: the absolute block cube it exists on, a typed value and never a delta pc.
No constructor omits it, and every derived heap keeps its source's. A heap enters an
instruction only through `fetch_heap` (`declaration.rs`), which homes it at the cube of
the phase being walked. That is the block's `cube` in the join phase as well as the body
phase. `assume_location_axioms` gates by the heap's region. `VerifyContext::current_cube` is
private to `context/`. Code outside it asks for `effective_pc` (an instruction's pc), so
the heap code cannot read the walker's ambient state again.

Cases: the three `join_merge_*` cases (row 3: the `Merge` at a join), and
`loop_head_nonaliasing_leak` (row 5: the loop head's re-inhale of the invariant, also
a join-phase `heap_union`).
