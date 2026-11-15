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
`heap::Region`, the absolute block cube it exists on; no `Heap` constructor omits it, and
every derived heap keeps its source's. Regions are built from the block's `cube` in
`declaration.rs`, for the join phase as well as the body phase, and a stored heap's region
only ever **narrows**: `fetch_heap` moves a heap onto the phase's region and asserts that
the phase's region lies within the heap's own (`Region::within`). The one place a heap
widens is a join: `Merge` fetches its arms at their own regions (`fetch_arm`), and
`merge_heaps` takes the join's region explicitly; each arm's path lives on in the
per-chunk guards. The viz display and tests build an empty region as well. An n-ary
join's synthetic blocks get the reach of the edges they merge as their cube, not the
whole join's. `Heap` and `Region` live in `heap/store.rs`, so the algebra reaches the
region only through `Heap::region`. `assume_location_axioms` gates by it, and
`union_heaps` asserts that its summands share a region.
`VerifyContext::current_cube` is private to `context/`. Code outside it asks for
`effective_pc` (an instruction's pc), so the heap code cannot read the walker's ambient
state again.

Cases: the three `join_merge_*` cases (row 3: the `Merge` at a join),
`nary_join_nonaliasing_leak` (a synthetic block of a three-way join), and
`loop_head_nonaliasing_leak` (row 5: the loop head's re-inhale of the invariant, also a
join-phase `heap_union`).
