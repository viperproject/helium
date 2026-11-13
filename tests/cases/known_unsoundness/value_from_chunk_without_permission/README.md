# A value is read off a chunk that holds no permission

Design class **D** (chunk contents read outside the condition they mean something under),
census row 2 of `plans/verifier-soundness-interfaces.md`. Sources: #30, #49 bug 2.

A chunk drained to zero on some paths stays in the heap with the value it had: a debit
gated by the path condition leaves `1 - (b ? 1 : 0)`, which does not fold to zero. Two
places read values off such chunks as if they held the location:

- the agreement rule of a summarized consume equates every member's value with the demand,
  whether or not the member holds permission;
- `unfold`, `fold` and function frame checks take the value of whichever chunk matched on
  ground, before consuming.

The agreement facts are also guarded by the member's guard only, so ground states them on
paths where the heap does not exist (`summary_cross_branch_leak`).

```viper
method m(x: Ref, y: Ref, b: Bool)
  requires acc(x.f) && x.f == 4
{
  if (b) {
    assume y == x
    exhale acc(y.f)
    inhale acc(y.f) && y.f == 5
    exhale acc(x.f)
    assert false   // accepted: the stale 4 is equated with the current 5
  }
}
```

**What a structural fix looks like.** `Chunk.value` is not readable. A value comes only out
of a `Read` or a `Taken` that carries the condition under which the chunk held positive
permission, so a value without that condition cannot be formed. See §4.6. The fix must not
lose the value a chunk that does hold permission provides (#49's
`zero_residue_value_from_holder.vpr` is such a must-verify control).
