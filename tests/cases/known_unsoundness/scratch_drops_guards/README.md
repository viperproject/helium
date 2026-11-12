# The block scratch drops guards it never assumed

Design class **B** (facts that lose their condition), census row 1 of
`plans/verifier-soundness-interfaces.md`. Sources: #48, #39, #49 bug 1.

`assume_guarded` records `guards ==> fact` on ground and the bare `fact` in the block
scratch. Dropping the guards is correct only when they are the block's cube, which the
scratch has already assumed. Many callers pass finer guards: an intra-block path condition
(ternary, implication, `unfolding` in one arm), a Σ-ite address equality from a summarized
exhale, a chunk's presence guard. The scratch then knows the fact on paths where it does
not hold.

```viper
method m(x: Ref, a: Ref, b: Ref)
  requires acc(a.f) && acc(b.f) && a.f == 1 && b.f == 2
  requires x == a || x == b
{
  exhale acc(x.f, 1/2)
  assert false   // accepted
}
```

The summarized exhale ties `x.f` to `a.f` under `x == a` and to `b.f` under `x == b`; the
scratch gets both unguarded, so `1 == 2`.

**What a structural fix looks like.** No caller decides what the scratch receives: a fact
is stated under an absolute region, and the prover projects that region onto the scratch
(the scratch can drop only the literals of its own cube). Raw `union` outside the context
goes away. See §4.3.

Cases: `scratch_guard_*` (unfolding under a finer pc), `summarized_exhale_*` and
`summary_*` (Σ-ite address guards), `pc_alias_intra_block_literal`.
