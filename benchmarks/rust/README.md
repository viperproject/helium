# Rust benchmark corpus

Realistic spec-less Rust, encoded to Viper by Prusti, for measuring verifier cost on
programs that do more computation than a bare enum match. Built to answer whether the
lazy / sticky / dominator-scoped scratch e-graph designs have anything to work with
(the block-VMIR two-e-graph design notes; those live outside the repo).

    src/*.rs        hand-written and generated sources (committed)
    vpr/*.vpr       their Prusti encodings (committed — measurement needs no Prusti)
    encode_all.sh   src -> vpr, skipping anything already up to date
    gen_depth.py    regenerates the depth_d{D}_m{M} family

Rules for every source here, same as `../../../cases/rust/structs_enums.rs`:

- **no `prusti_contracts`** — obligations come from Prusti's own type predicates
  (framing, fold/unfold, discriminant well-formedness), so the corpus measures the
  encoding every Prusti user pays rather than hand-written specs;
- **no loops, no recursion, no returned references**; `&mut` parameters are fine;
- **a borrow may be stored in a struct, but such a struct may not be a parameter** —
  Prusti encodes `fn f(c: &mut Cursor)` into a program its own Silicon run rejects with
  `insufficient.permission`, surfacing as `[Prusti internal error] ... could not be
  backtranslated`. `borrow_fields.rs` therefore packs the borrow inside the body;
  `borrow_fields_rejected.rs.txt` keeps the nine rejected members verbatim;
- every member must verify — a failing member stops at its failing instruction and has
  no stable cost.

## What each file is for

| file | axis |
|---|---|
| `mat3_mul.rs` | block length: unrolled 3x3 linear algebra, 30+ statement straight-line blocks |
| `vec3_math.rs` | block length x call density: every operation built from calls to smaller ones |
| `aabb_collide.rs` | dominator depth: if/else nested 3-5 deep, arms doing real work |
| `state_machine.rs` | dominator depth x arm count: a 5x5 `State`/`Event` match grid |
| `color_blend.rs` | branch cascades: many sequential two-way branches per member |
| `shape_area.rs` | payload enums: arms that unfold nested-struct payloads and write through them |
| `inventory.rs` | Option/Result paths: hand-rolled `Maybe`/`Res`, error arms, dispatch to one of three `&mut` fields |
| `bank_transfer.rs` | permission traffic: two `&mut` accounts in one call, guarded debits, swaps |
| `physics_step.rs` | composition: three bodies through integrate -> clamp -> bounce |
| `classify_tuple.rs` | many sequential blocks, shallow cubes: an 8-element buffer classified element by element |
| `borrow_fields.rs` | borrows *stored in* structs: `p_Ref_mutable` nested inside another type predicate, so a fold drags a borrow's permission through the footprint |
| `depth_d{D}_m{M}.rs` | the two axes separated and scaled: D nesting levels x M statements per block |
| `enum_v{N}_p{D}.rs` | payload-enum cost separated: N variants x payload nesting depth D, each arm writing through a `&mut` payload — cost is exponential in N, linear in payload size (`../../enum_scaling_2026-08-14.md`) |

## Encoding

    python3 gen_depth.py         # regenerate the depth_d{D}_m{M} family
    python3 gen_enum.py          # regenerate the enum_v{N}_p{D} family
    ./encode_all.sh              # all of src/
    ./encode_all.sh mat3_mul     # one file

Needs the local Prusti checkout (`../../tools/prusti_encode.sh`, override with
`PRUSTI_RUSTC`). Roughly 1-3 minutes and ~0.5-1 MB of Viper per source, which is why
`vpr/` is committed.

## Two `&mut`-into-a-call gaps this corpus uncovered

Every program here that calls a helper taking `&mut` — most of them — failed with
"insufficient permission" when the corpus was first built. The framing-miss trace
(demanded vs held addresses, now `verify --trace=fail`) showed two distinct causes:

1. **Unconditional call.** The reborrow's address term meets the held chunk's address
   only after a **full** saturation, while the miss retry ran the terminating reductions
   only. Fixed in `heap_subtract_inner`; the retry sits after the provably-zero check so
   only a would-fail obligation pays (retrying at every miss cost 36x).
   Pinned by `tests/cases/passing/permissions/mut_reborrow_call.vpr`.

2. **Call inside a branch arm.** The arm's reborrow mints its own ref, and
   `p_Ref_mutable_assign` states `snap(arm_ref) == arbitrary_value(param_addr, ..)` only
   *under the arm's pc* — invisible to ground-canonical address matching. Fixed by
   routing a consume miss through `chunk_under_pc` (already used on the read side) and
   taking the invariant-7 path with an empty partner set: sufficiency under the pc, debit
   gated by the pc. Pinned by
   `tests/cases/passing/permissions/mut_reborrow_call_in_branch.vpr`.

3. **Consecutive calls on a branch-selected reborrow.** Each `&mut` reborrow of a
   branch-selected place mints a fresh ref that is only *pc-equal* to the one before, and
   every call's `#requires` exhale / `#ensures` inhale pair leaves another alias behind.
   The ground-miss fallback from (2) then took the **first** chunk its probe matched —
   order-dependent, and the first hit was the chunk an earlier consume had already
   drained (a pc-gated debit leaves it as `pc ? 0 : 1/1`), while the full permission sat
   in the chunk the intervening inhale produced. Fixed in `894a300` by collecting the
   whole pc-alias set and proving sufficiency over its sum. Pinned by
   `tests/cases/passing/permissions/pc_alias_consume_prefers_holder.vpr` and, on the
   unsoundness side, `tests/cases/failing/pc_alias_ground_miss_double_spend.vpr`.
   Cleared `physics_step::m_world_kick_slowest` and `inventory::m_slots_move`.

4. **Re-reading a field after a call.** Prusti wraps every `&mut` field access in
   `make_concrete_T` / `unfold` / read / `fold` / `make_generic_T`, and that pair relates
   the before/after predicate snapshots only through the snap *function* `p_T_snap`. The
   `f%pre` token that gates unfolding `p_T_snap`'s body is emitted as an orphan recipe
   step, and `RecipeBuilder::slice` -- backward closure from the result -- pruned it as
   unreachable, so an occurrence introduced by a method contract had no token and stayed
   opaque. Two reads of one *unchanged* field were therefore provably unrelated. Fixed by
   `slice_with_tokens`; pinned by
   `tests/cases/passing/functions/resource_recipe_keeps_pre_token.vpr`. Cleared
   `physics_step::m_body_apply_impulse` and three `panic_free` probes
   (`frame_other_field`, `write_then_read`, `enum_rematch`).

All four are why the sources here are written as ordinary Rust rather than around the
verifier. If any gap returns, those cases fail before the corpus does.

## Checking the corpus

    ./check.sh                   # all of vpr/
    ./check.sh physics_step      # one stem

Holds the corpus to the rule above — every member verifies — against the reviewed
exception list in `expected_failures.txt`. Fails on a member that regressed *and* on a
listed member that now passes, so a closed gap has to be deleted from the list rather
than left to rot. Currently **2886 verified, 0 known-failing** — every member of the
corpus verifies, so `expected_failures.txt` is empty.

Not part of `cargo test`: the corpus takes minutes and its `vpr/` is large, so it is a
pre-merge check rather than a unit-test-suite member.
