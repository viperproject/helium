# Panic-freedom corpus

Where the boundary of our arithmetic reasoning sits, measured on the obligations a Prusti
user gets for free: every Rust program carries divide-by-zero, overflow, negation, shift,
and bounds checks whether or not it has a single specification. `benchmarks/rust/` turns
these off (`PRUSTI_CHECK_OVERFLOWS=false`) because it measures block structure; this
corpus turns them on because they *are* the measurement.

    src/*.rs           sources, one file per reasoning tier
    vpr/*.vpr          Prusti encodings with overflow checks ON
    encode_all.sh      src -> vpr
    expected.txt       per-member OK/FAIL expectation
    check.sh           run + diff against expected.txt

## Current state (2026-08-13, after the case split was retired)

    as expected: 53   incomplete: 12   unsound: 0

**Accepted regression, 2026-08-13: 55 → 53.** The goal-directed case split (the old
"tier 4") was deleted from `prove_under_pc`; see the tier ladder on that method. Two
members depended on it and are now incomplete by decision, not by accident:
`tier_a_no_arith::m_div_by_match_literal` and `tier_c_range_arith::m_div_after_join`.

Both are the same shape — a divisor that is a *join of constants* (`let d = if flag { 3 }
else { 5 }`, or the three-arm `match`), then `a / d`. The split used to fork the state and
close each arm separately. The replacement, a literal-generalized decomposition in
`IteReduceApplier` (class folds to `L`, an arm folds to something else ⇒ that arm is not
taken ⇒ the condition is pinned; both arms disagreeing pins both polarities, which is the
contradiction), closes the plain-Viper form `d := c ? 1 : 2; assert d != 0`, but **not**
these two: Prusti wraps the arms in snapshot constructors, so the goal is
`s_Int_i32_value(ite(flag, cons(3), cons(5)))`. Commuting an accessor through an `ite` is
implemented (`ProjApplier`) but registered only for real ADT declarations
(`func_registry.rs`), never for a *domain* `cons`/`value` pair — so the arms never become
literals and the decomposition has nothing to fire on.

Recovering them means registering a proj rule for the snapshot-domain `cons`/`value` pair
(its two round-trip axioms are an isomorphism declaration). Not done: this is
panic-freedom incompleteness, and the Prusti corpus is the constraint that governs.

- **Tier A: 9/10.** Every shape where the guard matches the check discharges; the
  divide-by-zero machinery itself is fine. `div_by_match_literal` is the one loss, see the
  accepted regression above.
- **Tier B: 10/10.** Was 2/10 until 2026-08-03. The eight order-guarded failures were
  **not** a missing `0 < d ==> d != 0` — that inference worked all along. They were a
  goal-*spelling* gap: Prusti emits the MIR assert as `_t == false`, negation lowers to
  `ite(e, false, true)`, and the case splitter only ever collected `Ite` conditions, so
  the `== false` spelling handed it nothing to split on. Fixed by four normalisation
  rewrites (`eq-false-is-not-{l,r}`, `eq-true-is-self-{l,r}`); see
  `../../findings_2026-07-30_panic_freedom.md` §2.
- **Tier C: 1/10.** `neg_bounded` — it needed one guard fact, not a range, so it was
  mis-tiered. `div_after_join` (literal divisors on both arms) passed here until the case
  split was retired; see the accepted regression above. Every genuine interval or sum
  bound still fails, as expected — this is the tier that motivates
  `../../z3_integration_plan.md`.
- **Tier D: 4/4**, counting the two must-fail members as correctly rejected.
  `index_guard_lt_len` came with the negation-proven-false mirror (`8f0125d`).
- **`probe_unreachable`: 4/5**, and the one rejection is the right one — a reachable
  `unreachable!()` is caught.
- **`probe_unreachable_nonarith`: 13/14.** Was 9/14. `frame_other_field`,
  `write_then_read` and `enum_rematch` were all one bug: a function application
  introduced by a *resource* body (a method contract) lost its `f%pre` token to
  `RecipeBuilder::slice`'s backward closure, so the predicate snap function never
  unfolded and two reads of one unchanged field stayed unrelated. `eq_two_literals` then
  fell to the negation-proven-false mirror (`8f0125d`): Prusti spells a taken branch as
  the *else* of `g == false`, so no guard ever produced its positive fact. Only
  `bool_after_join` remains -- a join-representation gap, not a decomposition one: both
  arms mint a fresh chunk value only *guardedly* equal to `cons(true)`, which no
  decomposition of the goal can reach.

The arithmetic boundary is no longer "order facts vs equality facts" — that reading was an
artefact of the spelling gap, which happened to hit every order-guarded case because
Prusti spells those asserts the same way. The boundary now is **one fact vs a range**:
anything discharged by a single guard fact works, anything needing interval or sum
propagation does not. The one remaining non-arithmetic blocker (`bool_after_join`) is
independent of both and is the reason an arithmetic tier alone would not finish the job.

## The tiers

The split is by *what a proof of panic-freedom needs*, not by what the Rust looks like.

| file | needs | expectation |
|---|---|---|
| `tier_a_no_arith.rs` | nothing beyond matching the check against the path condition (`d != 0` guarding `100 / d`) | should pass — a failure is a pc-propagation or goal-normalisation bug |
| `tier_b_trivial_arith.rs` | one order/equality fact (`0 < d ==> d != 0`, `d != i32::MIN`) | green since 2026-08-03; was the live frontier, and what it actually caught was goal spelling |
| `tier_c_range_arith.rs` | interval propagation and sum/product bounds (`a, b < 10000 ==> a + b` in range) | expected out of reach without an interval analysis or the Z3 tier |
| `tier_d_index.rs` | array bounds; also the only file mixing must-pass and must-fail members | Prusti's array support is thin, so this file may not encode at all |
| `probe_unreachable.rs` | contradictory path conditions, equality and order | mixed by design — establishes how `unreachable!()` differs from `panic!()` |
| `probe_unreachable_nonarith.rs` | contradictions with no arithmetic: booleans, discriminants, equality, framing | should pass — a failure is a gap the arithmetic work would not fix |
| `must_fail.rs` | nothing — every member CAN panic | must be rejected; an `OK` here is an unsoundness, not a win |

`must_fail.rs` is the point of the corpus as much as the tiers are. Failing to prove a
safe program is incompleteness and is expected in tiers B–C; *proving* an unsafe one is a
bug that no amount of arithmetic reasoning would excuse.

Prusti's own verdict, taken while encoding, is the oracle for which tier a case belongs
to: tiers A–C encode with no verification errors (so they are genuinely panic-free), and
`must_fail.rs` encodes with one error per member. Two design consequences fell out of
that check and are worth keeping in mind when adding cases:

- with overflow checks on, `a / d` carries **two** obligations, `d != 0` and
  `!(a == i32::MIN && d == -1)` — a case meant to isolate the first needs a literal
  dividend, or a guard like `d < -1` that excludes both;
- Prusti does not check reachability of an explicit `panic!` (see the note in
  `expected.txt`), so `panic!` cases measure the encoding, not us — but `unreachable!()`
  *is* checked, by a different mechanism entirely. See below.

## `panic!` vs `unreachable!` vs MIR `Unreachable`

`probe_unreachable.rs` exists because the three are encoded three different ways and only
two of them are obligations:

| construct | encoding | checked? |
|---|---|---|
| `panic!(..)` | call to **method** `m_std::rt::panic_fmt`, contract `ensures false` | **no** — the call inhales `false`, poisoning the state before the `exhale false` that follows, so the check is vacuous |
| `unreachable!()` | call to **function** `cf_core::panicking::panic`, whose contract is `requires false` (spelled as a `let`-chain bottoming out in `s_Bool_cons(false)`), then `exhale false` | **yes** — a function precondition is checked at the call site and cannot be self-discharged |
| MIR `Unreachable` terminator (dead arm after an exhaustive discriminant switch) | bare `exhale false; inhale false` | **yes** |

The distinction is method-vs-function, not panic-vs-unreachable: a method's postcondition
is assumed after the call, so `ensures false` makes everything downstream vacuous, while a
function's precondition is an obligation on the caller. Anyone adding cases should reach
for `unreachable!()` rather than `panic!()` when the point is to test that we *reject*
something.

We match Silicon on all five probe members: Prusti reports exactly one error
(`unreachable_actually_reachable`) and we reject that one too. The only difference is
`unreachable_contradiction_order`, where we are incomplete for the usual Tier B reason —
the contradiction is `0 < x` against `x < 0`, an order fact. Its equality twin
(`unreachable_contradiction_eq`, `x == 0` against `x != 0`) passes.

## Non-arithmetic unreachability gaps

`probe_unreachable_nonarith.rs` asks whether anything fails for a reason *other* than the
missing order fact. It did: five members failed with no arithmetic anywhere in the
contradiction. Three of them (`frame_other_field`, `write_then_read`, `enum_rematch`)
were one bug -- a resource-body function application losing its `f%pre` token to
`RecipeBuilder::slice` -- fixed 2026-08-03. Two remain, both boolean reification.

| member | contradiction | status |
|---|---|---|
| `bool_direct`, `bool_through_copy`, `bool_through_negation`, `bool_nested_deep` | `b` against `!b`, incl. through a copy, a stored negation, and 3 levels of nesting | pass |
| `eq_two_vars` | `x == y` against `x != y` | pass |
| `enum_via_bool` | discriminant equality carried in a `bool` | pass |
| `bool_after_join` | both arms of a join set the flag `true` | **fail** |
| `eq_transitive` | `x == y`, `y == z`, `x != z` | **fail** |
| `eq_two_literals` | `x == 5` against `x == 7` (distinct constants) | **fail** |
| `frame_other_field` | `p.x` unchanged by a write to `p.y` | **fail** |
| `write_then_read` | reading back the value just written through `&mut` | **fail** |
| `enum_rematch` | re-matching inside an arm | **fail**, and with a different error: `insufficient permission` on the `#ensures` exhale, not an assertion |

`call_congruence` (two calls to a non-`#[pure]` fn) and `guard_survives_call` (a helper
taking `&mut`) are listed FAIL because they are genuinely unprovable without a spec on the
helper — Prusti rejects exactly those two and nothing else, which is what makes the rest
of the file a clean measurement rather than a guess.

`isolate/` rebuilds these five shapes in hand-written Viper, one encoding layer at a time,
and splits them into two causes: boolean reification for `eq_transitive` and
`eq_two_literals` (confirmed — value boxing is innocent), and the
`make_generic`/`make_concrete` round trip for `frame_other_field` and `write_then_read`
(root cause not yet pinned). `bool_after_join` is not reproduced by any reduction there.
See `isolate/README.md`.

So the order-fact gap is **not** the only thing standing between us and panic-freedom.
Equality transitivity, constant distinctness, join reasoning, and same-object field
framing each block unreachability independently. `enum_rematch` is a fourth class again —
a permission failure rather than a proof failure.

## Two encoder changes this corpus needed

`tools/prusti_encode.sh` gained both, and `benchmarks/rust/` is unaffected:

1. `PRUSTI_CHECK_OVERFLOWS` is now an override (`${...:-false}`) rather than hardcoded, so
   this corpus can turn overflow checks on while the perf corpus keeps them off.
2. A non-zero exit from `prusti-rustc` is no longer fatal when it still dumped a `.vpr`.
   Prusti verifies as it encodes, so `must_fail.rs` — whose whole purpose is to be
   rejected — could not be encoded at all before this.

## Usage

    ./encode_all.sh                 # all of src/ (slow: Prusti, ~1-3 min per source)
    ./encode_all.sh tier_b_trivial_arith
    cargo build --release --bin verify
    ./check.sh                      # every stem in expected.txt
    ./check.sh tier_a_no_arith

`check.sh` classifies each mismatch and exits non-zero only on `UNSOUND`, so it can be run
while tiers B and C are still red.

## Why tier B was the interesting one

**Resolved 2026-08-03 — kept because the diagnosis it produced is the point.** Tier B is
now 10/10; what follows is what it caught and why the tier was worth building.

`physics_step.rs` in the perf corpus divides under `if 0 < den`. Prusti emits

    p_Bool_assign(_11p, mir_binop_Eq_Int_i32_Int_i32(p_Int_i32_snap(_10p), s_Int_i32_cons(0)))
    _tmp4 := p_Bool_snap(_11p)
    exhale s_Bool_value(_tmp4) == false

and we could not discharge it. Reduced, the failure was not the arithmetic but the goal
form: under `0 < x` we proved `assert !(x == 0)` but not `assert (x == 0) == false`, and
proving the former first made the latter go through. Negation lowers to
`ite(e, false, true)` and the case splitter collected only `Ite` conditions, so the
`== false` spelling gave it nothing to split on; four normalisation rewrites in
`verify/rewrite.rs` closed it and took tier B from 2/10 to 10/10 in one step.

The tier separation is what made that diagnosable. Tier A holds the shapes where the guard
matches the check directly, so the two causes stayed separable — tier A green with tier B
red said the missing piece was *not* pc propagation, which is what pointed at the goal
form rather than at the arithmetic everyone assumed. The same split still applies to what
is left: tier C red with A and B green says the missing piece is range reasoning.
