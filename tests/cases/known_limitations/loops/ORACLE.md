# Loop corpus — Silicon oracle

Every case below was run through Silicon and its outcome recorded. Silicon is
the reference: any disagreement between us and this table is a bug in one of the
two, and the burden is on us.

Reproduce:

```sh
Z3_EXE=/usr/sbin/z3 java -Xss512m \
  -cp ~/Documents/ethz/thesis/viperserver/target/scala-2.13/viperserver.jar \
  viper.silicon.SiliconRunner <case>.vpr
```

Silicon version: `1.1-SNAPSHOT (b4cdf753@(detached))`. Recorded 2026-08-04.

## Must verify — `known_limitations/loops/` until the stage that fixes them

Currently all fail: `viper/cfg.rs:266` rejects the back edge, `typecheck/mod.rs:1514`
rejects `while`. Promote each to `passing/loops/` as it starts verifying — the
`known_limitations_still_fail` test breaks the build to force this.

The loop cut itself now works; the cases still here need an order fact between
two non-constant terms (`i < n` gives `i + 1 <= n`; `i <= n` and `!(i < n)` give
`i == n`), which integer ranges cannot express. The four promoted by *ranges*
verify through the integer ranges of the e-class analysis and live in
`passing/loops/`.

| case | scenario | Silicon | promoted by |
|---|---|---|---|
| `scalar_sum.vpr` | S1 scalar accumulator | PASS | stage 4 |
| `list_walk.vpr` | S2 predicate traversal, no QP | PASS | ranges |
| `loop_under_if.vpr` | S3 loop under a path condition | PASS | stage 4 |
| `zero_iter.vpr` | S11 zero-iteration loop | PASS | stage 4 |
| `wildcard_ro.vpr` | S15 read-only framing via wildcard | PASS | stage 4 |
| `frame_survives.vpr` | frame preserved verbatim across the cut | PASS | stage 4 |
| `goto_loop.vpr` | Prusti's shape: label-carried invariant | PASS | stage 4 |
| `no_invariant.vpr` | S12 invariant-less head, still provable | PASS | ranges |
| `nested.vpr` | S7 nested loops | PASS | stage 7 |
| `break_simple.vpr` | S4 `break` out of a loop | PASS | ranges |
| `break_state.vpr` | S4 out edge carries live state | PASS | ranges |

## Structural (Peano) loops — `passing/loops/`

Peano naturals drive the loop by a **tag check**, so the cut is exercised with
no integer reasoning anywhere. This is what isolates loop correctness from the
arithmetic gap that blocks the integer cases above.

| case | what it pins down | Silicon | ours |
|---|---|---|---|
| `peano_countdown.vpr` | data-dependent guard on a havoc'd variable; the post-loop fact comes only from the exit edge's guard negation | PASS | PASS |
| `peano_invariant.vpr` | a **load-bearing** invariant: established, preserved across the body, consumed after | PASS | PASS |
| `peano_havoc_control.vpr` | same program, invariant dropped — must FAIL, or the cut is not havocing | FAIL | FAIL |

The last is the one that matters. `peano_invariant` passing proves the cut
carries the invariant; `peano_havoc_control` failing proves it does not carry
anything *else*. Without the pair, a cut that simply threaded the pre-loop state
through unchanged would pass the first and be silently unsound.

## Must be rejected — `failing/loops/`

Green today for the wrong reason (loops unsupported ⇒ pipeline error, which
counts as rejection). They become meaningful once stage 4 lands.

| case | why it must fail | Silicon error |
|---|---|---|
| `wildcard_ro_write.vpr` | write permission in the invariant ⇒ frame residual empty ⇒ value not retained | `assert.failed:assertion.false` — `box.f == 10` |
| `havoc_loses_value.vpr` | the cut havocs written vars even when the loop runs zero times | `assert.failed:assertion.false` — `i == 5` |
| `no_invariant_loses.vpr` | S12 — empty invariant ⇒ nothing relates `i` to `n` across the cut | `postcondition.violated:assertion.false` — `i == n` |

## The load-bearing pair

`wildcard_ro.vpr` (PASS) and `wildcard_ro_write.vpr` (FAIL) are the same program
differing only in the invariant's permission amount. Together they pin down the
mechanism that makes read-only framing work:

- **wildcard** — exhale leaves a *positive* residual in the frame, so the frame
  retains the value, and `box.f == 10` survives the loop.
- **write** — exhale takes everything, the frame residual is empty, no value
  agreement is available, and the assertion correctly fails.

Silicon's heap at the assertion in the wildcard case:

```
box@2@12.f -> ms@11@12 # $k@6@12 + W - $k@7@12
```

`ms@11@12` is a merged snapshot — `singleMerge` emits `snapEqs` equating it to
both sides. Permission is the body's wildcard plus the frame residual `W - $k`.

This pair is the highest-risk item in the whole plan (`design/loops/PLAN.md` §8):
if our wildcard exhale cannot leave a usable residual, read-only framing is lost
and every invariant gets heavier. It is in the corpus from day one so that
surfaces immediately rather than at stage 8.
