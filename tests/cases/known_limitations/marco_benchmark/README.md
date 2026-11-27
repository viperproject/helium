# marco_benchmark known limitations

Minimal programs, one per root cause, for the files of the
`marco_benchmark-importResolved` suite (`../HeliumBenchmarkData`) that Helium
does not verify. The file list is the union of two exports from the benchmark
database, `marco_benchmark_inconclusive.csv` (765 files, Helium crashed or
reported an unsupported construct) and `marco_benchmark_incomplete.csv` (252
files, Helium reported a failed check). The export names 1052 files; 35 Ru1
Prusti names in it match no file in the suite and are left out.

Measured on `main` at 4dfd65a6 with `verify --json`, 8 files at a time, 180 s
per file:

| | Files | Verify | Fail | Timeout |
|---|---|---|---|---|
| inconclusive | 765 | 0 | 745 | 20 |
| incomplete | 252 | 82 | 129 | 41 |
| **total** | **1017** | **82** | **874** | **61** |

The 82 files that now verify were fixed after the export was taken.

Each program fails in Helium with the error named in its header, and the
program is valid Viper that should verify. When one starts verifying,
`known_limitations_still_fail` fails: move it to `tests/cases/passing/`.
Silicon was not run on the new cases. They follow Viper semantics: chained
comparisons, `Int * Perm`, macros that take a field name and macro-bound
variables are part of Silver.

## Ranked by how many files they block

**Blocked** counts files that report the error at least once. **Sole** counts
files where it is the only error, so fixing it alone makes them verify.
**Cumulative** counts files that verify after fixing that row and every row
above it. A `SKIP` (a member that depends on a rejected one) is not counted as
an error. Fixing one cause can reveal a failure it hid, so treat the counts as
upper bounds.

| Case | Error | Blocked | Sole | Cumulative | Main sources |
|---|---|---|---|---|---|
| `axiom_without_trigger.vpr` | `forall` needs an explicit trigger | 261 | 95 | 95 | Nagini, Mendel, `Vi/viper`, `DaV/viper` |
| `inhale_exhale_expression.vpr` | inhale-exhale expression `[A, B]` | 178 | 38 | 218 | Nagini, Ru1 Prusti, Voila |
| *(no case: timeouts)* | over 180 s | 61 | 61 | 279 | Ru2 Prusti (30), Nagini (28), Gobra Dafny-book chap05 / chap10 |
| *(no case: see below)* | cannot find callable | 40 | 14 | 294 | reachability and flow preambles |
| `field_read_argument_in_trigger.vpr` | a trigger may only contain variables, literals and nested applications | 37 | 4 | 299 | Flow framework, Voila, Ru1 Prusti |
| `exists.vpr` | unsupported construct: `exists` | 31 | 1 | 311 | Voila, reachability, Nagini, Mendel |
| *(no case: Ru2 lifetime tokens)* | insufficient permission | 27 | 27 | 338 | Ru2 Prusti |
| *(no case: see below)* | assertion failed | 26 | 6 | 362 | Nagini, Ru2 Prusti, single files |
| `magic_wand.vpr` | magic wand `--*`, `package`, `apply` | 23 | 16 | 384 | Ru1 / Ru2 Prusti pledges |
| `pure_conjunct_in_quantified_permission.vpr` | pure conjunct inside a quantified permission | 23 | 0 | 388 | Flow framework |
| `macro_bound_variable_clash.vpr` | name already declared in this scope | 19 | 0 | 407 | Flow framework |
| `macro_field_parameter.vpr` | unknown field `$field$` | 17 | 1 | 420 | reachability preamble |
| `quasihavoc.vpr` | `quasihavoc`, `quasihavocall` | 17 | 0 | 437 | Voila |
| `chained_comparison.vpr` | cannot unify `Bool` and a number | 13 | 9 | 450 | `Vi/viper/dp` |
| `forperm.vpr` | unsupported construct: `forperm` | 13 | 13 | 463 | RSL2Viper |
| `acc_in_let.vpr` | unsupported pure expression: `Acc(..)` | 9 | 5 | 472 | Gobra |
| `int_times_perm.vpr` | cannot unify `Int` and `Perm` | 7 | 4 | 479 | 2vyper |
| *(no case)* | 11 smaller errors, listed below | 26 | 2 | 505 | |


### Not reduced

- **Ru2 lifetime tokens** (`insufficient permission`, 27 files, all sole):
  the failing exhale is a `lft_tok_sep_return$N` call or the exhale of the
  full `LifetimeToken` amount at the end of the method. Example:
  `Ru2/new/viper_programs/004/purification.rs_m_test5_fail-MemorySafety-Regular-MemorySafety`.
  It needs the whole method: cutting it after any earlier statement makes it
  verify, and the take/return pattern on its own (with `newlft`, symbolic
  amounts `p / 24`, `intersect(Set(..))` lifetimes) verifies too. The file
  takes about 2 minutes, which makes delta reduction slow.
- **`assertion failed`** (26 files): 15
  Nagini recursive functions fail `#requires` (`fib`, `fib4__rec`,
  `popcount`, ...). A reduction to the `int___ge__(n, __prim__int___box__(0))`
  subtype obligation verifies. The others are the single-file reductions in
  the next section.
- **`cannot find callable`** (40 files): mostly preamble files of
  `Rea/reachability` and `Flo/` that are not complete programs (for example
  `_preambles/common/zopg` uses `exists_DEP` without declaring it).
- **11 smaller errors** (26 files): `Undefined variable` (7, RSL2Viper),
  `undeclared type` (7, preambles and Dafny-book chapter 10),
  `` `child` is a macro, not a field`` (4, reachability), `cannot unify Perm and Int` (4, RSL2Viper), purify:
  deref outside footprint (3, the PVL kernels; fixed, see below), and one file each for
  `insufficient permission`, a heap read in a nested quantifier, a `wildcard`
  literal, a quantified permission in `assert` / `assume` and a sibling-scope
  redeclaration (`sibling_scope_redeclaration.vpr`, Nagini SIF `lookup`).
- **Timeouts** (61 files): a case that times out would only slow the suite
  down. With 8 files at a time, some of them may finish alone.

## Reasoning gaps (reduced from single benchmark files)

These files are still in the list and still fail as described, but with other
errors as well, so they are not ranked above.

| Case | Error | Source |
|---|---|---|
| `field_permission_implies_non_null.vpr` | assertion failed: `acc(t.h)` does not give `t != null` | AVLTree.nokeys `rebalance*` |
| `function_through_nested_fold.vpr` | assertion failed | linkedlists `append`, `insert` |
| `recursive_call_in_own_postcondition.vpr` | assertion failed | VerCors `Util.isPrefix` and its lemmas |
| `unfolding_reads_permission_held_outside_predicate.vpr` | insufficient permission | AVLTree.iterative `has` (invariant; this file is not in the current export) |
| `../domain_tag_exhaustiveness_with_partial_arg.vpr` (existing) | assertion failed | Dafny-book chap04 `Eval`, `EvalList` |
| `unfolding_in_quantifier.vpr` | unsupported: `unfolding` inside a quantifier | linkedlists `test02` (now rejected earlier, for a missing trigger) |

- `PP/mpp/toplas_p3_no_heap/pldi16_so_PokerHand_true` (`compare`): Silicon
  needs over 100 s, too slow as a reduction oracle. Helium fails the
  invariant-preservation check of the method's second loop, yet verifies that
  loop extracted into a method of its own, so the cause depends on the
  surrounding method.
