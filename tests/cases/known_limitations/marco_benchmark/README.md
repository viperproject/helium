# marco_benchmark known limitations

Minimal programs, one per root cause, for the files of the
`marco_benchmark-importResolved` suite (`../HeliumBenchmarkData`) that Silicon
verifies and Helium does not. Measured on `main` at 9683333 over the 838 listed
files: none verify.

Each program verifies in Silicon and fails in Helium with the error named in
its header. When one starts verifying, `known_limitations_still_fail` fails:
move it to `tests/cases/passing/`.

## Ranked by how many files they block

**Blocked** counts files that report the error at least once. **Sole** counts
files where it is the only error, so fixing it alone makes them verify.
**Cumulative** counts files that verify after fixing that row and every row
above it. Fixing one cause can reveal a failure it hid, so treat the counts as
upper bounds.

| Case | Error | Blocked | Sole | Cumulative |
|---|---|---|---|---|
| `heap_function_in_quantifier.vpr`, `old_in_quantifier.vpr` | heap-dependent function inside a `forall` | 440 | 0 | 676 |
| `qp_predicate_in_function_precondition.vpr` | Predicate `P` used in pure expression context (a predicate trigger) | 49 | 0 | 808 |
| *(no case: timeouts)* | over 180 s (4 Ru2, Dafny-book chap05 / chap10) | 6 | 6 | 814 |

The counts are from the measurement above. Since then, quantifier bodies
are typed with their host's rules (`typed::Forall`), so `old` inside a
`forall` and field-read triggers (`{ r.v }`) are accepted. The reductions
`qp_in_predicate_body.vpr` and `qp_field_in_method_contract.vpr` (quantified
permission row) and `result_in_quantifier.vpr` now verify and moved to
`tests/cases/passing/` (`qp/field_read_trigger_in_*.vpr`,
`functions/result_in_quantifier.vpr`), and so does `field_read_in_quantifier.vpr`
(now `qp/forall_read_in_contract.vpr`). The counts were not re-measured.

All Ru2 (Prusti) quantifier rows share one shape: the heap reads in Prusti's
`memory_block_join` / `memory_block_split` postconditions never mention the
bound variable (all 865 of them), so hoisting those reads out of the `forall`
is enough; general heap-dependent quantifiers are not needed.

## Unsupported constructs and front-end rejections

| Case | Error | Files | Source |
|---|---|---|---|
| `axiom_without_trigger.vpr` | `forall` needs an explicit trigger | 14 | Nagini prelude, Gobra |
| `inhale_exhale_expression.vpr`, `forperm.vpr` | inhale-exhale `[A, B]`, `forperm` | 9 | Nagini obligations |
| `acc_in_let.vpr` | unsupported pure expression: `Acc(..)` | 2 | Gobra struct allocation |
| `unfolding_in_quantifier.vpr` | unsupported: `unfolding` inside a quantifier | 1 | linkedlists `test02` |
| `sibling_scope_redeclaration.vpr` | name already declared in this scope | 1 | Nagini SIF `lookup` |

## Reasoning gaps (reduced from single benchmark files)

| Case | Error | Source |
|---|---|---|
| `field_permission_implies_non_null.vpr` | assertion failed: `acc(t.h)` does not give `t != null` | AVLTree.nokeys `rebalance*` |
| `read_covered_by_complementary_conditional_permissions.vpr` | purify: deref outside footprint | PVL kernels `Ref__main_barrier` |
| `function_through_nested_fold.vpr` | assertion failed | linkedlists `append`, `insert` |
| `conditional_precondition_function_after_refold.vpr` | assertion failed | Gobra Tree / binary_tree `deleteLeftMost` |
| `recursive_call_in_own_postcondition.vpr` | assertion failed | VerCors `Util.isPrefix` and its lemmas |
| `sequence_case_split_on_function_branch.vpr` | assertion failed (passes with `SILVER_OXIDE_Z3_RLIMIT=50000000`) | RingBufferRd `Push` |
| `unfolding_reads_permission_held_outside_predicate.vpr` | insufficient permission | AVLTree.iterative `has` (invariant) |
| `../domain_tag_exhaustiveness_with_partial_arg.vpr` (existing) | assertion failed | Dafny-book chap04 `Eval`, `EvalList` |

`Vi/viper/all/chalice/AVLTree.iterative` (`has`) has problems with `unfolding`.

- `PP/mpp/toplas_p3_no_heap/pldi16_so_PokerHand_true` (`compare`): Silicon
  needs over 100 s, too slow as a reduction oracle. Helium fails the
  invariant-preservation check of the method's second loop, yet verifies that
  loop extracted into a method of its own, so the cause depends on the
  surrounding method.
- The six timeouts: a case that times out would only slow the suite down.
