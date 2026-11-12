# Known unsoundness

Programs that **must be rejected** (Silicon rejects them) but that this verifier currently
accepts, with the Z3 rung on and off. `tests/suite.rs::known_unsoundness_still_verifies`
asserts the current, wrong verdict.

Each subdirectory is **one bug class**: one root cause, named in its `README.md`, together
with the root-cause class (A–G) of `plans/verifier-soundness-interfaces.md` §2.1.

## How a class is closed

The goal is not a point fix per program. A class is fixed by changing the structure of the
code (types, interfaces, who owns an invariant) so that the mistake behind it cannot be
made again at another call site. Such a fix rejects every case in the class at once; then
the whole directory moves to `tests/cases/failing/<class>/`.

The suite reports a class that is only partly rejected. That usually means the fix patched
the call sites these programs reach, not the root cause.

## Classes

| directory | root cause | design class | cases | source |
|---|---|---|---|---|
| `scratch_drops_guards/` | a guarded assumption reaches the block scratch without the guards the scratch has not assumed | B | 20 | #48, #39, #49 (bug 1) |
| `value_from_chunk_without_permission/` | a consume or read takes a value from a chunk that holds no permission on the path | D | 8 | #30, #49 (bug 2) |
| `join_merge_ambient_cube/` | a join's heap merge gates location axioms by the ambient block cube, which is empty in a join | A | 3 | #49 (bug 3) |
| `refute_on_failed_proof/` | a `refute` passes because its proof attempt was cut by a termination bound | §4.4 / §9.1 | 1 | #55 |
