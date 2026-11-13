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
| `value_from_chunk_without_permission/` | a consume or read takes a value from a chunk that holds no permission on the path | D | 8 | #30, #49 (bug 2) |
| `refute_on_failed_proof/` | a `refute` passes because its proof attempt was cut by a termination bound | §4.4 / §9.1 | 1 | #55 |

Closed classes, now under `tests/cases/failing/`: `scratch_drops_guards/` (B, 20 cases)
and `join_merge_ambient_cube/` (A, 4 cases).
