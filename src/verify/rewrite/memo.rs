//! Applier-memo and scratch-scope machinery.
//!
//! An applier that mints new nodes must not re-mint them every saturation
//! iteration, so each such rule keeps a [`Memo`] of the instances it has already
//! built. The memo is **unit-scoped** (a fresh e-graph makes remembered ids
//! meaningless) and carries a **scratch overlay** so a throwaway probe clone can
//! record instances without polluting the live run.

use crate::dhash::HashSet;
use std::sync::Mutex;

/// Mint a scope id to be `resume`d later — for a scratch graph that outlives a
/// single run (the per-block scratch).
pub(crate) fn new_scope_id() -> u64 {
    NEXT_SCOPE.with(|g| {
        let id = g.get() + 1;
        g.set(id);
        id
    })
}

/// Start a new memo unit. Call when a fresh e-graph is created for a unit.
pub(crate) fn new_memo_unit() {
    UNIT_GEN.with(|g| g.set(g.get() + 1));
}

/// RAII marker for a scratch (throwaway-clone) saturation scope.
pub(crate) struct ScratchScope;

impl ScratchScope {
    /// A one-shot scope: its overlay dies with the returned guard. For throwaway
    /// clones (`probe`-tier probes, WD checks). Such a clone is taken from the
    /// graph *as it is now*, so every base entry is already materialized in it —
    /// it reads the base.
    pub(crate) fn enter() -> Self {
        SCRATCH_STACK.with(|s| s.borrow_mut().push((new_scope_id(), false)));
        ScratchScope
    }

    /// Re-enter the scope `id`, so a graph that outlives one run keeps its memo
    /// across runs. Must nest LIFO with every other scope.
    ///
    /// **Detached**: the graph this scope belongs to was cloned at some earlier
    /// point and the live graph has moved on since, so a base entry recorded
    /// after the clone describes an instance this graph does *not* have. Runs
    /// under a detached scope therefore ignore the base and re-instantiate into
    /// their own overlay. Reading it instead loses the instance in both graphs:
    /// the live run claims the key, and the clone — which never saw the union,
    /// because an applier unions its e-graph directly and only
    /// `VerifyContext::union` mirrors — is told it already has it.
    pub(crate) fn resume(id: u64) -> Self {
        SCRATCH_STACK.with(|s| s.borrow_mut().push((id, true)));
        ScratchScope
    }
}

impl Drop for ScratchScope {
    fn drop(&mut self) {
        SCRATCH_STACK.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

pub(super) struct MemoInner<K> {
    pub(super) unit: u64,
    pub(super) base: HashSet<K>,
    /// One set per active scratch scope, innermost last, tagged with the scope id
    /// it belongs to. Pruned lazily against [`SCRATCH_STACK`] on each insert: a
    /// position whose id no longer matches belongs to a scope that has exited, so
    /// it and everything above it are dropped.
    pub(super) overlays: Vec<(u64, HashSet<K>)>,
}

/// A unit-scoped applier memo with a scratch overlay (see module docs above).
pub(crate) struct Memo<K>(Mutex<MemoInner<K>>);

// Applier-memo scoping ("already instantiated this call/σ"). A pure cost guard
// keyed on canonical e-class ids; re-instantiating is idempotent.
//
// - **Live** runs write to a **base** set that survives across runs: the live
//   e-graph of a unit only grows, so an entry stays valid for the whole unit.
// - **Scratch** runs (`probe`-tier probes, forall-WD checks) saturate a throwaway
//   clone and write to an **overlay** instead. A leaked scratch entry would be a
//   completeness bug: the clone's ids can collide with ids the live graph mints
//   later. Reading the base from a scratch run is fine.
//
// Scopes nest, and a nested clone contains everything its parent's graph does,
// so overlays form a stack: a run reads the base plus every active overlay and
// writes to the top one. A scope's overlay lives as long as its id is on the
// stack — the long-lived per-block scratch `resume`s one id across all of its
// runs so its instances stay remembered (a fresh scope per run measured ~2×
// slower on `structs_enums.vpr`).
//
// All state is thread-local; the `Mutex` in [`Memo`] only satisfies egg's
// `Send + Sync` bounds.
thread_local! {
    /// Bumped per verification unit (`VerifyContext::new`): unit boundaries
    /// switch to a fresh e-graph, so all remembered ids are meaningless.
    static UNIT_GEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// The active scratch scopes, innermost last: a scope id whose overlay is
    /// readable (and, for the last one, writable), plus whether that scope is
    /// **detached** from the live graph (see [`ScratchScope::resume`]). Empty = a
    /// live run.
    static SCRATCH_STACK: std::cell::RefCell<Vec<(u64, bool)>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// Source of scope ids. Fresh per [`ScratchScope::enter`]; a block scratch
    /// takes one at build time and `resume`s it for each of its runs.
    static NEXT_SCOPE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

impl<K: Eq + std::hash::Hash> Memo<K> {
    pub(super) fn new() -> Self {
        Self(Mutex::new(MemoInner {
            unit: u64::MAX,
            base: HashSet::default(),
            overlays: Vec::new(),
        }))
    }

    /// `true` when `key` has not been seen in the current scope — the caller
    /// should build the instance. Records the key in the base (live run) or
    /// the overlay (scratch run).
    pub(super) fn insert(&self, key: K) -> bool {
        let unit = UNIT_GEN.with(|g| g.get());
        let mut memo = self.0.lock().unwrap();
        if memo.unit != unit {
            memo.unit = unit;
            memo.base.clear();
            memo.overlays.clear();
        }
        let stack = SCRATCH_STACK.with(|s| s.borrow().clone());
        if stack.is_empty() && memo.base.contains(&key) {
            return false;
        }
        if stack.is_empty() {
            memo.overlays.clear();
            return memo.base.insert(key);
        }
        // Drop overlays whose scope has exited (id mismatch at its position, or
        // beyond the current depth), keeping the live prefix.
        let keep = stack
            .iter()
            .zip(memo.overlays.iter())
            .take_while(|((id, _), (oid, _))| id == oid)
            .count();
        memo.overlays.truncate(keep);
        if memo.overlays.iter().any(|(_, set)| set.contains(&key)) {
            return false;
        }
        // Materialize the remaining active scopes (each nested clone contains
        // everything its parent had, so an empty set for a scope is just "nothing
        // recorded there yet"), then record against the innermost.
        for (id, _) in &stack[memo.overlays.len()..] {
            memo.overlays.push((*id, HashSet::default()));
        }
        memo.overlays
            .last_mut()
            .expect("stack non-empty")
            .1
            .insert(key)
    }
}
