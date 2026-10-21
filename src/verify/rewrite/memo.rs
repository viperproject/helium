//! Applier-memo and scratch-scope machinery.
//!
//! An applier that mints new nodes must not re-mint them every saturation
//! iteration, so each such rule keeps a [`Memo`] of the instances it has already
//! built. The memo is **unit-scoped** (a fresh e-graph makes remembered ids
//! meaningless) and keeps a separate record per **scope** — a graph state other
//! than the live one: a probe that is rolled back, or a scope of the working graph
//! — so instances built there neither pollute the live record nor outlive the
//! state they were built in.

use crate::dhash::{HashMap, HashSet};
use std::sync::Mutex;

/// Mint a scope id to be [`ScratchScope::resume`]d later — for a scope that
/// outlives a single run (the working graph and its scopes). It stays live until
/// [`retire_scope`].
pub(crate) fn new_scope_id() -> u64 {
    let id = NEXT_SCOPE.with(|g| {
        let id = g.get() + 1;
        g.set(id);
        id
    });
    LIVE_SCOPES.with(|l| l.borrow_mut().insert(id));
    id
}

/// End scope `id`: what was recorded under it described a graph state that no
/// longer exists (rolled back), so every memo forgets it.
pub(crate) fn retire_scope(id: u64) {
    LIVE_SCOPES.with(|l| l.borrow_mut().remove(&id));
}

/// Start a new memo unit. Call when a fresh e-graph is created for a unit.
pub(crate) fn new_memo_unit() {
    UNIT_GEN.with(|g| g.set(g.get() + 1));
}

/// RAII marker for running inside a scope.
pub(crate) struct ScratchScope {
    /// Retire the scope when leaving it (a one-shot scope).
    retire: Option<u64>,
}

impl ScratchScope {
    /// A one-shot scope, retired when the returned guard drops. For a probe whose
    /// graph state is rolled back afterwards.
    pub(crate) fn enter() -> Self {
        let id = new_scope_id();
        SCRATCH_STACK.with(|s| s.borrow_mut().push(id));
        ScratchScope { retire: Some(id) }
    }

    /// Re-enter the live scope `id`, so a graph state that outlives one run keeps
    /// its memo across runs. Must nest LIFO with every other scope.
    pub(crate) fn resume(id: u64) -> Self {
        SCRATCH_STACK.with(|s| s.borrow_mut().push(id));
        ScratchScope { retire: None }
    }
}

impl Drop for ScratchScope {
    fn drop(&mut self) {
        SCRATCH_STACK.with(|s| {
            s.borrow_mut().pop();
        });
        if let Some(id) = self.retire {
            retire_scope(id);
        }
    }
}

pub(super) struct MemoInner<K> {
    pub(super) unit: u64,
    pub(super) base: HashSet<K>,
    /// One set per scope something was recorded under.
    pub(super) scoped: HashMap<u64, HashSet<K>>,
}

/// A unit-scoped applier memo with per-scope records (see module docs above).
pub(crate) struct Memo<K>(Mutex<MemoInner<K>>);

// Applier-memo scoping ("already instantiated this call/σ"). A pure cost guard
// keyed on canonical e-class ids; re-instantiating is idempotent.
//
// - **Live** runs (no scope active) write to a **base** set that survives across
//   runs: the live e-graph of a unit only grows, so an entry stays valid for the
//   whole unit.
// - Runs inside scopes read and write only scope records: the base describes the
//   live graph, which a scope's graph need not contain (the working graph is a
//   copy taken at some point and has moved on separately). A run reads the
//   records of every active scope — nested scopes are states derived from their
//   enclosing ones, so what was built there is still there — and writes to the
//   innermost. A record is dropped once its scope is retired.
//
// All state is thread-local; the `Mutex` in [`Memo`] only satisfies egg's
// `Send + Sync` bounds.
thread_local! {
    /// Bumped per verification unit (`VerifyContext::new`): unit boundaries
    /// switch to a fresh e-graph, so all remembered ids are meaningless.
    static UNIT_GEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// The active scopes, innermost last. Empty = a live run.
    static SCRATCH_STACK: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
    /// The scopes not yet retired.
    static LIVE_SCOPES: std::cell::RefCell<HashSet<u64>> =
        std::cell::RefCell::new(HashSet::default());
    /// Source of scope ids.
    static NEXT_SCOPE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

impl<K: Eq + std::hash::Hash> Memo<K> {
    pub(super) fn new() -> Self {
        Self(Mutex::new(MemoInner {
            unit: u64::MAX,
            base: HashSet::default(),
            scoped: HashMap::default(),
        }))
    }

    /// `true` when `key` has not been seen in the current scope — the caller
    /// should build the instance. Records the key in the base (live run) or
    /// the innermost scope's record.
    pub(super) fn insert(&self, key: K) -> bool {
        let unit = UNIT_GEN.with(|g| g.get());
        let mut memo = self.0.lock().unwrap();
        if memo.unit != unit {
            memo.unit = unit;
            memo.base.clear();
            memo.scoped.clear();
        }
        let innermost = SCRATCH_STACK.with(|s| s.borrow().last().copied());
        let Some(innermost) = innermost else {
            return memo.base.insert(key);
        };
        if !memo.scoped.is_empty() {
            LIVE_SCOPES.with(|live| {
                let live = live.borrow();
                memo.scoped.retain(|id, _| live.contains(id));
            });
        }
        let seen = SCRATCH_STACK.with(|s| {
            s.borrow()
                .iter()
                .any(|id| memo.scoped.get(id).is_some_and(|set| set.contains(&key)))
        });
        if seen {
            return false;
        }
        memo.scoped.entry(innermost).or_default().insert(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A record made in a scope is seen by the scopes nested in it and is gone
    /// once the scope is retired: the ids it was keyed on are rolled back and
    /// reused, so keeping it would skip a needed instance (S13).
    #[test]
    fn scope_records_die_with_their_scope() {
        new_memo_unit();
        let memo: Memo<u32> = Memo::new();
        let outer = new_scope_id();
        {
            let _outer = ScratchScope::resume(outer);
            let inner = new_scope_id();
            {
                let _inner = ScratchScope::resume(inner);
                assert!(memo.insert(7), "first instance is built");
                assert!(!memo.insert(7), "and remembered in its scope");
            }
            retire_scope(inner);
            {
                let _sibling = ScratchScope::enter();
                assert!(memo.insert(7), "a retired scope's record is forgotten");
            }
            assert!(memo.insert(8));
            {
                let _nested = ScratchScope::enter();
                assert!(!memo.insert(8), "an enclosing scope's record is seen");
            }
        }
        retire_scope(outer);
        // The live record is separate from every scope's.
        assert!(memo.insert(8), "scope records are not the live record");
        assert!(!memo.insert(8));
        let _probe = ScratchScope::enter();
        assert!(memo.insert(8), "a scope does not read the live record");
    }
}
