pub(crate) mod algebra;

use std::cell::Cell;
use std::rc::Rc;

use lasso::Spur;

use crate::verify::context::VerifyContext;
use crate::verify::lang::Symbolic;
use crate::vmir::{Bound, Literal, Polarity, Type};

/// A path condition in verify space: a single conjunctive cube of e-class
/// literals (the analog of a block's `PathConds`, already minimized by
/// `reach.rs` at lowering). Shared `Rc` so cloning a `Heap` per-inst stays O(1).
pub type HeapPc = Rc<[(egg::Id, Polarity)]>;

/// S0 merge instrumentation (gated by `SILVER_OXIDE_TRACE_MERGE`). Counts the
/// join-merge structures the pc-hoist plan aims to flatten: `Select` nodes
/// actually built (reachability towers), `0`-leaf constructions (absence encoded
/// as a `?:0` amount), and the deepest `Select` tree seen. Reset/dumped per
/// method by [`MergeTrace::take`]. Pure diagnostics — no effect when the flag is
/// off (the `enabled()` check short-circuits every hot-path bump).
#[derive(Default, Clone, Copy)]
pub struct MergeTrace {
    pub selects_built: u64,
    pub zero_leaves: u64,
    pub max_depth: u32,
}

thread_local! {
    static MERGE_TRACE: Cell<MergeTrace> = const { Cell::new(MergeTrace {
        selects_built: 0,
        zero_leaves: 0,
        max_depth: 0,
    }) };
    static MERGE_TRACE_ON: Cell<Option<bool>> = const { Cell::new(None) };
}

impl MergeTrace {
    fn enabled() -> bool {
        MERGE_TRACE_ON.with(|c| match c.get() {
            Some(b) => b,
            None => {
                let b = std::env::var_os("SILVER_OXIDE_TRACE_MERGE").is_some();
                c.set(Some(b));
                b
            }
        })
    }

    fn bump_select(depth: u32) {
        if !Self::enabled() {
            return;
        }
        MERGE_TRACE.with(|c| {
            let mut t = c.get();
            t.selects_built += 1;
            t.max_depth = t.max_depth.max(depth);
            c.set(t);
        });
    }

    pub(crate) fn bump_zero() {
        if !Self::enabled() {
            return;
        }
        MERGE_TRACE.with(|c| {
            let mut t = c.get();
            t.zero_leaves += 1;
            c.set(t);
        });
    }

    /// Read and reset the per-thread counters. `None` when tracing is off.
    pub fn take() -> Option<MergeTrace> {
        if !Self::enabled() {
            return None;
        }
        MERGE_TRACE.with(|c| {
            let t = c.get();
            c.set(MergeTrace::default());
            Some(t)
        })
    }
}

impl ChunkPerm {
    /// The `Select`-nesting depth of this perm tree (a `Leaf` is 0). Diagnostic.
    fn depth(&self) -> u32 {
        match self {
            ChunkPerm::Leaf { .. } => 0,
            ChunkPerm::Select { then, els, .. } => 1 + then.depth().max(els.depth()),
        }
    }
}

/// A chunk's permission as an explicit term, held OUTSIDE the union-find so a
/// branch structure collapses without saturation. `Leaf` is an e-class id (a
/// literal `1/1`, a symbolic real, a wildcard) and is treated as OPAQUE — never
/// structurally decomposed. A `Select` comes from either of two places, both via
/// the [`ChunkPerm::select`] smart constructor: a control-flow **join** merge, and
/// a **gated `acc`** — `Sink::gate_perm` keeps a path condition as a `PermInst::Ite`
/// structure rather than folding it into the amount, so `build_perm` lands it here.
/// An ungated frontend perm is still a `Leaf`, for which [`ChunkPerm::to_id`] is
/// the identity.
#[derive(Debug, Clone)]
pub enum ChunkPerm {
    Leaf {
        id: egg::Id,
        /// Whether this amount **came from** a `wildcard` — origin, not sign. A
        /// `1/2` is positive too, and must not trigger [`perm_add_wildcard`]'s
        /// fresh-share rule; a `wildcard` must, however its class const-folds.
        /// Carried structurally because the e-graph cannot answer it: congruence
        /// puts wildcard-bearing terms into a literal's class, and by the time the
        /// merge side sees two amounts their provenance is otherwise gone.
        wild: bool,
    },
    Select {
        cond: egg::Id,
        then: Box<ChunkPerm>,
        els: Box<ChunkPerm>,
    },
}

impl ChunkPerm {
    /// A concrete leaf: an amount with no `wildcard` in its provenance.
    pub(crate) fn leaf(id: egg::Id) -> Self {
        ChunkPerm::Leaf { id, wild: false }
    }

    /// A leaf minted from a `wildcard` (or from a term one flowed into). Selects
    /// the fresh-share rule on the produce side; see the `wild` field.
    pub(crate) fn wild_leaf(id: egg::Id) -> Self {
        ChunkPerm::Leaf { id, wild: true }
    }

    /// Whether any leaf of this tree came from a `wildcard`.
    pub(crate) fn has_wild(&self) -> bool {
        match self {
            ChunkPerm::Leaf { wild, .. } => *wild,
            ChunkPerm::Select { then, els, .. } => then.has_wild() || els.has_wild(),
        }
    }

    /// Structural equality with LEAVES compared by e-class `find` (so a `1/1`
    /// from either arm counts as equal). O(size), no saturation.
    pub(crate) fn same(ctx: &VerifyContext<'_>, a: &ChunkPerm, b: &ChunkPerm) -> bool {
        match (a, b) {
            (ChunkPerm::Leaf { id: x, .. }, ChunkPerm::Leaf { id: y, .. }) => {
                ctx.egraph.find(*x) == ctx.egraph.find(*y)
            }
            (
                ChunkPerm::Select {
                    cond: c1,
                    then: t1,
                    els: e1,
                },
                ChunkPerm::Select {
                    cond: c2,
                    then: t2,
                    els: e2,
                },
            ) => {
                ctx.egraph.find(*c1) == ctx.egraph.find(*c2)
                    && Self::same(ctx, t1, t2)
                    && Self::same(ctx, e1, e2)
            }
            _ => false,
        }
    }

    /// Descend into an arm while it re-branches on the SAME condition class,
    /// taking the then-side (`take_then`) or els-side. This is the `ite`
    /// idempotence identity `ite(c, ite(c, a, b), e) = ite(c, a, e)` (and the
    /// dual for the els arm): under `c` an inner `ite(c, …)` is decided, so only
    /// its matching branch survives. CFG lowering makes many joins in one match
    /// arm share the arm's reach `cond`, so without this the perm accretes a
    /// redundant `Select` layer per such join.
    fn collapse_same_cond(
        ctx: &VerifyContext<'_>,
        cond_c: egg::Id,
        arm: ChunkPerm,
        take_then: bool,
    ) -> ChunkPerm {
        match arm {
            ChunkPerm::Select {
                cond: c2,
                then,
                els,
            } if ctx.egraph.find(c2) == cond_c => {
                let inner = if take_then { *then } else { *els };
                Self::collapse_same_cond(ctx, cond_c, inner, take_then)
            }
            other => other,
        }
    }

    /// This tree as seen from *inside* one arm of `cond`: every re-branch on the
    /// same condition class is decided, so only the matching side survives. The
    /// public face of [`Self::collapse_same_cond`], used by `perm_add` to restrict
    /// one addend while descending into the other.
    pub(crate) fn restrict(
        ctx: &VerifyContext<'_>,
        cond: egg::Id,
        arm: ChunkPerm,
        take_then: bool,
    ) -> ChunkPerm {
        Self::collapse_same_cond(ctx, ctx.egraph.find(cond), arm, take_then)
    }

    /// The join-select smart constructor. `cond` is the then-edge reach value.
    /// Applies, in order: (0) `ite`-idempotence flattening of arms that branch on
    /// the same `cond`; (i) same-amount collapse (`then ≡ els ⇒ then`, the
    /// give-back / untouched kill); (ii) dead-arm drop when `cond` const-folds to
    /// a boolean literal; (iii) otherwise a `Select`. Deterministic, bounded.
    pub fn select(
        ctx: &mut VerifyContext<'_>,
        cond: egg::Id,
        then: ChunkPerm,
        els: ChunkPerm,
    ) -> Self {
        let cond_c = ctx.egraph.find(cond);
        let then = Self::collapse_same_cond(ctx, cond_c, then, true);
        let els = Self::collapse_same_cond(ctx, cond_c, els, false);
        if Self::same(ctx, &then, &els) {
            return then;
        }
        match ctx.egraph[cond_c].data.known() {
            Some(Literal::Bool(true)) => return then,
            Some(Literal::Bool(false)) => return els,
            _ => {}
        }
        let out = ChunkPerm::Select {
            cond,
            then: Box::new(then),
            els: Box::new(els),
        };
        MergeTrace::bump_select(out.depth());
        out
    }

    /// Lower to an e-graph id — ONLY where the prover needs an e-class
    /// (sufficiency, bound/non-alias axioms, `perm > 0` framing). For a `Leaf`
    /// this is the identity (no node added), so with the flag off the migrated
    /// heap ops are byte-identical to Stage 3.
    pub fn to_id(&self, ctx: &mut VerifyContext<'_>) -> egg::Id {
        match self {
            ChunkPerm::Leaf { id, .. } => *id,
            ChunkPerm::Select { cond, then, els } => {
                let t = then.to_id(ctx);
                let e = els.to_id(ctx);
                ctx.add(Symbolic::Ite([*cond, t, e]))
            }
        }
    }

    /// The leaf amount id if this is a bare `Leaf` (no branch structure).
    pub fn as_leaf(&self) -> Option<egg::Id> {
        match self {
            ChunkPerm::Leaf { id, .. } => Some(*id),
            ChunkPerm::Select { .. } => None,
        }
    }

    /// Visit every leaf amount id **with the branch conditions that reach it** —
    /// the `Select` conditions on the path from the root, each with the polarity
    /// of the arm taken.
    ///
    /// A leaf only describes the permission *under its own arm*, so any fact
    /// stated about it must be gated by that cube. [`Self::for_each_leaf`] drops
    /// this, which is sound only while every leaf independently satisfies the
    /// fact being stated.
    pub fn for_each_leaf_under(&self, f: &mut impl FnMut(egg::Id, &[(egg::Id, Polarity)])) {
        fn go(
            p: &ChunkPerm,
            path: &mut Vec<(egg::Id, Polarity)>,
            f: &mut impl FnMut(egg::Id, &[(egg::Id, Polarity)]),
        ) {
            match p {
                ChunkPerm::Leaf { id, .. } => f(*id, path),
                ChunkPerm::Select { cond, then, els } => {
                    path.push((*cond, Polarity::Positive));
                    go(then, path, f);
                    path.pop();
                    path.push((*cond, Polarity::Negative));
                    go(els, path, f);
                    path.pop();
                }
            }
        }
        go(self, &mut Vec::new(), f);
    }

    /// A representative e-class id for debug display, WITHOUT mutating the graph
    /// (a `Leaf`'s id, or a `Select`'s condition). Viz only — not a real perm id.
    pub fn repr_id(&self) -> egg::Id {
        match self {
            ChunkPerm::Leaf { id, .. } => *id,
            ChunkPerm::Select { cond, .. } => *cond,
        }
    }
}

// ---------------------------------------------------------------------------
// Guard cube algebra
//
// A presence guard is a flat conjunctive cube of e-class literals ([`HeapPc`]).
// These four operations are the whole vocabulary over it. They live here, beside
// the private [`Chunk::guard`] field they read, so the encapsulation the accessors
// below establish is not leaked back out to every caller that needs to compare or
// extend a cube.
// ---------------------------------------------------------------------------

/// The conjunction of two presence cubes, or `None` when they are **disjoint** —
/// some literal occurs in both with opposite polarity, so the two chunks are
/// never held in the same state and no joint fact about them may be stated.
///
/// Decided in Rust by e-class identity rather than left to saturation: the
/// complementary pair (`[c+]` and `[c−]`, the two one-sided chunks of a single
/// join) is exactly the case that must be dropped, and dropping it must not
/// depend on an `ite`-idempotence rewrite firing.
pub(crate) fn cube_meet(
    ctx: &VerifyContext<'_>,
    a: &[(egg::Id, Polarity)],
    b: &[(egg::Id, Polarity)],
) -> Option<Vec<(egg::Id, Polarity)>> {
    let mut out: Vec<(egg::Id, Polarity)> = a.to_vec();
    for (id, pol) in b {
        let c = ctx.egraph.find(*id);
        match out.iter().find(|(i, _)| ctx.egraph.find(*i) == c) {
            Some((_, p)) if p == pol => {}
            Some(_) => return None,
            None => out.push((*id, *pol)),
        }
    }
    Some(out)
}

/// Whether `cube` entails `sub`: every literal of `sub` already appears in `cube`
/// with the same polarity, so `sub` holds wherever `cube` does. Syntactic on
/// e-classes, like [`cube_eq`] — no prove, no saturation.
pub(crate) fn cube_entails(
    ctx: &VerifyContext<'_>,
    cube: &[(egg::Id, Polarity)],
    sub: &[(egg::Id, Polarity)],
) -> bool {
    sub.iter().all(|(id, pol)| {
        let c = ctx.egraph.find(*id);
        cube.iter()
            .any(|(i, p)| p == pol && ctx.egraph.find(*i) == c)
    })
}

/// Set-equality of two guard cubes under canonical e-classes (order-insensitive;
/// guards are small).
pub(crate) fn cube_eq(
    ctx: &VerifyContext<'_>,
    a: &[(egg::Id, Polarity)],
    b: &[(egg::Id, Polarity)],
) -> bool {
    a.len() == b.len()
        && a.iter().all(|(ia, pa)| {
            let ca = ctx.egraph.find(*ia);
            b.iter()
                .any(|(ib, pb)| pa == pb && ctx.egraph.find(*ib) == ca)
        })
}

/// Append `lit` to a guard cube (idempotent under canonical e-classes).
pub(crate) fn cube_push(
    ctx: &VerifyContext<'_>,
    base: &[(egg::Id, Polarity)],
    lit: (egg::Id, Polarity),
) -> HeapPc {
    let c = ctx.egraph.find(lit.0);
    if base
        .iter()
        .any(|(i, p)| *p == lit.1 && ctx.egraph.find(*i) == c)
    {
        return Rc::from(base.to_vec());
    }
    let mut v = base.to_vec();
    v.push(lit);
    Rc::from(v)
}

/// Materialize `guard ? perm : 0`. With an empty guard this is `perm` unchanged
/// (no node minted), which is why the unconditional hot path pays nothing for
/// gating. Prefer [`Chunk::gated_perm`]; this free form is for the two callers
/// that hold a [`ChunkPerm`] without its chunk.
pub(crate) fn gate_perm_by_guard(
    ctx: &mut VerifyContext<'_>,
    perm: &ChunkPerm,
    guard: &[(egg::Id, Polarity)],
) -> ChunkPerm {
    let mut acc = perm.clone();
    for (id, pol) in guard {
        let zero = ChunkPerm::leaf(expr!(ctx, 0 / 1));
        acc = match pol {
            Polarity::Positive => ChunkPerm::select(ctx, *id, acc, zero),
            Polarity::Negative => ChunkPerm::select(ctx, *id, zero, acc),
        };
    }
    acc
}

/// The **kind** of a heap location: a field/predicate group, the held value
/// type, and the permission bound. This is exactly the content of a location's
/// `Type::Addr` — the chunks of one kind share a group in the heap. Sourced
/// straight from VMIR (the address's `Type::Addr`), never from e-graph inference.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocationKind {
    pub group: Spur,
    pub value: Type,
    pub bound: Bound,
}

impl LocationKind {
    /// Extract the kind from an address type. `None` for a non-`Addr` type.
    pub fn from_addr_type(ty: &Type) -> Option<Self> {
        match ty {
            Type::Addr {
                group,
                value,
                bound,
            } => Some(Self {
                group: *group,
                value: (**value).clone(),
                bound: bound.clone(),
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Chunk {
    /// The address e-class this chunk sits at (its identity within a group).
    pub(crate) addr: egg::Id,
    /// The **ungated** amount. Private: a chunk's real permission is
    /// `guard ? perm : 0`, and reading this raw reports a conditionally-held
    /// chunk as fully held. Go through [`Chunk::gated_perm`] unless the call site
    /// re-attaches the guard itself — see [`Chunk::ungated_perm`].
    perm: ChunkPerm,
    pub(crate) value: egg::Id,
    /// Reachability guard: the flat cube of branch literals under which this
    /// chunk is present (empty = unconditional). A held-on-one-arm conditional
    /// footprint carries its arm's reach cube here (appended per join by
    /// `merge_heaps`), instead of nesting a `?:0` `Select` into `perm` — so
    /// `perm` stays a guard-free amount and no `0` leaf is ever built. Presence is
    /// `guard ∧` the consuming instruction's pc; the amount holds unconditionally
    /// under it, and the consume sites reconstruct the `guard?perm:0` obligation
    /// transiently ([`Chunk::gated_perm`]).
    ///
    /// Private for the same reason as `perm`: the two are only meaningful
    /// together. Read it through [`Chunk::guard`], and prefer the derived
    /// [`Chunk::presence_cube`] / [`Chunk::pc_entails_guard`] where they fit.
    guard: HeapPc,
    /// Recipe provenance of `value` — the recipe-space temp a certificate walk
    /// (function/resource verification) associates with the held value, so a
    /// later `Deref` purifies to the pure term this chunk was produced from.
    /// `None` in method bodies (no recipe is built) and for values without a
    /// pure recipe (fresh, merged).
    pub(crate) recipe: Option<crate::vmir::Val>,
}

impl Chunk {
    /// Build a chunk from a bare permission id, wrapped as a concrete
    /// `ChunkPerm::Leaf`. Test-only: production code builds the permission as a
    /// [`ChunkPerm`] first, so that a wildcard's origin and a gate's branch
    /// structure are not lost on the way in.
    #[cfg(test)]
    pub fn new(addr: egg::Id, perm: egg::Id, value: egg::Id) -> Self {
        Self::new_perm(addr, ChunkPerm::leaf(perm), value)
    }

    /// Build a chunk from an explicit permission term (the join merge's select).
    pub fn new_perm(addr: egg::Id, perm: ChunkPerm, value: egg::Id) -> Self {
        Self {
            addr,
            perm,
            value,
            guard: Rc::from(Vec::new()),
            recipe: None,
        }
    }

    pub fn with_recipe(mut self, recipe: Option<crate::vmir::Val>) -> Self {
        self.recipe = recipe;
        self
    }

    /// This chunk's residual presence guard (relative to the heap `pc`).
    pub(crate) fn guard(&self) -> &[(egg::Id, Polarity)] {
        &self.guard
    }

    /// This chunk's guard as a shared [`HeapPc`], for re-attaching to a chunk
    /// derived from it ([`Chunk::with_guard`]). `Rc` clone — O(1), unlike
    /// rebuilding one from [`Chunk::guard`].
    pub(crate) fn guard_pc(&self) -> HeapPc {
        self.guard.clone()
    }

    /// Return this chunk with residual presence guard `guard` (structural share).
    pub(crate) fn with_guard(mut self, guard: HeapPc) -> Self {
        self.guard = guard;
        self
    }

    /// Return this chunk holding permission `perm` (guard and value unchanged).
    pub(crate) fn with_perm(mut self, perm: ChunkPerm) -> Self {
        self.perm = perm;
        self
    }

    /// **The permission this chunk actually holds**: `guard ? perm : 0`. With an
    /// empty guard this is the bare amount and mints no node, so the
    /// unconditional path is free.
    ///
    /// This is the default reading. A conditionally-held chunk keeps a flat cube
    /// and a guard-free amount, so anything that states a *fact* about the
    /// permission — a bound, a disequality, a sufficiency proof, a framing check —
    /// must ask for it this way or it will treat a conditional footprint as fully
    /// held. That mistake has produced two soundness bugs
    /// (`tests/cases/failing/guarded_nonaliasing_two_arms.vpr`,
    /// `guarded_merge_chunks.vpr`).
    pub(crate) fn gated_perm(&self, ctx: &mut VerifyContext<'_>) -> ChunkPerm {
        gate_perm_by_guard(ctx, &self.perm, &self.guard)
    }

    /// The **raw** amount, guard NOT folded in.
    ///
    /// Valid only when the call site discharges the guard itself, in one of two
    /// ways, and says which in a comment:
    /// (a) it re-attaches this chunk's guard to whatever it builds
    ///     ([`Chunk::with_guard`]), so the conditionality is preserved, or
    /// (b) it gates the result by a cube that entails the guard — including the
    ///     case where the instruction's `pc` already does
    ///     ([`Chunk::pc_entails_guard`]), which makes the gate a no-op.
    ///
    /// If neither holds, use [`Chunk::gated_perm`].
    pub(crate) fn ungated_perm(&self) -> &ChunkPerm {
        &self.perm
    }

    /// The cube this chunk is actually present under here: `pc ∧ guard`, or
    /// `None` when the two are disjoint — the chunk is unreachable in this state
    /// and states nothing.
    pub(crate) fn presence_cube(
        &self,
        ctx: &VerifyContext<'_>,
        pc: &[(egg::Id, Polarity)],
    ) -> Option<Vec<(egg::Id, Polarity)>> {
        cube_meet(ctx, pc, &self.guard)
    }

    /// Whether `pc` already entails this chunk's guard, i.e. the consume happens
    /// inside the very region the chunk is present in — so gating would be a
    /// no-op and the flat cube should be carried through untouched.
    pub(crate) fn pc_entails_guard(
        &self,
        ctx: &VerifyContext<'_>,
        pc: &[(egg::Id, Polarity)],
    ) -> bool {
        cube_entails(ctx, pc, &self.guard)
    }

    /// A representative e-class id for this chunk's amount, WITHOUT mutating the
    /// graph. Debug/tests only — not a real permission id.
    pub(crate) fn perm_repr_id(&self) -> egg::Id {
        self.perm.repr_id()
    }
}

/// Symbolic heap: chunks partitioned by [`LocationKind`]. Each group is an
/// `Rc<[Chunk]>` so cloning a heap (frequent — one per instruction) shares the
/// group slices; mutating one group copies just that slice (copy-on-write). The
/// outer `im::HashMap` is itself structurally shared.
///
/// Within a group, identity is the chunk's `addr` e-class — today at most one
/// chunk per address (the same-address merge happens in `declaration.rs`); the vec
/// shape is a prerequisite for the lazy Σ-ite permission model.
///
/// A chunk's reachability is carried by its own [`Chunk::guard`] (a flat cube),
/// not by a heap-level path condition — the join merge appends the branch literal
/// per chunk, and the consume sites gate against `guard ∧` the instruction pc.
#[derive(Debug, Clone)]
pub struct Heap {
    /// Unseeded hasher: iteration order (and so the order join/consume code mints
    /// terms in) must be a function of the heap alone, not of the process.
    groups: im::HashMap<LocationKind, Rc<[Chunk]>, rustc_hash::FxBuildHasher>,
}

impl Heap {
    pub fn empty() -> Self {
        Self {
            groups: im::HashMap::default(),
        }
    }

    /// The chunks of one location kind (empty slice if none held).
    pub fn chunks_of(&self, kind: &LocationKind) -> &[Chunk] {
        self.groups.get(kind).map(|r| &r[..]).unwrap_or(&[])
    }

    /// The chunk held at `addr` within `kind`'s group (exact e-class match).
    pub fn chunk(&self, kind: &LocationKind, addr: egg::Id) -> Option<&Chunk> {
        self.chunks_of(kind).iter().find(|c| c.addr == addr)
    }

    /// Insert `chunk` into `kind`'s group, replacing any chunk already at the
    /// same `addr` (preserving today's one-chunk-per-address semantics).
    pub fn with_chunk(&self, kind: &LocationKind, chunk: Chunk) -> Self {
        let mut v: Vec<Chunk> = self
            .groups
            .get(kind)
            .map(|r| r.to_vec())
            .unwrap_or_default();
        match v.iter_mut().find(|c| c.addr == chunk.addr) {
            Some(slot) => *slot = chunk,
            None => v.push(chunk),
        }
        Self {
            groups: self.groups.update(kind.clone(), v.into()),
        }
    }

    /// Drop the chunk at `addr` in `kind`'s group (removing the group if empty).
    pub fn without_chunk(&self, kind: &LocationKind, addr: egg::Id) -> Self {
        let Some(group) = self.groups.get(kind) else {
            return self.clone();
        };
        let v: Vec<Chunk> = group.iter().filter(|c| c.addr != addr).cloned().collect();
        let groups = if v.is_empty() {
            self.groups.without(kind)
        } else {
            self.groups.update(kind.clone(), v.into())
        };
        Self { groups }
    }

    /// Every chunk in the heap, paired with its location kind.
    pub fn entries(&self) -> impl Iterator<Item = (&LocationKind, &Chunk)> {
        self.groups
            .iter()
            .flat_map(|(k, cs)| cs.iter().map(move |c| (k, c)))
    }

    /// The location kinds (group keys) present in the heap.
    pub fn kinds(&self) -> impl Iterator<Item = &LocationKind> {
        self.groups.keys()
    }

    /// The chunk at `addr` within `kind`'s group, matched by **canonical**
    /// e-class (`ctx.egraph.find`) so an address that only aliases the stored
    /// key under the currently-assumed context still resolves. Used by the
    /// Stage-4 join merge (a chunk held on both arms must line up by identity).
    pub fn chunk_canon(
        &self,
        ctx: &VerifyContext<'_>,
        kind: &LocationKind,
        addr: egg::Id,
    ) -> Option<&Chunk> {
        let canon = ctx.egraph.find(addr);
        self.chunks_of(kind)
            .iter()
            .find(|c| ctx.egraph.find(c.addr) == canon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two heaps built by the same insertions iterate their groups in the same
    /// order. With a per-map random seed they did not, even within one process.
    #[test]
    fn group_iteration_order_is_deterministic() {
        let mut rodeo = lasso::Rodeo::default();
        let kinds: Vec<LocationKind> = (0..32)
            .map(|i| LocationKind {
                group: rodeo.get_or_intern(format!("g{i}")),
                value: Type::Int,
                bound: Bound::Unbounded,
            })
            .collect();
        let build = || {
            kinds.iter().enumerate().fold(Heap::empty(), |h, (i, k)| {
                let id = egg::Id::from(i);
                h.with_chunk(k, Chunk::new(id, id, id))
            })
        };
        let order = |h: &Heap| h.kinds().cloned().collect::<Vec<_>>();
        assert_eq!(order(&build()), order(&build()));
    }
}
