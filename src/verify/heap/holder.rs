//! What a heap holds: single chunks ([`Chunk`]) and quantified chunk sets
//! ([`QuantChunk`]), both a [`Holder`] — a body under a **presence** cube.
//!
//! Presence is the condition, relative to the heap's region, under which the
//! holder is held at all. It is private to this module, and so is a single
//! chunk's amount: the two are only meaningful together, and reading the
//! amount without its presence reports a conditionally-held chunk as fully held
//! (`tests/cases/failing/guarded_nonaliasing_two_arms.vpr`,
//! `guarded_merge_chunks.vpr`). Everything outside reads an amount through a
//! method that has folded the presence in, or keeps it attached:
//!
//! | need                                   | method                        |
//! |----------------------------------------|-------------------------------|
//! | the amount held                        | [`Chunk::gated_perm`]         |
//! | the amount as it counts under a pc     | [`Chunk::perm_under`]         |
//! | a new amount, same presence            | [`Chunk::map_perm`]           |
//! | two chunks at one address combined     | [`Chunk::zip`] + a [`Cell`] op |
//! | per-leaf facts (location axioms)       | [`Chunk::amount_and_presence`] |
//! | a chunk set's amount at an address     | [`Holder::gate`]              |
//!
//! A chunk's amount, the recipe of its positivity and its value travel as one
//! [`Cell`], and every way two of them combine is a `Cell` operation
//! ([`Cell::join`], [`Cell::sum`]); [`Chunk::zip`] only lines the presences up
//! first.
//!
//! This module is a leaf on purpose: Rust privacy is per module *and its
//! descendants*, so the private fields are private to everything else in the
//! heap too.

use std::ops::{Deref, DerefMut};
use std::rc::Rc;

use crate::verify::context::VerifyContext;
use crate::verify::heap::algebra::{perm_add, perm_add_wildcard, prove_perm_positive};
use crate::verify::heap::cond::{Cube, Lit};
use crate::verify::heap::interface::Value;
use crate::verify::heap::value::{Scalar, Sum, held_sum};
use crate::verify::heap::{ChunkPerm, Handover, QPTaken, Restored, SetId, TakeId, Taken};
use crate::vmir;

/// A body held under a presence cube. Derefs to the body, which never holds an
/// ungated amount in a public field.
#[derive(Debug, Clone)]
pub(crate) struct Holder<F> {
    /// The branch literals under which this is held, relative to the heap's
    /// region: empty when it was produced in the region's own block, extended
    /// per join by `merge_heaps` (the arm's literal) and per out-of-region use
    /// by [`crate::verify::heap::Heap::at`].
    presence: Cube,
    body: F,
}

impl<F> Deref for Holder<F> {
    type Target = F;
    fn deref(&self) -> &F {
        &self.body
    }
}

impl<F> DerefMut for Holder<F> {
    fn deref_mut(&mut self) -> &mut F {
        &mut self.body
    }
}

impl<F: Clone> Holder<F> {
    /// `body`, held wherever its heap is.
    pub(super) fn unconditional(body: F) -> Self {
        Holder {
            presence: Cube::empty(),
            body,
        }
    }

    /// The cube this is held under. Reading it alone is fine; it is the amount
    /// that must not be read without it.
    pub(super) fn presence(&self) -> &Cube {
        &self.presence
    }

    /// This holder, held only where `lit` also holds: a join arm's literal, or
    /// a region literal an out-of-region use does not establish.
    pub(super) fn under(&self, ctx: &VerifyContext<'_>, lit: Lit) -> Self {
        Holder {
            presence: self.presence.with(ctx, lit),
            body: self.body.clone(),
        }
    }

    /// `pc ∧ presence`, the cube this is actually held under in a state with
    /// path condition `pc`; `None` when the two are disjoint and it is not held
    /// there at all.
    pub(super) fn presence_within(&self, ctx: &VerifyContext<'_>, pc: &[Lit]) -> Option<Vec<Lit>> {
        self.presence.within(ctx, pc)
    }

    /// Whether `pc` syntactically entails the presence: inside `pc` this is held
    /// unconditionally.
    pub(super) fn present_under(&self, ctx: &VerifyContext<'_>, pc: &[Lit]) -> bool {
        self.presence.entailed_by(ctx, pc)
    }

    /// `body`, held exactly where this is: a record derived from this holder
    /// (what an exhale of it took) that must count only where it does.
    pub(super) fn with_body<G>(&self, body: G) -> Holder<G> {
        Holder {
            presence: self.presence.clone(),
            body,
        }
    }

    /// `presence ? perm : 0`: an amount computed from this holder's body, as
    /// it is actually held.
    pub(super) fn gate(&self, ctx: &VerifyContext<'_>, perm: ChunkPerm) -> ChunkPerm {
        ChunkPerm::gate(ctx, &self.presence, perm)
    }
}

// ---------------------------------------------------------------------------
// Single chunks
// ---------------------------------------------------------------------------

/// What a single chunk holds at its location: an amount, the recipe of its
/// being positive, and the value. The three are about one another, so they
/// change together ([`Cell::join`], [`Cell::sum`], [`Cell::debited`]). All
/// private: see the module docs.
#[derive(Debug, Clone)]
pub(crate) struct Cell {
    /// The amount, without the holder's presence.
    perm: ChunkPerm,
    /// In a certificate walk, the recipe of `0 < perm` (of the bare amount):
    /// what the sum of two cells needs to mirror the value it picks
    /// (`Value::sum`). `None` outside a certificate walk, and once the amount
    /// changes in a way it does not follow (a debit, a presence folded in).
    held: Option<vmir::Val>,
    /// The value it was given, with its recipe (see [`Value`]): in a
    /// certificate walk (function/resource verification) the recipe-space term
    /// that names it, so a later `Deref` purifies to the pure term this chunk
    /// was produced from. It is the location's value only where the cell holds
    /// a positive share, and a cell drained to zero on some path keeps its old
    /// one. Reads and takes get a value through `view::Shares`, which resolves
    /// it under that condition; see [`Chunk::stored_value`] for the code that
    /// only carries it along.
    value: Value,
}

impl Cell {
    /// The cell at `addr` at a join on `cond` of an arm holding `a` and one
    /// holding `b`: the amounts as a `Select` on `cond`, the values joined
    /// ([`Value::join`]). Method bodies only, so there is no positivity
    /// recipe.
    pub(super) fn join(
        ctx: &mut VerifyContext<'_>,
        addr: egg::Id,
        cond: egg::Id,
        a: &Cell,
        b: &Cell,
    ) -> Cell {
        let value = Value::join(ctx, &Scalar { addr }, cond, &a.value, &b.value);
        Cell {
            perm: ChunkPerm::select(ctx, cond, a.perm.clone(), b.perm.clone()),
            held: None,
            value,
        }
    }

    /// The cell holding `a` and `b` at once at one location: `pa + pb`, the
    /// location's value with the agreement between the two assumed
    /// ([`Value::sum`]; `known` says which amounts are known positive under
    /// `pc`), and the recipe of `0 < pa + pb` from the operands'.
    pub(super) fn sum(
        ctx: &mut VerifyContext<'_>,
        a: &Cell,
        b: &Cell,
        known: (bool, bool),
        pc: &[Lit],
    ) -> Cell {
        let (pa, pb) = (&a.perm, &b.perm);
        // Structural sum: `perm_add` distributes over the gates' `Select`s, so
        // the total keeps one concrete amount per branch (`Select(b, 1/1+1/2,
        // 0+1/2)`) rather than collapsing to an opaque `AddR` whose positivity
        // nothing can decide. Safe only because the bound axiom gates each leaf
        // by the arm that reaches it (`for_each_leaf_under`): otherwise an
        // over-bound arm would fold the bound ungated and make the unit
        // inconsistent.
        let (p0, p1) = (pa.to_id(ctx), pb.to_id(ctx));
        // A sum with a wildcard in it is an opaque `AddR` leaf nothing can
        // decide, so it is not built: `perm_add_wildcard` replaces the
        // wildcard-bearing leaves with fresh shares carrying the facts instead.
        // Gated on `has_wildcard` first, so a wildcard-free program builds
        // byte-identical terms for one `bool` test.
        let perm = if ctx.has_wildcard && (pa.has_wild() || pb.has_wild()) {
            perm_add_wildcard(ctx, pa, pb)
        } else {
            perm_add(ctx, pa, pb)
        };
        let value = Value::sum(
            ctx,
            Sum {
                perm: p0,
                held: a.held.as_ref(),
                value: &a.value,
            },
            Sum {
                perm: p1,
                held: b.held.as_ref(),
                value: &b.value,
            },
            known,
            pc,
        );
        let held = held_sum(ctx, a.held.as_ref(), b.held.as_ref());
        Cell { perm, held, value }
    }

    /// This cell holding `f(amount)` instead, its value unchanged: a debit.
    /// The positivity recipe is dropped, since a debit does not follow it.
    pub(super) fn debited(&self, f: impl FnOnce(&ChunkPerm) -> ChunkPerm) -> Cell {
        Cell {
            perm: f(&self.perm),
            held: None,
            value: self.value.clone(),
        }
    }

    /// Whether the amount is positive under `pc` (proven per leaf).
    pub(super) fn positive(&self, ctx: &mut VerifyContext<'_>, pc: &[Lit]) -> bool {
        prove_perm_positive(ctx, &self.perm, pc)
    }
}

/// One location's permission and value.
#[derive(Debug, Clone)]
pub(crate) struct Point {
    /// The address e-class this chunk sits at (its identity within a group).
    pub(crate) addr: egg::Id,
    /// What it holds there. Private: see the module docs.
    cell: Cell,
}

/// A single location's permission, held under a presence cube.
pub(crate) type Chunk = Holder<Point>;

impl Chunk {
    /// Build a chunk from a bare permission id, wrapped as a concrete
    /// `ChunkPerm::Leaf`. Test-only: production code builds the permission as a
    /// [`ChunkPerm`] first, so that a wildcard's origin and a gate's branch
    /// structure are not lost on the way in.
    #[cfg(test)]
    pub fn new(addr: egg::Id, perm: egg::Id, value: egg::Id) -> Self {
        Self::new_perm(addr, ChunkPerm::leaf(perm), None, Value::plain(value))
    }

    /// An unconditional chunk holding `perm`: an inhaled `acc`, or a demand.
    /// `held` is the recipe of `0 < perm` (see [`Cell`]).
    pub(super) fn new_perm(
        addr: egg::Id,
        perm: ChunkPerm,
        held: Option<vmir::Val>,
        value: Value,
    ) -> Self {
        Holder::unconditional(Point {
            addr,
            cell: Cell { perm, held, value },
        })
    }

    /// The value this chunk was given, **not** the location's value: that is
    /// this one only where the chunk holds a positive share (see `value`). For
    /// code that carries a value along with its chunk and states nothing about
    /// the location — a share being built, a join's value phi, a heap sum whose
    /// agreement is gated on both amounts, a write's `ite`, a debug dump. To
    /// *use* the location's value, ask `view::Shares::value`.
    pub(crate) fn stored_value(&self) -> egg::Id {
        self.body.cell.value.id
    }

    /// [`Chunk::stored_value`] with its recipe: what a share of this chunk
    /// names, or what a combinator builds the next value from.
    pub(super) fn value(&self) -> &Value {
        &self.body.cell.value
    }

    /// The recipe of the stored value (certificate walks only).
    pub(super) fn recipe(&self) -> Option<&vmir::Val> {
        self.body.cell.value.recipe.as_ref()
    }

    /// This chunk holding `value` instead (a write, a join, a heap sum). The
    /// recipe goes with it: a new value never keeps the old one's name.
    pub(super) fn set_value(&mut self, value: Value) {
        self.body.cell.value = value;
    }

    /// **The permission this chunk actually holds**: `presence ? perm : 0`.
    /// With an empty presence this is the bare amount and mints nothing.
    pub(super) fn gated_perm(&self, ctx: &VerifyContext<'_>) -> ChunkPerm {
        self.gate(ctx, self.body.cell.perm.clone())
    }

    /// The amount as it counts in a state with path condition `pc`: bare when
    /// `pc` entails the presence (the gate would be a no-op there, and the bare
    /// amount keeps the term small), gated otherwise.
    pub(super) fn perm_under(&self, ctx: &VerifyContext<'_>, pc: &[Lit]) -> ChunkPerm {
        if self.present_under(ctx, pc) {
            self.body.cell.perm.clone()
        } else {
            self.gated_perm(ctx)
        }
    }

    /// This chunk in the form a consume under `pc` works on.
    ///
    /// When `pc` entails the presence, the chunk unchanged: the consume happens
    /// inside the very region it is held in, and its presence must survive the
    /// consume as a flat cube — folded into the amount, a later join-level
    /// consume would have no structure to case-split on (how the `&mut`
    /// reborrow shape lost its proof). Otherwise the presence is folded into
    /// the amount and the chunk is unconditional.
    ///
    /// Either way [`Chunk::perm_under`] on the result is its bare amount.
    pub(super) fn normalized_for(&self, ctx: &VerifyContext<'_>, pc: &[Lit]) -> Chunk {
        if self.present_under(ctx, pc) {
            return self.clone();
        }
        Holder::unconditional(Point {
            addr: self.body.addr,
            cell: self.gated_cell(ctx),
        })
    }

    /// This chunk holding `f(amount)` under the same presence. The new amount
    /// stays conditional on it, so `f` may work on the bare amount. Its
    /// positivity recipe is dropped: `f` is a debit ([`Cell::debited`]).
    pub(super) fn map_perm(&self, f: impl FnOnce(&ChunkPerm) -> ChunkPerm) -> Chunk {
        self.with_body(Point {
            addr: self.body.addr,
            cell: self.body.cell.debited(f),
        })
    }

    /// The cell with the presence folded into its amount: what this chunk
    /// holds, as an unconditional one would. The positivity recipe is of the
    /// bare amount, not of the gated one, so it is dropped.
    fn gated_cell(&self, ctx: &VerifyContext<'_>) -> Cell {
        Cell {
            perm: self.gated_perm(ctx),
            held: None,
            value: self.body.cell.value.clone(),
        }
    }

    /// Combine two chunks at one address into one, at `a`'s address. `f`
    /// builds the new cell from the two operands' (a [`Cell`] operation).
    ///
    /// - **presences equal** — the two are held in exactly the same region, so
    ///   `f` gets the cells as they are and the result keeps the shared
    ///   presence. Flat, no `ite`, and free on the unconditional hot path.
    /// - **presences differ** — no single flat cube describes the result, so
    ///   each presence is folded into its own amount ([`Chunk::gated_cell`])
    ///   and the result is unconditional, its conditionality inside the
    ///   amount.
    pub(super) fn zip(
        ctx: &mut VerifyContext<'_>,
        a: &Chunk,
        b: &Chunk,
        f: impl FnOnce(&mut VerifyContext<'_>, &Cell, &Cell) -> Cell,
    ) -> Chunk {
        let addr = a.body.addr;
        if a.presence.same(ctx, &b.presence) {
            let cell = f(ctx, &a.body.cell, &b.body.cell);
            a.with_body(Point { addr, cell })
        } else {
            let (ca, cb) = (a.gated_cell(ctx), b.gated_cell(ctx));
            let cell = f(ctx, &ca, &cb);
            Holder::unconditional(Point { addr, cell })
        }
    }

    /// The bare amount **together with** the presence it is only meaningful
    /// under. For the location axioms, which need the amount structural (the
    /// bound is assumed per leaf, non-aliasing sums two bare leaves) and gate
    /// each fact by the cube instead. Anything stated about the amount must be
    /// gated by the cube.
    pub(super) fn amount_and_presence(&self) -> (&ChunkPerm, &Cube) {
        (&self.body.cell.perm, &self.presence)
    }

    /// Whether this chunk holds nothing on any branch (const-fold only), so it
    /// can be dropped. Where its presence fails it holds nothing anyway.
    pub(super) fn is_emptied(&self, ctx: &VerifyContext<'_>) -> bool {
        crate::verify::heap::algebra::perm_all_zero(ctx, &self.body.cell.perm)
    }

    /// A representative e-class id for this chunk's amount, WITHOUT mutating the
    /// graph. Debug/tests only — not a real permission id.
    pub(crate) fn perm_repr_id(&self) -> Option<egg::Id> {
        self.body.cell.perm.repr_id()
    }
}

// ---------------------------------------------------------------------------
// Quantified chunk sets
// ---------------------------------------------------------------------------

/// A quantified permission `forall xs :: c(xs) ==> acc(e(xs).f, p(xs))` as held:
/// its body with the captures it was inhaled with, its inverses, and what was
/// taken from it since. Its amount at an address is computed
/// (`quantified_permissions::qp_perm_at`) and gated by [`Holder::gate`].
#[derive(Debug, Clone)]
pub(crate) struct SetBody {
    /// Which inhale this set is a version of (see [`SetId`]).
    pub(super) origin: SetId,
    /// The body, still in vmir temp space.
    pub(super) q: Rc<vmir::QuantChunks>,
    /// `free_temps()` resolved at inhale: enclosing temp -> its e-class then.
    pub(super) caps: crate::verify::heap::interface::Captures,
    /// The inverse functions minted at inhale, one per binder, each from the
    /// arguments of the location's address call to that binder (see
    /// `quantified_permissions::mint_and_assume_inverse_axiom`).
    pub(super) inv: Rc<[crate::verify::lang::FuncId]>,
    /// The path condition of the inhale. With the presence, what the body is
    /// evaluated and reasoned about under ([`QuantChunk::eval_pc`]); it does
    /// not gate the set's permission: every use of the heap inside the region
    /// it was built in establishes it, and a use outside that region is
    /// rehomed by [`crate::verify::heap::Heap::at`], which extends the presence
    /// instead.
    pub(super) pc: Cube,
    /// What was exhaled from this set, each under its own presence guard:
    /// whole chunk sets and single locations. A lookup subtracts their
    /// permission at the address from this set's own, so the set's permission
    /// is never rebuilt.
    pub(super) taken: Rc<[Debit]>,
    /// The set's value map, with its recipe (same role as [`Point`]'s): in a
    /// resource body the footprint seed `SlotValue(i)`, so a read through this
    /// set purifies to `lookup(seed, addr)`.
    pub(super) value_map: Value,
    /// The maps minted when a taken set was given back to this one (see
    /// [`Restored`]); each is defined at an address where it is read.
    pub(super) restored: Rc<[Restored]>,
    /// What this set learned from other sets an exhale took from while this
    /// one kept its share (see [`Handover`]); each is assumed at an address
    /// where this set is read.
    pub(super) handovers: Rc<[Handover]>,
}

/// A quantified chunk set, held under a presence cube.
pub(crate) type QuantChunk = Holder<SetBody>;

impl QuantChunk {
    /// This set with `what` recorded as taken from it, as a new exhale of an
    /// unconditional demand (a single location).
    pub(super) fn with_taken(&self, ctx: &mut VerifyContext<'_>, what: Taken) -> QuantChunk {
        let id = TakeId::fresh(ctx);
        self.with_debit(Holder::unconditional(QPTaken { id, what }))
    }

    /// This set with `what` recorded as taken from it, as a new exhale of the
    /// holder `by` (an exhaled chunk set): it counts where `by` is held.
    pub(super) fn with_taken_by<G: Clone>(
        &self,
        ctx: &mut VerifyContext<'_>,
        by: &Holder<G>,
        what: Taken,
    ) -> QuantChunk {
        let id = TakeId::fresh(ctx);
        self.with_debit(by.with_body(QPTaken { id, what }))
    }

    fn with_debit(&self, debit: Debit) -> QuantChunk {
        let mut taken = self.taken.to_vec();
        taken.push(debit);
        let mut out = self.clone();
        out.taken = taken.into();
        out
    }

    /// What this set's body is evaluated under: the inhale's path condition
    /// and the branch literals it is held under since.
    pub(super) fn eval_pc(&self) -> Vec<Lit> {
        self.pc
            .iter()
            .chain(self.presence.iter())
            .copied()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// What was taken from a chunk set
// ---------------------------------------------------------------------------

/// One exhale recorded against a chunk set (its `taken` list), held under the
/// presence it counts under. A join extends it like any holder's
/// ([`Holder::under`]), and its amount at an address is gated through
/// [`Holder::gate`].
pub(crate) type Debit = Holder<QPTaken>;

impl Debit {
    /// A whole set this exhale took, as the set it is: held under this debit's
    /// presence. `None` for any other take.
    pub(super) fn as_set(&self) -> Option<QuantChunk> {
        match &self.what {
            Taken::Set(body) => Some(self.with_body(body.clone())),
            _ => None,
        }
    }
}
