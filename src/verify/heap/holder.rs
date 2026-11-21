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
//! | two chunks at one address combined     | [`Chunk::zip`]                |
//! | per-leaf facts (location axioms)       | [`Chunk::amount_and_presence`] |
//! | a chunk set's amount at an address     | [`Holder::gate`]              |
//!
//! This module is a leaf on purpose: Rust privacy is per module *and its
//! descendants*, so the private fields are private to everything else in the
//! heap too.

use std::ops::{Deref, DerefMut};
use std::rc::Rc;

use crate::verify::context::VerifyContext;
use crate::verify::heap::cond::{Cube, Lit};
use crate::verify::heap::{ChunkPerm, QPTaken, Restored, SetId, TakeId, Taken};
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

/// One location's permission and value.
#[derive(Debug, Clone)]
pub(crate) struct Point {
    /// The address e-class this chunk sits at (its identity within a group).
    pub(crate) addr: egg::Id,
    /// The amount, without the presence. Private: see the module docs.
    perm: ChunkPerm,
    /// The value it was given. Private, like the amount: it is the location's
    /// value only where this chunk holds a positive share, and a chunk drained
    /// to zero on some path keeps its old one. Reads and takes get a value
    /// through `view::Shares`, which resolves it under that condition; see
    /// [`Chunk::stored_value`] for the code that only carries it along.
    value: egg::Id,
    /// Recipe provenance of `value` — the recipe-space temp a certificate walk
    /// (function/resource verification) associates with the held value, so a
    /// later `Deref` purifies to the pure term this chunk was produced from.
    /// `None` in method bodies (no recipe is built) and for values without a
    /// pure recipe (fresh, merged).
    pub(crate) recipe: Option<vmir::Val>,
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
        Self::new_perm(addr, ChunkPerm::leaf(perm), value)
    }

    /// An unconditional chunk holding `perm`: an inhaled `acc`, or a demand.
    pub(super) fn new_perm(addr: egg::Id, perm: ChunkPerm, value: egg::Id) -> Self {
        Holder::unconditional(Point {
            addr,
            perm,
            value,
            recipe: None,
        })
    }

    pub(super) fn with_recipe(mut self, recipe: Option<vmir::Val>) -> Self {
        self.body.recipe = recipe;
        self
    }

    /// The value this chunk was given, **not** the location's value: that is
    /// this one only where the chunk holds a positive share (see `value`). For
    /// code that carries a value along with its chunk and states nothing about
    /// the location — a share being built, a join's value phi, a heap sum whose
    /// agreement is gated on both amounts, a write's `ite`, a debug dump. To
    /// *use* the location's value, ask `view::Shares::value`.
    pub(crate) fn stored_value(&self) -> egg::Id {
        self.body.value
    }

    /// This chunk holding `value` instead (a write, a join, a heap sum).
    pub(super) fn set_value(&mut self, value: egg::Id) {
        self.body.value = value;
    }

    /// **The permission this chunk actually holds**: `presence ? perm : 0`.
    /// With an empty presence this is the bare amount and mints nothing.
    pub(super) fn gated_perm(&self, ctx: &VerifyContext<'_>) -> ChunkPerm {
        self.gate(ctx, self.body.perm.clone())
    }

    /// The amount as it counts in a state with path condition `pc`: bare when
    /// `pc` entails the presence (the gate would be a no-op there, and the bare
    /// amount keeps the term small), gated otherwise.
    pub(super) fn perm_under(&self, ctx: &VerifyContext<'_>, pc: &[Lit]) -> ChunkPerm {
        if self.present_under(ctx, pc) {
            self.body.perm.clone()
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
            perm: self.gated_perm(ctx),
            ..self.body.clone()
        })
    }

    /// This chunk holding `f(amount)` under the same presence. The new amount
    /// stays conditional on it, so `f` may work on the bare amount.
    pub(super) fn map_perm(&self, f: impl FnOnce(&ChunkPerm) -> ChunkPerm) -> Chunk {
        let perm = f(&self.body.perm);
        Holder {
            presence: self.presence.clone(),
            body: Point {
                perm,
                ..self.body.clone()
            },
        }
    }

    /// Combine two chunks at one address into one, at `a`'s address and with
    /// `a`'s value and recipe (the caller sets others). `f` builds the new
    /// amount from the two operand amounts, and may return a by-product.
    ///
    /// - **presences equal** — the two are held in exactly the same region, so
    ///   `f` gets the bare amounts and the result keeps the shared presence.
    ///   Flat, no `ite`, and free on the unconditional hot path.
    /// - **presences differ** — no single flat cube describes the result, so each
    ///   presence is folded into its own amount and the result is
    ///   unconditional, its conditionality inside the amount.
    pub(super) fn zip<T>(
        ctx: &mut VerifyContext<'_>,
        a: &Chunk,
        b: &Chunk,
        f: impl FnOnce(&mut VerifyContext<'_>, &ChunkPerm, &ChunkPerm) -> (ChunkPerm, T),
    ) -> (Chunk, T) {
        let (pa, pb, presence) = if a.presence.same(ctx, &b.presence) {
            (a.body.perm.clone(), b.body.perm.clone(), a.presence.clone())
        } else {
            (a.gated_perm(ctx), b.gated_perm(ctx), Cube::empty())
        };
        let (perm, out) = f(ctx, &pa, &pb);
        let chunk = Holder {
            presence,
            body: Point {
                perm,
                ..a.body.clone()
            },
        };
        (chunk, out)
    }

    /// The bare amount **together with** the presence it is only meaningful
    /// under. For the location axioms, which need the amount structural (the
    /// bound is assumed per leaf, non-aliasing sums two bare leaves) and gate
    /// each fact by the cube instead. Anything stated about the amount must be
    /// gated by the cube.
    pub(super) fn amount_and_presence(&self) -> (&ChunkPerm, &Cube) {
        (&self.body.perm, &self.presence)
    }

    /// Whether this chunk holds nothing on any branch (const-fold only), so it
    /// can be dropped. Where its presence fails it holds nothing anyway.
    pub(super) fn is_emptied(&self, ctx: &VerifyContext<'_>) -> bool {
        crate::verify::heap::algebra::perm_all_zero(ctx, &self.body.perm)
    }

    /// A representative e-class id for this chunk's amount, WITHOUT mutating the
    /// graph. Debug/tests only — not a real permission id.
    pub(crate) fn perm_repr_id(&self) -> Option<egg::Id> {
        self.body.perm.repr_id()
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
    pub(super) caps: Rc<[(usize, egg::Id)]>,
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
    /// The value map associated with that chunk
    pub(super) value_map: egg::Id,
    /// Recipe provenance of `value_map`, same role as [`Point::recipe`]: in a
    /// resource body the footprint seed `SlotValue(i)`, so a read through
    /// this set purifies to `lookup(seed, addr)`. `None` in a method body.
    pub(super) recipe: Option<vmir::Val>,
    /// The maps minted when a taken set was given back to this one (see
    /// [`Restored`]); each is defined at an address where it is read.
    pub(super) restored: Rc<[Restored]>,
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
