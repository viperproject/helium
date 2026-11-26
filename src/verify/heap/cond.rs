//! Conditions: conjunctive cubes of e-class literals.
//!
//! Every condition the heap stores (a chunk's presence guard, a chunk set's
//! guard and inhale pc, a heap's region, the guard of a recorded exhale) is a
//! [`Cube`]. The instruction pc arrives as a plain `&[Lit]` slice; the methods
//! here take it as such, so a stored cube and a pc never need converting.
//!
//! All comparisons are on **canonical** e-classes (`egraph.find`), decided in
//! Rust without saturation. [`Cube::holds_under`] is the one semantic check: it
//! falls back to the prover for a literal the pc does not mention verbatim.

use std::ops::Deref;
use std::rc::Rc;

use crate::verify::context::VerifyContext;
use crate::vmir::Polarity;

/// One literal of a cube: an e-class and the polarity it is assumed with.
pub(crate) type Lit = (egg::Id, Polarity);

/// A conjunction of literals; empty is `true`. Shared, so cloning one (per
/// chunk, per heap clone) is O(1).
#[derive(Debug, Clone)]
pub(crate) struct Cube(Rc<[Lit]>);

impl Deref for Cube {
    type Target = [Lit];
    fn deref(&self) -> &[Lit] {
        &self.0
    }
}

impl From<&[Lit]> for Cube {
    fn from(lits: &[Lit]) -> Self {
        Cube(Rc::from(lits))
    }
}

impl From<Vec<Lit>> for Cube {
    fn from(lits: Vec<Lit>) -> Self {
        Cube(Rc::from(lits))
    }
}

impl Default for Cube {
    fn default() -> Self {
        Cube::empty()
    }
}

impl Cube {
    /// The cube `true`.
    pub(crate) fn empty() -> Self {
        Cube(Rc::from([]))
    }

    /// `self ∧ lit` (idempotent under canonical e-classes).
    pub(crate) fn with(&self, ctx: &VerifyContext<'_>, lit: Lit) -> Cube {
        if contains(ctx, self, lit) {
            return self.clone();
        }
        let mut v = self.to_vec();
        v.push(lit);
        Cube::from(v)
    }

    /// Set-equality under canonical e-classes (order-insensitive).
    pub(crate) fn same(&self, ctx: &VerifyContext<'_>, other: &[Lit]) -> bool {
        self.len() == other.len() && self.iter().all(|&l| contains(ctx, other, l))
    }

    /// Whether `pc` syntactically entails this cube: every literal of it occurs
    /// in `pc` with the same polarity.
    pub(crate) fn entailed_by(&self, ctx: &VerifyContext<'_>, pc: &[Lit]) -> bool {
        entails(ctx, pc, self)
    }

    /// Whether this cube holds wherever `pc` does: each literal is in `pc` or
    /// proven under it. The semantic counterpart of [`Cube::entailed_by`], for
    /// cubes `reach.rs` minimised differently.
    pub(crate) fn holds_under(&self, ctx: &mut VerifyContext<'_>, pc: &[Lit]) -> bool {
        self.iter().all(|&lit| lit_holds_under(ctx, pc, lit))
    }

    /// `pc ∧ self`, `pc`'s literals first (the order the block scratch's prefix
    /// test expects), or `None` when the two are disjoint.
    pub(crate) fn within(&self, ctx: &VerifyContext<'_>, pc: &[Lit]) -> Option<Vec<Lit>> {
        meet(ctx, pc, self)
    }
}

/// Whether `cube` mentions `lit` (same class, same polarity).
fn contains(ctx: &VerifyContext<'_>, cube: &[Lit], (id, pol): Lit) -> bool {
    let c = ctx.egraph.find(id);
    cube.iter()
        .any(|&(i, p)| p == pol && ctx.egraph.find(i) == c)
}

/// Whether `cube` entails `sub`: every literal of `sub` occurs in `cube`.
fn entails(ctx: &VerifyContext<'_>, cube: &[Lit], sub: &[Lit]) -> bool {
    sub.iter().all(|&l| contains(ctx, cube, l))
}

/// Whether the single literal `lit` holds wherever `pc` does.
pub(crate) fn lit_holds_under(ctx: &mut VerifyContext<'_>, pc: &[Lit], (id, pol): Lit) -> bool {
    if contains(ctx, pc, (id, pol)) {
        return true;
    }
    let goal = match pol {
        Polarity::Positive => id,
        Polarity::Negative => expr!(ctx, { id } == false),
    };
    ctx.prove_under_pc(goal, pc)
}

/// The conjunction `a ∧ b` (`a`'s literals first), or `None` when they are
/// **disjoint**: some literal occurs in both with opposite polarity, so the two
/// are never true in the same state.
///
/// Decided by e-class identity rather than left to saturation: the
/// complementary pair (`[c+]` and `[c−]`, the two one-sided chunks of a single
/// join) is exactly the case that must be dropped, and dropping it must not
/// depend on an `ite`-idempotence rewrite firing.
pub(crate) fn meet(ctx: &VerifyContext<'_>, a: &[Lit], b: &[Lit]) -> Option<Vec<Lit>> {
    let mut out: Vec<Lit> = a.to_vec();
    for &(id, pol) in b {
        let c = ctx.egraph.find(id);
        match out.iter().find(|(i, _)| ctx.egraph.find(*i) == c) {
            Some((_, p)) if *p == pol => {}
            Some(_) => return None,
            None => out.push((id, pol)),
        }
    }
    Some(out)
}
