//! How a heap combinator builds a new [`Value`] from held ones.
//!
//! A [`Value`] is a value term together with its recipe: in a certificate walk
//! (a function, resource or contract body, `ctx.recipe` set) the recipe-space
//! term that names it. The recipe is the value term's **mirror**: whenever the
//! heap builds a value from values that all have recipes, it emits the same
//! term in recipe space, and otherwise the result has none. Every rule for that
//! lives here, once, so a combinator cannot keep a recipe that no longer names
//! its value.
//!
//! | operation | value                                  | recipe                       |
//! |-----------|----------------------------------------|------------------------------|
//! | join      | one of them, or `ite` / the maps' join | method bodies only: none     |
//! | sum       | `joint_value` (see [`Value::sum`])     | `Ternary(0 < pa, ra, rb)`    |
//! | write     | the new value                          | method bodies only: none     |
//!
//! A certificate walk never joins, sums two heaps or writes: `HeapInst::Merge`,
//! `Union` and `Assign` are lowered only in method bodies, and a resource or
//! function body encodes its branches as gated permissions instead
//! (`translate::spatial::lower_spatial`). The operations that only method bodies
//! reach assert it ([`assert_no_recipe_walk`]) rather than build a mirror no
//! certificate would read: one that ever reached them would fail loudly instead
//! of naming a value wrongly.

use crate::verify::context::VerifyContext;
use crate::verify::heap::agree::joint_value;
use crate::verify::heap::cond::Lit;
use crate::verify::heap::interface::Value;
use crate::verify::rewrite::AxiomPure;
use crate::vmir::{self, Val};

/// Panics when `ctx` is in a certificate walk: `what` is an operation only a
/// method body performs (see the module docs).
pub(super) fn assert_no_recipe_walk(ctx: &VerifyContext<'_>, what: &str) {
    assert!(
        ctx.recipe.is_none(),
        "{what} in a certificate walk: only method bodies join, sum or write heaps"
    );
}

impl Value {
    /// The value at a join of two arms holding `a` and `b`: `a` when the two
    /// are one e-class already, else `build`'s term (an `ite` on the join's
    /// condition, or the value maps' `join`). Method bodies only.
    pub(super) fn join(
        ctx: &mut VerifyContext<'_>,
        a: &Value,
        b: &Value,
        build: impl FnOnce(&mut VerifyContext<'_>) -> egg::Id,
    ) -> Value {
        assert_no_recipe_walk(ctx, "a heap join");
        if ctx.egraph.find(a.id) == ctx.egraph.find(b.id) {
            return a.clone();
        }
        Value::plain(build(ctx))
    }

    /// The value `id` written over a held one: the old recipe never survives.
    /// Method bodies only.
    pub(super) fn written(ctx: &VerifyContext<'_>, id: egg::Id) -> Value {
        assert_no_recipe_walk(ctx, "a field write");
        Value::plain(id)
    }

    /// The value of a location two holders hold at once: `a` under amount
    /// `pa`, `b` under `pb`, with the agreement between them assumed
    /// ([`joint_value`]; `known` says which amounts are known positive under
    /// `pc`).
    ///
    /// The recipe mirrors the value [`joint_value`] picks: the known side's, or
    /// for `ite(0 < pa, va, vb)` the term `Ternary(held_a, ra, rb)`, where
    /// `held_a` is the recipe of `0 < pa` ([`Sum::held`]). Without every recipe
    /// that needs, the value has none.
    pub(super) fn sum(
        ctx: &mut VerifyContext<'_>,
        a: Sum<'_>,
        b: Sum<'_>,
        known: (bool, bool),
        pc: &[Lit],
    ) -> Value {
        let id = joint_value(ctx, (a.perm, a.value.id), (b.perm, b.value.id), known, pc);
        let recipe = match known {
            (true, _) => a.value.recipe.clone(),
            (false, true) => b.value.recipe.clone(),
            (false, false) => match (a.held, &a.value.recipe, &b.value.recipe) {
                (Some(held), Some(ra), Some(rb)) if ctx.recipe.is_some() => Some(emit(
                    ctx,
                    AxiomPure::Ternary(held.clone(), ra.clone(), rb.clone()),
                )),
                _ => None,
            },
        };
        Value { id, recipe }
    }
}

/// One operand of [`Value::sum`]: a holder's amount as an e-class, the recipe
/// of its being positive, and its value.
pub(super) struct Sum<'a> {
    pub(super) perm: egg::Id,
    pub(super) held: Option<&'a Val>,
    pub(super) value: &'a Value,
}

/// The recipe of `0 < pa + pb` from those of `0 < pa` and `0 < pb`: amounts are
/// never negative, so it is `0 < pa ∨ 0 < pb`. `None` unless both are known.
pub(super) fn held_sum(
    ctx: &mut VerifyContext<'_>,
    a: Option<&Val>,
    b: Option<&Val>,
) -> Option<Val> {
    let (a, b) = (a?, b?);
    ctx.recipe.as_ref()?;
    Some(emit(
        ctx,
        AxiomPure::Ternary(a.clone(), vmir::TRUE, b.clone()),
    ))
}

/// Emit `p` into the certificate under construction.
fn emit(ctx: &mut VerifyContext<'_>, p: AxiomPure) -> Val {
    ctx.recipe
        .as_mut()
        .expect("a recipe step is emitted only in a certificate walk")
        .emit(p)
}
