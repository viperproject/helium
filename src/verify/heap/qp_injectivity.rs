//! Receiver injectivity of a quantified permission.
//!
//! `forall xs :: c(xs) ==> acc(f(e(xs)), p(xs))` is well-formed only if distinct
//! instances that actually hold permission name distinct locations:
//!
//! ```text
//! forall xs1, xs2 ::
//!     c(xs1) ∧ c(xs2) ∧ 0 < p(xs1) ∧ 0 < p(xs2) ∧ e(xs1) = e(xs2)  ==>  xs1 = xs2
//! ```
//!
//! This is what later licenses the inverse-function view of the chunk set (one
//! binding per location). Without it, an inhale could split one location's
//! permission across several bindings and the inverse axiom would assume `false`.
//!
//! The universal is proven by **skolemization**: both binder tuples are fresh
//! constants, the antecedents go into the path condition, and the binder
//! equalities are the goals.
//!
//! The antecedent compares the address call's **arguments** (`e(xs)`, the
//! receiver of a field, the argument tuple of a predicate), not the addresses
//! themselves, as Viper does. Address functions are uninterpreted here, so
//! `f(a) = f(b)` would not give the prover `a = b`; since every instance of one
//! chunk set applies the same address function, equal arguments is exactly
//! equal locations.

use egg::Id;

use crate::verify::context::VerifyContext;
use crate::verify::error::VerifyError;
use crate::verify::heap::ChunkPerm;
use crate::vmir::{Polarity, QuantChunks};

/// One instance of a quantified permission's body at given binder values: what
/// the injectivity check reads off it.
pub(crate) struct QpInstance {
    /// `c(xs)`: whether this binding contributes a chunk.
    pub(crate) cond: Id,
    /// The arguments of the location's address call: the receiver of `e.f`, the
    /// argument tuple of `P(..)`.
    pub(crate) loc_args: Vec<Id>,
    /// `p(xs)`, with the enclosing branch conditions folded in.
    pub(crate) perm: ChunkPerm,
}

/// Evaluate the body of `q` at the given binder values. Supplied by the caller,
/// which owns the evaluation state the body's captured temps resolve against.
pub(crate) type Instantiate<'s> =
    dyn FnMut(&mut VerifyContext<'_>, &[Id]) -> Result<QpInstance, VerifyError> + 's;

/// Prove the receiver injectivity of `q` under the path condition `pc`, or fail
/// with a [`VerifyError::SideCondition`].
///
/// Runs on a scratch e-graph: the fresh binders and both body instances exist
/// only for this proof and must not leak into the live graph. The certificate
/// recipe (if one is being built) is paused for the same reason, exactly as for
/// a pure `forall`'s well-definedness check.
pub(crate) fn check_qp_injectivity(
    ctx: &mut VerifyContext<'_>,
    q: &QuantChunks,
    pc: &[(Id, Polarity)],
    instantiate: &mut Instantiate<'_>,
) -> Result<(), VerifyError> {
    let recipe = ctx.recipe.take();
    let res = ctx.with_scratch_graph(|ctx| check_in_scratch(ctx, q, pc, instantiate));
    ctx.recipe = recipe;
    res
}

fn check_in_scratch(
    ctx: &mut VerifyContext<'_>,
    q: &QuantChunks,
    pc: &[(Id, Polarity)],
    instantiate: &mut Instantiate<'_>,
) -> Result<(), VerifyError> {
    let x1 = fresh_binders(ctx, q);
    let x2 = fresh_binders(ctx, q);
    let i1 = instantiate(ctx, &x1)?;
    let i2 = instantiate(ctx, &x2)?;
    assert_eq!(
        i1.loc_args.len(),
        i2.loc_args.len(),
        "two instances of one quantified permission must apply the same address function"
    );

    // pc ++ antecedents: the order `prove_under_pc` (and the block scratch's
    // cube prefix test) expects, as in a pure `forall`'s WD check.
    let mut lits = pc.to_vec();
    for inst in [&i1, &i2] {
        lits.push((inst.cond, Polarity::Positive));
        let p = inst.perm.to_id(ctx);
        lits.push((expr!(ctx, (0 / 1) <r { p }), Polarity::Positive));
    }
    for (&a1, &a2) in i1.loc_args.iter().zip(&i2.loc_args) {
        lits.push((expr!(ctx, { a1 } == { a2 }), Polarity::Positive));
    }

    // One goal per binder: `xs1 = xs2` component-wise.
    for (&b1, &b2) in x1.iter().zip(&x2) {
        let goal = expr!(ctx, { b1 } == { b2 });
        if !ctx.prove_under_pc(goal, &lits) {
            return Err(VerifyError::SideCondition(
                "quantified permission might not be injective",
            ));
        }
    }
    Ok(())
}

/// One fresh constant per binder of `q`.
fn fresh_binders(ctx: &mut VerifyContext<'_>, q: &QuantChunks) -> Vec<Id> {
    q.bound
        .iter()
        .map(|ty| ctx.fresh_symbolic_value(ty.clone()))
        .collect()
}
