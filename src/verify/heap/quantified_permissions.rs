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

use std::rc::Rc;
use std::sync::Arc;

use egg::Id;

use crate::verify::context::VerifyContext;
use crate::verify::error::VerifyError;
use crate::verify::heap::ChunkPerm;
use crate::verify::lang::{FuncId, Symbolic};
use crate::verify::quant::{Canonical, QuantRecipe};
use crate::verify::rewrite::AxiomPure;
use crate::vmir::{BinOp, InstKind, PermInst, PermVal, Polarity, QuantChunks, Val};

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
        lits.push((expr!(ctx, (0 / 1) < r { p }), Polarity::Positive));
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

/// The permission a held chunk set has at an address, its guard folded in, or
/// `None` when the set cannot name that address. Supplied by the caller, which
/// owns the lookup (`qp_perm_at`).
pub(crate) type HeldAt<'s> =
    dyn FnMut(&mut VerifyContext<'_>, Id) -> Result<Option<ChunkPerm>, VerifyError> + 's;

/// Whether the exhaled quantified permission `q` is covered by a held chunk set
/// under the exhale's path condition `pc`: every location it names has at least
/// the amount it wants available there (own permission minus what was already
/// taken),
///
/// ```text
/// forall ys ::  c'(ys)  ==>  p'(ys) <= perm_held(e'(ys))
/// ```
///
/// Proven by skolemization, as the injectivity check: `ys` are fresh constants
/// on a scratch graph (nothing leaks into the live one; the certificate recipe
/// is paused likewise), `c'(ys)` joins the path condition, and the comparison
/// is discharged per branch by `prove_sufficient_aligned`. This is what makes
/// recording the exhaled set as taken from the held one sound: no location
/// goes negative.
pub(crate) fn check_qp_contained(
    ctx: &mut VerifyContext<'_>,
    q: &QuantChunks,
    pc: &[(Id, Polarity)],
    instantiate: &mut Instantiate<'_>,
    held_at: &mut HeldAt<'_>,
) -> Result<bool, VerifyError> {
    let recipe = ctx.recipe.take();
    let res = ctx.with_scratch_graph(|ctx| {
        let ys = fresh_binders(ctx, q);
        let inst = instantiate(ctx, &ys)?;
        let call = q.loc_call();
        let addr = ctx.add(Symbolic::FuncApp(
            crate::verify::func_registry::func_id_for_member(call.function),
            call.type_args.clone().into(),
            inst.loc_args.into(),
        ));
        let Some(avail) = held_at(ctx, addr)? else {
            return Ok(false);
        };
        let mut lits = pc.to_vec();
        lits.push((inst.cond, Polarity::Positive));
        Ok(crate::verify::heap::algebra::prove_sufficient_aligned(
            ctx, &avail, &inst.perm, &lits,
        ))
    });
    ctx.recipe = recipe;
    res
}

/// One fresh constant per binder of `q`.
fn fresh_binders(ctx: &mut VerifyContext<'_>, q: &QuantChunks) -> Vec<Id> {
    q.bound
        .iter()
        .map(|ty| ctx.fresh_symbolic_value(ty.clone()))
        .collect()
}

/// Mint the inverse functions of an inhaled chunk set and assume its inverse
/// axiom under the inhale's path condition `pc`. Returns the inverses, one per
/// binder, for the chunk to carry (a lookup at an address applies them to the
/// address call's arguments).
///
/// With `e(xs)` the arguments of the location's address call (as in the
/// injectivity check above) and `inv_j` fresh, the axiom is
///
/// ```text
/// forall xs :: {triggers}   c(xs) ∧ 0 < p(xs)  ==>  ⋀_j inv_j(e(xs)) = xs_j
/// ```
///
/// Its antecedent is the injectivity check's, which is what makes it
/// consistent; so it must only be assumed once [`check_qp_injectivity`] passed
/// under the same `pc`. Silicon's second axiom `e(inv(r)) = r` is left to the
/// lookup, which conjoins it into the chunk's condition at the address it asks
/// about.
///
/// The body is canonicalized exactly as a pure `forall`'s ([`Canonical`]) and
/// the axiom's steps are appended to it. `caps` is [`QuantChunks::free_temps`],
/// resolved.
pub(crate) fn mint_and_assume_inverse_axiom(
    ctx: &mut VerifyContext<'_>,
    q: &QuantChunks,
    caps: &[(usize, Id)],
    pc: &[(Id, Polarity)],
) -> Result<Rc<[FuncId]>, VerifyError> {
    // One fresh inverse per binder, from the address function's parameters.
    let loc_call = q.loc_call();
    let (params, _) = ctx.function_signature(loc_call.function, &loc_call.type_args);
    let inv: Rc<[FuncId]> = q
        .bound
        .iter()
        .map(|ret| {
            let f = ctx.alloc.qp_inverse(params.clone(), ret.clone());
            ctx.func_ret_types.insert(f, ret.clone());
            f
        })
        .collect();
    // Without a trigger the axiom could never fire; assuming nothing is sound.
    if q.triggers.is_empty() {
        return Ok(inv);
    }

    let free: Vec<usize> = caps.iter().map(|&(k, _)| k).collect();
    let (interner, decls) = (ctx.interner, ctx.decls);
    let names = move |m| crate::verify::context::member_name_in(interner, decls, m);
    let table = Arc::clone(ctx.alloc.quant_table());
    let mut c = Canonical::new(
        ctx.alloc,
        &table.read().expect("recipe table lock"),
        &names,
        q.binder_base,
        q.bound.len(),
        &free,
        &q.triggers,
        &q.body.insts,
    )?;

    // `c(xs) ∧ 0 < p(xs)  ==>  ⋀_j inv_j(e(xs)) = xs_j`
    let perms: Vec<&PermInst> = q
        .body
        .insts
        .iter()
        .filter_map(|i| match &i.kind {
            InstKind::Perm(p) => Some(p),
            _ => None,
        })
        .collect();
    let pos = positive(&mut c, &q.body.perm, &perms);
    let cond = c.rename(&q.body.cond);
    let ante = and(&mut c, cond, pos);
    let args: Vec<Val> = loc_call.args.iter().map(|a| c.rename(a)).collect();
    let eqs: Vec<Val> = inv
        .iter()
        .enumerate()
        .map(|(j, &func)| {
            let binder = c.rename(&Val::Temp(q.binder_base + j));
            let app = c.push(AxiomPure::App {
                func,
                type_args: vec![],
                args: args.clone(),
            });
            c.push(AxiomPure::Binary(BinOp::Eq, app, binder))
        })
        .collect();
    let concl = eqs
        .into_iter()
        .reduce(|a, b| and(&mut c, a, b))
        .expect("a quantified permission binds a variable");
    let res = c.push(AxiomPure::Ternary(ante, concl, crate::vmir::TRUE));

    // Ground (a QP cannot be generic) and fresh, so pushed without dedup.
    let rid = table
        .write()
        .expect("recipe table lock")
        .push_fresh(QuantRecipe {
            n_caps: c.n_caps(),
            bound: q.bound.clone(),
            n_tys: 0,
            own_tys: vec![],
            groups: c.groups,
            insts: c.insts,
            res,
            guard: None,
        });
    let children = caps.iter().map(|&(_, id)| id).collect();
    let node = ctx.add(Symbolic::Forall(rid, Box::new([]), children));
    ctx.assume_guarded(node, pc.iter().rev().copied());
    Ok(inv)
}

/// `a ∧ b`, as `a ? b : false` (the shape the e-graph's `ite` rule splits).
fn and(c: &mut Canonical, a: Val, b: Val) -> Val {
    c.push(AxiomPure::Ternary(a, b, crate::vmir::FALSE))
}

/// `0 < p` as a boolean, pushed through the permission's `ite`s; a `wildcard`
/// is positive by definition.
fn positive(c: &mut Canonical, p: &PermVal, perms: &[&PermInst]) -> Val {
    match p {
        PermVal::Amount(v) => {
            let v = c.rename(v);
            c.push(AxiomPure::Binary(BinOp::LtR, crate::vmir::none(), v))
        }
        PermVal::Wildcard => crate::vmir::TRUE,
        PermVal::Temp(i) => {
            let PermInst::Ite(cond, t, e) = perms[*i];
            let (t, e) = (positive(c, t, perms), positive(c, e, perms));
            let cond = c.rename(cond);
            c.push(AxiomPure::Ternary(cond, t, e))
        }
    }
}
