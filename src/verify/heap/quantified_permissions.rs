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

use crate::verify::eval::{
    EvalState, eval_perm_inst, eval_perm_structural, eval_pure_inst, pc_lits_raw,
};
use crate::verify::heap::holders::{HolderAt, greedy_takes, holders_at, sum_perms};
use crate::verify::heap::{
    Chunk, Heap, LocationKind, QuantChunk, SetId, Taken, gate_perm_by_guard,
};
use crate::vmir;

use crate::verify::analysis::ConstFold;
use crate::verify::context::VerifyContext;
use crate::verify::error::VerifyError;
use crate::verify::heap::ChunkPerm;
use crate::verify::lang::{FuncId, RecipeId, Symbolic};
use crate::verify::quant::{Canonical, QuantRecipe};
use crate::verify::rewrite::AxiomPure;
use crate::vmir::{BinOp, InstKind, PermInst, PermVal, Polarity, QuantChunks, Val};

/// One instance of a quantified permission's body at given binder values: what
/// the injectivity check reads off it.
pub(super) struct QpInstance {
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
type Instantiate<'s> =
    dyn FnMut(&mut VerifyContext<'_>, &[Id]) -> Result<QpInstance, VerifyError> + 's;

/// Prove the receiver injectivity of `q` under the path condition `pc`, or fail
/// with a [`VerifyError::SideCondition`].
///
/// Runs on a scratch e-graph: the fresh binders and both body instances exist
/// only for this proof and must not leak into the live graph. The certificate
/// recipe (if one is being built) is paused for the same reason, exactly as for
/// a pure `forall`'s well-definedness check.
fn check_qp_injectivity(
    ctx: &mut VerifyContext<'_>,
    q: &QuantChunks,
    pc: &[(Id, Polarity)],
    instantiate: &mut Instantiate<'_>,
) -> Result<(), VerifyError> {
    ctx.pause_recipe(|ctx| ctx.with_scratch_graph(|ctx| check_in_scratch(ctx, q, pc, instantiate)))
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
type HeldAt<'s> =
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
fn check_qp_contained(
    ctx: &mut VerifyContext<'_>,
    q: &QuantChunks,
    pc: &[(Id, Polarity)],
    instantiate: &mut Instantiate<'_>,
    held_at: &mut HeldAt<'_>,
) -> Result<bool, VerifyError> {
    ctx.pause_recipe(|ctx| {
        ctx.with_scratch_graph(|ctx| {
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
        })
    })
}

/// One fresh constant per binder of `q`.
/// Whether the amount of `q` (captured temps resolved through `caps`) is never
/// negative where it holds: `forall xs :: c(xs) ==> not(p(xs) < 0)` under the
/// path condition `pc`. Proven for fresh binders on a scratch graph, with the
/// certificate recipe paused, as [`check_qp_injectivity`].
pub(super) fn check_qp_nonnegative(
    ctx: &mut VerifyContext<'_>,
    q: &QuantChunks,
    caps: &[(usize, Id)],
    pc: &[(Id, Polarity)],
) -> Result<bool, VerifyError> {
    ctx.pause_recipe(|ctx| {
        ctx.with_scratch_graph(|ctx| {
            let xs = fresh_binders(ctx, q);
            let inst = eval_qp_instance(ctx, caps, q, &xs, pc)?;
            let mut lits = pc.to_vec();
            lits.push((inst.cond, Polarity::Positive));
            let p = inst.perm.to_id(ctx);
            let zero = expr!(ctx, 0 / 1);
            let goal = expr!(ctx, not({ p } < r { zero }));
            Ok(ctx.prove_under_pc(goal, &lits))
        })
    })
}

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
fn mint_and_assume_inverse_axiom(
    ctx: &mut VerifyContext<'_>,
    q: &QuantChunks,
    caps: &[(usize, Id)],
    pc: &[(Id, Polarity)],
) -> Result<Rc<[FuncId]>, VerifyError> {
    let loc_call = q.loc_call();
    // Without a trigger the axiom could never fire; assuming nothing is sound.
    if q.triggers.is_empty() {
        return Ok(mint_inverses(ctx, q));
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

    // A set with the same axiom up to its inverses, over the same captures,
    // already has inverses: reuse them, so its forall is the same node.
    let shape = InverseShape {
        bound: q.bound.clone(),
        groups: c.groups.clone(),
        insts: c.insts.clone(),
        ante: ante.clone(),
        args: args.clone(),
    };
    let cap_ids: Vec<Id> = caps.iter().map(|&(_, id)| id).collect();
    if let Some((rid, inv)) = ctx.inverse_axioms.find(&mut ctx.egraph, &shape, &cap_ids) {
        let node = ctx.add(Symbolic::Forall(rid, Box::new([]), cap_ids.into()));
        ctx.assume_guarded(node, pc.iter().rev().copied());
        return Ok(inv);
    }
    let inv = mint_inverses(ctx, q);
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
    ctx.inverse_axioms
        .insert(&ctx.egraph, shape, &cap_ids, rid, inv.clone());
    let node = ctx.add(Symbolic::Forall(rid, Box::new([]), cap_ids.into()));
    ctx.assume_guarded(node, pc.iter().rev().copied());
    Ok(inv)
}

/// One fresh inverse per binder of `q`, from its address function's parameters.
fn mint_inverses(ctx: &mut VerifyContext<'_>, q: &QuantChunks) -> Rc<[FuncId]> {
    let loc_call = q.loc_call();
    let (params, _) = ctx.function_signature(loc_call.function, &loc_call.type_args);
    q.bound
        .iter()
        .map(|ret| {
            let f = ctx.alloc.qp_inverse(params.clone(), ret.clone());
            ctx.func_ret_types.insert(f, ret.clone());
            f
        })
        .collect()
}

/// An inverse axiom with its inverses left out: everything
/// [`mint_and_assume_inverse_axiom`] canonicalizes before it names them. Two
/// chunk sets of one shape over the same captures state the same axiom up to
/// the inverses, so they can share them.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct InverseShape {
    bound: Box<[vmir::Type]>,
    groups: Vec<Vec<crate::verify::rewrite::PreparedTerm>>,
    insts: Vec<crate::verify::rewrite::AxiomInst>,
    ante: Val,
    args: Vec<Val>,
}

/// The inverse axioms assumed in one member's e-graph, by shape and captures.
///
/// Sharing is what keeps a run of loops linear: each loop's invariant set
/// would otherwise mint its own forall, and every one of them keeps firing on
/// every later trigger term. It is sound because one shape over the same
/// captures is one proposition, licensed by each set's own injectivity check;
/// the forall is assumed again under each set's path condition. The captures
/// must agree, not just the shape: inverses shared across different captures
/// would equate the binders of two different sets naming one location.
#[derive(Clone, Default)]
pub(crate) struct InverseAxioms(
    std::collections::HashMap<InverseShape, Vec<(Vec<Id>, RecipeId, Rc<[FuncId]>)>>,
);

impl InverseAxioms {
    fn find(
        &self,
        egraph: &mut egg::EGraph<Symbolic, ConstFold>,
        shape: &InverseShape,
        caps: &[Id],
    ) -> Option<(RecipeId, Rc<[FuncId]>)> {
        self.0.get(shape)?.iter().find_map(|(cs, rid, inv)| {
            cs.iter()
                .zip(caps)
                .all(|(&a, &b)| egraph.find(a) == egraph.find(b))
                .then(|| (*rid, inv.clone()))
        })
    }

    fn insert(
        &mut self,
        egraph: &egg::EGraph<Symbolic, ConstFold>,
        shape: InverseShape,
        caps: &[Id],
        rid: RecipeId,
        inv: Rc<[FuncId]>,
    ) {
        let caps = caps.iter().map(|&c| egraph.find(c)).collect();
        self.0.entry(shape).or_default().push((caps, rid, inv));
    }
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

/// Evaluate a quantified permission's body at `binders`, with its captured
/// temps resolved through `caps` (the values fixed at the inhale, see
/// [`EvalState::frame_from_caps`]), never the live state: a lookup may run in
/// another temp space or after the captured variables were reassigned. Body
/// permission temps count from 0, so the frame starts with no permissions.
///
/// The body holds only `Pure` and `Perm` instructions and its location is an
/// address call (both guaranteed by `lower_quant_chunks`); anything else is a
/// lowering bug and panics.
fn eval_qp_instance(
    ctx: &mut VerifyContext<'_>,
    caps: &[(usize, egg::Id)],
    q: &vmir::QuantChunks,
    binders: &[egg::Id],
    host_pc: &[(egg::Id, Polarity)],
) -> Result<QpInstance, VerifyError> {
    // The instance only feeds proofs (injectivity, containment, a lookup's
    // permission); its terms never enter a certificate, and its temps carry no
    // recipes. A read's certificate term is `lookup(seed, addr)`, built by the
    // `Deref` itself.
    ctx.pause_recipe(|ctx| eval_qp_instance_unrecorded(ctx, caps, q, binders, host_pc))
}

fn eval_qp_instance_unrecorded(
    ctx: &mut VerifyContext<'_>,
    caps: &[(usize, egg::Id)],
    q: &vmir::QuantChunks,
    binders: &[egg::Id],
    host_pc: &[(egg::Id, Polarity)],
) -> Result<QpInstance, VerifyError> {
    debug_assert_eq!(binders.len(), q.bound.len());
    let mut state = EvalState::frame_from_caps(q.binder_base, caps);
    for (&id, ty) in binders.iter().zip(q.bound.iter()) {
        state.push_val(id, ty.clone(), None);
    }

    for inst in &q.body.insts {
        let mut pc_lits = host_pc.to_vec();
        pc_lits.extend(pc_lits_raw(ctx, &state, &inst.pc));
        match &inst.kind {
            InstKind::Pure(ty, pi) => {
                let (id, _) = eval_pure_inst(ctx, &state, ty, pi, &pc_lits, None)?;
                state.push_val(id, ty.clone(), None);
            }
            InstKind::Perm(pi) => {
                let p = eval_perm_inst(ctx, &state, pi);
                state.push_perm(p, pi.clone());
            }
            other => panic!("quantified permission body holds a non-pure inst: {other:?}"),
        }
    }

    Ok(QpInstance {
        cond: state.get_val(ctx, &q.body.cond),
        loc_args: q
            .loc_call()
            .args
            .iter()
            .map(|a| state.get_val(ctx, a))
            .collect(),
        perm: eval_perm_structural(ctx, &state, &q.body.perm),
    })
}

/// The permission a quantified chunk set holds at `addr`, or `None` when `addr`
/// is not an application of the chunk's address function (no binding can name
/// it). Not yet gated by `chunk.guard`, the inhale's path condition: the caller
/// gates it exactly as a [`Chunk`]'s guard.
///
/// The binding that names `addr = f(as)` is solved per binder `x_j`:
/// - **projection**: if some address argument is `x_j` itself, `x_j = as_i`;
/// - otherwise **inversion**: `x_j = inv_j(as)`, with the inverse minted at the
///   inhale (whose axiom, when assumed, lets the prover reduce it).
///
/// The body is evaluated at that binding from the chunk's captures, and its
/// permission is gated by `c(xs)` and by the rebuilt address arguments being
/// `as` (Silicon's `e(inv(r)) = r`). That gate is what makes the answer sound
/// for any solution, even an unconstrained inverse: the permission is returned
/// only where the instance really names `addr`. Arguments that already coincide
/// with `as` (every projected one) add no gate.
pub(super) fn qp_perm_at(
    ctx: &mut VerifyContext<'_>,
    chunk: &QuantChunk,
    addr: egg::Id,
) -> Result<Option<ChunkPerm>, VerifyError> {
    let q = &*chunk.q;
    let call = q.loc_call();
    let Some(args) = addr_call_args(ctx, call, addr) else {
        return Ok(None);
    };
    assert_eq!(
        args.len(),
        call.args.iter().count(),
        "address call arity mismatch"
    );

    let binders: Vec<egg::Id> = (0..q.bound.len())
        .map(|j| {
            let me = Val::Temp(q.binder_base + j);
            match call.args.iter().position(|a| *a == me) {
                Some(i) => args[i],
                None => ctx.add(Symbolic::FuncApp(chunk.inv[j], Box::new([]), args.clone())),
            }
        })
        .collect();
    let inst = eval_qp_instance(ctx, &chunk.caps, q, &binders, &chunk.eval_pc())?;

    let perm = gate_on_args(ctx, inst.perm, &inst.loc_args, &args);
    let zero = ChunkPerm::leaf(expr!(ctx, 0 / 1));
    let mut perm = ChunkPerm::select(ctx, inst.cond, perm, zero);
    // What was exhaled from this set, each gated by its presence guard.
    for taken in chunk.taken.iter() {
        let p = match &taken.what {
            // A taken set applies the same address function, so it always answers.
            Taken::Set(set) => {
                let p = qp_perm_at(ctx, set, addr)?
                    .expect("a taken chunk set applies the same address function");
                gate_perm_by_guard(ctx, &p, &set.guard)
            }
            // A single location counts only where it is `addr`. Address
            // functions are injective (`rewrite::address_injectivity_rule`), so
            // `at == addr` reduces to its arguments.
            Taken::Single {
                addr: at,
                perm: p,
                guard,
            } => {
                let p = gate_perm_by_guard(ctx, p, guard);
                if ctx.egraph.find(*at) == ctx.egraph.find(addr) {
                    p
                } else {
                    let eq = expr!(ctx, { *at } == { addr });
                    let zero = ChunkPerm::leaf(expr!(ctx, 0 / 1));
                    ChunkPerm::select(ctx, eq, p, zero)
                }
            }
            Taken::Share {
                demand,
                before,
                holder,
                pc,
                guard,
            } => {
                let at_pc: Vec<_> = pc.iter().chain(guard.iter()).copied().collect();
                let take = share_at(ctx, demand, before, HolderAt::Set(*holder), addr, &at_pc)?;
                gate_perm_by_guard(ctx, &ChunkPerm::leaf(take), guard)
            }
        };
        perm = crate::verify::heap::algebra::perm_sub_aligned(ctx, &perm, &p);
    }
    Ok(Some(perm))
}

/// The arguments of `addr` as an application of `call`'s address function (the
/// receiver of a field, the argument tuple of a predicate), or `None` when its
/// e-class holds no such application.
fn addr_call_args(
    ctx: &VerifyContext<'_>,
    call: &vmir::FunctionCall,
    addr: egg::Id,
) -> Option<Box<[egg::Id]>> {
    let func = crate::verify::func_registry::func_id_for_member(call.function);
    let class = ctx.egraph.find(addr);
    ctx.egraph[class].nodes.iter().find_map(|n| match n {
        Symbolic::FuncApp(f, tys, args) if *f == func && **tys == call.type_args[..] => {
            Some(args.clone())
        }
        _ => None,
    })
}

/// `perm` where the two address argument tuples agree, `0` elsewhere. Arguments
/// already in one e-class add no gate.
fn gate_on_args(
    ctx: &mut VerifyContext<'_>,
    mut perm: ChunkPerm,
    got: &[egg::Id],
    want: &[egg::Id],
) -> ChunkPerm {
    let zero = ChunkPerm::leaf(expr!(ctx, 0 / 1));
    for (&g, &w) in got.iter().zip(want).rev() {
        if ctx.egraph.find(g) != ctx.egraph.find(w) {
            let eq = expr!(ctx, { g } == { w });
            perm = ChunkPerm::select(ctx, eq, perm, zero.clone());
        }
    }
    perm
}

/// Index of the first held set of `kind` that covers `exhale`
pub(super) fn find_covering_qp(
    ctx: &mut VerifyContext<'_>,
    heap: &Heap,
    kind: &LocationKind,
    exhaled: &QuantChunk,
) -> Result<Option<usize>, VerifyError> {
    let vmir_chunk = &exhaled.q;
    let pc = exhaled.eval_pc();
    for (i, held) in heap.qp_chunks_of(&kind).iter().enumerate() {
        let contained = check_qp_contained(
            ctx,
            vmir_chunk,
            &pc,
            &mut |ctx, binders| eval_qp_instance(ctx, &exhaled.caps, vmir_chunk, binders, &pc),
            &mut |ctx, addr| {
                Ok(qp_perm_at(ctx, held, addr)?.map(|p| gate_perm_by_guard(ctx, &p, &held.guard)))
            },
        )?;

        if contained {
            return Ok(Some(i));
        }
    }
    Ok(None)
}

/// Whether `a` and `b` name the same locations with the same amounts, each
/// contained in the other ([`check_qp_contained`]). What was taken from either
/// counts, so a set that gave something away is not the same as its origin.
pub(super) fn same_footprint(ctx: &mut VerifyContext<'_>, a: &QuantChunk, b: &QuantChunk) -> bool {
    let within = |ctx: &mut VerifyContext<'_>, sub: &QuantChunk, held: &QuantChunk| {
        let pc = sub.eval_pc();
        check_qp_contained(
            ctx,
            &sub.q,
            &pc,
            &mut |ctx, binders| eval_qp_instance(ctx, &sub.caps, &sub.q, binders, &pc),
            &mut |ctx, addr| {
                Ok(qp_perm_at(ctx, held, addr)?.map(|p| gate_perm_by_guard(ctx, &p, &held.guard)))
            },
        )
        // The bodies evaluated once already, at their inhales.
        .unwrap_or_else(|e| panic!("chunk set body failed to re-evaluate: {e:?}"))
    };
    within(ctx, a, b) && within(ctx, b, a)
}

/// Whether the exhaled set is covered by everything `heap` holds together: at
/// each location it names, its amount is at most the sum over all holders
/// there ([`holders_at`]). The fallback when no single held set covers it.
pub(super) fn covered_by_sum(
    ctx: &mut VerifyContext<'_>,
    heap: &Heap,
    kind: &LocationKind,
    exhaled: &QuantChunk,
) -> Result<bool, VerifyError> {
    let q = &exhaled.q;
    let pc = exhaled.eval_pc();
    check_qp_contained(
        ctx,
        q,
        &pc,
        &mut |ctx, binders| eval_qp_instance(ctx, &exhaled.caps, q, binders, &pc),
        &mut |ctx, addr| {
            let holders = holders_at(ctx, heap, kind, addr, &pc)?;
            Ok(Some(sum_perms(ctx, &holders)))
        },
    )
}

/// What the holder `at` of `before` gives up at `addr` when `demand` is split
/// greedily over all holders there ([`greedy_takes`]); `0` when `demand` does
/// not name `addr` or `at` does not hold it. Evaluated under `guard`, the
/// exhale's path condition.
fn share_at(
    ctx: &mut VerifyContext<'_>,
    demand: &QuantChunk,
    before: &Heap,
    at: HolderAt,
    addr: Id,
    guard: &[(Id, Polarity)],
) -> Result<Id, VerifyError> {
    let zero = expr!(ctx, 0 / 1);
    let Some(d) = qp_perm_at(ctx, demand, addr)? else {
        return Ok(zero);
    };
    let d = d.to_id(ctx);
    let kind = LocationKind::from_addr_type(&demand.q.loc_ty).expect("QP loc_ty must be Addr");
    let holders = holders_at(ctx, before, &kind, addr, guard)?;
    let takes = greedy_takes(ctx, &holders, d);
    Ok(holders
        .iter()
        .position(|h| h.at == at)
        .map_or(zero, |i| takes[i]))
}

/// Take the exhaled set from all holders together, after [`covered_by_sum`]
/// proved they hold enough. The split is greedy in `holders_at` order: a
/// single chunk gives up its share at its own address at once; a held set
/// records a [`Taken::Share`], evaluated at each later lookup.
pub(super) fn take_spread(
    ctx: &mut VerifyContext<'_>,
    heap: &Heap,
    kind: &LocationKind,
    exhaled: QuantChunk,
) -> Result<Heap, VerifyError> {
    let pc = exhaled.eval_pc();
    let mut out = heap.clone();
    for (i, c) in heap.chunks_of(kind).iter().enumerate() {
        let take = share_at(ctx, &exhaled, heap, HolderAt::Chunk(i), c.addr, &pc)?;
        let take = gate_perm_by_guard(ctx, &ChunkPerm::leaf(take), &exhaled.guard);
        let rest = crate::verify::heap::algebra::perm_sub_aligned(ctx, c.ungated_perm(), &take);
        out = out.with_chunk(
            kind,
            Chunk::new_perm(c.addr, rest, c.value)
                .with_recipe(c.recipe.clone())
                .with_guard(c.guard_pc()),
        );
    }
    for (i, set) in heap.qp_chunks_of(kind).iter().enumerate() {
        let set = set.with_taken(
            ctx,
            Taken::Share {
                demand: exhaled.clone(),
                before: heap.clone(),
                holder: i,
                pc: pc.clone().into(),
                guard: exhaled.guard.clone(),
            },
        );
        out = out.with_qp_chunk_replaced(kind, i, set);
    }
    Ok(out)
}

/// Subtracts the `exhaled` chunk from the `i`th chunk of `kind` in heap. Returns the heap with the subtraction
/// This function just removes it, no checks are performed.
///
/// When the exhale takes everything the set still holds (the usual hand-off
/// of a whole set to a loop or a callee), the set is dropped instead: it holds
/// no permission, so nothing can be read through it, and later unions and
/// reads do not have to look past it.
pub(super) fn take_from_qp(
    ctx: &mut VerifyContext<'_>,
    heap: &Heap,
    kind: &LocationKind,
    i: usize,
    exhaled: QuantChunk,
) -> Heap {
    let held = &heap.qp_chunks_of(kind)[i];
    if emptied_by(ctx, held, &exhaled) {
        return heap.with_qp_chunk_removed(kind, i);
    }
    let held = held.with_taken(
        ctx,
        Taken::Set(QuantChunk {
            value_map: held.value_map,
            ..exhaled
        }),
    );
    heap.with_qp_chunk_replaced(kind, i, held)
}

/// Whether taking `exhaled` leaves `held` empty on the exhale's path: what
/// `held` still holds is contained in `exhaled`. The other direction is what
/// made `held` the covering set ([`find_covering_qp`]). Only the exhale's path
/// matters, as the heap is dropped from only there on, so the containment is
/// proven under both guards.
fn emptied_by(ctx: &mut VerifyContext<'_>, held: &QuantChunk, exhaled: &QuantChunk) -> bool {
    let pc: Vec<_> = held
        .eval_pc()
        .into_iter()
        .chain(exhaled.eval_pc())
        .collect();
    check_qp_contained(
        ctx,
        &held.q,
        &pc,
        &mut |ctx, binders| eval_qp_instance(ctx, &held.caps, &held.q, binders, &pc),
        &mut |ctx, addr| {
            Ok(
                qp_perm_at(ctx, exhaled, addr)?
                    .map(|p| gate_perm_by_guard(ctx, &p, &exhaled.guard)),
            )
        },
    )
    // The body evaluated once already, at its inhale.
    .unwrap_or_else(|e| panic!("chunk set body failed to re-evaluate: {e:?}"))
}

/// The chunk set of `q` with its captured temps already resolved to `caps`:
/// either from the live state ([`quant_chunk_of`]) or rebuilt from a resource
/// footprint slot's recipes at a call site (`walk_footprint`). Proves the
/// receiver injectivity under `pc_lits` and assumes the inverse axiom that
/// proof licenses. The result carries no recipe; the caller sets one.
pub(super) fn quant_chunk_from_caps(
    ctx: &mut VerifyContext<'_>,
    q: &Rc<vmir::QuantChunks>,
    caps: Rc<[(usize, egg::Id)]>,
    pc_lits: &[(egg::Id, Polarity)],
    value_map: egg::Id,
) -> Result<QuantChunk, VerifyError> {
    // Distinct bindings that hold permission must name distinct locations.
    check_qp_injectivity(ctx, q, pc_lits, &mut |ctx, binders| {
        eval_qp_instance(ctx, &caps, q, binders, pc_lits)
    })?;
    quant_chunk_injective(ctx, q, caps, pc_lits, value_map)
}

/// The chunk set of `q` at `caps` whose receiver injectivity is already
/// established, so only the inverse axiom is assumed here.
///
/// The one such caller is a footprint slot *produced* by `walk_footprint`
/// (inhale, unfold, from_snap): the resource's certificate walk proved
/// injectivity for arbitrary arguments under the body's earlier conjuncts, and
/// the produce assumes those conjuncts (the body boolean) under the same
/// guard. It cannot re-prove it there, since the boolean is assumed only after
/// all slots are walked. A *consumed* slot must not use this: there the
/// boolean is asserted after the slots, and an inverse axiom assumed for a
/// non-injective receiver would make that assertion vacuously provable.
pub(super) fn quant_chunk_injective(
    ctx: &mut VerifyContext<'_>,
    q: &Rc<vmir::QuantChunks>,
    caps: Rc<[(usize, egg::Id)]>,
    pc_lits: &[(egg::Id, Polarity)],
    value_map: egg::Id,
) -> Result<QuantChunk, VerifyError> {
    // One binding per location: assume the inverse axiom that names it.
    let inv = mint_and_assume_inverse_axiom(ctx, q, &caps, pc_lits)?;
    Ok(QuantChunk {
        origin: SetId::fresh(ctx),
        q: q.clone(),
        caps,
        inv,
        pc: pc_lits.into(),
        guard: Rc::from([]),
        taken: Rc::from([]),
        value_map,
        recipe: None,
        restored: Rc::from([]),
    })
}
