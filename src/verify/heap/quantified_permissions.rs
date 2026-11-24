//! Well-formedness of a quantified permission, and the chunk set it denotes.
//!
//! `forall xs :: c(xs) ==> acc(f(e(xs)), p(xs))` is well-formed when, for every
//! binding (checked in this order, as Silicon does):
//!
//! 1. its body is **well-defined**: `c` everywhere, `e` and `p` where `c` holds
//!    (divisors, indices, map keys, callees' preconditions);
//! 2. its amount is **non-negative** where `c` holds;
//! 3. distinct instances that hold permission name distinct locations
//!    (**receiver injectivity**):
//!
//! ```text
//! forall xs1, xs2 ::
//!     c(xs1) ∧ c(xs2) ∧ 0 < p(xs1) ∧ 0 < p(xs2) ∧ f(e(xs1)) = f(e(xs2))  ==>  xs1 = xs2
//! ```
//!
//! Injectivity is what later licenses the inverse-function view of the chunk
//! set (one binding per location). Without it, an inhale could split one
//! location's permission across several bindings and the inverse axiom would
//! assume `false`. Well-definedness is what lets every later instance of the
//! body (`eval_qp_instance`) skip its side conditions.
//!
//! All three are proven by **skolemization** in one scratch graph
//! ([`check_qp_well_formed`]): the binder tuples are fresh constants, the
//! antecedents go into the path condition, and the side conditions, the sign
//! and the binder equalities are the goals.
//!
//! The injectivity antecedent compares the **addresses**, as Viper does.
//! Address functions are injective (`rewrite::address_injectivity_rule`), so
//! the prover gets the arguments' equality (`e(xs)`, the receiver of a field,
//! the argument tuple of a predicate) from it.

use std::rc::Rc;
use std::sync::Arc;

use egg::Id;

use crate::verify::eval::{
    EvalState, OnRead, eval_body_checked, eval_perm_inst, eval_perm_structural, eval_pure_inst,
    pc_lits_raw,
};
use crate::verify::heap::QpCheck;
use crate::verify::heap::algebra::{prove_framed_aligned, prove_sufficient_aligned};
use crate::verify::heap::interface::Captures;
use crate::verify::heap::view::HolderAt;
use crate::verify::heap::{Heap, LocationKind, QuantChunk, SetId, Taken, holder::SetBody};
use crate::vmir;

use crate::verify::analysis::ConstFold;
use crate::verify::context::VerifyContext;
use crate::verify::error::VerifyError;
use crate::verify::heap::ChunkPerm;
use crate::verify::lang::{FuncId, RecipeId, Symbolic};
use crate::verify::quant::{Canonical, QuantRecipe, ResolvedRead, Stand};
use crate::verify::rewrite::AxiomPure;
use crate::vmir::{BinOp, InstKind, PermInst, PermVal, Polarity, PureInst, QuantChunks, Val};

/// One instance of a quantified permission's body at given binder values: what
/// the injectivity check reads off it.
pub(super) struct QpInstance {
    /// `c(xs)`: whether this binding contributes a chunk.
    pub(crate) cond: Id,
    /// The location's address `f(e(xs))`: the address call applied to its
    /// arguments (the receiver of `e.f`, the argument tuple of `P(..)`).
    pub(crate) addr: Id,
    /// `p(xs)`, with the enclosing branch conditions folded in.
    pub(crate) perm: ChunkPerm,
}

/// Evaluate the body of `q` at the given binder values. Supplied by the caller,
/// which owns the evaluation state the body's captured temps resolve against.
type Instantiate<'s> =
    dyn FnMut(&mut VerifyContext<'_>, &[Id]) -> Result<QpInstance, VerifyError> + 's;

/// Prove that `q` (captured temps resolved through `caps`) is well-formed under
/// the path condition `pc`, as far as `check` asks (see the module docs and
/// [`QpCheck`]), or fail with the side condition that does not hold.
///
/// Runs on a scratch e-graph: the fresh binders and the body instances exist
/// only for this proof and must not leak into the live graph. The certificate
/// recipe (if one is being built) is paused for the same reason, exactly as for
/// a pure `forall`'s well-definedness check.
fn check_qp_well_formed(
    ctx: &mut VerifyContext<'_>,
    q: &QuantChunks,
    caps: &Captures,
    pc: &[(Id, Polarity)],
    check: QpCheck,
) -> Result<(), VerifyError> {
    if check == QpCheck::Established {
        return Ok(());
    }
    ctx.pause_recipe(|ctx| {
        ctx.with_scratch_graph(|ctx| check_well_formed_in_scratch(ctx, q, caps, pc, check))
    })
}

fn check_well_formed_in_scratch(
    ctx: &mut VerifyContext<'_>,
    q: &QuantChunks,
    caps: &Captures,
    pc: &[(Id, Polarity)],
    check: QpCheck,
) -> Result<(), VerifyError> {
    // The first instance is evaluated with its side conditions discharged
    // (well-definedness for an arbitrary binding); the second, for
    // injectivity only, then needs none.
    let x1 = fresh_binders(ctx, q);
    let i1 = if check == QpCheck::Body {
        let i1 = eval_qp_instance_checked(ctx, caps, q, &x1, pc)?;
        // Non-negative where the binding holds: `c(xs) ==> not(p(xs) < 0)`.
        let mut lits = pc.to_vec();
        lits.push((i1.cond, Polarity::Positive));
        let p = i1.perm.to_id(ctx);
        let zero = expr!(ctx, 0 / 1);
        let goal = expr!(ctx, not({ p } < r { zero }));
        if !ctx.prove_under_pc(goal, &lits) {
            return Err(VerifyError::SideCondition("permission may be negative"));
        }
        i1
    } else {
        eval_qp_instance(ctx, caps, q, &x1, pc)?
    };
    let x2 = fresh_binders(ctx, q);
    let i2 = eval_qp_instance(ctx, caps, q, &x2, pc)?;

    // pc ++ antecedents: the order `prove_under_pc` (and the block scratch's
    // cube prefix test) expects, as in a pure `forall`'s WD check.
    let mut lits = pc.to_vec();
    for inst in [&i1, &i2] {
        lits.push((inst.cond, Polarity::Positive));
        let p = inst.perm.to_id(ctx);
        lits.push((expr!(ctx, (0 / 1) < r { p }), Polarity::Positive));
    }
    // Same location: the address functions are injective
    // (`rewrite::address_injectivity_rule`), so this gives the arguments' equality.
    lits.push((expr!(ctx, { i1.addr } == { i2.addr }), Polarity::Positive));

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

/// When what is held at a location covers what a set has there, per branch:
/// `prove_sufficient_aligned` compares amounts; `prove_framed_aligned` also
/// lets a `wildcard` demand ask for a positive share only, for a consume's
/// demand that a frame checks. A *held* set's wildcard is a real amount
/// (`Captures::wildcard`), so comparing two held sets compares amounts.
type Covers = fn(&mut VerifyContext<'_>, &ChunkPerm, &ChunkPerm, &[(Id, Polarity)]) -> bool;

/// The permission held at an address, guards folded in. Supplied by the caller,
/// which owns the lookup (`qp_perm_at`, or a sum over all holders).
type HeldAt<'s> = dyn FnMut(&mut VerifyContext<'_>, Id) -> Result<ChunkPerm, VerifyError> + 's;

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
/// is discharged per branch by `covers`. This is what makes recording the
/// exhaled set as taken from the held one sound: no location goes negative.
fn check_qp_contained(
    ctx: &mut VerifyContext<'_>,
    q: &QuantChunks,
    pc: &[(Id, Polarity)],
    instantiate: &mut Instantiate<'_>,
    held_at: &mut HeldAt<'_>,
    covers: Covers,
) -> Result<bool, VerifyError> {
    ctx.pause_recipe(|ctx| {
        ctx.with_scratch_graph(|ctx| {
            let ys = fresh_binders(ctx, q);
            let inst = instantiate(ctx, &ys)?;
            let avail = held_at(ctx, inst.addr)?;
            let mut lits = pc.to_vec();
            lits.push((inst.cond, Polarity::Positive));
            Ok(covers(ctx, &avail, &inst.perm, &lits))
        })
    })
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
/// binder, for the chunk to carry (a lookup at an address applies them to it).
///
/// With `f(e(xs))` the location's address and `inv_j` fresh, the axiom is
///
/// ```text
/// forall xs :: {triggers}   c(xs) ∧ 0 < p(xs)  ==>  ⋀_j inv_j(f(e(xs))) = xs_j
/// ```
///
/// Its antecedent is the injectivity check's, which is what makes it
/// consistent; so it must only be assumed once [`check_qp_well_formed`] passed
/// under the same `pc`. Silicon's second axiom `f(e(inv(r))) = r` is left to the
/// lookup, which conjoins it into the chunk's condition at the address it asks
/// about.
///
/// The body is canonicalized exactly as a pure `forall`'s ([`Canonical`]) and
/// the axiom's steps are appended to it. `caps` is [`QuantChunks::free_temps`],
/// resolved.
fn mint_and_assume_inverse_axiom(
    ctx: &mut VerifyContext<'_>,
    q: &QuantChunks,
    caps: &Captures,
    pc: &[(Id, Polarity)],
) -> Result<Rc<[FuncId]>, VerifyError> {
    // Without a trigger the axiom could never fire; assuming nothing is sound.
    if q.triggers.is_empty() && caps.reads.is_empty() {
        return Ok(mint_inverses(ctx, q));
    }

    let free: Vec<usize> = caps.temps.iter().map(|&(k, _)| k).collect();
    let by_value: Vec<bool> = caps
        .reads
        .iter()
        .map(|r| matches!(r.stand, Stand::Value(_)))
        .collect();
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
        &q.heap_reads(),
        &by_value,
        &q.triggers,
        &q.body.insts,
    )?;
    // A receiver that reads the heap has no VMIR location trigger (typecheck's
    // `location_can_trigger`): the compiled location, which reads the stand-ins,
    // triggers instead.
    if c.groups.is_empty() {
        let loc = c.rename(&q.body.loc);
        match c.pattern_of(&loc, q.bound.len()) {
            Some(pattern) => c.groups = vec![vec![pattern]],
            None => return Ok(mint_inverses(ctx, q)),
        }
    }

    // `c(xs) ∧ 0 < p(xs)  ==>  ⋀_j inv_j(f(e(xs))) = xs_j`
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
    let loc = c.rename(&q.body.loc);

    // A set with the same axiom up to its inverses, over the same captures,
    // already has inverses: reuse them, so its forall is the same node.
    let shape = InverseShape {
        bound: q.bound.clone(),
        groups: c.groups.clone(),
        insts: c.insts.clone(),
        ante: ante.clone(),
        loc: loc.clone(),
    };
    // The children, in `Canonical`'s slot order: captured temps, then reads.
    let temp_ids = caps.temps.iter().map(|&(_, id)| id);
    let cap_ids: Vec<Id> = temp_ids
        .chain(caps.reads.iter().map(|r| r.stand.id()))
        .collect();
    let (rid, inv) = match ctx.inverse_axioms.find(&mut ctx.egraph, &shape, &cap_ids) {
        Some(found) => found,
        None => {
            let inv = mint_inverses(ctx, q);
            let mut concl = crate::vmir::TRUE;
            for (j, &func) in inv.iter().enumerate() {
                let binder = c.rename(&Val::Temp(q.binder_base + j));
                let args = vec![loc.clone()];
                let app = c.push(AxiomPure::App {
                    func,
                    type_args: vec![],
                    args,
                });
                let eq = c.push(AxiomPure::Binary(BinOp::Eq, app, binder));
                concl = and(&mut c, eq, concl);
            }
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
            (rid, inv)
        }
    };
    let node = ctx.add(Symbolic::Forall(rid, Box::new([]), cap_ids.into()));
    ctx.assume_guarded(node, pc.iter().rev().copied());
    Ok(inv)
}

/// The fresh wildcard function of a produced set of `q`, over its locations
/// (see [`Captures::wildcard`]).
pub(super) fn mint_wildcard(ctx: &mut VerifyContext<'_>, q: &QuantChunks) -> FuncId {
    let f = ctx.alloc.qp_wildcard(q.loc_ty.clone());
    ctx.func_ret_types.insert(f, vmir::Type::Real);
    f
}

/// One fresh inverse per binder of `q`, from its location's address type.
fn mint_inverses(ctx: &mut VerifyContext<'_>, q: &QuantChunks) -> Rc<[FuncId]> {
    q.bound
        .iter()
        .map(|ret| {
            let f = ctx.alloc.qp_inverse(vec![q.loc_ty.clone()], ret.clone());
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
    loc: Val,
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
/// The body holds `Pure` and `Perm` instructions, and the `Assert`/`Assume` of
/// a call to a contract-bearing function; its location is an address call
/// (both guaranteed by `lower_quant_chunks`). Anything else is a lowering bug
/// and panics. Its side conditions and `Assert`s are not checked here: a set
/// exists only once [`check_qp_well_formed`] showed them for an arbitrary
/// binding (or its resource's own check did, see [`QpCheck`]).
fn eval_qp_instance(
    ctx: &mut VerifyContext<'_>,
    caps: &Captures,
    q: &vmir::QuantChunks,
    binders: &[egg::Id],
    host_pc: &[(egg::Id, Polarity)],
) -> Result<QpInstance, VerifyError> {
    // The instance only feeds proofs (well-formedness, containment, a lookup's
    // permission); its terms never enter a certificate, and its temps carry no
    // recipes. A read's certificate term is `lookup(seed, addr)`, built by the
    // `Deref` itself.
    ctx.pause_recipe(|ctx| {
        let state = eval_body_at(ctx, caps, q, binders, host_pc, None)?;
        Ok(instance_of(ctx, &state, q))
    })
}

/// [`eval_qp_instance`]'s walk: the body's state at `binders`. A `Deref` goes
/// to `on_read` when one is given, else to the resolved [`Captures::reads`].
fn eval_body_at(
    ctx: &mut VerifyContext<'_>,
    caps: &Captures,
    q: &vmir::QuantChunks,
    binders: &[egg::Id],
    host_pc: &[(egg::Id, Polarity)],
    mut on_read: Option<&mut OnRead<'_>>,
) -> Result<EvalState, VerifyError> {
    let mut state = instance_frame(caps, q, binders);
    for inst in &q.body.insts {
        let pc_lits = body_inst_pc(ctx, &state, q, inst, host_pc);
        match &inst.kind {
            InstKind::Pure(ty, PureInst::Deref(heap, addr)) if let Some(f) = &mut on_read => {
                let id = f(ctx, &state, heap, addr, &pc_lits)?;
                state.push_val(id, ty.clone(), None);
            }
            InstKind::Pure(ty, pi) => {
                let (id, _) = eval_pure_inst(ctx, &state, ty, pi, &pc_lits, None)?;
                state.push_val(id, ty.clone(), None);
            }
            InstKind::Perm(pi) => {
                let p = eval_perm_inst(ctx, &state, pi);
                state.push_perm(p, pi.clone());
            }
            // A callee's precondition: shown by the well-formedness check.
            InstKind::Assert(_) => {}
            // A callee's postcondition: it holds where the call is
            // well-defined, which is what the pc (with the binding's
            // condition) says.
            InstKind::Assume(val) => {
                let id = state.get_val(ctx, val);
                ctx.assume_guarded(id, pc_lits.iter().rev().copied());
            }
            other => panic!("quantified permission body holds a non-pure inst: {other:?}"),
        }
    }
    Ok(state)
}

/// Resolves the heap reads of a quantifier body ([`vmir::HeapRead`]) to the
/// value maps that stand in for their heaps ([`ResolvedRead`]). Shared by
/// quantified permissions ([`resolve_reads`]) and pure `forall`s
/// (`eval::check_forall_wd`).
///
/// A read must be framed, for every binding that reaches it, by **one** holder
/// of its heap. That is shown as the well-formedness checks show their goals:
/// a walk of the body at fresh binders on a scratch graph meets each read
/// under its own pc ([`Self::on_read`]), and the holder's value there stands
/// for the read for the rest of the walk. After the walk, in the live graph,
/// [`Self::finish`] hands back what stands for each read ([`Stand`]): a chunk
/// set's value map, or a single chunk's value, which the read is for every
/// binding since every address it names is that chunk's. Only e-classes that
/// predate the scratch graph are kept across it: a set's map, a chunk's value.
pub(crate) struct ReadResolver<F> {
    /// The heap a body `Deref` names, as it is at the quantifier.
    heap_of: F,
    /// Per read in body order: what resolves it.
    covers: Vec<ResolvedRead>,
}

impl<F> ReadResolver<F>
where
    F: FnMut(&mut VerifyContext<'_>, &vmir::HeapVal) -> Result<Heap, VerifyError>,
{
    pub(crate) fn new(heap_of: F) -> Self {
        ReadResolver {
            heap_of,
            covers: Vec::new(),
        }
    }

    /// The read `Deref(heap, addr)` met by the walk in `state` under `pc`: its
    /// value at the walked binding.
    pub(crate) fn on_read(
        &mut self,
        ctx: &mut VerifyContext<'_>,
        state: &EvalState,
        heap: &vmir::HeapVal,
        addr: &Val,
        pc: &[(Id, Polarity)],
    ) -> Result<Id, VerifyError> {
        let loc = state
            .loc(ctx, addr)
            .expect("a read's operand is an address");
        let h = (self.heap_of)(ctx, heap)?;
        let Some((at, value)) = h.covering_holder(ctx, &loc, pc)? else {
            return Err(VerifyError::Unimplemented(
                "heap read in a quantifier framed by several holders",
            ));
        };
        let (stand, recipe) = match at {
            HolderAt::Set(i) => {
                let set = &h.qp_chunks_of(&loc.kind)[i];
                (Stand::Map(set.value_map), set.recipe.clone())
            }
            HolderAt::Chunk(at) => {
                let chunk = h.chunks_of(&loc.kind).iter().find(|c| c.addr == at);
                (Stand::Value(value), chunk.and_then(|c| c.recipe.clone()))
            }
        };
        self.covers.push(ResolvedRead {
            heap: *heap,
            addr: addr.clone(),
            kind: loc.kind,
            stand,
            recipe,
        });
        Ok(value)
    }

    /// The reads met, resolved.
    pub(crate) fn finish(self) -> Rc<[ResolvedRead]> {
        self.covers.into()
    }
}

/// The captures of `q`'s body: the captured `temps`, and its heap reads
/// resolved by one walk at fresh binders ([`ReadResolver`]).
pub(crate) fn resolve_reads(
    ctx: &mut VerifyContext<'_>,
    q: &vmir::QuantChunks,
    temps: Rc<[(usize, Id)]>,
    heap_of: &mut dyn FnMut(&mut VerifyContext<'_>, &vmir::HeapVal) -> Result<Heap, VerifyError>,
    pc: &[(Id, Polarity)],
) -> Result<Captures, VerifyError> {
    let mut caps = Captures {
        temps,
        reads: Rc::from([]),
        wildcard: None,
    };
    if q.heap_reads().is_empty() {
        return Ok(caps);
    }
    let mut resolver = ReadResolver::new(heap_of);
    ctx.pause_recipe(|ctx| {
        ctx.with_scratch_graph(|ctx| {
            let xs = fresh_binders(ctx, q);
            let mut on_read =
                |ctx: &mut VerifyContext<'_>, state: &EvalState, heap: &_, addr: &_, pc: &_| {
                    resolver.on_read(ctx, state, heap, addr, pc)
                };
            eval_body_at(ctx, &caps, q, &xs, pc, Some(&mut on_read)).map(|_| ())
        })
    })?;
    caps.reads = resolver.finish();
    Ok(caps)
}

/// [`eval_qp_instance`] with the body's side conditions discharged as it is
/// evaluated ([`eval_body_checked`]): the instance at fresh `binders` that
/// shows the body well-defined for every binding. The receiver and amount are
/// checked only where the binding's condition holds ([`body_inst_pc`]).
fn eval_qp_instance_checked(
    ctx: &mut VerifyContext<'_>,
    caps: &Captures,
    q: &vmir::QuantChunks,
    binders: &[egg::Id],
    host_pc: &[(egg::Id, Polarity)],
) -> Result<QpInstance, VerifyError> {
    let mut state = instance_frame(caps, q, binders);
    let mut pc_of = |ctx: &mut VerifyContext<'_>, state: &EvalState, inst: &vmir::Inst| {
        body_inst_pc(ctx, state, q, inst, host_pc)
    };
    eval_body_checked(ctx, &mut state, &q.body.insts, &mut pc_of, None)?;
    Ok(instance_of(ctx, &state, q))
}

/// The body's frame at `binders`: the captures, then the binders.
fn instance_frame(caps: &Captures, q: &vmir::QuantChunks, binders: &[egg::Id]) -> EvalState {
    debug_assert_eq!(binders.len(), q.bound.len());
    let mut state = EvalState::frame_from_caps(q.binder_base, &caps.temps);
    state.read_maps = caps.reads.clone();
    state.wildcard = caps.wildcard.map(|w| (w, q.body.loc.clone()));
    for (&id, ty) in binders.iter().zip(q.bound.iter()) {
        state.push_val(id, ty.clone(), None);
    }
    state
}

/// The instance an evaluated body `state` denotes.
fn instance_of(
    ctx: &mut VerifyContext<'_>,
    state: &EvalState,
    q: &vmir::QuantChunks,
) -> QpInstance {
    QpInstance {
        cond: state.get_val(ctx, &q.body.cond),
        addr: state.get_val(ctx, &q.body.loc),
        perm: eval_perm_structural(ctx, state, &q.body.perm),
    }
}

/// The path condition of one body instruction `inst`, evaluated in `state`
/// under the host's `host_pc`: the host's, the instruction's own guards, and
/// the binding's condition `c(xs)` once it is evaluated. The body evaluates the
/// condition first, then the location and the amount, which Viper evaluates
/// only where the condition holds (`forall xs :: c(xs) ==> acc(e(xs).f,
/// p(xs))`): that guard is what lets `c(xs)` establish their well-definedness
/// and their callees' postconditions.
fn body_inst_pc(
    ctx: &mut VerifyContext<'_>,
    state: &EvalState,
    q: &vmir::QuantChunks,
    inst: &vmir::Inst,
    host_pc: &[(egg::Id, Polarity)],
) -> Vec<(egg::Id, Polarity)> {
    let mut pc = host_pc.to_vec();
    pc.extend(pc_lits_raw(ctx, state, &inst.pc));
    let cond_evaluated = match &q.body.cond {
        Val::Temp(k) => *k < state.vals.len(),
        Val::Literal(_) => true,
    };
    if cond_evaluated {
        pc.push((state.get_val(ctx, &q.body.cond), Polarity::Positive));
    }
    pc
}

/// The permission a quantified chunk set holds at `addr`, gated by its presence
/// guard `chunk.guard` exactly as a [`Chunk`]'s.
///
/// The binding that names `addr` is `x_j = inv_j(addr)`, with the inverses
/// minted at the inhale (whose axiom, once it fires on `addr`, lets the prover
/// reduce them). The body is evaluated at that binding from the chunk's
/// captures, and its permission is gated by `c(xs)` and by the rebuilt address
/// being `addr` (Silicon's `f(e(inv(r))) = r`). That gate is what makes the
/// answer sound for any inverse, even an unconstrained one: the permission
/// counts only where the instance really names `addr`.
pub(super) fn qp_perm_at(
    ctx: &mut VerifyContext<'_>,
    chunk: &QuantChunk,
    addr: egg::Id,
) -> Result<ChunkPerm, VerifyError> {
    let binders: Vec<egg::Id> = chunk
        .inv
        .iter()
        .map(|&inv| ctx.add(Symbolic::FuncApp(inv, Box::new([]), Box::new([addr]))))
        .collect();
    let inst = eval_qp_instance(ctx, &chunk.caps, &chunk.q, &binders, &chunk.eval_pc())?;

    let perm = gate_at(ctx, inst.perm, inst.addr, addr);
    let mut perm = ChunkPerm::select(ctx, inst.cond, perm, ChunkPerm::Zero);
    // What was exhaled from this set, each gated by its debit's presence.
    for debit in chunk.taken.iter() {
        let p = match &debit.what {
            // A set, held under the debit's presence: `qp_perm_at` gates it.
            Taken::Set(_) => {
                let set = debit
                    .as_set()
                    .expect("a whole-set take reads back as a set");
                qp_perm_at(ctx, &set, addr)?
            }
            // A single location counts only where it is `addr`.
            Taken::Single { addr: at, perm: p } => {
                let p = debit.gate(ctx, p.clone());
                gate_at(ctx, p, *at, addr)
            }
            Taken::Share {
                demand,
                before,
                holder,
                pc,
            } => {
                let at_pc: Vec<_> = pc.iter().chain(debit.presence().iter()).copied().collect();
                let take = share_at(ctx, demand, before, HolderAt::Set(*holder), addr, &at_pc)?;
                debit.gate(ctx, ChunkPerm::leaf(take))
            }
        };
        perm = crate::verify::heap::algebra::perm_sub_aligned(ctx, &perm, &p);
    }
    Ok(chunk.gate(ctx, perm))
}

/// `perm` where the address `at` is `addr`, `0` elsewhere; no gate when they are
/// one e-class. Address functions are injective
/// (`rewrite::address_injectivity_rule`), so the gate reduces to the arguments.
fn gate_at(ctx: &mut VerifyContext<'_>, perm: ChunkPerm, at: Id, addr: Id) -> ChunkPerm {
    if ctx.egraph.find(at) == ctx.egraph.find(addr) {
        return perm;
    }
    let eq = expr!(ctx, { at } == { addr });
    ChunkPerm::select(ctx, eq, perm, ChunkPerm::Zero)
}

/// Whether the set `sub` is within what `held_at` says is held: at every
/// location `sub` names, its amount is at most that ([`check_qp_contained`]).
/// Proven under `sub`'s own evaluation pc and `extra_pc`. Every containment
/// question about chunk sets is this one, with a different `held_at` and
/// [`Covers`].
fn set_within(
    ctx: &mut VerifyContext<'_>,
    sub: &QuantChunk,
    extra_pc: &[(Id, Polarity)],
    held_at: &mut HeldAt<'_>,
    covers: Covers,
) -> Result<bool, VerifyError> {
    let pc: Vec<_> = sub
        .eval_pc()
        .into_iter()
        .chain(extra_pc.iter().copied())
        .collect();
    check_qp_contained(
        ctx,
        &sub.q,
        &pc,
        &mut |ctx, binders| eval_qp_instance(ctx, &sub.caps, &sub.q, binders, &pc),
        held_at,
        covers,
    )
}

/// [`set_within`] the set `held`.
fn set_within_set(
    ctx: &mut VerifyContext<'_>,
    sub: &QuantChunk,
    extra_pc: &[(Id, Polarity)],
    held: &QuantChunk,
    covers: Covers,
) -> Result<bool, VerifyError> {
    set_within(
        ctx,
        sub,
        extra_pc,
        &mut |ctx, addr| qp_perm_at(ctx, held, addr),
        covers,
    )
}

/// Index of the first held set of `kind` that covers `exhaled` by itself.
pub(super) fn find_covering_qp(
    ctx: &mut VerifyContext<'_>,
    heap: &Heap,
    kind: &LocationKind,
    exhaled: &QuantChunk,
) -> Result<Option<usize>, VerifyError> {
    for (i, held) in heap.qp_chunks_of(kind).iter().enumerate() {
        if set_within_set(ctx, exhaled, &[], held, prove_framed_aligned)? {
            return Ok(Some(i));
        }
    }
    Ok(None)
}

/// Whether `a` and `b` name the same locations with the same amounts, each
/// within the other. What was taken from either counts, so a set that gave
/// something away is not the same as its origin.
pub(super) fn same_footprint(ctx: &mut VerifyContext<'_>, a: &QuantChunk, b: &QuantChunk) -> bool {
    let within = |ctx: &mut VerifyContext<'_>, sub, held| {
        set_within_set(ctx, sub, &[], held, prove_sufficient_aligned)
            // The bodies evaluated once already, at their inhales.
            .unwrap_or_else(|e| panic!("chunk set body failed to re-evaluate: {e:?}"))
    };
    within(ctx, a, b) && within(ctx, b, a)
}

/// Whether the exhaled set is covered by everything `heap` holds together: at
/// each location it names, its amount is at most the total of all holders
/// there ([`View::all_shares`]). The fallback when no single held set covers it.
pub(super) fn covered_by_sum(
    ctx: &mut VerifyContext<'_>,
    heap: &Heap,
    kind: &LocationKind,
    exhaled: &QuantChunk,
) -> Result<bool, VerifyError> {
    let pc = exhaled.eval_pc();
    let mut held_at = |ctx: &mut VerifyContext<'_>, addr| {
        Ok(heap.view(&pc).all_amounts(ctx, kind, addr)?.total(ctx))
    };
    set_within(ctx, exhaled, &[], &mut held_at, prove_framed_aligned)
}

/// What the holder `at` of `before` gives up at `addr` when `demand` is split
/// greedily over all holders there (`Shares::greedy_takes`); `0` when `demand` does
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
    let d = qp_perm_at(ctx, demand, addr)?.to_id(ctx);
    let kind = LocationKind::from_addr_type(&demand.q.loc_ty).expect("QP loc_ty must be Addr");
    let shares = before.view(guard).all_amounts(ctx, &kind, addr)?;
    let takes = shares.greedy_takes(ctx, d);
    Ok(shares
        .iter()
        .position(|h| h.at == at)
        .map_or(zero, |i| takes[i]))
}

/// Take the exhaled set from all holders together, after [`covered_by_sum`]
/// proved they hold enough. The split is greedy in `View::all_shares` order: a
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
    for c in heap.chunks_of(kind) {
        let take = share_at(ctx, &exhaled, heap, HolderAt::Chunk(c.addr), c.addr, &pc)?;
        let take = exhaled.gate(ctx, ChunkPerm::leaf(take));
        // Taken from the bare amount, the remainder keeping the chunk's presence.
        let rest =
            c.map_perm(|held| crate::verify::heap::algebra::perm_sub_aligned(ctx, held, &take));
        out = out.with_chunk(kind, rest);
    }
    for (i, set) in heap.qp_chunks_of(kind).iter().enumerate() {
        let set = set.with_taken_by(
            ctx,
            &exhaled,
            Taken::Share {
                demand: exhaled.clone(),
                before: heap.clone(),
                holder: i,
                pc: pc.as_slice().into(),
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
    let mut exhaled = exhaled;
    exhaled.value_map = held.value_map;
    let held = held.with_taken_by(ctx, &exhaled, Taken::Set(SetBody::clone(&exhaled)));
    heap.with_qp_chunk_replaced(kind, i, held)
}

/// Whether taking `exhaled` leaves `held` empty on the exhale's path: what
/// `held` still holds is contained in `exhaled`. The other direction is what
/// made `held` the covering set ([`find_covering_qp`]). Only the exhale's path
/// matters, as the heap is dropped from only there on, so the containment is
/// proven under both guards.
fn emptied_by(ctx: &mut VerifyContext<'_>, held: &QuantChunk, exhaled: &QuantChunk) -> bool {
    set_within_set(
        ctx,
        held,
        &exhaled.eval_pc(),
        exhaled,
        prove_sufficient_aligned,
    )
    // The body evaluated once already, at its inhale.
    .unwrap_or_else(|e| panic!("chunk set body failed to re-evaluate: {e:?}"))
}

/// The chunk set of `q` with its captured temps already resolved to `caps`:
/// either from the live state (an inhale or exhale of the set) or rebuilt from
/// a resource footprint slot's recipes at a call site (`walk_footprint`).
/// Proves what `check` asks of it under `pc_lits` ([`check_qp_well_formed`])
/// and assumes the inverse axiom that injectivity licenses. The result carries
/// no recipe; the caller sets one.
pub(super) fn quant_chunk_from_caps(
    ctx: &mut VerifyContext<'_>,
    q: &Rc<vmir::QuantChunks>,
    caps: Captures,
    pc_lits: &[(egg::Id, Polarity)],
    value_map: egg::Id,
    check: QpCheck,
) -> Result<QuantChunk, VerifyError> {
    check_qp_well_formed(ctx, q, &caps, pc_lits, check)?;
    // One binding per location: assume the inverse axiom that names it.
    let inv = mint_and_assume_inverse_axiom(ctx, q, &caps, pc_lits)?;
    Ok(QuantChunk::unconditional(SetBody {
        origin: SetId::fresh(ctx),
        q: q.clone(),
        caps,
        inv,
        pc: pc_lits.into(),
        taken: Rc::from([]),
        value_map,
        recipe: None,
        restored: Rc::from([]),
        handovers: Rc::from([]),
    }))
}
