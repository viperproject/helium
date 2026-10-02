//! Lower `typed::TypedPureExp<Ext>` into VMIR `PureInst` chains.

use crate::dhash::HashMap;

use lasso::Spur;

use crate::translate::sink::{PcKind, Sink};
use crate::translate::{TranslationContext, TranslationError};
use crate::viper::typed;
use crate::vmir::{
    self, FALSE, HeapInst, HeapVal, Literal, Polarity, PureInst, ResourceCall, TRUE, Val,
};

/// Earlier heap states an `old(...)` expression can read from, in a method
/// body. `baseline` is the post-requires-inhale heap (target of unlabeled
/// `old`); `labeled` maps each `label L` to the heap captured at that point.
pub(crate) struct OldHeaps<'a> {
    pub baseline: HeapVal,
    pub labeled: &'a HashMap<Spur, HeapVal>,
}

/// Heaps a pure expression reads from. `value` is used by field derefs and
/// heap-dependent functions; `perm` is used by `perm(loc)`. They differ only on
/// `exhale`, where value reads use the pre-exhale heap but `perm` tracks the
/// running (subtracted) heap; elsewhere both are the same heap. `old`, when
/// present, lets `old(...)` reach back to earlier heap states (method bodies
/// only); `None` outside a method body.
#[derive(Clone, Copy)]
pub(crate) struct HeapCtx<'a> {
    pub value: HeapVal,
    pub perm: HeapVal,
    pub old: Option<&'a OldHeaps<'a>>,
    /// The function result `Val`, available only while lowering a function
    /// postcondition (`FuncEnsuresExt::Result`); `None` everywhere else. Held by
    /// reference so `HeapCtx` stays `Copy` (`Val` is not `Copy`).
    pub result: Option<&'a Val>,
    /// Set while lowering a `forall` body. There is no heap in there (`value` and
    /// `perm` are the inert `Empty`) and there cannot be one: the body is compiled
    /// into a recipe the verifier rebuilds inside a rewrite rule, which can
    /// neither read the symbolic heap nor discharge an obligation. It gates the
    /// one construct that would need both — a heap-dependent call's `Snap`.
    pub in_quantifier: bool,
}

impl<'a> HeapCtx<'a> {
    /// Both reads from `heap`, with `old` reachable (method-body lowering).
    pub(crate) fn same_with_old(heap: HeapVal, old: &'a OldHeaps<'a>) -> Self {
        Self {
            value: heap,
            perm: heap,
            old: Some(old),
            result: None,
            in_quantifier: false,
        }
    }
}

/// Emit a desugared `unfold acc(P(args), pe)`: the slot-level consume that hands
/// back the predicate's snapshot, then the resource-level produce bound to it.
///
/// ```text
/// h1, e1 : Option<Snap(P)> := h0 - acc(P@loc(args), pe)
/// h2                       := h1 inhale P(args) pe with e1
/// ```
///
/// The order is forced by permission accounting, not by the boolean: the
/// predicate chunk must be gone before its body's slots are produced, or a
/// self-referential predicate would briefly hold both. `Sub` and `Add` carry no
/// boolean at all, so the body's facts land only on the resource op -- assumed
/// here, asserted on the `fold` side -- which is why neither ordering can
/// produce a boolean the other instruction was supposed to have justified.
pub(crate) fn emit_unfold_pair(
    b: &TranslationContext<'_>,
    sink: &mut Sink,
    base: HeapVal,
    name: Spur,
    call: ResourceCall,
    perm: vmir::PermVal,
) -> HeapVal {
    // The predicate's address: an ordinary application of its own id, typed
    // `&[group] Snap(P) @ unbounded` (see `resource::lower_resource_addr`).
    let group = b.group_tag(name);
    let addr_ty = vmir::Type::addr(
        group,
        vmir::Type::Snap(call.resource),
        vmir::Bound::Unbounded,
    );
    let addr = sink.emit_pure(
        addr_ty,
        PureInst::FunctionCall(vmir::FunctionCall {
            function: call.resource,
            type_args: Vec::new(),
            args: call.args.clone().into(),
            // A predicate `@addr`, not a Silver `function`: nothing to unfold.
            export: false,
        }),
    );
    let (h_sub, opt) = sink.emit_sub_yielding(base, addr, perm.clone());
    // The seam, named rather than implicit: `Sub` yields `Option<Snap(P)>`
    // because it discovers whether anything was there; the resource produce needs
    // a plain `Snap(P)`.
    let snap = sink.emit_pure(vmir::Type::Snap(call.resource), PureInst::OptionUnwrap(opt));
    sink.emit_resource_inhale(h_sub, call, perm, vmir::Bind::Bound(snap))
}

/// Emit a desugared `fold acc(P(args), pe)`: the resource-level consume that
/// asserts the body and hands back its snapshot, then the slot-level produce
/// bound to it.
///
/// ```text
/// h1, e1 : Snap(P) := h0 exhale P(args) pe
/// h2               := h1 + acc(P@loc(args), pe) with e1
/// ```
///
/// The mirror of [`emit_unfold_pair`], and the ordering is forced the same way:
/// the field chunks must be consumed before the predicate chunk is added, so a
/// self-referential predicate never briefly holds both. No `unwrap` seam here --
/// `Exhale`'s yield and `Add`'s bind are both plain, because a consume of a
/// resource at `perm > 0` cannot come up empty.
pub(crate) fn emit_fold_pair(
    b: &TranslationContext<'_>,
    sink: &mut Sink,
    base: HeapVal,
    name: Spur,
    call: ResourceCall,
    perm: vmir::PermVal,
) -> HeapVal {
    let group = b.group_tag(name);
    let addr_ty = vmir::Type::addr(
        group,
        vmir::Type::Snap(call.resource),
        vmir::Bound::Unbounded,
    );
    let addr = sink.emit_pure(
        addr_ty,
        PureInst::FunctionCall(vmir::FunctionCall {
            function: call.resource,
            type_args: Vec::new(),
            args: call.args.clone().into(),
            // A predicate `@addr`, not a Silver `function`: nothing to unfold.
            export: false,
        }),
    );
    let resource = call.resource;
    let (h_ex, snap) = sink.emit_resource_exhale(base, call, perm.clone(), true);
    let snap = snap.expect("a predicate is self-framed, so its exhale yields a snapshot");
    let _ = resource;
    sink.emit_heap_guarded(vmir::HeapInst::Add {
        base: h_ex,
        loc: addr,
        perm,
        bind: vmir::Bind::Bound(snap),
    })
}

/// Lower a predicate-with-perm (`P(args)` + permission) into a self-framed
/// `ResourceCall` plus the permission (a source `wildcard` → [`PermVal::Wildcard`],
/// otherwise the amount through the read-only policy — see [`Sink::perm_amount`];
/// **not** pc-gated — the caller gates). Shared by `unfolding` expressions and
/// method-body `fold`/`unfold` statements.
pub(crate) fn lower_pred_call<Ext: PureExt>(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    sink: &mut Sink,
    hctx: HeapCtx<'_>,
    pwp: &typed::PredicateWithPerm<Ext>,
) -> Result<(ResourceCall, vmir::PermVal), TranslationError> {
    let name = pwp.pred_call.name.0;
    if !b.name_map.contains_key(&name) {
        return Err(TranslationError::UnknownIdent(
            b.interner.resolve(&name).to_string(),
        ));
    }
    // Only a *concrete* predicate has a `Resource` id. An abstract one is a
    // `Function` + `Domain` pair, so the miss here is the whole abstractness
    // test — there is no `ResourceCall` to build.
    let pred_id = *b.pred_resources.get(&name).ok_or_else(|| {
        TranslationError::AbstractPredicateNotFoldable(b.interner.resolve(&name).to_string())
    })?;
    let mut args = Vec::with_capacity(pwp.pred_call.args.len());
    for a in &pwp.pred_call.args {
        args.push(lower(b, env, sink, hctx, a)?);
    }
    let perm = if crate::translate::spatial::is_wildcard(&pwp.perm) {
        vmir::PermVal::Wildcard
    } else {
        let perm_val = lower(b, env, sink, hctx, &pwp.perm)?;
        sink.perm_amount(perm_val)
    };
    let call = ResourceCall {
        resource: pred_id,
        args,
    };
    Ok((call, perm))
}

pub(crate) fn lower<Ext: PureExt>(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    sink: &mut Sink,
    hctx: HeapCtx<'_>,
    exp: &typed::TypedPureExp<Ext>,
) -> Result<Val, TranslationError> {
    use typed::PureExpKind as P;
    let ty = b.lower_type(&exp.ty);
    match &*exp.exp {
        P::Ident(id) => env
            .get(&id.0)
            .cloned()
            .ok_or_else(|| TranslationError::UnknownIdent(b.interner.resolve(&id.0).to_string())),
        P::Const(lit) => {
            let lit = lower_literal(lit)?;
            // An Int literal used in a Real (permission) context becomes the
            // equivalent Real literal directly — no `real(..)` cast needed.
            match lit {
                Literal::Int(n) if ty == vmir::Type::Real => {
                    Ok(Val::Literal(Literal::Real(num::BigRational::from(n))))
                }
                _ => Ok(Val::Literal(lit)),
            }
        }
        P::Unary(op, x) => {
            let v = lower(b, env, sink, hctx, x)?;
            match op {
                // !v  =  v ? false : true
                typed::UnOp::Not => Ok(sink.emit_pure(ty, PureInst::Ternary(v, FALSE, TRUE))),
                // -v  =  0 - v
                typed::UnOp::Neg => Ok(sink.emit_pure(
                    ty.clone(),
                    PureInst::Binary(vmir::BinOp::sub(&ty), zero_literal(&ty), v),
                )),
                typed::UnOp::Cardinality => Err(TranslationError::Unsupported("cardinality")),
            }
        }
        P::Binary(op, l, r) => lower_binary(b, env, sink, hctx, ty, op, l, r),
        P::Ternary { if_, then, else_ } => {
            let c = lower(b, env, sink, hctx, if_)?;
            let t = sink.with_cond(c.clone(), Polarity::Positive, PcKind::Branch, |sink| {
                lower(b, env, sink, hctx, then)
            })?;
            let e = sink.with_cond(c.clone(), Polarity::Negative, PcKind::Branch, |sink| {
                lower(b, env, sink, hctx, else_)
            })?;
            Ok(sink.emit_pure(ty, PureInst::Ternary(c, t, e)))
        }
        // A domain function call (pure, heap-free). Domains are monomorphic, so
        // the application carries no type arguments — only ADT constructors and
        // projections do.
        P::DomainFunctionCall(call) => {
            let type_args = Vec::new();
            let mut args = Vec::with_capacity(call.args.len());
            for a in &call.args {
                args.push(lower(b, env, sink, hctx, a)?);
            }
            let function = *b.name_map.get(&call.name.0).ok_or_else(|| {
                TranslationError::UnknownIdent(b.interner.resolve(&call.name.0).to_string())
            })?;
            Ok(sink.emit_pure(
                ty,
                PureInst::FunctionCall(vmir::FunctionCall {
                    function,
                    type_args,
                    args: args.into(),
                    // A domain function: no body recipe, no precondition token.
                    export: false,
                }),
            ))
        }
        // A constructor lowers to the semantic `AdtCons`; its type arguments are
        // its result type (`exp.ty`), the variant tag from `ctor_tag`.
        P::AdtConstructor(call) => {
            let mut args = Vec::with_capacity(call.args.len());
            for a in &call.args {
                args.push(lower(b, env, sink, hctx, a)?);
            }
            let &(adt_spur, variant) = b.adt.ctor_tag.get(&call.name.0).ok_or_else(|| {
                TranslationError::UnknownIdent(b.interner.resolve(&call.name.0).to_string())
            })?;
            let adt = b.name_map[&adt_spur];
            let type_args = adt_type_args(b, &exp.ty);
            Ok(sink.emit_pure(
                ty,
                PureInst::AdtCons {
                    adt,
                    type_args,
                    variant,
                    args,
                },
            ))
        }
        P::LetIn { binder, value, exp } => {
            let val = lower(b, env, sink, hctx, value)?;
            let mut inner_env = env.clone();
            inner_env.insert(binder.0, val);
            lower(b, &inner_env, sink, hctx, exp)
        }
        P::AdtDestructor(base, field) => {
            // `e.f` ⇒ `AdtProj{adt, variant, field}(e)`. The verifier's
            // projection reduction folds it when `e` is a known constructor.
            // Type args come from the scrutinee's type (`base.ty`).
            let type_args = adt_type_args(b, &base.ty);
            let base_v = lower(b, env, sink, hctx, base)?;
            let (adt, variant, field) = dtor_projection(b, &base.ty, field)?;
            Ok(sink.emit_pure(
                ty,
                PureInst::AdtProj {
                    adt,
                    type_args,
                    variant,
                    field,
                    base: base_v,
                },
            ))
        }
        P::AdtDiscriminator(base, variant) => {
            // `e.is<Ctor>` ⇒ `AdtTag{adt}(e) == tag_index`. The verifier's tag
            // reduction folds this to a literal when `e` is a known constructor.
            // Type args come from the scrutinee's type (`base.ty`).
            let type_args = adt_type_args(b, &base.ty);
            let base_v = lower(b, env, sink, hctx, base)?;
            let &(adt_spur, tag) = b.adt.ctor_tag.get(&variant.0).ok_or_else(|| {
                TranslationError::UnknownIdent(b.interner.resolve(&variant.0).to_string())
            })?;
            let adt = b.name_map[&adt_spur];
            let tag_call = sink.emit_pure(
                vmir::Type::Int,
                PureInst::AdtTag {
                    adt,
                    type_args,
                    base: base_v,
                },
            );
            let idx = Val::Literal(Literal::Int(num::BigInt::from(tag)));
            Ok(sink.emit_pure(
                vmir::Type::Bool,
                PureInst::Binary(vmir::BinOp::Eq, tag_call, idx),
            ))
        }
        P::Ext(ext) => Ext::lower_ext(b, env, sink, hctx, ty, ext),
    }
}

/// The type arguments of an ADT/domain-typed expression — its head's type
/// parameters at this use site. `Domain(_, args)` → lower each; any other type
/// (a non-generic / non-ADT result) → empty.
/// The `(adt id, variant, field)` that destructor `field` projects on a
/// receiver of type `base_ty`. The receiver's ADT is part of the key: several
/// ADTs may declare a field of the same name.
fn dtor_projection(
    b: &TranslationContext<'_>,
    base_ty: &typed::Type,
    field: &typed::Ident,
) -> Result<(vmir::MemberId, usize, usize), TranslationError> {
    let typed::Type::Domain(adt, _) = base_ty else {
        return Err(TranslationError::UnknownIdent(
            b.interner.resolve(&field.0).to_string(),
        ));
    };
    b.adt
        .dtor_sem
        .get(&(adt.0, field.0))
        .copied()
        .ok_or_else(|| TranslationError::UnknownIdent(b.interner.resolve(&field.0).to_string()))
}

fn adt_type_args(b: &TranslationContext<'_>, ty: &typed::Type) -> Vec<vmir::Type> {
    match ty {
        typed::Type::Domain(_, args) => args.iter().map(|t| b.lower_type(t)).collect(),
        _ => Vec::new(),
    }
}

/// The single fold kept during lowering: a division of literals in a `Real`
/// (permission) context becomes one `Real` fraction literal, so `1/2` is a
/// literal rather than a `real(1) / real(2)` division. No other arithmetic is
/// folded — e.g. `1/2 + 1/3` stays a real addition for the e-graph to handle.
fn fold_real_div(op: &typed::BinOp, l: &Literal, r: &Literal, ty: &vmir::Type) -> Option<Literal> {
    use num::BigRational;
    if !matches!(op, typed::BinOp::Div) || *ty != vmir::Type::Real {
        return None;
    }
    let rat = |lit: &Literal| -> Option<BigRational> {
        match lit {
            Literal::Int(n) => Some(BigRational::from(n.clone())),
            Literal::Real(x) => Some(x.clone()),
            _ => None,
        }
    };
    Some(Literal::Real(rat(l)? / rat(r)?))
}

/// Wrap `v` in `real(..)` when an `Int` operand is used where a `Real` is
/// expected, keeping every operation's e-graph operands homogeneous.
fn real_cast_if(sink: &mut Sink, v: Val, operand_ty: &vmir::Type, target_ty: &vmir::Type) -> Val {
    if *target_ty == vmir::Type::Real && *operand_ty == vmir::Type::Int {
        sink.emit_pure(vmir::Type::Real, PureInst::RealCast(v))
    } else {
        v
    }
}

fn lower_binary<Ext: PureExt>(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    sink: &mut Sink,
    hctx: HeapCtx<'_>,
    ty: vmir::Type,
    op: &typed::BinOp,
    l: &typed::TypedPureExp<Ext>,
    r: &typed::TypedPureExp<Ext>,
) -> Result<Val, TranslationError> {
    use typed::BinOp as B;
    use vmir::BinOp as V;
    let lv = lower(b, env, sink, hctx, l)?;
    // Short-circuiting boolean ops only evaluate `r` on the path where `l`
    // takes the guarding value, so `r` is lowered under that guard.
    match op {
        B::And => {
            // l && r  =  l ? r : false
            let rv = sink.with_cond(lv.clone(), Polarity::Positive, PcKind::Branch, |sink| {
                lower(b, env, sink, hctx, r)
            })?;
            return Ok(sink.emit_pure(ty, PureInst::Ternary(lv, rv, FALSE)));
        }
        B::Or => {
            // l || r  =  l ? true : r
            let rv = sink.with_cond(lv.clone(), Polarity::Negative, PcKind::Branch, |sink| {
                lower(b, env, sink, hctx, r)
            })?;
            return Ok(sink.emit_pure(ty, PureInst::Ternary(lv, TRUE, rv)));
        }
        B::Implies => {
            // l ==> r  =  l ? r : true
            let rv = sink.with_cond(lv.clone(), Polarity::Positive, PcKind::Branch, |sink| {
                lower(b, env, sink, hctx, r)
            })?;
            return Ok(sink.emit_pure(ty, PureInst::Ternary(lv, rv, TRUE)));
        }
        _ => {}
    }
    // Strict ops: both operands always evaluate, so `r` is lowered under the
    // outer path condition unchanged.
    let rv = lower(b, env, sink, hctx, r)?;
    // Only fold a literal division in a `Real` context to its fraction (`1/2`);
    // all other literal arithmetic stays symbolic for the e-graph.
    if let (Val::Literal(la), Val::Literal(lb)) = (&lv, &rv) {
        if let Some(folded) = fold_real_div(op, la, lb, &ty) {
            return Ok(Val::Literal(folded));
        }
    }
    // Homogenize: a `Real`-result arithmetic op with an `Int` operand gets that
    // operand wrapped in `real(..)` so the e-graph operands share a type.
    let lty = b.lower_type(&l.ty);
    let lv = real_cast_if(sink, lv, &lty, &ty);
    let rv = real_cast_if(sink, rv, &b.lower_type(&r.ty), &ty);
    // Operand sort for the sort-tagged operators. An arithmetic op's result type
    // *is* its operand sort (that is what the casts above just established), but
    // a comparison produces `Bool`, so its sort comes from an operand instead —
    // either one, since the typechecker equates the two
    // (`viper::typecheck`, `Lt | Le | Gt | Ge` imposes `lk.equate_with(rk)`).
    let cmp = &lty;
    Ok(match op {
        B::Plus => sink.emit_pure(ty.clone(), PureInst::Binary(V::add(&ty), lv, rv)),
        B::Minus => sink.emit_pure(ty.clone(), PureInst::Binary(V::sub(&ty), lv, rv)),
        B::Mult => sink.emit_pure(ty.clone(), PureInst::Binary(V::mul(&ty), lv, rv)),
        // The divisor≠0 obligation is checked in the current value heap.
        // `IntDiv` (`\`) collapses into the `Int` VMIR division: `\`'s operands
        // are `Int` by type checking, so it lands on the same variant `/` does
        // over integers.
        B::Div | B::IntDiv => sink.with_heap(hctx.value, |sink| {
            sink.emit_pure_guarded(ty.clone(), PureInst::Binary(V::div(&ty), lv, rv))
        }),
        B::Mod => sink.with_heap(hctx.value, |sink| {
            sink.emit_pure_guarded(ty, PureInst::Binary(V::Mod, lv, rv))
        }),
        B::Eq => sink.emit_pure(ty, PureInst::Binary(V::Eq, lv, rv)),
        B::Lt => sink.emit_pure(ty, PureInst::Binary(V::lt(cmp), lv, rv)),
        // Desugarings:
        B::Neq => {
            let eq = sink.emit_pure(vmir::Type::Bool, PureInst::Binary(V::Eq, lv, rv));
            sink.emit_pure(ty, PureInst::Ternary(eq, FALSE, TRUE))
        }
        B::Le => {
            // l <= r  <=>  !(r < l)
            let gt = sink.emit_pure(vmir::Type::Bool, PureInst::Binary(V::lt(cmp), rv, lv));
            sink.emit_pure(ty, PureInst::Ternary(gt, FALSE, TRUE))
        }
        B::Gt => sink.emit_pure(ty, PureInst::Binary(V::lt(cmp), rv, lv)),
        B::Ge => {
            // l >= r  <=>  !(l < r)
            let lt = sink.emit_pure(vmir::Type::Bool, PureInst::Binary(V::lt(cmp), lv, rv));
            sink.emit_pure(ty, PureInst::Ternary(lt, FALSE, TRUE))
        }
        B::And | B::Or | B::Implies => unreachable!("handled above"),
        B::Iff => sink.emit_pure(ty, PureInst::Binary(V::Eq, lv, rv)),
        B::In | B::Union | B::SetMinus | B::Intersection | B::Subset | B::Concat | B::Range => {
            return Err(TranslationError::Unsupported("collection operator"));
        }
    })
}

/// Type-appropriate zero used to desugar arithmetic negation `-v` as
/// `Binary(Minus, 0, v)`. Non-numeric types panic; upstream typechecking
/// rejects them before lowering.
pub(crate) fn zero_literal(ty: &vmir::Type) -> Val {
    match ty {
        vmir::Type::Int => Val::Literal(Literal::Int(num::BigInt::from(0))),
        vmir::Type::Real => Val::Literal(Literal::Real(num::BigInt::from(0).into())),
        other => panic!("Neg on non-numeric type {other:?}"),
    }
}

pub(crate) fn lower_literal(lit: &typed::Literal) -> Result<Literal, TranslationError> {
    match lit {
        typed::Literal::Bool(b) => Ok(Literal::Bool(*b)),
        typed::Literal::Int(n) => Ok(Literal::Int(n.clone())),
        typed::Literal::Real(r) => Ok(Literal::Real(r.clone())),
        typed::Literal::Null => Ok(Literal::Null),
        typed::Literal::Wildcard => Err(TranslationError::Unsupported("wildcard literal")),
    }
}

/// Lower a Silver `function` application to a VMIR `FunctionCall` (monomorphic,
/// no `type_args`). Calls are always pure: a **heap-dependent** callee (one
/// whose `requires` grants permission) receives the snapshot of its `#requires`
/// resource — built here by a frame-only `exhale` of its `#requires`, which
/// implicitly checks the precondition (footprint sufficiency + resource bool) —
/// as an extra trailing argument. A **heap-free** callee's precondition is
/// instead asserted as a boolean contract call.
fn lower_func_app<Ext: PureExt>(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    sink: &mut Sink,
    hctx: HeapCtx<'_>,
    ty: vmir::Type,
    call: &typed::Call<Ext>,
) -> Result<Val, TranslationError> {
    let mut args = Vec::with_capacity(call.args.len());
    for a in &call.args {
        args.push(lower(b, env, sink, hctx, a)?);
    }
    let func = *b.name_map.get(&call.name.0).ok_or_else(|| {
        TranslationError::UnknownIdent(b.interner.resolve(&call.name.0).to_string())
    })?;
    let contracts = b.contracts.get(&call.name.0);
    let requires = contracts.and_then(|c| c.requires);
    let heap_dep = contracts.is_some_and(|c| c.heap_dep);
    // Use-side precondition, and the call's actual argument list.
    let mut call_args = args.clone();
    if heap_dep {
        // Narrow the current value heap to the callee's precondition snapshot.
        // `Snap` implicitly asserts the precondition under the running pc.
        let req_id = requires.expect("heap-dep implies a requires");
        // A quantifier body has no heap to narrow, and could not have one (see
        // `HeapCtx::in_quantifier`). A binder-dependent footprint would need
        // quantified permissions besides.
        //
        // The obvious workaround — emit the `Snap` in the *host* scope and let
        // `vmir::Forall::free_temps` capture it — does not work, for two reasons
        // worth recording so they are not re-derived:
        //
        //  * `Snap` bundles two jobs: it narrows the footprint to a snapshot
        //    *and* asserts the resource's boolean. Those pull opposite ways here.
        //    The snapshot wants to be binder-free (one captured value), but the
        //    boolean must be checked per-binding **under the quantifier's
        //    antecedent** — for `get(xs, i)` it is typically `0 <= i && i < len(xs)`,
        //    which only the antecedent establishes. Hoisting asserts it outside the
        //    antecedent, so it fails exactly where the feature would be wanted.
        //  * `f#requires` is parameterized by the callee's *whole* param list, so a
        //    `Snap` for `get(xs, i)` carries `args = [xs, i]` and mentions the
        //    binder even when no footprint slot does. Hoisting is therefore only
        //    ever applicable to calls that are already fully binder-free.
        //
        // Supporting this properly means splitting those two jobs: a
        // param-independence analysis over the footprint recipes, a snapshot term
        // that omits the irrelevant params, and per-binding checking of the
        // resource boolean under the antecedent.
        if hctx.in_quantifier {
            return Err(TranslationError::HeapDepFunctionInQuantifier(
                b.interner.resolve(&call.name.0).to_string(),
            ));
        }
        // The implicit precondition check: a frame-only exhale of the callee's
        // `#requires` resource, yielding its snapshot but no heap. `1/1`, not
        // `wildcard`: the scale multiplies each footprint slot's own permission,
        // so `1 * p` reproduces the amounts the dedicated `Snap` instruction used.
        let s = sink.emit_resource_frame_exhale(
            hctx.value,
            vmir::ResourceCall {
                resource: req_id,
                args: args.clone(),
            },
            vmir::PermVal::write(),
        );
        call_args.push(s);
    } else if let Some(req_id) = requires {
        // Heap-free: assert `f#requires(args)` before the call.
        let check = call_contract(sink, req_id, args.clone());
        sink.emit_assert(check);
    }
    // No use-site `assume f#ensures(..)`: the postcondition is delivered by the
    // verifier as a guarded rewrite keyed on `f` (the pre-token is the passed
    // `assert f#requires(args)` above), so transitive call sites get it too.
    // `emit_call`, not `emit_pure`: the inst must carry its lowering pc so the
    // verifier can assume the callee's `f%pre` token under it (the token's truth
    // is what releases the callee's body equality and postcondition facts).
    // The one call in this file that exports: a genuine value-position use whose
    // result flows onward, so the callee's application really is materialized at
    // a client of the enclosing body and really does need its `%pre` token there.
    // A contract definition (`f#requires` / `f#ensures`) is no exception — it is
    // an ordinary function, and a call in its body is a value position like any
    // other. Without the token the fact the contract carries is stranded: a
    // client of `f` learns `f(3) == g(3)` and cannot unfold `g`.
    //
    // A **method body** is the one place the mark is meaningless: nothing ever
    // replays it, so there is no later site for the token to be re-released at.
    let ret = sink.emit_call(
        ty,
        PureInst::FunctionCall(vmir::FunctionCall {
            function: func,
            type_args: Vec::new(),
            args: call_args.into(),
            export: !sink.in_method_body,
        }),
    );
    Ok(ret)
}

/// Lower a heap-reading node (`e.f`, a `function` call, `unfolding`). Shared by
/// every heap-bearing context's `lower_ext`.
pub(crate) fn lower_heap_node<Ext: PureExt>(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    sink: &mut Sink,
    hctx: HeapCtx<'_>,
    ty: vmir::Type,
    node: &typed::HeapNode<Ext>,
) -> Result<Val, TranslationError> {
    use typed::HeapNode as H;
    match node {
        H::Field(base, id) => {
            let base = lower(b, env, sink, hctx, base)?;
            let addr = crate::translate::resource::field_addr(b, sink, base, id.0)?;
            Ok(sink.emit_pure_guarded(ty, PureInst::Deref(hctx.value, addr)))
        }
        // A heap-dependent Silver `function` — not generic yet, so no type args.
        H::FunctionCall(call) => lower_func_app(b, env, sink, hctx, ty, call),
        H::Unfolding(pwp, body) => {
            // `unfolding acc(P(args), perm) in body`: a scoped unfold. Emit an
            // `Unfold`, evaluate `body` against the unfolded heap, then discard it
            // (the surrounding expression keeps reading the original `hctx`).
            let (call, perm) = lower_pred_call(b, env, sink, hctx, pwp)?;
            let h = emit_unfold_pair(b, sink, hctx.value, pwp.pred_call.name.0, call, perm);
            let inner = HeapCtx {
                value: h,
                perm: h,
                ..hctx
            };
            lower(b, env, sink, inner, body)
        }
    }
}

pub(crate) trait PureExt: Sized + Clone + std::fmt::Debug {
    fn lower_ext(
        b: &TranslationContext<'_>,
        env: &HashMap<Spur, Val>,
        sink: &mut Sink,
        hctx: HeapCtx<'_>,
        ty: vmir::Type,
        ext: &Self,
    ) -> Result<Val, TranslationError>;
}

impl PureExt for ! {
    fn lower_ext(
        _b: &TranslationContext<'_>,
        _env: &HashMap<Spur, Val>,
        _sink: &mut Sink,
        _hctx: HeapCtx<'_>,
        _ty: vmir::Type,
        ext: &Self,
    ) -> Result<Val, TranslationError> {
        match *ext {}
    }
}

impl PureExt for typed::HeapExt {
    fn lower_ext(
        b: &TranslationContext<'_>,
        env: &HashMap<Spur, Val>,
        sink: &mut Sink,
        hctx: HeapCtx<'_>,
        ty: vmir::Type,
        ext: &Self,
    ) -> Result<Val, TranslationError> {
        match ext {
            typed::HeapExt::Heap(node) => lower_heap_node(b, env, sink, hctx, ty, node),
            typed::HeapExt::Forall(q) => lower_forall(b, env, sink, q),
        }
    }
}

/// Domain axioms: the only extension is a call to a precondition-free Silver
/// `function` (typecheck-enforced), so `lower_func_app` never touches the
/// (inert `Empty`) heap in `hctx` — no `Snap`, no `requires` assert; only the
/// callee's `#ensures` assume is stitched.
impl PureExt for typed::AxiomExt {
    fn lower_ext(
        b: &TranslationContext<'_>,
        env: &HashMap<Spur, Val>,
        sink: &mut Sink,
        hctx: HeapCtx<'_>,
        ty: vmir::Type,
        ext: &Self,
    ) -> Result<Val, TranslationError> {
        match ext {
            typed::AxiomExt::FunctionCall(call) => lower_func_app(b, env, sink, hctx, ty, call),
            typed::AxiomExt::Forall(q) => lower_forall(b, env, sink, q),
        }
    }
}

/// Lower a pure `forall` into an inline [`PureInst::Forall`] step of the
/// enclosing stream. Capture is **implicit**: the body is lowered in an inner sink
/// that continues *this* sink's temp numbering, so a free identifier resolves
/// through the enclosing `env` to the very temp it already has — nothing to
/// collect, nothing to remap.
///
/// The step's own temp `p` is allocated first, since the body is numbered *from*
/// it (`binder_base == p`, see [`vmir::Forall`]): the first binder shadows the
/// `forall`'s own boolean, so the quantifier cannot refer to itself. This sink's
/// counter is left at `p + 1`, so the enclosing stream shadows the body's temps
/// rather than skipping past them; the two scopes never overlap in time, so that
/// is sound and it keeps the verifier's positional value table dense.
fn lower_forall(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    sink: &mut Sink,
    q: &typed::Forall,
) -> Result<Val, TranslationError> {
    let n = q.bound.len();
    let bound: Vec<vmir::Type> = q.bound.iter().map(|bv| b.lower_type(&bv.ty)).collect();

    // The step's temp, then the frame it opens.
    let step = sink.next_val_temp();
    let Val::Temp(p) = step else {
        unreachable!("next_val_temp yields a temp")
    };
    let binder_base = p;

    // The body's environment: the enclosing one, with the binders bound to the
    // frame's leading temps. Shadowing an enclosing name is exactly what `insert`
    // does here, and matches Silver's scoping.
    let mut inner_env = env.clone();
    for (k, bv) in q.bound.iter().enumerate() {
        inner_env.insert(bv.name.0, Val::Temp(binder_base + k));
    }

    // Trigger groups: alternatives, each a conjunctive multi-pattern. Typecheck
    // has already established every group's shape (application-rooted terms over
    // variables/literals/nested applications) and that it covers all binders.
    let mut triggers = Vec::with_capacity(q.triggers.len());
    for group in &q.triggers {
        let terms = group
            .iter()
            .map(|t| lower_trig_term(b, t, &inner_env))
            .collect::<Result<Vec<_>, _>>()?;
        triggers.push(vmir::QuantTrigger {
            terms: terms.into(),
        });
    }

    // Lower the body in an inner sink whose params (everything up to and including
    // the binders) occupy `Val::Temp(0..binder_base + n)`.
    let mut inner = Sink::new(binder_base + n, 0);
    let hctx = HeapCtx {
        value: HeapVal::Empty,
        perm: HeapVal::Empty,
        old: None,
        result: None,
        in_quantifier: true,
    };
    let res = lower(b, &inner_env, &mut inner, hctx, &q.body)?;
    let body = vmir::FunctionBody {
        insts: inner.insts,
        res,
    };

    sink.emit_pure_at(
        &step,
        vmir::Type::Bool,
        PureInst::Forall(Box::new(vmir::Forall {
            binder_base,
            bound: bound.into(),
            triggers: triggers.into(),
            body,
        })),
    );
    Ok(step)
}

/// Lower one trigger term into its VMIR pattern tree. A variable resolves through
/// `env` to the temp it names — a binder or an enclosing value, told apart later
/// by `binder_base` alone; a literal to `Lit`; anything else is an application,
/// lowered to the same head the body's `PureInst` would produce, with its
/// arguments lowered recursively (a trigger may nest arbitrarily). Shapes outside
/// this grammar were rejected at typecheck (`check_triggers`).
fn lower_trig_term(
    b: &TranslationContext<'_>,
    term: &typed::TypedPureExp<typed::AxiomExt>,
    env: &HashMap<Spur, Val>,
) -> Result<vmir::TrigTerm, TranslationError> {
    use typed::PureExpKind as P;
    let lower_args = |args: &[typed::TypedPureExp<typed::AxiomExt>]| {
        args.iter()
            .map(|a| lower_trig_term(b, a, env))
            .collect::<Result<Vec<_>, _>>()
    };
    match term.exp.as_ref() {
        P::Ident(id) => match env.get(&id.0) {
            Some(Val::Temp(k)) => Ok(vmir::TrigTerm::Var(*k)),
            // A trigger variable bound to a literal cannot happen: `env` maps
            // params, locals and binders, all of them temps.
            Some(Val::Literal(lit)) => Ok(vmir::TrigTerm::Lit(lit.clone())),
            None => Err(TranslationError::UnknownIdent(
                b.interner.resolve(&id.0).to_string(),
            )),
        },
        P::Const(lit) => Ok(vmir::TrigTerm::Lit(lower_literal(lit)?)),
        P::DomainFunctionCall(call) | P::Ext(typed::AxiomExt::FunctionCall(call)) => {
            let function = *b.name_map.get(&call.name.0).ok_or_else(|| {
                TranslationError::UnknownIdent(b.interner.resolve(&call.name.0).to_string())
            })?;
            Ok(vmir::TrigTerm::App {
                head: vmir::TrigHead::Func(function),
                type_args: Vec::new(),
                args: lower_args(&call.args)?.into(),
            })
        }
        P::AdtConstructor(call) => {
            let &(adt_spur, variant) = b.adt.ctor_tag.get(&call.name.0).ok_or_else(|| {
                TranslationError::UnknownIdent(b.interner.resolve(&call.name.0).to_string())
            })?;
            Ok(vmir::TrigTerm::App {
                head: vmir::TrigHead::AdtCons {
                    adt: b.name_map[&adt_spur],
                    variant,
                },
                type_args: adt_type_args(b, &term.ty),
                args: lower_args(&call.args)?.into(),
            })
        }
        P::AdtDestructor(base, field) => {
            let (adt, variant, field) = dtor_projection(b, &base.ty, field)?;
            Ok(vmir::TrigTerm::App {
                head: vmir::TrigHead::AdtProj {
                    adt,
                    variant,
                    field,
                },
                type_args: adt_type_args(b, &base.ty),
                args: Box::new([lower_trig_term(b, base, env)?]),
            })
        }
        // `{ e.isCons }` triggers on the tag application the discriminator reads
        // (the `== tag` comparison around it is not a matchable shape).
        P::AdtDiscriminator(base, variant) => {
            let &(adt_spur, _) = b.adt.ctor_tag.get(&variant.0).ok_or_else(|| {
                TranslationError::UnknownIdent(b.interner.resolve(&variant.0).to_string())
            })?;
            Ok(vmir::TrigTerm::App {
                head: vmir::TrigHead::AdtTag {
                    adt: b.name_map[&adt_spur],
                },
                type_args: adt_type_args(b, &base.ty),
                args: Box::new([lower_trig_term(b, base, env)?]),
            })
        }
        P::Unary(..)
        | P::Binary(..)
        | P::Ternary { .. }
        | P::LetIn { .. }
        | P::Ext(typed::AxiomExt::Forall(_)) => {
            unreachable!("typecheck rejects non-matchable trigger subterms")
        }
    }
}

impl PureExt for typed::MethodEnsuresExt {
    fn lower_ext(
        b: &TranslationContext<'_>,
        env: &HashMap<Spur, Val>,
        sink: &mut Sink,
        hctx: HeapCtx<'_>,
        ty: vmir::Type,
        ext: &Self,
    ) -> Result<Val, TranslationError> {
        match ext {
            typed::MethodEnsuresExt::Heap(node) => lower_heap_node(b, env, sink, hctx, ty, node),
            // old(e): re-read `e` against the method pre-state. For a two-state
            // ensures that heap is the one its entry bound `inhale` reconstructs
            // from the trailing snapshot parameter, supplied as the `old`
            // baseline by `lower_spatial_ensures`. Ensures-`old` is always
            // unlabeled (`old[L]` is a type error). A self-framed ensures has no
            // pre-state (`hctx.old == None`): `old` there needs a `requires` to
            // frame it.
            typed::MethodEnsuresExt::Old(inner) => {
                let old = hctx.old.ok_or(TranslationError::Unsupported(
                    "`old` in method ensures needs a precondition framing it",
                ))?;
                let heap = old.baseline;
                lower(
                    b,
                    env,
                    sink,
                    HeapCtx {
                        value: heap,
                        perm: heap,
                        ..hctx
                    },
                    inner,
                )
            }
            typed::MethodEnsuresExt::Forall(q) => lower_forall(b, env, sink, q),
        }
    }
}

impl PureExt for typed::MethodBodyExt {
    fn lower_ext(
        b: &TranslationContext<'_>,
        env: &HashMap<Spur, Val>,
        sink: &mut Sink,
        hctx: HeapCtx<'_>,
        ty: vmir::Type,
        ext: &Self,
    ) -> Result<Val, TranslationError> {
        match ext {
            typed::MethodBodyExt::Heap(node) => lower_heap_node(b, env, sink, hctx, ty, node),
            // old(e) / old[L](e): re-read `e` against an earlier heap. Unlabeled
            // → the post-requires-inhale baseline; labeled → the heap captured
            // at `label L`. The `old` context is carried along so nested `old`s
            // still resolve.
            typed::MethodBodyExt::Old(label, inner) => {
                let old = hctx
                    .old
                    .ok_or(TranslationError::Unsupported("`old` outside method body"))?;
                let heap = match label {
                    None => old.baseline,
                    Some(l) => *old.labeled.get(l).ok_or_else(|| {
                        TranslationError::UnknownIdent(b.interner.resolve(l).to_string())
                    })?,
                };
                lower(
                    b,
                    env,
                    sink,
                    HeapCtx {
                        value: heap,
                        perm: heap,
                        ..hctx
                    },
                    inner,
                )
            }
            // perm(loc): query the permission held at `loc` in the perm heap.
            typed::MethodBodyExt::Perm(res) => {
                let addr =
                    crate::translate::resource::lower_resource_addr(b, env, sink, hctx, res)?;
                Ok(sink.emit_pure(ty, PureInst::Perm(hctx.perm, addr)))
            }
            typed::MethodBodyExt::Forall(q) => lower_forall(b, env, sink, q),
        }
    }
}

impl PureExt for typed::FuncEnsuresExt {
    fn lower_ext(
        b: &TranslationContext<'_>,
        env: &HashMap<Spur, Val>,
        sink: &mut Sink,
        hctx: HeapCtx<'_>,
        ty: vmir::Type,
        ext: &Self,
    ) -> Result<Val, TranslationError> {
        match ext {
            typed::FuncEnsuresExt::Heap(node) => lower_heap_node(b, env, sink, hctx, ty, node),
            // `result`: the function's return value, supplied by the ensures
            // function's caller in the last parameter slot.
            typed::FuncEnsuresExt::Result => hctx.result.cloned().ok_or(
                TranslationError::Unsupported("`result` outside a function postcondition"),
            ),
            // A function is single-state: `old(e)` reads the same (only) heap the
            // postcondition is framed by, so it re-reads `e` against the current
            // context.
            typed::FuncEnsuresExt::Old(inner) => lower(b, env, sink, hctx, inner),
            typed::FuncEnsuresExt::Forall(q) => lower_forall(b, env, sink, q),
        }
    }
}

/// The contract functions to stitch around a function body: `assume
/// requires(params)` at entry, `assert ensures(params, result)` at exit.
/// `requires`/`ensures` are `None` when the function omits that clause; for a
/// heap-dependent function `requires` is `None` (the precondition is assumed
/// implicitly by the entry bound `inhale`) and `ensures` is `None` too (the
/// snapshot-taking exit check is deferred). `params` are the function's
/// parameter `Val`s (`Temp(0..n_params)`).
///
/// There is deliberately **no** use-site `assume ensures(..)`: postconditions
/// are delivered by the verifier as guarded rewrites keyed on the function
/// symbol — see the facts replay in `verify::rewrite`.
pub(crate) struct FnContract {
    pub requires: Option<vmir::MemberId>,
    pub ensures: Option<vmir::MemberId>,
    pub params: Vec<Val>,
    /// The trailing snapshot param of a heap-dependent function, appended to the
    /// exit `assert #ensures(params, result, s)`.
    pub snap: Option<Val>,
}

/// The entry bound `inhale` of a heap-dependent function (or ensures-function)
/// body: reconstruct the precondition heap from the snapshot parameter `snap`
/// of `resource(args)`. The produced heap becomes the body's value/perm heap.
pub(crate) struct SnapEntry {
    pub resource: vmir::MemberId,
    pub args: Vec<Val>,
    pub snap: Val,
}

/// Emit a heap-free boolean contract call `func(args)` and return its `Val`.
///
/// `emit_call`, not `emit_pure`: a contract function is an ordinary function, so
/// the verifier assumes its `%pre` token here, and that token's truth is what
/// releases the callees the contract names. A `g#requires(1)` check under `b`
/// must therefore carry `b` — otherwise the release is unconditional and `g`'s
/// own postcondition leaks onto sibling paths.
fn call_contract(sink: &mut Sink, func: vmir::MemberId, args: Vec<Val>) -> Val {
    sink.emit_call(
        vmir::Type::Bool,
        PureInst::FunctionCall(vmir::FunctionCall {
            function: func,
            type_args: Vec::new(),
            args: args.into(),
            // Obligation position (an entry `assume`, a `g#requires` check, the
            // exit `f#ensures` assert): feeds no result, so no *enclosing* body
            // needs to re-mint this token when it is replayed at its own call
            // sites. The token minted here, at this occurrence, is unaffected.
            export: false,
        }),
    )
}

/// Lower a pure expression into a standalone [`vmir::FunctionBody`] (function
/// body / contract-function definition). `val_base` is the first free pure-temp
/// counter (params, plus `result`/snapshot slots for a postcondition /
/// heap-dependent function, occupy the lower temps); `heap` is the context heap
/// the body reads from (`HeapVal::Empty` for a heap-free body); `result` is
/// `Some` only for a postcondition function. When `snap_entry` is `Some`, the
/// body opens with its bound `inhale` — reconstructing the precondition heap from
/// the snapshot parameter (implicitly assuming the resource bool) — and that
/// heap replaces `heap` as the body's value/perm heap. When `contract` is
/// `Some`, the body **assumes** `requires(params)` at entry and **asserts**
/// `ensures(params ++ [body_result] ++ [snap])` at exit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_function_body<Ext: PureExt>(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    exp: &typed::TypedPureExp<Ext>,
    val_base: usize,
    heap: HeapVal,
    result: Option<Val>,
    contract: Option<FnContract>,
    snap_entry: Option<SnapEntry>,
) -> Result<vmir::FunctionBody, TranslationError> {
    let mut sink = Sink::new(val_base, 0);
    // Function bodies are read-only: every `unfolding`/`acc` permission is
    // weakened to a wildcard (a function only needs *some* positive share).
    sink.read_only = true;
    // A heap-dependent body reads the heap reconstructed from its snapshot
    // parameter; a heap-free body reads the inert `heap` (`Empty`).
    let heap = match snap_entry {
        Some(SnapEntry {
            resource,
            args,
            snap,
        }) => sink.emit_heap(HeapInst::Inhale {
            base: HeapVal::Empty,
            bind: vmir::Bind::Bound(snap),
            call: vmir::ResourceCall { resource, args },
            // `1/1`, NOT `wildcard`: the scale multiplies each footprint
            // slot's own permission, so `1/1` reproduces the amounts the
            // dedicated instruction used (`1 * p` folds away). A wildcard
            // scale would silently rewrite every slot to `w * p`.
            perm: vmir::PermVal::write(),
        }),
        None => heap,
    };
    let hctx = HeapCtx {
        value: heap,
        perm: heap,
        old: None,
        result: result.as_ref(),
        in_quantifier: false,
    };
    // Entry: assume the precondition. (Heap-dependent bodies skip this — the
    // the bound `inhale` above assumes the requires resource's bool implicitly.)
    if let Some(FnContract {
        requires: Some(req),
        params,
        ..
    }) = &contract
    {
        let check = call_contract(&mut sink, *req, params.clone());
        sink.emit_assume(check);
    }
    let res = lower(b, env, &mut sink, hctx, exp)?;
    // Exit: assert the postcondition — the definition-side check, and (via the
    // verifier's facts export) the source of the post fact replayed at every
    // occurrence of the function. A heap-dependent function's `#ensures` takes
    // the snapshot too (`(params ++ [result, s])`), so it can read the
    // precondition heap.
    if let Some(FnContract {
        ensures: Some(ens),
        params,
        snap,
        ..
    }) = &contract
    {
        let mut args = params.clone();
        args.push(res.clone());
        args.extend(snap.clone());
        let check = call_contract(&mut sink, *ens, args);
        sink.emit_assert(check);
    }
    Ok(vmir::FunctionBody {
        insts: sink.insts,
        res,
    })
}

/// Lower a domain-axiom body (heap-free, no contract). Any `forall` it contains
/// is an inline [`PureInst::Forall`] step of the returned stream.
pub(crate) fn lower_axiom_body(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    exp: &typed::TypedPureExp<typed::AxiomExt>,
) -> Result<vmir::FunctionBody, TranslationError> {
    let mut sink = Sink::new(0, 0);
    let hctx = HeapCtx {
        value: HeapVal::Empty,
        perm: HeapVal::Empty,
        old: None,
        result: None,
        in_quantifier: false,
    };
    let res = lower(b, env, &mut sink, hctx, exp)?;
    Ok(vmir::FunctionBody {
        insts: sink.insts,
        res,
    })
}
