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
    /// `Some` while lowering a `forall` body: the heap reads of the body, already
    /// evaluated at the quantifier's position (see [`lower_forall`]). There is no
    /// heap in the body itself (`value` and `perm` are the inert `Empty`) and
    /// there cannot be one: the body is compiled into a recipe the verifier
    /// rebuilds inside a rewrite rule, which can neither read the symbolic heap
    /// nor discharge an obligation. A heap read missing from the map is one that
    /// depends on a binder.
    pub hoisted: Option<&'a Hoisted>,
}

/// The heap reads of a `forall` body that mention no binder, each evaluated
/// once at the quantifier's position, keyed by the address of its extension
/// node in the typed body (the body is borrowed, so the addresses are stable
/// for the whole lowering). A body occurrence lowers to the value here, which
/// the quantifier then captures like any other enclosing value.
pub(crate) type Hoisted = HashMap<*const (), Val>;

/// The key of an extension node in [`Hoisted`].
fn hoist_key<Ext>(ext: &Ext) -> *const () {
    std::ptr::from_ref(ext).cast()
}

impl<'a> HeapCtx<'a> {
    /// Both reads from `heap`, with `old` reachable (method-body lowering).
    pub(crate) fn same_with_old(heap: HeapVal, old: &'a OldHeaps<'a>) -> Self {
        Self {
            value: heap,
            perm: heap,
            old: Some(old),
            result: None,
            hoisted: None,
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
        // its result type (`exp.ty`), the variant index from `ctor_variant`.
        P::AdtConstructor(call) => {
            let mut args = Vec::with_capacity(call.args.len());
            for a in &call.args {
                args.push(lower(b, env, sink, hctx, a)?);
            }
            let &(adt_spur, variant) = b.adt.ctor_variant.get(&call.name.0).ok_or_else(|| {
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
            // `e.is<Ctor>` ⇒ the `Bool` tester `AdtIs{adt, variant}(e)`. The
            // verifier's `is` reduction folds it to a literal when `e` is a known
            // constructor; `is-excl` relates the testers of one value.
            // Type args come from the scrutinee's type (`base.ty`).
            let type_args = adt_type_args(b, &base.ty);
            let base_v = lower(b, env, sink, hctx, base)?;
            let &(adt_spur, variant) = b.adt.ctor_variant.get(&variant.0).ok_or_else(|| {
                TranslationError::UnknownIdent(b.interner.resolve(&variant.0).to_string())
            })?;
            Ok(sink.emit_pure(
                vmir::Type::Bool,
                PureInst::AdtIs {
                    adt: b.name_map[&adt_spur],
                    type_args,
                    variant,
                    base: base_v,
                },
            ))
        }
        P::Ext(ext) => match hctx.hoisted.and_then(|h| h.get(&hoist_key(ext))) {
            Some(v) => Ok(v.clone()),
            None => Ext::lower_ext(b, env, sink, hctx, ty, ext),
        },
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
        // A quantifier body has no heap to narrow (see `HeapCtx::hoisted`). A
        // call mentioning no binder was evaluated at the quantifier's position
        // instead, so one reaching here mentions a binder: its snapshot would
        // have to be taken per instance (quantified permissions).
        if hctx.hoisted.is_some() {
            return Err(TranslationError::BinderDependentHeapRead(format!(
                "function `{}`",
                b.interner.resolve(&call.name.0)
            )));
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
    // A quantifier body cannot read a heap; the reads it may contain were
    // evaluated at the quantifier's position (`HeapCtx::hoisted`). A call is
    // decided by `lower_func_app`: a heap-free callee is fine here.
    if hctx.hoisted.is_some() && !matches!(node, H::FunctionCall(_)) {
        return Err(binder_dependent_read(b, &ExtNode::Heap(node)));
    }
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

pub(crate) trait PureExt: Sized + Clone + PartialEq + std::fmt::Debug {
    fn lower_ext(
        b: &TranslationContext<'_>,
        env: &HashMap<Spur, Val>,
        sink: &mut Sink,
        hctx: HeapCtx<'_>,
        ty: vmir::Type,
        ext: &Self,
    ) -> Result<Val, TranslationError>;

    /// What this node is to a quantifier body: which heap it reads and where its
    /// sub-expressions are (see [`Hoister`]).
    fn node(&self) -> ExtNode<'_, Self>;

    /// The heap context an [`ExtNode::Old`] node reads its operand in.
    fn old_ctx<'h>(
        &self,
        _b: &TranslationContext<'_>,
        _hctx: HeapCtx<'h>,
    ) -> Result<HeapCtx<'h>, TranslationError> {
        unreachable!("old_ctx on a node that is not `old`")
    }
}

/// A context extension node, as seen by a quantifier body.
pub(crate) enum ExtNode<'e, Ext> {
    /// `e.f`, a Silver `function` call, `unfolding`: reads the context's heap
    /// (a call only if its callee is heap-dependent).
    Heap(&'e typed::HeapNode<Ext>),
    /// `old(e)` / `old[L](e)`: `e` is read in the heap [`PureExt::old_ctx`] names.
    Old(&'e typed::TypedPureExp<Ext>),
    /// `perm(loc)`: reads the context's permission heap.
    Perm(&'e typed::ResourceExp<Ext>),
    /// `result`: a value, reads nothing.
    Result,
    /// A call to a precondition-free Silver `function` in an axiom: reads nothing.
    PureCall(&'e typed::Call<Ext>),
    Forall(&'e typed::Forall<Ext>),
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
    fn node(&self) -> ExtNode<'_, Self> {
        match *self {}
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
            typed::HeapExt::Forall(q) => lower_forall(b, env, sink, hctx, q),
        }
    }
    fn node(&self) -> ExtNode<'_, Self> {
        match self {
            typed::HeapExt::Heap(node) => ExtNode::Heap(node),
            typed::HeapExt::Forall(q) => ExtNode::Forall(q),
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
            typed::AxiomExt::Forall(q) => lower_forall(b, env, sink, hctx, q),
        }
    }
    fn node(&self) -> ExtNode<'_, Self> {
        match self {
            typed::AxiomExt::FunctionCall(call) => ExtNode::PureCall(call),
            typed::AxiomExt::Forall(q) => ExtNode::Forall(q),
        }
    }
}

/// Lower a pure `forall` into an inline [`PureInst::Forall`] step of the
/// enclosing stream. Capture is **implicit**: the body is lowered in an inner sink
/// that continues *this* sink's temp numbering, so a free identifier resolves
/// through the enclosing `env` to the very temp it already has — nothing to
/// collect, nothing to remap.
///
/// The body may read the heap of the quantifier's position (and `old` heaps),
/// as Silver allows in an assertion, but only through reads that mention no
/// binder: those are evaluated here, in the enclosing stream, before the step
/// is allocated ([`Hoister`]), and the body uses their values, which the
/// quantifier captures. A closed read has one value for every instance, and its
/// well-definedness (permission, a callee's precondition) is checked here under
/// the enclosing path condition: what Silicon checks per instance, minus the
/// body's own guards on the way to the read. Dropping those guards strengthens
/// the check, so it stays sound; it rejects only a read that the enclosing state
/// does not frame but every instance reaching it does, through its guard (a
/// vacuous guard, or one whose satisfiability implies the permission). A read
/// that mentions a binder needs a per-instance footprint, i.e. quantified
/// permissions, and is rejected.
///
/// The step's own temp `p` is allocated after those reads, since the body is
/// numbered *from* it (`binder_base == p`, see [`vmir::Forall`]): the first
/// binder shadows the `forall`'s own boolean, so the quantifier cannot refer to
/// itself. This sink's counter is left at `p + 1`, so the enclosing stream
/// shadows the body's temps rather than skipping past them; the two scopes never
/// overlap in time, so that is sound and it keeps the verifier's positional
/// value table dense.
fn lower_forall<Ext: PureExt>(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    sink: &mut Sink,
    hctx: HeapCtx<'_>,
    q: &typed::Forall<Ext>,
) -> Result<Val, TranslationError> {
    let n = q.bound.len();
    let bound: Vec<vmir::Type> = q.bound.iter().map(|bv| b.lower_type(&bv.ty)).collect();

    // The closed reads, in the enclosing stream. A nested quantifier's were
    // collected with its encloser's (the walk descends into nested bodies), so
    // it reuses that map.
    let mut own = Hoisted::default();
    let hoisted = match hctx.hoisted {
        Some(outer) => outer,
        None => {
            let mut hoister = Hoister {
                b,
                env,
                sink: &mut *sink,
                local: Vec::new(),
                done: Vec::new(),
                out: &mut own,
            };
            hoister.forall(hctx, q)?;
            &own
        }
    };
    let body_hctx = HeapCtx {
        value: HeapVal::Empty,
        perm: HeapVal::Empty,
        hoisted: Some(hoisted),
        ..hctx
    };

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
            .map(|t| match &*t.exp {
                // A closed heap read at a term's root was evaluated to a value
                // above, leaving no application to match on.
                typed::PureExpKind::Ext(ext) if hoisted.contains_key(&hoist_key(ext)) => {
                    Err(TranslationError::Unsupported(
                        "a trigger term that reads the heap and mentions no quantified variable",
                    ))
                }
                _ => lower_trig_term(b, t, &inner_env, hoisted),
            })
            .collect::<Result<Vec<_>, _>>()?;
        triggers.push(vmir::QuantTrigger {
            terms: terms.into(),
        });
    }

    // Lower the body in an inner sink whose params (everything up to and including
    // the binders) occupy `Val::Temp(0..binder_base + n)`.
    let mut inner = Sink::new(binder_base + n, 0);
    let res = lower(b, &inner_env, &mut inner, body_hctx, &q.body)?;
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

/// The walk that evaluates a quantifier's closed heap reads in the enclosing
/// stream (see [`lower_forall`]). It descends top-down, so a read is taken as a
/// whole as soon as it mentions no name bound inside the quantifier; only a read
/// that does mention one is looked into: `old(e)` for closed reads of `e` in the
/// old heap, anything else is rejected.
struct Hoister<'b, 'c, 's, 'e, Ext> {
    b: &'b TranslationContext<'c>,
    env: &'b HashMap<Spur, Val>,
    sink: &'s mut Sink,
    /// The names bound between the outermost quantifier and the current node:
    /// binders of it and of nested quantifiers, and `let` binders.
    local: Vec<Spur>,
    /// The reads evaluated so far, with the heaps they were read in: a read
    /// written twice (in a trigger and in the body, say) is evaluated, and
    /// checked, once.
    done: Vec<(HeapVal, HeapVal, &'e typed::TypedPureExp<Ext>, Val)>,
    out: &'s mut Hoisted,
}

impl<'e, Ext: PureExt> Hoister<'_, '_, '_, 'e, Ext> {
    fn forall(
        &mut self,
        hctx: HeapCtx<'_>,
        q: &'e typed::Forall<Ext>,
    ) -> Result<(), TranslationError> {
        let depth = self.local.len();
        self.local.extend(q.bound.iter().map(|bv| bv.name.0));
        let r = q
            .triggers
            .iter()
            .flatten()
            .chain(std::iter::once(&q.body))
            .try_for_each(|e| self.exp(hctx, e));
        self.local.truncate(depth);
        r
    }

    fn exp(
        &mut self,
        hctx: HeapCtx<'_>,
        exp: &'e typed::TypedPureExp<Ext>,
    ) -> Result<(), TranslationError> {
        use typed::PureExpKind as P;
        match &*exp.exp {
            P::Ident(_) | P::Const(_) => Ok(()),
            P::Unary(_, e) | P::AdtDestructor(e, _) | P::AdtDiscriminator(e, _) => {
                self.exp(hctx, e)
            }
            P::Binary(_, l, r) => {
                self.exp(hctx, l)?;
                self.exp(hctx, r)
            }
            P::Ternary { if_, then, else_ } => {
                self.exp(hctx, if_)?;
                self.exp(hctx, then)?;
                self.exp(hctx, else_)
            }
            P::LetIn { binder, value, exp } => {
                self.exp(hctx, value)?;
                self.local.push(binder.0);
                let r = self.exp(hctx, exp);
                self.local.pop();
                r
            }
            P::DomainFunctionCall(call) | P::AdtConstructor(call) => self.args(hctx, call),
            P::Ext(ext) => match ext.node() {
                ExtNode::Forall(q) => self.forall(hctx, q),
                ExtNode::PureCall(call) => self.args(hctx, call),
                ExtNode::Result => Ok(()),
                ExtNode::Heap(typed::HeapNode::FunctionCall(call))
                    if !self
                        .b
                        .contracts
                        .get(&call.name.0)
                        .is_some_and(|c| c.heap_dep) =>
                {
                    self.args(hctx, call)
                }
                _ if !mentions(exp, &self.local) => self.hoist(hctx, exp, ext),
                ExtNode::Old(inner) => {
                    let old = ext.old_ctx(self.b, hctx)?;
                    self.exp(old, inner)
                }
                node => Err(binder_dependent_read(self.b, &node)),
            },
        }
    }

    fn args(
        &mut self,
        hctx: HeapCtx<'_>,
        call: &'e typed::Call<Ext>,
    ) -> Result<(), TranslationError> {
        call.args.iter().try_for_each(|a| self.exp(hctx, a))
    }

    /// Evaluate the closed read `exp` (extension node `ext`) here, in the
    /// enclosing stream, exactly as it would be evaluated outside the quantifier.
    fn hoist(
        &mut self,
        hctx: HeapCtx<'_>,
        exp: &'e typed::TypedPureExp<Ext>,
        ext: &'e Ext,
    ) -> Result<(), TranslationError> {
        let seen = self
            .done
            .iter()
            .find(|(v, p, e, _)| *v == hctx.value && *p == hctx.perm && *e == exp);
        let val = match seen {
            Some((.., val)) => val.clone(),
            None => {
                let val = lower(self.b, self.env, self.sink, hctx, exp)?;
                self.done.push((hctx.value, hctx.perm, exp, val.clone()));
                val
            }
        };
        self.out.insert(hoist_key(ext), val);
        Ok(())
    }
}

/// The error for a heap read that mentions a binder, naming what it reads.
fn binder_dependent_read<Ext>(
    b: &TranslationContext<'_>,
    node: &ExtNode<'_, Ext>,
) -> TranslationError {
    let what = match node {
        ExtNode::Heap(typed::HeapNode::Field(_, f)) => {
            format!("field `{}`", b.interner.resolve(&f.0))
        }
        ExtNode::Heap(typed::HeapNode::FunctionCall(call)) => {
            format!("function `{}`", b.interner.resolve(&call.name.0))
        }
        ExtNode::Heap(typed::HeapNode::Unfolding(pwp, _)) => {
            format!(
                "`unfolding` of `{}`",
                b.interner.resolve(&pwp.pred_call.name.0)
            )
        }
        ExtNode::Perm(_) => "`perm`".to_string(),
        ExtNode::Old(_) | ExtNode::Result | ExtNode::PureCall(_) | ExtNode::Forall(_) => {
            unreachable!("not a heap read")
        }
    };
    TranslationError::BinderDependentHeapRead(what)
}

/// Whether `exp` mentions any of `names`. Shadowing inside `exp` is ignored, so
/// this may say yes for a name rebound before its use; the caller then treats
/// the expression as binder-dependent, which only ever rejects more.
fn mentions<Ext: PureExt>(exp: &typed::TypedPureExp<Ext>, names: &[Spur]) -> bool {
    use typed::PureExpKind as P;
    let any = |es: &[typed::TypedPureExp<Ext>]| es.iter().any(|e| mentions(e, names));
    match &*exp.exp {
        P::Ident(id) => names.contains(&id.0),
        P::Const(_) => false,
        P::Unary(_, e) | P::AdtDestructor(e, _) | P::AdtDiscriminator(e, _) => mentions(e, names),
        P::Binary(_, l, r)
        | P::LetIn {
            value: l, exp: r, ..
        } => mentions(l, names) || mentions(r, names),
        P::Ternary { if_, then, else_ } => {
            mentions(if_, names) || mentions(then, names) || mentions(else_, names)
        }
        P::DomainFunctionCall(call) | P::AdtConstructor(call) => any(&call.args),
        P::Ext(ext) => match ext.node() {
            ExtNode::Heap(typed::HeapNode::Field(e, _)) | ExtNode::Old(e) => mentions(e, names),
            ExtNode::Heap(typed::HeapNode::FunctionCall(call)) | ExtNode::PureCall(call) => {
                any(&call.args)
            }
            ExtNode::Heap(typed::HeapNode::Unfolding(pwp, body)) => {
                any(&pwp.pred_call.args) || mentions(&pwp.perm, names) || mentions(body, names)
            }
            ExtNode::Perm(res) => match &*res.0 {
                typed::ResourceExpKind::Field(e, _) => mentions(e, names),
                typed::ResourceExpKind::PredicateCall(call) => any(&call.args),
            },
            ExtNode::Result => false,
            ExtNode::Forall(q) => q.triggers.iter().any(|g| any(g)) || mentions(&q.body, names),
        },
    }
}

/// Lower one trigger term into its VMIR pattern tree. A variable resolves through
/// `env` to the temp it names — a binder or an enclosing value, told apart later
/// by `binder_base` alone; a literal to `Lit`; a closed heap read to the value it
/// was evaluated to (`hoisted`), an enclosing value like a variable; anything
/// else is an application, lowered to the same head the body's `PureInst` would
/// produce, with its arguments lowered recursively (a trigger may nest
/// arbitrarily). Shapes outside this grammar were rejected at typecheck
/// (`check_triggers`).
fn lower_trig_term<Ext: PureExt>(
    b: &TranslationContext<'_>,
    term: &typed::TypedPureExp<Ext>,
    env: &HashMap<Spur, Val>,
    hoisted: &Hoisted,
) -> Result<vmir::TrigTerm, TranslationError> {
    use typed::PureExpKind as P;
    let lower_args = |args: &[typed::TypedPureExp<Ext>]| {
        args.iter()
            .map(|a| lower_trig_term(b, a, env, hoisted))
            .collect::<Result<Vec<_>, _>>()
    };
    let func = |call: &typed::Call<Ext>| -> Result<vmir::TrigTerm, TranslationError> {
        let function = *b.name_map.get(&call.name.0).ok_or_else(|| {
            TranslationError::UnknownIdent(b.interner.resolve(&call.name.0).to_string())
        })?;
        Ok(vmir::TrigTerm::App {
            head: vmir::TrigHead::Func(function),
            type_args: Vec::new(),
            args: lower_args(&call.args)?.into(),
        })
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
        P::DomainFunctionCall(call) => func(call),
        P::Ext(ext) => match (hoisted.get(&hoist_key(ext)), ext.node()) {
            (Some(Val::Temp(k)), _) => Ok(vmir::TrigTerm::Var(*k)),
            (Some(Val::Literal(lit)), _) => Ok(vmir::TrigTerm::Lit(lit.clone())),
            // Not hoisted, so heap-free: the head is the plain function.
            (
                None,
                ExtNode::Heap(typed::HeapNode::FunctionCall(call)) | ExtNode::PureCall(call),
            ) => func(call),
            (None, _) => unreachable!("typecheck rejects non-matchable trigger subterms"),
        },
        P::AdtConstructor(call) => {
            let &(adt_spur, variant) = b.adt.ctor_variant.get(&call.name.0).ok_or_else(|| {
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
                args: Box::new([lower_trig_term(b, base, env, hoisted)?]),
            })
        }
        // `{ e.isCons }` triggers on the tester the discriminator lowers to, the
        // same `AdtIs` head a body builds, so it fires only on a test of that
        // variant.
        P::AdtDiscriminator(base, variant) => {
            let &(adt_spur, variant) = b.adt.ctor_variant.get(&variant.0).ok_or_else(|| {
                TranslationError::UnknownIdent(b.interner.resolve(&variant.0).to_string())
            })?;
            Ok(vmir::TrigTerm::App {
                head: vmir::TrigHead::AdtIs {
                    adt: b.name_map[&adt_spur],
                    variant,
                },
                type_args: adt_type_args(b, &base.ty),
                args: Box::new([lower_trig_term(b, base, env, hoisted)?]),
            })
        }
        P::Unary(..) | P::Binary(..) | P::Ternary { .. } | P::LetIn { .. } => {
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
            typed::MethodEnsuresExt::Old(inner) => {
                lower(b, env, sink, ext.old_ctx(b, hctx)?, inner)
            }
            typed::MethodEnsuresExt::Forall(q) => lower_forall(b, env, sink, hctx, q),
        }
    }
    fn node(&self) -> ExtNode<'_, Self> {
        match self {
            typed::MethodEnsuresExt::Heap(node) => ExtNode::Heap(node),
            typed::MethodEnsuresExt::Old(inner) => ExtNode::Old(inner),
            typed::MethodEnsuresExt::Forall(q) => ExtNode::Forall(q),
        }
    }
    // old(e): re-read `e` against the method pre-state. For a two-state
    // ensures that heap is the one its entry bound `inhale` reconstructs
    // from the trailing snapshot parameter, supplied as the `old`
    // baseline by `lower_spatial_ensures`. Ensures-`old` is always
    // unlabeled (`old[L]` is a type error). A self-framed ensures has no
    // pre-state (`hctx.old == None`): `old` there needs a `requires` to
    // frame it.
    fn old_ctx<'h>(
        &self,
        _b: &TranslationContext<'_>,
        hctx: HeapCtx<'h>,
    ) -> Result<HeapCtx<'h>, TranslationError> {
        let old = hctx.old.ok_or(TranslationError::Unsupported(
            "`old` in method ensures needs a precondition framing it",
        ))?;
        Ok(HeapCtx {
            value: old.baseline,
            perm: old.baseline,
            ..hctx
        })
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
            typed::MethodBodyExt::Old(_, inner) => {
                lower(b, env, sink, ext.old_ctx(b, hctx)?, inner)
            }
            // perm(loc): query the permission held at `loc` in the perm heap. A
            // quantifier body has none (`HeapCtx::hoisted`).
            typed::MethodBodyExt::Perm(res) if hctx.hoisted.is_some() => {
                Err(binder_dependent_read(b, &ExtNode::Perm(res)))
            }
            typed::MethodBodyExt::Perm(res) => {
                let addr =
                    crate::translate::resource::lower_resource_addr(b, env, sink, hctx, res)?;
                Ok(sink.emit_pure(ty, PureInst::Perm(hctx.perm, addr)))
            }
            typed::MethodBodyExt::Forall(q) => lower_forall(b, env, sink, hctx, q),
        }
    }
    fn node(&self) -> ExtNode<'_, Self> {
        match self {
            typed::MethodBodyExt::Heap(node) => ExtNode::Heap(node),
            typed::MethodBodyExt::Old(_, inner) => ExtNode::Old(inner),
            typed::MethodBodyExt::Perm(res) => ExtNode::Perm(res),
            typed::MethodBodyExt::Forall(q) => ExtNode::Forall(q),
        }
    }
    // old(e) / old[L](e): re-read `e` against an earlier heap. Unlabeled
    // → the post-requires-inhale baseline; labeled → the heap captured
    // at `label L`. The `old` context is carried along so nested `old`s
    // still resolve.
    fn old_ctx<'h>(
        &self,
        b: &TranslationContext<'_>,
        hctx: HeapCtx<'h>,
    ) -> Result<HeapCtx<'h>, TranslationError> {
        let typed::MethodBodyExt::Old(label, _) = self else {
            unreachable!("old_ctx on a node that is not `old`")
        };
        let old = hctx
            .old
            .ok_or(TranslationError::Unsupported("`old` outside method body"))?;
        let heap = match label {
            None => old.baseline,
            Some(l) => *old
                .labeled
                .get(l)
                .ok_or_else(|| TranslationError::UnknownIdent(b.interner.resolve(l).to_string()))?,
        };
        Ok(HeapCtx {
            value: heap,
            perm: heap,
            ..hctx
        })
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
            typed::FuncEnsuresExt::Old(inner) => lower(b, env, sink, ext.old_ctx(b, hctx)?, inner),
            typed::FuncEnsuresExt::Forall(q) => lower_forall(b, env, sink, hctx, q),
        }
    }
    fn node(&self) -> ExtNode<'_, Self> {
        match self {
            typed::FuncEnsuresExt::Heap(node) => ExtNode::Heap(node),
            typed::FuncEnsuresExt::Result => ExtNode::Result,
            typed::FuncEnsuresExt::Old(inner) => ExtNode::Old(inner),
            typed::FuncEnsuresExt::Forall(q) => ExtNode::Forall(q),
        }
    }
    // A function is single-state: `old(e)` reads the same (only) heap the
    // postcondition is framed by, so it re-reads `e` against the current
    // context.
    fn old_ctx<'h>(
        &self,
        _b: &TranslationContext<'_>,
        hctx: HeapCtx<'h>,
    ) -> Result<HeapCtx<'h>, TranslationError> {
        Ok(hctx)
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
        hoisted: None,
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
        hoisted: None,
    };
    let res = lower(b, env, &mut sink, hctx, exp)?;
    Ok(vmir::FunctionBody {
        insts: sink.insts,
        res,
    })
}
