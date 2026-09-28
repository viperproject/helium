//! Lower `typed::SpatialExp` into a `vmir::ResourceBody` (or, for source-level
//! `assert`/`assume`, a single boolean over the held heap).

use crate::dhash::HashMap;

use lasso::Spur;

use crate::translate::pure_exp::{self, HeapCtx, OldHeaps, PureExt};
use crate::translate::resource::lower_resource_addr;
use crate::translate::sink::{PcKind, Sink};
use crate::translate::{TranslationContext, TranslationError};
use crate::viper::typed;
use crate::vmir::{self, FALSE, HeapInst, HeapVal, PermVal, Polarity, PureInst, TRUE, Type, Val};

/// Direction and heap semantics of a spatial lowering.
///
/// - `Inhale`: `acc`s are **added** to the accumulation heap; both value and
///   `perm` reads track that growing heap (also the mode for resource bodies).
/// - `Exhale { value_heap }`: `acc`s are **subtracted** (left-to-right), so a
///   later `perm` observes the reduced heap; `perm` reads thus track the
///   shrinking accumulation heap, while value reads use the fixed pre-exhale
///   `value_heap`.
#[derive(Clone, Copy)]
pub(crate) enum SpatialMode {
    Inhale,
    Exhale { value_heap: HeapVal },
}

impl SpatialMode {
    /// Heaps to read sub-expressions from, given the current accumulation heap.
    /// `perm` always tracks `acc_heap`; values track it on inhale but use the
    /// fixed pre-exhale heap on exhale. `old` (method bodies only) is carried
    /// through so `old(...)` sub-expressions can reach earlier heaps.
    fn heap_ctx<'a>(self, acc_heap: HeapVal, old: Option<&'a OldHeaps<'a>>) -> HeapCtx<'a> {
        match self {
            SpatialMode::Inhale => HeapCtx {
                value: acc_heap,
                perm: acc_heap,
                old,
                result: None,
                in_quantifier: false,
            },
            SpatialMode::Exhale { value_heap } => HeapCtx {
                value: value_heap,
                perm: acc_heap,
                old,
                result: None,
                in_quantifier: false,
            },
        }
    }
}

/// Lower a resource body. `initial_heap` is the body's starting heap
/// reference; pass `HeapVal::Empty` when the owning `Resource.requires`
/// is `None`, and `HeapVal::Temp(0)` (with `heap_base = 1`) once the
/// resource has its own precondition resource. `heap_base` is the first
/// heap counter the body's emitted heap insts will use. `read_only` weakens
/// every `acc` permission to a wildcard — set for a **function** precondition
/// resource (a function only needs *some* positive share), not for a predicate
/// body or a method contract.
pub(crate) fn lower_spatial_never(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    exp: &typed::SpatialExp<typed::HeapExt>,
    val_base: usize,
    initial_heap: HeapVal,
    heap_base: usize,
    read_only: bool,
) -> Result<vmir::ResourceBody, TranslationError> {
    let mut sink = Sink::new(val_base, heap_base);
    sink.read_only = read_only;
    sink.in_resource_body = true;
    let (h, bv) = lower_spatial(
        b,
        env,
        &mut sink,
        initial_heap,
        SpatialMode::Inhale,
        None,
        exp,
    )?;
    Ok(vmir::ResourceBody {
        insts: sink.insts,
        res: (h, bv.unwrap_or(TRUE)),
    })
}

/// Whether a spatial assertion mentions any `acc` (heap permission). An
/// Acc-free assertion is purely logical, hence heap-free.
pub(crate) fn spatial_contains_acc<Ext>(exp: &typed::SpatialExp<Ext>) -> bool {
    use typed::SpatialExpKind as S;
    match &*exp.0 {
        S::Acc(..) => true,
        S::Pure(_) => false,
        S::Conj(l, r) => spatial_contains_acc(l) || spatial_contains_acc(r),
        S::Implies(_, r) => spatial_contains_acc(r),
        S::Ternary { then, else_, .. } => spatial_contains_acc(then) || spatial_contains_acc(else_),
    }
}

/// Lower an **Acc-free** (purely logical) precondition into a boolean
/// [`vmir::FunctionBody`] — the body of a heap-free function's `f#requires`
/// contract function. The caller must have rejected any `acc` first (see
/// [`spatial_contains_acc`]); with none present `lower_assertion_bool` never
/// touches the heap, so the passed `HeapVal::Empty` is inert.
pub(crate) fn lower_pure_precond_body(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    exp: &typed::SpatialExp<typed::HeapExt>,
    val_base: usize,
) -> Result<vmir::FunctionBody, TranslationError> {
    let mut sink = Sink::new(val_base, 0);
    let bv = lower_assertion_bool(b, env, &mut sink, HeapVal::Empty, None, exp)?;
    Ok(vmir::FunctionBody {
        insts: sink.insts,
        res: bv.unwrap_or(TRUE),
    })
}

pub(crate) fn lower_spatial_ensures(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    exp: &typed::SpatialExp<typed::MethodEnsuresExt>,
    val_base: usize,
    initial_heap: HeapVal,
    snap_entry: Option<pure_exp::SnapEntry>,
) -> Result<vmir::ResourceBody, TranslationError> {
    let mut sink = Sink::new(val_base, 0);
    sink.in_resource_body = true;
    // A two-state ensures opens with a bound `inhale` of `req(args)`:
    // reconstruct the method pre-state from the trailing snapshot parameter.
    // That heap is the (unlabeled) `old` baseline `old(e)` reads. A self-framed
    // ensures has no pre-state, so `old` is rejected at lowering (see
    // `MethodEnsuresExt`).
    let pre_state = snap_entry.map(
        |pure_exp::SnapEntry {
             resource,
             args,
             snap,
         }| {
            sink.emit_heap(HeapInst::Inhale {
                base: HeapVal::Empty,
                bind: vmir::Bind::Bound(snap),
                call: vmir::ResourceCall { resource, args },
                // `1/1`, NOT `wildcard`: the scale multiplies each footprint
                // slot's own permission, so `1/1` reproduces the amounts the
                // dedicated instruction used (`1 * p` folds away). A wildcard
                // scale would silently rewrite every slot to `w * p`.
                perm: vmir::PermVal::write(),
            })
        },
    );
    let labeled: HashMap<Spur, HeapVal> = HashMap::default();
    let old = pre_state.map(|baseline| OldHeaps {
        baseline,
        labeled: &labeled,
    });
    let (h, bv) = lower_spatial(
        b,
        env,
        &mut sink,
        initial_heap,
        SpatialMode::Inhale,
        old.as_ref(),
        exp,
    )?;
    Ok(vmir::ResourceBody {
        insts: sink.insts,
        res: (h, bv.unwrap_or(TRUE)),
    })
}

/// Lower a `SpatialExp` into a heap delta and an optional boolean.
///
/// `None` means the spatial expression carries no logical content beyond its
/// heap chunks — i.e. the boolean is trivially `true`, so an all-permission
/// expression like `acc(..) && acc(..)` lowers to the heap delta alone.
///
/// A branch is encoded into permission fractions rather than a heap multiplexer:
/// the enclosing `with_cond` path condition gates every `acc`'s permission (see
/// `Sink::gate_perm`) and chunks are added to a single monotonic heap timeline
/// (no `HeapInst::Ternary`). The verifier's agreement axiom keeps values from
/// mutually-exclusive branches isolated, since their fractions are never both
/// positive.
pub(crate) fn lower_spatial<Ext: PureExt>(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    sink: &mut Sink,
    acc_heap: HeapVal,
    mode: SpatialMode,
    old: Option<&OldHeaps>,
    exp: &typed::SpatialExp<Ext>,
) -> Result<(HeapVal, Option<Val>), TranslationError> {
    use typed::SpatialExpKind as S;
    // Heaps to read heap-dependent sub-expressions from at this point.
    let hctx = mode.heap_ctx(acc_heap, old);
    match &*exp.0 {
        S::Acc(res, perm) => {
            let (loc, perm) = lower_acc(b, env, sink, hctx, res, perm)?;
            // Inhale adds the chunk; exhale subtracts it (so a later `perm`
            // observes the reduced heap).
            let inst = match mode {
                SpatialMode::Inhale => HeapInst::Add {
                    base: acc_heap,
                    loc,
                    perm,
                    // A footprint `acc` in a resource body declares that
                    // resource's next own slot; in a method body it produces an
                    // unconstrained value.
                    bind: sink.acc_bind(),
                },
                SpatialMode::Exhale { .. } => HeapInst::Sub {
                    base: acc_heap,
                    loc,
                    perm,
                    // A plain `exhale acc(..)` discards what it removed.
                    yields_value: false,
                },
            };
            // Always carry the path condition: an `acc` has a permission ≥ 0
            // side condition that must be discharged under the conditions
            // reaching it (and an inhale produces a pc-gated assumption). E.g.
            // in `p >= none && acc(x.f, p)` the threaded `p >= none` is what
            // makes `p >= 0` provable.
            let h_out = sink.emit_heap_guarded(inst);
            Ok((h_out, None))
        }
        S::Conj(l, r) => {
            let (h_mid, b_l) = lower_spatial(b, env, sink, acc_heap, mode, old, l)?;
            // `A && B`: by short-circuit semantics B is only reached when A
            // holds, so A's boolean is part of B's path condition. Threading it
            // lets B's side conditions (e.g. an `acc` permission ≥ 0) assume the
            // facts of A — e.g. `p >= none && acc(x.f, p)` discharges `p >= 0`.
            let (h_out, b_r) = match b_l.clone() {
                Some(v) => sink.with_cond(v, Polarity::Positive, PcKind::Fact, |sink| {
                    lower_spatial(b, env, sink, h_mid, mode, old, r)
                })?,
                None => lower_spatial(b, env, sink, h_mid, mode, old, r)?,
            };

            let b_sum = match (b_l, b_r) {
                (None, None) => None,
                (Some(v), None) | (None, Some(v)) => Some(v),
                (Some(vl), Some(vr)) => {
                    Some(sink.emit_pure(Type::Bool, PureInst::Ternary(vl, vr, FALSE)))
                }
            };

            Ok((h_out, b_sum))
        }
        S::Implies(cond, body) => {
            let c = pure_exp::lower(b, env, sink, hctx, cond)?;
            // `with_cond` pushes `c` onto the path condition; that stack is both
            // the body's side-condition guard and the source of the permission
            // gating (see `Sink::gate_perm`), so no separate guard value is built.
            let (h_out, b_b) =
                sink.with_cond(c.clone(), Polarity::Positive, PcKind::Branch, |sink| {
                    lower_spatial(b, env, sink, acc_heap, mode, old, body)
                })?;
            // c ==> b_b  =  c ? b_b : true. When body has no boolean, the
            // whole implication is trivially true.
            let bv = b_b.map(|v| sink.emit_pure(Type::Bool, PureInst::Ternary(c, v, TRUE)));
            Ok((h_out, bv))
        }
        S::Ternary { if_, then, else_ } => {
            let c = pure_exp::lower(b, env, sink, hctx, if_)?;
            // Both arms add their gated chunks to the same timeline, in order:
            // then onto `acc_heap`, else onto the then-result. No heap ternary;
            // the else arm's negative polarity is applied by flipping ternary
            // branches in `Sink::gate_perm`, not by materializing `!c`.
            let (h_mid, b_t) =
                sink.with_cond(c.clone(), Polarity::Positive, PcKind::Branch, |sink| {
                    lower_spatial(b, env, sink, acc_heap, mode, old, then)
                })?;
            let (h_out, b_e) =
                sink.with_cond(c.clone(), Polarity::Negative, PcKind::Branch, |sink| {
                    lower_spatial(b, env, sink, h_mid, mode, old, else_)
                })?;
            let bv = match (b_t, b_e) {
                (None, None) => None,
                (Some(vt), None) => {
                    Some(sink.emit_pure(Type::Bool, PureInst::Ternary(c, vt, TRUE)))
                }
                (None, Some(ve)) => {
                    Some(sink.emit_pure(Type::Bool, PureInst::Ternary(c, TRUE, ve)))
                }
                (Some(vt), Some(ve)) => {
                    Some(sink.emit_pure(Type::Bool, PureInst::Ternary(c, vt, ve)))
                }
            };
            Ok((h_out, bv))
        }
        S::Pure(p) => {
            let v = pure_exp::lower(b, env, sink, hctx, p)?;
            Ok((acc_heap, Some(v)))
        }
    }
}

/// Lower `acc(res, perm)` to its location and (pc-gated) permission. The caller
/// emits the `HeapInst::Add`/`HeapInst::Sub` that adds/subtracts the chunk. A source-level
/// `wildcard` maps directly to [`PermVal::Wildcard`]; otherwise the amount is
/// lowered and passed through the read-only policy (see [`Sink::perm_amount`]).
fn lower_acc<Ext: PureExt>(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    sink: &mut Sink,
    hctx: HeapCtx<'_>,
    res: &typed::ResourceExp<Ext>,
    perm: &typed::TypedPureExp<Ext>,
) -> Result<(Val, PermVal), TranslationError> {
    let perm = if is_wildcard(perm) {
        PermVal::Wildcard
    } else {
        let perm_val = pure_exp::lower(b, env, sink, hctx, perm)?;
        sink.perm_amount(perm_val)
    };
    let perm = sink.gate_perm(perm);
    let addr = lower_resource_addr(b, env, sink, hctx, res)?;
    Ok((addr, perm))
}

/// Whether a permission expression is the `wildcard` literal.
pub(crate) fn is_wildcard<Ext>(perm: &typed::TypedPureExp<Ext>) -> bool {
    matches!(
        &*perm.exp,
        typed::PureExpKind::Const(typed::Literal::Wildcard)
    )
}

/// Lower an assertion used by source-level `assert`/`assume` into a single
/// boolean over `heap` (returns `None` when trivially true). Permission is
/// **not** moved: each `acc(loc, p)` becomes the boolean `perm(loc) >= p`.
pub(crate) fn lower_assertion_bool<Ext: PureExt>(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    sink: &mut Sink,
    heap: HeapVal,
    old: Option<&OldHeaps>,
    exp: &typed::SpatialExp<Ext>,
) -> Result<Option<Val>, TranslationError> {
    use typed::SpatialExpKind as S;
    let hctx = HeapCtx {
        value: heap,
        perm: heap,
        old,
        result: None,
        in_quantifier: false,
    };
    match &*exp.0 {
        // acc(loc, p)  ==>  perm(loc) >= p  ==  not(perm(loc) < p)
        S::Acc(res, perm) => {
            let p = pure_exp::lower(b, env, sink, hctx, perm)?;
            let addr = lower_resource_addr(b, env, sink, hctx, res)?;
            let held = sink.emit_pure(Type::Real, PureInst::Perm(heap, addr));
            let lt = sink.emit_pure(Type::Bool, PureInst::Binary(vmir::BinOp::LtR, held, p));
            Ok(Some(
                sink.emit_pure(Type::Bool, PureInst::Ternary(lt, FALSE, TRUE)),
            ))
        }
        S::Conj(l, r) => {
            let bl = lower_assertion_bool(b, env, sink, heap, old, l)?;
            let br = lower_assertion_bool(b, env, sink, heap, old, r)?;
            Ok(match (bl, br) {
                (None, None) => None,
                (Some(v), None) | (None, Some(v)) => Some(v),
                // l && r  =  l ? r : false
                (Some(vl), Some(vr)) => {
                    Some(sink.emit_pure(Type::Bool, PureInst::Ternary(vl, vr, FALSE)))
                }
            })
        }
        S::Implies(cond, body) => {
            let c = pure_exp::lower(b, env, sink, hctx, cond)?;
            let bb = sink.with_cond(c.clone(), Polarity::Positive, PcKind::Branch, |sink| {
                lower_assertion_bool(b, env, sink, heap, old, body)
            })?;
            // c ==> bb  =  c ? bb : true
            Ok(bb.map(|v| sink.emit_pure(Type::Bool, PureInst::Ternary(c, v, TRUE))))
        }
        S::Ternary { if_, then, else_ } => {
            let c = pure_exp::lower(b, env, sink, hctx, if_)?;
            let bt = sink.with_cond(c.clone(), Polarity::Positive, PcKind::Branch, |sink| {
                lower_assertion_bool(b, env, sink, heap, old, then)
            })?;
            let be = sink.with_cond(c.clone(), Polarity::Negative, PcKind::Branch, |sink| {
                lower_assertion_bool(b, env, sink, heap, old, else_)
            })?;
            Ok(match (bt, be) {
                (None, None) => None,
                (Some(vt), None) => {
                    Some(sink.emit_pure(Type::Bool, PureInst::Ternary(c, vt, TRUE)))
                }
                (None, Some(ve)) => {
                    Some(sink.emit_pure(Type::Bool, PureInst::Ternary(c, TRUE, ve)))
                }
                (Some(vt), Some(ve)) => {
                    Some(sink.emit_pure(Type::Bool, PureInst::Ternary(c, vt, ve)))
                }
            })
        }
        S::Pure(p) => Ok(Some(pure_exp::lower(b, env, sink, hctx, p)?)),
    }
}
