//! Lower a `typed::Method` body into a **block-structured** `vmir::Method`.
//! The basic-block CFG (`viper::cfg`) is preserved (not linearized): blocks are
//! emitted in topological order, each with a join phase (phi `ite` nodes
//! reconciling predecessors) and a body phase (its lowered statements), threaded
//! by a single linear heap for now. Joins are binary; an n-ary (multi-goto)
//! merge is normalised into a chain of synthetic binary-join blocks.

use crate::dhash::{HashMap, HashSet};
use std::marker::PhantomData;

use itertools::Itertools;
use lasso::Spur;
use typed_index_collections::TiVec;

use crate::translate::reach::{Reach, ReachRef, merge_two_envs};
use crate::translate::sink::{PcKind, Sink};
use crate::translate::spatial::{self, SpatialMode};
use crate::translate::{DeclSlot, Declarator, Definer};
use crate::translate::{
    Declared, Metaed, MethodContracts, TranslationContext, TranslationError, lower_type, pure_exp,
    resource,
};
use crate::viper::cfg::{self, BlockId, EdgeSide, Terminator};
use crate::viper::typed;
use crate::vmir::{
    self, HeapInst, HeapVal, Inst, PathConds, Polarity, PureInst, ResourceCall, Type, Val,
};

/// One translator per Silver `method`: owns its `#requires` / `#ensures`
/// contract resources **and** its `vmir::Method` body slot. Merging them is
/// sound because define order is free — a method body inhales/exhales any
/// callee's contract by *id* (recorded in `ctx.contracts` at declare), never
/// its filled slot.
pub(crate) struct MethodTranslator<'a, P = Declared> {
    src: &'a typed::Method,
    silver_name: Spur,
    requires_slot: Option<DeclSlot<vmir::Resource>>,
    ensures_slot: Option<DeclSlot<vmir::Resource>>,
    body_slot: Option<DeclSlot<vmir::Method>>,
    /// `#requires` resource id (the pre-state a two-state `#ensures` reads).
    requires: Option<vmir::MemberId>,
    _p: PhantomData<P>,
}

impl<'a> MethodTranslator<'a, Declared> {
    pub(crate) fn declare(
        m: &'a typed::Method,
        ctx: &mut TranslationContext<'_>,
        d: &mut impl Declarator,
    ) -> Self {
        let name = ctx.interner.resolve(&m.name.0).to_owned();
        let (requires, requires_slot) = if m.requires.is_some() {
            let (id, slot) = d.alloc_slot::<vmir::Resource>(&format!("{name}#requires"));
            (Some(id), Some(slot))
        } else {
            (None, None)
        };
        let (ensures, ensures_slot) = if m.ensures.is_some() {
            let (id, slot) = d.alloc_slot::<vmir::Resource>(&format!("{name}#ensures"));
            (Some(id), Some(slot))
        } else {
            (None, None)
        };
        let body_slot = if m.body.is_some() {
            let (id, slot) = d.alloc_slot::<vmir::Method>(&name);
            ctx.name_map.insert(m.name.0, id);
            Some(slot)
        } else {
            None
        };
        ctx.contracts.insert(
            m.name.0,
            MethodContracts {
                requires,
                ensures,
                heap_dep: false,
            },
        );
        MethodTranslator {
            src: m,
            silver_name: m.name.0,
            requires_slot,
            ensures_slot,
            body_slot,
            requires,
            _p: PhantomData,
        }
    }

    /// No `name_map`-dependent metadata to publish (contracts were published at
    /// declare).
    pub(crate) fn meta(self, _ctx: &mut TranslationContext<'_>) -> MethodTranslator<'a, Metaed> {
        MethodTranslator {
            src: self.src,
            silver_name: self.silver_name,
            requires_slot: self.requires_slot,
            ensures_slot: self.ensures_slot,
            body_slot: self.body_slot,
            requires: self.requires,
            _p: PhantomData,
        }
    }
}

impl MethodTranslator<'_, Metaed> {
    pub(crate) fn define(
        self,
        ctx: &TranslationContext<'_>,
        definer: &mut impl Definer,
    ) -> Result<(), TranslationError> {
        let MethodTranslator {
            src: m,
            silver_name,
            requires_slot,
            ensures_slot,
            body_slot,
            requires: meta_requires,
            _p,
        } = self;

        // #requires: a self-framed Resource.
        if let Some(requires) = &m.requires {
            let slot = requires_slot.expect("declared when m.requires is Some");
            let params: Vec<vmir::Type> = m.params.iter().map(|p| ctx.lower_type(&p.ty)).collect();
            let mut env: HashMap<Spur, vmir::Val> = HashMap::default();
            for (i, p) in m.params.iter().enumerate() {
                env.insert(p.name.0, vmir::Val::Temp(i));
            }
            // Self-framed: accumulate from `Empty`, heaps start at `HeapVal::Temp(0)`.
            let body = spatial::lower_spatial_never(
                ctx,
                &env,
                requires,
                params.len(),
                vmir::HeapVal::Empty,
                0,
                // Method precondition: real permissions.
                false,
            )?;
            let name =
                definer.intern_name(&format!("{}#requires", ctx.interner.resolve(&silver_name)));
            definer.define_resource(
                slot,
                vmir::Resource {
                    name,
                    params,
                    precond: vmir::Precond::SelfFramed,
                    body,
                },
            );
        }

        // #ensures: a Resource, two-state (`Ctx`) when the method has a requires.
        if let Some(ensures) = &m.ensures {
            let slot = ensures_slot.expect("declared when m.ensures is Some");
            let mut params: Vec<vmir::Type> =
                m.params.iter().map(|p| ctx.lower_type(&p.ty)).collect();
            params.extend(m.rets.iter().map(|r| ctx.lower_type(&r.ty)));
            let mut env: HashMap<Spur, vmir::Val> = HashMap::default();
            for (i, p) in m.params.iter().enumerate() {
                env.insert(p.name.0, vmir::Val::Temp(i));
            }
            for (i, r) in m.rets.iter().enumerate() {
                env.insert(r.name.0, vmir::Val::Temp(m.params.len() + i));
            }
            // The ensures precondition is `m#requires` (when present). Its delta
            // accumulates from `Empty` so a resource in both contracts isn't
            // double-counted.
            let precond = meta_requires
                .map(|req_id| {
                    let req_args: Vec<vmir::Val> =
                        (0..m.params.len()).map(vmir::Val::Temp).collect();
                    vmir::Precond::Ctx(req_id, req_args)
                })
                .unwrap_or(vmir::Precond::SelfFramed);
            // A two-state (`Ctx`) ensures receives the pre-state as a trailing
            // snapshot parameter `s : Snap(req)`; its body opens with a
            // a bound `inhale` reconstructing the pre-state heap `old(...)` reads.
            let snap_entry = match &precond {
                vmir::Precond::Ctx(req_id, req_args) => {
                    let snap = vmir::Val::Temp(params.len());
                    params.push(vmir::Type::Snap(*req_id));
                    Some(pure_exp::SnapEntry {
                        resource: *req_id,
                        args: req_args.clone(),
                        snap,
                    })
                }
                vmir::Precond::SelfFramed => None,
            };
            let body = spatial::lower_spatial_ensures(
                ctx,
                &env,
                ensures,
                params.len(),
                vmir::HeapVal::Empty,
                snap_entry,
            )?;
            let name =
                definer.intern_name(&format!("{}#ensures", ctx.interner.resolve(&silver_name)));
            definer.define_resource(
                slot,
                vmir::Resource {
                    name,
                    params,
                    precond,
                    body,
                },
            );
        }

        // The method's own body (when present).
        if let Some(slot) = body_slot {
            let body = m.body.as_ref().expect("slot implies a body");
            let name = definer.intern_name(ctx.interner.resolve(&silver_name));
            let method = lower_method(ctx, m, name, body)?;
            definer.define_method(slot, method);
        }

        Ok(())
    }
}

pub(crate) fn lower_method(
    b: &TranslationContext<'_>,
    m: &typed::Method,
    name: Spur,
    body: &typed::StmtBlock,
) -> Result<vmir::Method, TranslationError> {
    // The basic-block CFG is preserved as a `vmir::Block` graph: blocks are
    // emitted in topological order, each lowered under its reaching path
    // condition, with phi (`ite`) nodes reconciling variables at joins. The heap
    // is *not* phi'd — it threads linearly through the walk (a structural
    // per-predecessor merge is a later stage), off-path contributions gated to 0
    // permission (see `Sink::gate_perm`).
    let cfg = cfg::build_cfg(body).map_err(|_| {
        TranslationError::Unsupported("method control flow (irreducible or undefined label)")
    })?;

    let mut sink = Sink::new(0, 0);
    // Nothing replays a method body, so its calls carry no `export` mark.
    sink.in_method_body = true;

    // Initial environment: fresh values for params and rets. The method has no
    // signature on the VMIR side — params/rets are just initial Fresh insts.
    let mut init_env: HashMap<Spur, Val> = HashMap::default();
    let mut param_vals: Vec<Val> = Vec::with_capacity(m.params.len());
    for p in &m.params {
        let v = sink.emit_pure(b.lower_type(&p.ty), PureInst::Fresh);
        init_env.insert(p.name.0, v.clone());
        param_vals.push(v);
    }
    let mut ret_names: Vec<Spur> = Vec::with_capacity(m.rets.len());
    for r in &m.rets {
        let v = sink.emit_pure(b.lower_type(&r.ty), PureInst::Fresh);
        init_env.insert(r.name.0, v);
        ret_names.push(r.name.0);
    }

    // Types of every method-scoped variable, needed to type phi nodes at joins.
    let mut var_types: HashMap<Spur, Type> = HashMap::default();
    for p in &m.params {
        var_types.insert(p.name.0, b.lower_type(&p.ty));
    }
    for r in &m.rets {
        var_types.insert(r.name.0, b.lower_type(&r.ty));
    }
    collect_var_types(&b.name_map, &body.0, &mut var_types);

    // Inhale this method's own precondition into the linear heap that every
    // block threads. The pre-state handle is **minted here** and bound into the
    // inhale, rather than being read back off it: an inhale takes its values in
    // through `bind` and yields nothing. Binding to `s` is what makes this heap
    // and the pre-state the `#ensures` reconstructs from `s` name the *same*
    // terms (`unwrap(proj_i(s))`), which is how `old(...)` connects.
    //
    // `s` is opaque (`fresh`) because a method's caller decides the pre-state:
    // it is deterministic — two reconstructions from the same `s` agree — but
    // otherwise unconstrained, which per-slot `Fresh` could not express, since
    // there would be no handle to thread to the `#ensures` exhale.
    let mut current_heap: HeapVal = HeapVal::Empty;
    let mut req_snap: Option<Val> = None;
    if let Some(req_id) = b.method_requires(m.name.0) {
        let s = sink.emit_pure(vmir::Type::Snap(req_id), PureInst::Fresh);
        current_heap = emit_resource_inhale(
            &mut sink,
            req_id,
            current_heap,
            param_vals.clone(),
            vmir::Bind::Bound(s.clone()),
        );
        req_snap = Some(s);
    }
    // Baseline for unlabeled `old`: the post-requires-inhale heap.
    let baseline = current_heap;
    let mut labeled: HashMap<Spur, HeapVal> = HashMap::default();

    // Structural heap merge (fork model): each block's `h_in` is derived from
    // its `Preds` (a `HeapInst::Merge` at a `Join`) and arms lower unguarded.
    // `h_out_of` records every pushed block's exit heap so a later join's
    // `Merge` can name its predecessors' heaps.
    let mut h_out_of: HashMap<vmir::BlockId, HeapVal> = HashMap::default();
    // Per loop head (CFG block id): the exhale residual set aside at the cut.
    // Any edge leaving that loop unions it back — see the head/exit handling in
    // the block walk below.
    let mut frame_of: HashMap<BlockId, HeapVal> = HashMap::default();

    let order = cfg.topo_order();
    let cfg_preds = cfg.predecessors();
    let reachable = cfg.reachable();

    // Per-block reaching condition, as a BDD node (see `Reach`) and as the
    // lowering guard (`pc`) built from it, and the branch condition `Val` of a
    // `Branch` block.
    let n = cfg.blocks.len();
    let mut reach = Reach::new();
    let mut reach_bdd: TiVec<BlockId, ReachRef> = (0..n).map(|_| Reach::TRUE).collect();
    let mut reach_pc: TiVec<BlockId, PathConds> = (0..n).map(|_| PathConds::default()).collect();
    // A disjunctive reach split into the conditions of the edges that make it
    // disjunctive (see the assert distribution below); empty for a cube reach.
    let mut reach_split: TiVec<BlockId, Vec<PathConds>> = (0..n).map(|_| Vec::new()).collect();
    let mut cond_val: TiVec<BlockId, Option<Val>> = (0..n).map(|_| None).collect();
    let mut exit_env: HashMap<BlockId, HashMap<Spur, Val>> = HashMap::default();

    // Block-structured output: VMIR blocks in topological order plus the synthetic
    // binary-join blocks that normalise n-ary (multi-goto) merges, and the
    // cfg→vmir id map so a `Preds` can name its predecessors. `blocks` order is a
    // valid topo order (a block's preds — real or synthetic — are pushed first).
    let mut blocks: Vec<vmir::Block> = Vec::new();
    let mut vmir_id: HashMap<BlockId, vmir::BlockId> = HashMap::default();
    // The method prologue (param/ret fresh temps + the `#requires` inhale) is the
    // entry block's join prefix.
    let prologue = sink.take_since(0);
    let mut prologue = Some(prologue);

    for bid in order {
        if !reachable.contains(&bid) {
            continue;
        }

        let is_entry = bid == cfg.entry;

        // --- join phase: reach/edge emissions ---
        let join_mark = sink.insts.len();
        let (pc, edges) = if is_entry {
            (PathConds::default(), Vec::new())
        } else {
            // The reach is the OR of the incoming edges' conditions; in the same
            // pass build the per-pred materialized `Val` that phis reconcile over.
            let mut f = Reach::FALSE;
            let mut edges: Vec<(BlockId, Val)> = Vec::new();
            // Each edge's condition as a pc: the pred's pc plus the taken literal.
            let mut edge_pcs: Vec<PathConds> = Vec::new();
            let mut single_edge = None;
            let mut edge_fs: Vec<(BlockId, ReachRef)> = Vec::new();
            for (p, side) in cfg_preds[bid].iter().filter(|(p, _)| reachable.contains(p)) {
                let lit = match side {
                    EdgeSide::Goto => None,
                    EdgeSide::Then | EdgeSide::Else => {
                        let c = cond_val[*p].clone().expect("branch pred has a condition");
                        let pol = if matches!(side, EdgeSide::Then) {
                            Polarity::Positive
                        } else {
                            Polarity::Negative
                        };
                        Some((c, pol))
                    }
                };
                let mut ef = reach_bdd[*p];
                let mut epc = reach_pc[*p].clone();
                if let Some((c, pol)) = &lit {
                    let l = reach.lit(c, *pol);
                    ef = reach.and(ef, l);
                    epc.conds.push((c.clone(), *pol));
                }
                f = reach.or(f, ef);
                edge_fs.push((*p, ef));
                edge_pcs.push(epc);
                single_edge = Some((*p, lit));
            }
            reach_bdd[bid] = f;
            // An edge's `Val` selects its pred's heap and values at this join,
            // in the right fold `ite(e0, v0, ite(e1, v1, … v_last))` below, so it
            // need only be exact where that level is reached: where one of edges
            // `i..` is taken (an execution enters a join by exactly one edge).
            // Simplified against that, a diamond's edge is its branch literal
            // rather than the whole decision history above it.
            let mut care = vec![Reach::FALSE; edge_fs.len() + 1];
            for i in (0..edge_fs.len()).rev() {
                care[i] = reach.or(edge_fs[i].1, care[i + 1]);
            }
            for (i, (p, ef)) in edge_fs.into_iter().enumerate() {
                let g = reach.simplify(ef, care[i]);
                edges.push((p, reach.val(&mut sink, g)));
            }
            let (pc, disjunctive) = reach.pc(&mut sink, f);
            // A join splits into its incoming edges. A single edge adds its
            // literal to each part of the pred's split: the disjunction came
            // from the pred, and `(∨ᵢ eᵢ) ∧ l = ∨ᵢ (eᵢ ∧ l)`.
            reach_split[bid] = match single_edge {
                _ if !disjunctive => Vec::new(),
                Some((p, lit)) if edge_pcs.len() == 1 => reach_split[p]
                    .iter()
                    .map(|part| {
                        let mut part = part.clone();
                        part.conds.extend(lit.clone());
                        part
                    })
                    .collect(),
                _ => edge_pcs,
            };
            (pc, edges)
        };
        reach_pc[bid] = pc.clone();
        // Reach insts precede every phi; carried into the real block's join
        // (k ≤ 2) or the innermost synthetic block (n-ary), so all `ev` defs come
        // first in the flattened stream.
        let reach_insts = sink.take_since(join_mark);

        // --- entry env + Preds (+ synthetic binary-join blocks for n-ary) ---
        // `then_` is guarded by `cond`, `els` is the unguarded fall-through arm.
        let (env, preds_kind, mut real_join): (HashMap<Spur, Val>, vmir::Preds, Vec<Inst>) =
            if is_entry {
                (
                    init_env.clone(),
                    vmir::Preds::Entry,
                    prologue.take().unwrap(),
                )
            } else {
                match edges.as_slice() {
                    [] => unreachable!("a reachable non-entry block has a predecessor"),
                    // Single predecessor: inherit its env, no phi.
                    [(p, _)] => (
                        exit_env.get(p).cloned().unwrap_or_default(),
                        vmir::Preds::From(vmir_id[p]),
                        reach_insts,
                    ),
                    // Diamond: one binary phi.
                    [(p0, ev0), (p1, _)] => {
                        let phi_mark = sink.insts.len();
                        let env = merge_two_envs(
                            &mut sink,
                            ev0.clone(),
                            &exit_env.get(p0).cloned().unwrap_or_default(),
                            &exit_env.get(p1).cloned().unwrap_or_default(),
                            &var_types,
                        );
                        let mut join = reach_insts;
                        join.extend(sink.take_since(phi_mark));
                        (
                            env,
                            vmir::Preds::Join {
                                cond: ev0.clone(),
                                then_: vmir_id[p0],
                                els: vmir_id[p1],
                            },
                            join,
                        )
                    }
                    // n-ary (multi-goto label): right-fold the arms into a chain of
                    // synthetic binary joins, mirroring `ite(ev0, v0, ite(ev1, v1,
                    // … v_last))`. Innermost synthetic is created first (carries the
                    // reach insts); the outermost fold level is the real block.
                    _ => {
                        let k = edges.len();
                        let (last_p, _) = &edges[k - 1];
                        let mut acc_env = exit_env.get(last_p).cloned().unwrap_or_default();
                        let mut acc_ref = vmir_id[last_p];
                        let mut reach_insts = Some(reach_insts);
                        let mut real: Option<(HashMap<Spur, Val>, vmir::Preds, Vec<Inst>)> = None;
                        for i in (0..k - 1).rev() {
                            let (p, ev) = &edges[i];
                            let phi_mark = sink.insts.len();
                            let new_env = merge_two_envs(
                                &mut sink,
                                ev.clone(),
                                &exit_env.get(p).cloned().unwrap_or_default(),
                                &acc_env,
                                &var_types,
                            );
                            let mut join = reach_insts.take().unwrap_or_default();
                            join.extend(sink.take_since(phi_mark));
                            let then_ = vmir_id[p];
                            let els = acc_ref;
                            let kind = vmir::Preds::Join {
                                cond: ev.clone(),
                                then_,
                                els,
                            };
                            if i == 0 {
                                real = Some((new_env, kind, join));
                            } else {
                                let sid = vmir::BlockId(blocks.len());
                                // A synthetic join has an empty body, so its exit
                                // heap is exactly its `Merge`.
                                let syn_h_out = {
                                    let mark = sink.insts.len();
                                    let m = sink.with_conds(&pc, |sink| {
                                        sink.emit_heap_guarded(HeapInst::Merge {
                                            cond: ev.clone(),
                                            then_h: h_out_of[&then_],
                                            els_h: h_out_of[&els],
                                        })
                                    });
                                    join.extend(sink.take_since(mark));
                                    m
                                };
                                blocks.push(vmir::Block {
                                    cube: pc.clone(),
                                    preds: kind,
                                    join,
                                    body: Vec::new(),
                                    h_out: syn_h_out,
                                });
                                h_out_of.insert(sid, syn_h_out);
                                acc_ref = sid;
                                acc_env = new_env;
                            }
                        }
                        real.expect("n-ary fold produces the outermost (real) join")
                    }
                }
            };

        // Derive this block's entry heap from its `Preds`. A `Join` emits a
        // `HeapInst::Merge` into the join phase, before the body reads it.
        let h_in = match &preds_kind {
            vmir::Preds::Entry => baseline,
            vmir::Preds::From(p) => h_out_of[p],
            vmir::Preds::Join { cond, then_, els } => {
                let mark = sink.insts.len();
                // Emit under the block cube so the Merge inst carries it as its
                // pc (the verifier assumes the join cube before the merge).
                let m = sink.with_conds(&pc, |sink| {
                    sink.emit_heap_guarded(HeapInst::Merge {
                        cond: cond.clone(),
                        then_h: h_out_of[then_],
                        els_h: h_out_of[els],
                    })
                });
                real_join.extend(sink.take_since(mark));
                m
            }
        };

        // --- loop head: cut the back edge ---------------------------------
        // Establish the invariant from the (single) forward predecessor's heap,
        // keep the residual as the frame, havoc everything the body writes, then
        // rebuild the entry heap from the invariant alone. Anything the
        // invariant does not mention stays in the frame, untouched — which is
        // the whole framing story, and it falls out of permission arithmetic
        // rather than any modifies analysis.
        let mut env = env;
        // An edge that leaves a loop restores that loop's frame: the body holds
        // the invariant's footprint, the head set the rest aside, and after the
        // loop we hold their sum. Innermost first, since leaving nested loops
        // must restore inner before outer.
        let h_in = {
            let mut h = h_in;
            let exits: Vec<BlockId> = cfg_preds[bid]
                .iter()
                .filter(|(p, _)| reachable.contains(p))
                .flat_map(|(p, _)| cfg.loops.classify(*p, bid).exits.clone())
                .map(|li| cfg.loops.loops[li].head)
                .collect();
            for head in exits {
                let Some(&frame) = frame_of.get(&head) else {
                    continue;
                };
                let mark = sink.insts.len();
                h = sink.with_conds(&pc, |sink| {
                    sink.emit_heap_guarded(HeapInst::Union { a: h, b: frame })
                });
                real_join.extend(sink.take_since(mark));
            }
            h
        };
        let h_in = if let Some(l) = cfg.loops.at_head(bid) {
            let mark = sink.insts.len();
            let invs = cfg.blocks[bid].invs.clone();
            // The cube rides in the pc — these insts land in the *join* phase,
            // where the verifier has no ambient cube, so the pc is the only proof
            // context for the exhale's sufficiency check and the re-inhale's
            // `perm >= 0`. It must NOT gate the permissions: the cube's effect is
            // already carried by dataflow (this block's `h_out` reaches successors
            // only under the cube; the other arm carries its own heap through the
            // join's `Merge`), so gating would emit `e ? 1/1 : 0/1` for an exhale
            // that always needs the full share.
            let (frame, h_in) = sink.with_conds_kind(&pc, PcKind::Cube, |sink| {
                let old = pure_exp::OldHeaps {
                    baseline,
                    labeled: &labeled,
                };
                // Established at the *pre*-havoc values: this is the entry
                // obligation, about the state control actually arrives in.
                // The residual is the FRAME — everything the invariant does not
                // mention. It is an ordinary `HeapVal`, so any edge leaving this
                // loop simply names it (Silicon needs an `invariantContexts`
                // stack here only because its heap threads implicitly).
                let frame = lower_invariant(b, &env, sink, h_in, &invs, true, &old)?;
                for name in loop_written_vars(&cfg, l) {
                    if let Some(ty) = var_types.get(&name) {
                        let v = sink.emit_pure(ty.clone(), PureInst::Fresh);
                        env.insert(name, v);
                    }
                }
                // The body's heap *is* the invariant: an empty heap plus what the
                // invariant grants, at the havoc'd values.
                let h = lower_invariant(b, &env, sink, HeapVal::Empty, &invs, false, &old)?;
                Ok::<_, TranslationError>((frame, h))
            })?;
            real_join.extend(sink.take_since(mark));
            frame_of.insert(bid, frame);
            h_in
        } else {
            h_in
        };

        // `label L` captures the heap at block entry for later `old[L]`.
        if let Some(l) = cfg.blocks[bid].label {
            labeled.insert(l, h_in);
        }

        // --- body phase: lower the block's statements and terminator under pc ---
        let blk = &cfg.blocks[bid];
        let body_mark = sink.insts.len();
        // Fork model: the cube is this phase's **ambient** — proof context for every
        // inst in it (it does NOT gate permissions; the join merge SELECTs), and
        // elided from each `Inst.pc`, which is a delta over it. `Block.cube` below
        // is the single place it is recorded.
        let (new_heap, cond): (HeapVal, Option<Val>) = sink.with_ambient_cube(&pc, |sink| {
            let mut heap = h_in;
            for stmt in &blk.stmts {
                heap = lower_stmt(b, &mut env, sink, heap, baseline, &mut labeled, stmt)?;
            }
            let cond = match &blk.term {
                Terminator::Branch { cond, .. } => {
                    let old = pure_exp::OldHeaps {
                        baseline,
                        labeled: &labeled,
                    };
                    let cv = pure_exp::lower(
                        b,
                        &env,
                        sink,
                        pure_exp::HeapCtx::same_with_old(heap, &old),
                        cond,
                    )?;
                    Some(cv)
                }
                // At each exit, exhale the postcondition gated by this block's
                // pc against the final values of the return variables.
                Terminator::Return => {
                    if let Some(ens_id) = b.method_ensures(m.name.0) {
                        let mut ens_args = param_vals.clone();
                        for name in &ret_names {
                            ens_args.push(env.get(name).cloned().expect("return var bound"));
                        }
                        // A two-state `#ensures` (this method has its own
                        // `#requires`) reads the pre-state through its trailing
                        // snapshot parameter — the snapshot yielded by the entry
                        // `#requires` inhale.
                        let is_ctx = b.method_requires(m.name.0).is_some();
                        if is_ctx {
                            let s = req_snap
                                .clone()
                                .expect("two-state ensures implies an inhaled requires");
                            ens_args.push(s);
                        }
                        // `base` is the exit heap (delta subtracted from it). A
                        // self-framed callee (`!is_ctx`) yields its snapshot.
                        (heap, _) = emit_resource_exhale(sink, ens_id, heap, ens_args, !is_ctx);
                    }
                    None
                }
                // A back edge re-establishes the invariant and stops. Nothing
                // flows out: `Cfg::predecessors` excludes back edges, so this
                // block is a leaf and no successor ever reads `heap`. Matching
                // Silicon, the residual is not required to be empty — permission
                // left over at the back edge is simply lost.
                Terminator::Goto(t) if cfg.loops.is_back_edge(bid, *t) => {
                    let old = pure_exp::OldHeaps {
                        baseline,
                        labeled: &labeled,
                    };
                    let invs = &cfg.blocks[*t].invs;
                    heap = lower_invariant(b, &env, sink, heap, invs, true, &old)?;
                    None
                }
                Terminator::Goto(_) => None,
            };
            Ok::<_, TranslationError>((heap, cond))
        })?;
        current_heap = new_heap;
        if let Some(c) = &cond {
            reach.declare(c);
        }
        cond_val[bid] = cond;
        exit_env.insert(bid, env);
        let mut body = sink.take_since(body_mark);
        // Assert distribution. When this block's reach is a genuine
        // disjunction, its pc ends in one opaque materialized literal, and a pure
        // `Assert` under it is provable only by refuting the whole disjunction at
        // once — case analysis the prover does not perform (the shape: the
        // `assert false` of a `match`'s shared unreachable otherwise-target,
        // reached by one impossible edge per match; or an obligation on a value
        // merged by a match, which is constant per arm). For a pure obligation
        // `(∨ᵢ eᵢ) ⇒ φ ⟺ ∧ᵢ (eᵢ ⇒ φ)`, so emit one copy per part of the reach's
        // split (`reach_split`) under that part's pc. `Inst.pc` is a delta over
        // the (elided) ambient cube, so the part's pc is *prepended*: a copy's
        // effective pc is `reach ∧ eᵢ ∧ delta`, which equals `eᵢ ∧ delta`.
        // Positionally safe: `Assert` produces no temp.
        if reach_split[bid].len() > 1 {
            let mut out: Vec<Inst> = Vec::with_capacity(body.len());
            for inst in body {
                if matches!(inst.kind, vmir::InstKind::Assert(_)) {
                    for epc in &reach_split[bid] {
                        let mut copy = inst.clone();
                        let mut conds = epc.conds.clone();
                        conds.extend(copy.pc.conds.iter().cloned());
                        copy.pc.conds = conds;
                        out.push(copy);
                    }
                } else {
                    out.push(inst);
                }
            }
            body = out;
        }

        // Push the real block after its synthetic join blocks (if any), so preds
        // always precede successors in `blocks`.
        let rid = vmir::BlockId(blocks.len());
        blocks.push(vmir::Block {
            cube: pc,
            preds: preds_kind,
            join: real_join,
            body,
            h_out: current_heap,
        });
        vmir_id.insert(bid, rid);
        h_out_of.insert(rid, current_heap);
    }

    Ok(vmir::Method {
        name,
        entry: vmir_id[&cfg.entry],
        blocks: blocks.into(),
    })
}

/// Collect the VMIR type of every method-scoped `var` declaration (plus the
/// already-seeded params/rets), recursing through `if`/block statements. Viper
/// locals are method-scoped, so a single flat map suffices for phi typing.
fn collect_var_types(
    names: &HashMap<Spur, vmir::MemberId>,
    stmts: &[typed::Statement],
    out: &mut HashMap<Spur, Type>,
) {
    use typed::Statement as S;
    for s in stmts {
        match s {
            S::Var(idents, _) => {
                for id in idents {
                    out.insert(id.name.0, lower_type(names, &[], &id.ty));
                }
            }
            S::If(_, then, els) => {
                collect_var_types(names, &then.0, out);
                if let Some(e) = els {
                    collect_var_types(names, &e.0, out);
                }
            }
            S::While(_, _, body) => collect_var_types(names, &body.0, out),
            S::Block(inner) => collect_var_types(names, &inner.0, out),
            _ => {}
        }
    }
}

/// Inhale or exhale a loop invariant against `heap`. The clauses are
/// conjoined into a single assertion tree (`A && B && ...`) and lowered in
/// one `lower_spatial` call — exactly as a single `invariant A && B` would be
/// — rather than lowered clause by clause with a separate `lower_spatial`
/// call per clause. The latter would re-fix `value_heap` (the heap pure
/// sub-expressions read from) to the *post-previous-clause* heap on every
/// iteration, so an exhale's later clause would see the permission its
/// earlier clause had just subtracted and reject a field read the invariant,
/// read as a whole, plainly justifies (e.g. `invariant acc(a.f) invariant
/// a.f == 7`). A single `lower_spatial` call keeps `value_heap` fixed for the
/// whole invariant on exhale, matching `S::Conj`'s own handling of `&&`.
fn lower_invariant(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    sink: &mut Sink,
    heap: HeapVal,
    invs: &[typed::SpatialMethodExp],
    exhale: bool,
    old: &pure_exp::OldHeaps<'_>,
) -> Result<HeapVal, TranslationError> {
    // Balanced, like `combine_spatial`'s contract clauses.
    let Some(conj) = invs
        .iter()
        .cloned()
        .tree_reduce(|l, r| typed::SpatialExp(Box::new(typed::SpatialExpKind::Conj(l, r))))
    else {
        return Ok(heap);
    };
    let mode = if exhale {
        SpatialMode::Exhale { value_heap: heap }
    } else {
        SpatialMode::Inhale
    };
    let (h_out, bv) = spatial::lower_spatial(b, env, sink, heap, mode, Some(old), &conj)?;
    if let Some(v) = bv {
        if exhale {
            sink.with_heap(heap, |sink| sink.emit_assert(v));
        } else {
            sink.emit_assume(v);
        }
    }
    Ok(h_out)
}

/// Every variable a loop body assigns to or declares — the havoc set.
///
/// Over-approximating here is sound (a needlessly havoc'd variable only loses
/// information); under-approximating is not, since a variable the body mutates
/// would keep its pre-loop value across the cut.
fn loop_written_vars(cfg: &cfg::Cfg, l: &cfg::Loop) -> HashSet<Spur> {
    let mut out = HashSet::default();
    for bid in &l.body {
        for stmt in &cfg.blocks[*bid].stmts {
            match stmt {
                typed::Statement::Var(idents, _) => {
                    out.extend(idents.iter().map(|i| i.name.0));
                }
                typed::Statement::Assign(lhss, _) => {
                    out.extend(lhss.iter().filter_map(|l| match l {
                        typed::AssignLhs::Var(i) => Some(i.0),
                        // A field assignment mutates the heap, which the
                        // invariant's permissions already account for.
                        typed::AssignLhs::Field(..) => None,
                    }));
                }
                _ => {}
            }
        }
    }
    out
}

fn lower_stmt(
    b: &TranslationContext<'_>,
    env: &mut HashMap<Spur, Val>,
    sink: &mut Sink,
    current_heap: HeapVal,
    baseline: HeapVal,
    labeled: &mut HashMap<Spur, HeapVal>,
    stmt: &typed::Statement,
) -> Result<HeapVal, TranslationError> {
    use typed::Statement as S;
    match stmt {
        S::Var(idents, None) => {
            for id in idents {
                let ty = b.lower_type(&id.ty);
                let v = sink.emit_pure(ty, PureInst::Fresh);
                env.insert(id.name.0, v);
            }
            Ok(current_heap)
        }
        S::Var(idents, Some(typed::AssignRhs::Exp(pure))) => {
            if idents.len() != 1 {
                return Err(TranslationError::Unsupported("multi-LHS var := exp"));
            }
            let old = pure_exp::OldHeaps {
                baseline,
                labeled: &*labeled,
            };
            let v = pure_exp::lower(
                b,
                env,
                sink,
                pure_exp::HeapCtx::same_with_old(current_heap, &old),
                pure,
            )?;
            env.insert(idents[0].name.0, v);
            Ok(current_heap)
        }
        S::Var(idents, Some(typed::AssignRhs::MethodCall(call))) => {
            let ret_names: Vec<Spur> = idents.iter().map(|i| i.name.0).collect();
            let ret_types: Vec<vmir::Type> = idents.iter().map(|i| b.lower_type(&i.ty)).collect();
            lower_method_call(
                b,
                env,
                sink,
                current_heap,
                baseline,
                labeled,
                call,
                &ret_names,
                &ret_types,
            )
        }
        S::Assign(lhss, typed::AssignRhs::MethodCall(call)) => {
            let mut ret_names = Vec::with_capacity(lhss.len());
            for lhs in lhss {
                match lhs {
                    typed::AssignLhs::Var(name) => ret_names.push(name.0),
                    typed::AssignLhs::Field(_, _) => {
                        return Err(TranslationError::Unsupported("field lvalue"));
                    }
                }
            }
            let ret_types: Vec<vmir::Type> = ret_names
                .iter()
                .map(|spur| {
                    env.get(spur)
                        .map(|_| ())
                        .ok_or_else(|| {
                            TranslationError::UnknownIdent(b.interner.resolve(spur).to_string())
                        })
                        // type is recovered from env via name lookup, but env stores Val.
                        // For Phase 2 minimal scope, accept that the LHS variables have been
                        // declared earlier with `var` and assume their types match the
                        // callee's return signature. We use Fresh-typed Int as a placeholder
                        // only when we cannot resolve — better solution lands with proper type
                        // tracking in the env.
                        .map(|_| vmir::Type::Int)
                })
                .collect::<Result<_, _>>()?;
            lower_method_call(
                b,
                env,
                sink,
                current_heap,
                baseline,
                labeled,
                call,
                &ret_names,
                &ret_types,
            )
        }
        S::Assign(lhss, typed::AssignRhs::Exp(pure)) => {
            if lhss.len() != 1 {
                return Err(TranslationError::Unsupported("multi-LHS assign := exp"));
            }
            let old = pure_exp::OldHeaps {
                baseline,
                labeled: &*labeled,
            };
            let hctx = pure_exp::HeapCtx::same_with_old(current_heap, &old);
            let v = pure_exp::lower(b, env, sink, hctx, pure)?;
            match &lhss[0] {
                typed::AssignLhs::Var(n) => {
                    env.insert(n.0, v);
                    Ok(current_heap)
                }
                // `e.f := v`: mutate the heap at `field@addr(e)` to `v`. The
                // `Assign` is guarded — it requires write permission at the loc.
                typed::AssignLhs::Field(base, fname) => {
                    let base_val = pure_exp::lower(b, env, sink, hctx, base)?;
                    let addr = resource::field_addr(b, sink, base_val, fname.0)?;
                    // Inside an `if` arm the heap is a single timeline, so the
                    // write must be conditional in its *value*: `pc ? v : old`.
                    let val = if sink.branch_conds().is_empty() {
                        v
                    } else {
                        let ty = b.lower_type(&pure.ty);
                        let old = sink.emit_pure_guarded(
                            ty.clone(),
                            PureInst::Deref(current_heap, addr.clone()),
                        );
                        sink.gate_value(v, old, ty)
                    };
                    Ok(sink.emit_heap_guarded(HeapInst::Assign(
                        current_heap,
                        vmir::Assign { loc: addr, val },
                    )))
                }
            }
        }
        S::Var(idents, Some(typed::AssignRhs::New(sof))) => {
            if idents.len() != 1 {
                return Err(TranslationError::Unsupported("multi-LHS var := new"));
            }
            lower_new(b, env, sink, current_heap, idents[0].name.0, sof)
        }
        S::Assign(lhss, typed::AssignRhs::New(sof)) => {
            if lhss.len() != 1 {
                return Err(TranslationError::Unsupported("multi-LHS assign := new"));
            }
            let name = match &lhss[0] {
                typed::AssignLhs::Var(n) => n.0,
                typed::AssignLhs::Field(_, _) => {
                    return Err(TranslationError::Unsupported("field lvalue"));
                }
            };
            lower_new(b, env, sink, current_heap, name, sof)
        }
        // Control flow never reaches straight-line lowering. `viper::cfg`'s
        // `Builder::process` consumes every one of these into block structure —
        // branches and terminators — and only the statements it *pushes* into a
        // `BasicBlock` land here. `old[L]` is likewise bound from the block's
        // own `label` field at the top of the block walk, not from `S::Label`.
        S::If(..) | S::While(..) | S::Block(..) | S::Label(..) | S::Goto(..) => {
            unreachable!("control flow is resolved into blocks by viper::cfg: {stmt:?}")
        }
        S::Fold(pwp) => lower_fold_unfold(b, env, sink, current_heap, baseline, labeled, pwp, true),
        S::Unfold(pwp) => {
            lower_fold_unfold(b, env, sink, current_heap, baseline, labeled, pwp, false)
        }
        // Source-level assert/assume are non-destructive: the assertion is
        // reduced to a boolean over the current heap (each `acc(loc, p)` becomes
        // `perm(loc) >= p`) and asserted/assumed. The heap is unchanged.
        S::Assert(e) => {
            let old = pure_exp::OldHeaps {
                baseline,
                labeled: &*labeled,
            };
            if let Some(v) =
                spatial::lower_assertion_bool(b, env, sink, current_heap, Some(&old), e)?
            {
                // The obligation is checked in the current heap.
                sink.with_heap(current_heap, |sink| sink.emit_assert(v));
            }
            Ok(current_heap)
        }
        S::Refute(e) => {
            let old = pure_exp::OldHeaps {
                baseline,
                labeled: &*labeled,
            };
            if let Some(v) =
                spatial::lower_assertion_bool(b, env, sink, current_heap, Some(&old), e)?
            {
                sink.with_heap(current_heap, |sink| sink.emit_refute(v));
            }
            Ok(current_heap)
        }
        S::Assume(e) => {
            let old = pure_exp::OldHeaps {
                baseline,
                labeled: &*labeled,
            };
            if let Some(v) =
                spatial::lower_assertion_bool(b, env, sink, current_heap, Some(&old), e)?
            {
                sink.emit_assume(v);
            }
            Ok(current_heap)
        }
        // Inhale: add the assertion's heap delta to the current heap and assume
        // its boolean. Heap-dependent sub-expressions are evaluated against the
        // growing heap (`ReadHeap::Track`), so later conjuncts can observe the
        // permissions just inhaled.
        S::Inhale(e) => {
            let old = pure_exp::OldHeaps {
                baseline,
                labeled: &*labeled,
            };
            let (h_out, bv) = spatial::lower_spatial(
                b,
                env,
                sink,
                current_heap,
                SpatialMode::Inhale,
                Some(&old),
                e,
            )?;
            if let Some(v) = bv {
                sink.emit_assume(v);
            }
            Ok(h_out)
        }
        // Exhale: subtract the assertion's heap delta (accumulated from
        // `Empty`) from the current heap and assert its boolean. All
        // heap-dependent sub-expressions are evaluated against the heap from
        // *before* the exhale (`ReadHeap::Fixed(current_heap)`), since the
        // permissions are still held at that point.
        S::Exhale(e) => {
            // Subtraction happens inside `lower_spatial` (left-to-right), so a
            // `perm` in the assertion observes the running reduced heap, while
            // value reads use the fixed pre-exhale `current_heap`.
            let old = pure_exp::OldHeaps {
                baseline,
                labeled: &*labeled,
            };
            let (h_out, bv) = spatial::lower_spatial(
                b,
                env,
                sink,
                current_heap,
                SpatialMode::Exhale {
                    value_heap: current_heap,
                },
                Some(&old),
                e,
            )?;
            if let Some(v) = bv {
                // The exhale's boolean is over the pre-exhale state, so it is
                // checked in `current_heap` (not the reduced `h_out`).
                sink.with_heap(current_heap, |sink| sink.emit_assert(v));
            }
            Ok(h_out)
        }
    }
}

/// Lower `lhs := new(fields)`: allocate a fresh `Ref`, bind it to `lhs`, and
/// inhale full permission to each listed field (`acc(lhs.f, write)`), adding the
/// chunks to the current heap. `new(*)` is not yet supported.
fn lower_new(
    b: &TranslationContext<'_>,
    env: &mut HashMap<Spur, Val>,
    sink: &mut Sink,
    current_heap: HeapVal,
    lhs: Spur,
    sof: &typed::StarOrFields,
) -> Result<HeapVal, TranslationError> {
    let v = sink.emit_pure(Type::Ref, PureInst::Fresh);
    env.insert(lhs, v.clone());

    // `new` always yields a fresh *non-null* reference.
    let is_null = sink.emit_pure(
        Type::Bool,
        PureInst::Binary(vmir::BinOp::Eq, v.clone(), vmir::NULL),
    );
    let non_null = sink.emit_pure(
        Type::Bool,
        PureInst::Ternary(is_null, vmir::FALSE, vmir::TRUE),
    );
    sink.emit_assume(non_null);

    match sof {
        typed::StarOrFields::Fields(fields) => {
            let mut heap = current_heap;
            for f in fields {
                let (loc, perm) = resource::field_acc(b, sink, v.clone(), f.0, vmir::write())?;
                let perm = sink.gate_perm(vmir::PermVal::Amount(perm));
                heap = sink.emit_heap(HeapInst::Add {
                    base: heap,
                    loc,
                    perm,
                    // `new(...)` genuinely mints values: the canonical
                    // legitimate `with fresh`.
                    bind: vmir::Bind::Fresh,
                });
            }
            Ok(heap)
        }
        typed::StarOrFields::Star => Err(TranslationError::Unsupported("new(*)")),
    }
}

/// Lower `fold P(args)` / `unfold P(args)` into its bind-point pair -- see
/// [`pure_exp::emit_fold_pair`] / [`pure_exp::emit_unfold_pair`]. Neither is a
/// heap instruction: the predicate id, args, and perm come from the statement,
/// and the pair's second half yields the new working heap.
fn lower_fold_unfold(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    sink: &mut Sink,
    current_heap: HeapVal,
    baseline: HeapVal,
    labeled: &HashMap<Spur, HeapVal>,
    pwp: &typed::PredicateWithPerm<typed::MethodBodyExt>,
    is_fold: bool,
) -> Result<HeapVal, TranslationError> {
    let old = pure_exp::OldHeaps { baseline, labeled };
    let hctx = pure_exp::HeapCtx::same_with_old(current_heap, &old);
    let (call, perm) = pure_exp::lower_pred_call(b, env, sink, hctx, pwp)?;
    let perm = sink.gate_perm(perm);
    if is_fold {
        Ok(pure_exp::emit_fold_pair(
            b,
            sink,
            current_heap,
            pwp.pred_call.name.0,
            call,
            perm,
        ))
    } else {
        Ok(pure_exp::emit_unfold_pair(
            b,
            sink,
            current_heap,
            pwp.pred_call.name.0,
            call,
            perm,
        ))
    }
}

fn lower_method_call(
    b: &TranslationContext<'_>,
    env: &mut HashMap<Spur, Val>,
    sink: &mut Sink,
    current_heap: HeapVal,
    baseline: HeapVal,
    labeled: &HashMap<Spur, HeapVal>,
    call: &typed::Call<typed::MethodBodyExt>,
    ret_names: &[Spur],
    ret_types: &[vmir::Type],
) -> Result<HeapVal, TranslationError> {
    // Lower argument expressions (may contain `old(...)`).
    let old = pure_exp::OldHeaps { baseline, labeled };
    let mut args: Vec<Val> = Vec::with_capacity(call.args.len());
    for a in &call.args {
        args.push(pure_exp::lower(
            b,
            env,
            sink,
            pure_exp::HeapCtx::same_with_old(current_heap, &old),
            a,
        )?);
    }

    let mut heap = current_heap;

    // Exhale precondition (if present): `h, s := heap - acc m#requires(args)`
    // (implicitly asserts the requires bool). The yielded snapshot captures the
    // callee's pre-state (the consumed chunk values) for the ensures inhale.
    let mut req_snap: Option<Val> = None;
    if let Some(req_id) = b.method_requires(call.name.0) {
        // `#requires` is always self-framed, so it always yields its snapshot.
        let (h, s) = emit_resource_exhale(sink, req_id, heap, args.clone(), true);
        heap = h;
        req_snap = s;
    }

    // Allocate fresh return values BEFORE the post-condition inhale.
    let mut ret_vals: Vec<Val> = Vec::with_capacity(ret_names.len());
    for (spur, ty) in ret_names.iter().zip(ret_types.iter()) {
        let v = sink.emit_pure(ty.clone(), PureInst::Fresh);
        env.insert(*spur, v.clone());
        ret_vals.push(v);
    }

    // Inhale postcondition (if present): `h := heap + acc m#ensures(args, rets[, s])`
    // (implicitly assumes the ensures bool). A two-state `#ensures` reads the
    // callee's pre-state through its trailing snapshot argument — the snapshot
    // yielded by the precondition exhale above.
    if let Some(ens_id) = b.method_ensures(call.name.0) {
        let mut ens_args = args.clone();
        ens_args.extend(ret_vals.iter().cloned());
        // Two-state iff the callee has its own `#requires` (`is_ctx_resource`
        // on `ens_id` — the callee's own resource, never queried on `req_id`).
        let is_ctx = b.method_requires(call.name.0).is_some();
        if is_ctx {
            let s = req_snap
                .clone()
                .expect("two-state ensures implies an exhaled requires");
            ens_args.push(s);
        }
        // A call's post-state values are unknown to the caller.
        heap = emit_resource_inhale(sink, ens_id, heap, ens_args, vmir::Bind::Fresh);
    }

    Ok(heap)
}

/// Emit `h[, s] := base <sign> acc <resource>(args) write`: combine the
/// resource's full-permission delta onto `base`, implicitly assuming (`Add`) or
/// asserting (`Sub`) its boolean. Returns the resulting heap plus, for a
/// **self-framed** callee, the snapshot `Val` the inst yields (the pre-state
/// handle passed on as the trailing argument of a two-state resource call —
/// e.g. `m#requires`'s snapshot feeding `m#ensures`). A two-state callee
/// yields no snapshot. `yields_snap` is the caller's `!is_ctx` for `resource` —
/// always `true` for a `#requires` id (always self-framed), and
/// `method_requires(owner).is_none()` for a `#ensures` id.
fn emit_resource_inhale(
    sink: &mut Sink,
    resource: vmir::MemberId,
    base: HeapVal,
    args: Vec<Val>,
    bind: vmir::Bind,
) -> HeapVal {
    let perm = contract_perm(sink);
    sink.emit_resource_inhale(base, ResourceCall { resource, args }, perm, bind)
}

/// The consume counterpart of [`emit_resource_inhale`].
fn emit_resource_exhale(
    sink: &mut Sink,
    resource: vmir::MemberId,
    base: HeapVal,
    args: Vec<Val>,
    yields_snap: bool,
) -> (HeapVal, Option<Val>) {
    let perm = contract_perm(sink);
    sink.emit_resource_exhale(base, ResourceCall { resource, args }, perm, yields_snap)
}

/// Gate the permission by the current branch path condition so a contract
/// inhaled/exhaled inside an `if` arm contributes nothing on the other path
/// (the empty top-level pc leaves `write` unchanged).
fn contract_perm(sink: &mut Sink) -> vmir::PermVal {
    sink.gate_perm(vmir::PermVal::write())
}
