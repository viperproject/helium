use crate::dhash::HashMap;
use crate::verify::eval::{
    EvalState, OPERAND_RECIPE, check_forall_wd, collect_pc_lits, display_heaps, eval_perm,
    eval_perm_inst, eval_perm_structural, eval_pure_inst, format_inst, get_heap,
    get_heap_unrehomed, pc_lits_raw, perm_recipe, pure_obligations,
};
use crate::verify::heap::{Access, ConsumeMode, Loc, QpCheck, QuantSpec, ReadRecipe, Value};
use std::rc::Rc;

use crate::verify::cert::{BodyRecipe, PermRecipe, SeedRef, SlotShape};
use crate::vmir::CollKind;
use crate::{
    verify::{
        cert::{FunctionDefinition, ResourceDefinition},
        context::VerifyContext,
        error::VerifyError,
        heap::{ChunkPerm, Heap, LocationKind, Scale},
        lang::Symbolic,
        stats,
        viz::Snapshotter,
    },
    vmir::{
        self, Assign, Bind, Function, HeapInst, Inst, InstKind, Literal, MemberId, Method,
        PathConds, Polarity, PureInst, Resource, Type, Val,
    },
};

/// What a `wildcard` leaf becomes when a footprint slot's permission is grafted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WildcardAs {
    /// A freshly minted positive share (`0 < w` assumed) — the real permission.
    Fresh,
    /// Full permission `1`, so a gated wildcard builds the slot's **presence**
    /// indicator `ite(guard, 1, 0)` whose `0 < …` folds to `guard`. Used by a
    /// frame-only exhale, which needs presence but no amount: building the wildcard
    /// would leave un-collapsible `ite` residue in the persistent graph for
    /// `ite-reduce` to churn on.
    One,
}

/// Graft a footprint slot's permission: [`eval_perm_structural`]'s descent, over a
/// slot recipe instead of an [`EvalState`]. One wildcard is picked per `wildcard`
/// **leaf** reached, eagerly — a leaf under a false guard still mints, and the
/// enclosing `ite` discards it.
///
/// Keeps the **branch structure** as a [`ChunkPerm`] rather than flattening it into
/// one `Ite` term.
///
/// The structure is what lets a consume align the demand's arms against the held
/// permission's (see `heap_subtract`): `ChunkPerm::restrict` takes the matching arm
/// when both branch on the same condition and carries the whole term in otherwise, so
/// precision falls out of the shapes agreeing rather than out of any case analysis
/// here. Flattened into an `Ite`, that structure survives only inside the e-graph term
/// and the consume cannot see it.
///
/// `ChunkPerm::select` is the smart constructor: it flattens arms branching on the
/// same condition, collapses `then ≡ els`, and drops an arm whose condition
/// const-folds — so a wildcard-free program still builds exactly what it did before.
fn build_perm(
    ctx: &mut VerifyContext<'_>,
    perm: &PermRecipe,
    resolve: &impl Fn(&crate::verify::cert::SeedRef) -> egg::Id,
    pc: &[(egg::Id, Polarity)],
    changed: &mut Vec<egg::Id>,
    wildcard_as: WildcardAs,
) -> ChunkPerm {
    fn leaf(
        ctx: &mut VerifyContext<'_>,
        pv: &vmir::PermVal<crate::verify::cert::BodyRecipe>,
        built: &[ChunkPerm],
        resolve: &impl Fn(&crate::verify::cert::SeedRef) -> egg::Id,
        pc: &[(egg::Id, Polarity)],
        changed: &mut Vec<egg::Id>,
        wildcard_as: WildcardAs,
    ) -> ChunkPerm {
        match pv {
            vmir::PermVal::Amount(r) => {
                ChunkPerm::leaf(r.build(&mut ctx.egraph, resolve, pc, changed))
            }
            // `WildcardAs::One` builds a **presence** indicator, not a share: it is
            // the literal `1`, with no wildcard left in it, so the leaf is concrete.
            vmir::PermVal::Wildcard => match wildcard_as {
                WildcardAs::Fresh => ChunkPerm::wild_leaf(ctx.fresh_wildcard()),
                WildcardAs::One => ChunkPerm::leaf(
                    ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into()))),
                ),
            },
            // A step only names earlier ones, so this is always already built.
            vmir::PermVal::Temp(i) => built[*i].clone(),
        }
    }
    let mut built: Vec<ChunkPerm> = Vec::with_capacity(perm.steps.len());
    for step in &perm.steps {
        let vmir::PermInst::Ite(c, t, e) = step;
        let c = c.build(&mut ctx.egraph, resolve, pc, changed);
        let t = leaf(ctx, t, &built, resolve, pc, changed, wildcard_as);
        let e = leaf(ctx, e, &built, resolve, pc, changed, wildcard_as);
        built.push(ChunkPerm::select(ctx, c, t, e));
    }
    leaf(ctx, &perm.res, &built, resolve, pc, changed, wildcard_as)
}

/// Invariant 6 of the two-egraph block model (`design/block-vmir/82-*.md`):
/// statement-level heap instructions operate at the **block-PC level only** — their
/// effective pc is exactly the block cube, never a cube plus a suffix — because they
/// cannot appear as sub-expressions. Expression-embedded obligations (a function
/// precondition, a division check) are the only things allowed an extra suffix.
///
/// A violation means a heap effect landing under a condition the block model does
/// not know about.
///
/// Since `Inst.pc` became a **delta** over the phase's ambient cube, the "is the
/// cube" half of this is by construction: the body phase elides the ambient at
/// emission ([`Sink::with_ambient_cube`](crate::translate::sink)), so there is no
/// representation in which a statement inst could restate or contradict it. What is
/// left to check is the *suffix* half, and that is a plain emptiness test on the
/// delta — no e-graph, no `cube_eq`.
///
/// Still gated on `SILVER_OXIDE_ASSERT_BLOCK_PC` rather than promoted to an
/// always-on `debug_assert`, because the suffix half has a **known, deliberate
/// counterexample**: `unfolding p in e` is a Viper *expression* that lowers to
/// statement-level instructions, so under a ternary it legitimately carries a
/// suffix (`tests/cases/passing/predicates/unfolding_expr_scoped_read.vpr`, and
/// `tests/cases/failing/functions/predicate_graft_token_guard.vpr`). Plan 82
/// proposed rejecting that construct; measurement says the lowering re-folds and
/// fabricates no permission, so it is the invariant that is too strong. Until
/// `unfolding` becomes a scoped node, this stays an opt-in corpus check.
fn assert_statement_pc_is_block_cube(ctx: &mut VerifyContext<'_>, inst: &Inst) {
    if !ctx.in_block() || std::env::var_os("SILVER_OXIDE_ASSERT_BLOCK_PC").is_none() {
        return;
    }
    let statement_level = matches!(
        inst.kind,
        InstKind::Heap(HeapInst::Inhale { .. } | HeapInst::Exhale { .. } | HeapInst::Assign(..))
    );
    if !statement_level {
        return;
    }
    assert!(
        inst.pc.conds.is_empty(),
        "invariant 6: statement-level inst carries a pc suffix beyond the block cube \
         ({} extra literals) — kind {:?}",
        inst.pc.conds.len(),
        std::mem::discriminant(&inst.kind),
    );
}

/// Evaluate a heap inst. `Sub` may fail with `InsufficientPermission`.
fn eval_heap_inst(
    ctx: &mut VerifyContext<'_>,
    state: &EvalState,
    inst: &HeapInst,
    pc: &PathConds,
) -> Result<Heap, VerifyError> {
    match inst {
        // Heap SUM: both heaps are held at once, so permissions at a shared
        // location add and values there must agree. Contrast `Merge`, which
        // selects between mutually exclusive predecessor states.
        HeapInst::Union { a, b } => {
            let region = state.region(ctx);
            let ha = get_heap(ctx, state, a, &region)?;
            let hb = get_heap(ctx, state, b, &region)?;
            // The block cube is load-bearing here, not decorative: a loop exit's
            // union runs under `<!guard>`, and both the positivity decision and the
            // guarded value equality have to be relative to it.
            let pc_lits: Vec<(egg::Id, Polarity)> = pc
                .conds
                .iter()
                .map(|(v, p)| (state.get_val(ctx, v), *p))
                .collect();
            Ok(ha.union(ctx, &hb, &pc_lits))
        }
        // The heap at a `label`: unchanged, but defined in the label's block,
        // so its region is that block's cube (see `HeapInst::Label`).
        HeapInst::Label { base } => {
            let region = state.region(ctx);
            get_heap(ctx, state, base, &region)
        }
        // Block-IR heap join: structural per-chunk SELECT of the two predecessor
        // exit heaps. Emitted at every CFG join by the block lowering.
        HeapInst::Merge {
            cond,
            then_h,
            els_h,
        } => {
            // An unreachable predecessor arm (its cube refuted, e.g. the enum
            // `bb_unreach` after `inhale false`) is dropped outright — selecting
            // a `0`-leaf against it would need an exhaustiveness case split.
            match state.join_arms_dead.unwrap_or((false, false)) {
                (false, true) => {
                    let region = state.region(ctx);
                    return get_heap(ctx, state, then_h, &region);
                }
                (true, false) => {
                    let region = state.region(ctx);
                    return get_heap(ctx, state, els_h, &region);
                }
                _ => {}
            }
            let cond_id = state.get_val(ctx, cond);
            // Unrehomed: the merge rehomes each arm itself, appending the arm
            // literal to every chunk that arm holds.
            let h_then = get_heap_unrehomed(state, then_h);
            let h_els = get_heap_unrehomed(state, els_h);
            let region = state.region(ctx);
            Ok(Heap::merge(ctx, cond_id, &h_then, &h_els, &region))
        }
        // `base + acc(loc, perm)` / `base - acc(loc, perm)`.
        HeapInst::Add {
            base, loc, perm, ..
        }
        | HeapInst::Sub {
            base, loc, perm, ..
        } => {
            // `Add` carries a `Bind`; `Sub` is read-shaped and carries none.
            let bind = match inst {
                HeapInst::Add { bind, .. } => Some(bind),
                _ => None,
            };
            let region = state.region(ctx);
            let base_h = get_heap(ctx, state, base, &region)?;
            let l = state
                .loc(ctx, loc)
                .expect("acc location must be Addr-typed");
            let mut access = Access::Single {
                loc: l.clone(),
                perm: eval_perm_structural(ctx, state, perm),
                held: None,
            };
            let pc_lits = pc_lits_raw(ctx, state, pc);
            let Some(bind) = bind else {
                if ctx.recipe.is_some() {
                    return Err(VerifyError::Unimplemented(
                        "purify: unsupported heap inst in a resource body",
                    ));
                }
                return Ok(base_h.consume(ctx, access, ConsumeMode::Take, &pc_lits)?.0);
            };
            // The bind selects the value source.
            let mut value = match bind {
                // A resource body's footprint slot: a placeholder value, named
                // by the seed recorded below.
                Bind::SelfSlot => {
                    debug_assert!(
                        ctx.recipe.is_some(),
                        "`with self` outside a resource-body (certificate) walk"
                    );
                    Value::plain(ctx.fresh_symbolic_value(l.kind.value.clone()))
                }
                // A method-body produce: an unconstrained value, and no
                // certificate is under construction to record it in.
                Bind::Fresh => {
                    debug_assert!(
                        ctx.recipe.is_none(),
                        "`with fresh` inside a resource body — a body that \
                         mints an observable value is not deterministic"
                    );
                    Value::plain(ctx.fresh_symbolic_value(l.kind.value.clone()))
                }
                // The produce half of a desugared `fold`: the chunk holds the
                // snapshot the paired `exhale` handed back, rather than a fresh
                // value. That shared term IS the equation `snap(P(x)) =
                // cons(body)`.
                Bind::Bound(v) => Value {
                    id: state.get_val(ctx, v),
                    recipe: state.recipe_of(v),
                },
            };
            // A resource body's `acc` records a footprint slot in the
            // certificate under construction: its value is the seed placeholder
            // `SlotValue(i)` (supplied at graft time), its address/permission
            // recipes are sliced at the end of the walk. The seed doubles as the
            // chunk's recipe provenance, so a later `Deref` purifies to the slot
            // value.
            if ctx.recipe.is_some() {
                let addr_r = state.require_recipe(loc, OPERAND_RECIPE)?;
                let perm_r = perm_recipe(state, perm)?;
                let rb = ctx.recipe.as_mut().unwrap();
                // The chunk keeps the recipe of its amount's positivity, so a
                // second `acc` at its location can name the summed value.
                if let Access::Single { held, .. } = &mut access {
                    *held = Some(rb.emit_positive(&perm_r));
                }
                let i = rb.pending_slots.len();
                let seed = rb.emit_seed(SeedRef::SlotValue(i));
                rb.pending_slots.push((
                    l.kind.clone(),
                    l.kind.value.clone(),
                    SlotShape::Single {
                        addr: addr_r,
                        perm: perm_r,
                    },
                ));
                value.recipe = Some(seed);
            }
            base_h.produce(ctx, access, value, &pc_lits)
        }
        HeapInst::AddForall { base, q, bind } => {
            let map_ty = Type::ValueMap(Box::new(q.loc_ty.clone()));
            let spec = quant_spec_of(ctx, state, q, pc)?;
            let value = match bind {
                Bind::Fresh => {
                    debug_assert!(ctx.recipe.is_none(), "`with fresh` inside a resource body");
                    Value::plain(ctx.fresh_symbolic_value(map_ty))
                }
                Bind::Bound(v) => Value {
                    id: state.get_val(ctx, v),
                    recipe: state.recipe_of(v),
                },
                // A resource body's footprint slot: record the chunk set in the
                // certificate under construction. Its value map is the seed
                // `SlotValue(i)`, supplied at graft time; the captured values
                // are kept as recipes so a call site can re-evaluate the body.
                Bind::SelfSlot => {
                    let caps = q
                        .free_temps()
                        .into_iter()
                        .map(|k| Ok((k, state.require_recipe(&Val::Temp(k), OPERAND_RECIPE)?)))
                        .collect::<Result<Vec<_>, VerifyError>>()?;
                    let rb = ctx
                        .recipe
                        .as_mut()
                        .expect("`with self` outside a resource-body (certificate) walk");
                    let i = rb.pending_slots.len();
                    let seed = rb.emit_seed(SeedRef::SlotValue(i));
                    rb.pending_slots.push((
                        spec.kind(),
                        map_ty.clone(),
                        SlotShape::Quant {
                            q: spec.q.clone(),
                            caps,
                        },
                    ));
                    Value {
                        id: ctx.fresh_symbolic_value(map_ty),
                        recipe: Some(seed),
                    }
                }
            };
            let pc_lits = pc_lits_raw(ctx, state, pc);
            let region = state.region(ctx);
            get_heap(ctx, state, base, &region)?.produce(ctx, Access::Set(spec), value, &pc_lits)
        }
        // Exhale a chunk set: one held set must cover it.
        HeapInst::SubForall { base, q } => {
            if ctx.recipe.is_some() {
                return Err(VerifyError::Unimplemented(
                    "quantified permission in a resource footprint",
                ));
            }
            let spec = quant_spec_of(ctx, state, q, pc)?;
            let pc_lits = pc_lits_raw(ctx, state, pc);
            let region = state.region(ctx);
            let base_h = get_heap(ctx, state, base, &region)?;
            Ok(base_h
                .consume(ctx, Access::Set(spec), ConsumeMode::Take, &pc_lits)?
                .0)
        }
        // Resource inhale/exhale need the program + certificates; method-only.
        HeapInst::Inhale { .. } | HeapInst::Exhale { .. } => Err(VerifyError::Unimplemented(
            "resource inhale/exhale outside method body",
        )),
        // Field assignment `loc := val`: write permission at `loc`, then every
        // holder of `loc` takes the new value (permissions unchanged).
        HeapInst::Assign(heap, Assign { loc, val }) => {
            let l = state
                .loc(ctx, loc)
                .expect("assign location must be Addr-typed");
            let new_val = state.get_val(ctx, val);
            let pc_lits = pc_lits_raw(ctx, state, pc);
            let region = state.region(ctx);
            get_heap(ctx, state, heap, &region)?.write(ctx, &l, new_val, &pc_lits)
        }
    }
}

/// Evaluate one instruction of a resource body (well-formedness pass). Resource
/// bodies emit `Pure`/`Heap`; `unfold` is the one heap op handled specially
/// (shared with method bodies), the rest go through `eval_heap_inst`. The
/// effectful variants (`Assume`/`Assert`/`Refute`) are not produced here.
fn eval_resource_body_inst(
    ctx: &mut VerifyContext<'_>,
    program: &vmir::Program,
    state: &mut EvalState,
    inst: &Inst,
    certs: &HashMap<MemberId, ResourceDefinition>,
) -> Result<(), VerifyError> {
    match &inst.kind {
        InstKind::Pure(ty, pi) => {
            let pc_lits = collect_pc_lits(ctx, state, &inst.pc);
            let (id, recipe) = eval_pure_inst(ctx, state, ty, pi, &pc_lits, Some(&inst.pc))?;
            state.push_val(id, ty.clone(), recipe);
        }
        // The consume half of a desugared `unfold`: yields what it removed.
        InstKind::Heap(HeapInst::Sub {
            yields_value: true, ..
        }) => {
            eval_sub_yield(ctx, state, inst, &inst.pc)?;
        }
        // A heap-dependent function call's implicit precondition check, which
        // may appear in a contract body. Non-consuming, so it is legal here where
        // a consuming resource op is not.
        InstKind::Heap(HeapInst::Exhale {
            frame_only: true, ..
        }) => {
            eval_resource_op(ctx, program, state, inst, certs)?;
        }
        // The entry of a two-state resource body: reconstruct the pre-state
        // heap from the snapshot parameter (implicitly assuming the precondition
        // resource's boolean). A *bound* inhale, so it reconstructs rather than
        // havocs.
        InstKind::Heap(HeapInst::Inhale {
            bind: Bind::Bound(_),
            ..
        }) => {
            let heap = eval_from_snap(ctx, state, inst, certs)?;
            state.push_heap(heap);
        }
        InstKind::Heap(hi) => {
            let heap = eval_heap_inst(ctx, state, hi, &inst.pc)?;
            state.push_heap(heap);
        }
        InstKind::Perm(pi) => {
            let p = eval_perm_inst(ctx, state, pi);
            state.push_perm(p, pi.clone());
        }
        InstKind::Assert(val) => {
            let id = state.get_val(ctx, val);
            let pc_lits: Vec<(egg::Id, Polarity)> = inst
                .pc
                .conds
                .iter()
                .map(|(v, p)| (state.get_val(ctx, v), *p))
                .collect();
            if !ctx.prove_under_pc(id, &pc_lits) {
                if std::env::var_os("SILVER_OXIDE_TRACE_ASSERT").is_some() {
                    eprintln!(
                        "[assert-fail] Assert inst, pc={} lits:\n{}",
                        pc_lits.len(),
                        crate::verify::viz::dump_term(ctx, id, 40),
                    );
                }
                return Err(VerifyError::AssertionFailed);
            }
        }
        InstKind::Assume(_) | InstKind::Refute(_) => {
            return Err(VerifyError::Unimplemented(
                "effectful inst in resource body",
            ));
        }
    }
    Ok(())
}

fn eval_method_inst(
    ctx: &mut VerifyContext<'_>,
    program: &vmir::Program,
    state: &mut EvalState,
    inst: &Inst,
    certs: &HashMap<MemberId, ResourceDefinition>,
) -> Result<(), VerifyError> {
    match &inst.kind {
        InstKind::Pure(ty, pi) => {
            let pc_lits = collect_pc_lits(ctx, state, &inst.pc);
            let (id, recipe) = eval_pure_inst(ctx, state, ty, pi, &pc_lits, Some(&inst.pc))?;
            state.push_val(id, ty.clone(), recipe);
        }
        // `base inhale <resource>(args) perm`: produce the resource's footprint
        // (fresh values), scaled by `perm`, into `base` and **assume** its
        // boolean; `base exhale ...` consumes the footprint from `base` and
        // **asserts** it. A self-framed callee additionally yields the snapshot
        // of its footprint as a pure `Val` (the pre-state handle a two-state call
        // receives). Both route through `walk_footprint`.
        // The consume half of a desugared `unfold`: yields what it removed.
        InstKind::Heap(HeapInst::Sub {
            yields_value: true, ..
        }) => {
            eval_sub_yield(ctx, state, inst, &inst.pc)?;
        }
        InstKind::Heap(HeapInst::Inhale { .. } | HeapInst::Exhale { .. }) => {
            eval_resource_op(ctx, program, state, inst, certs)?;
        }
        InstKind::Heap(hi) => {
            let heap = eval_heap_inst(ctx, state, hi, &inst.pc)?;
            state.push_heap(heap);
        }
        InstKind::Perm(pi) => {
            let p = eval_perm_inst(ctx, state, pi);
            state.push_perm(p, pi.clone());
        }
        InstKind::Assume(val) => {
            let id = state.get_val(ctx, val);
            // Guard by the path condition: an `assume` inside a branch holds only
            // on that branch (rev to match `implication`'s innermost-first fold).
            let pc_lits = collect_pc_lits(ctx, state, &inst.pc);
            ctx.assume_guarded(id, pc_lits.iter().rev().copied());
        }
        InstKind::Assert(val) => {
            // TODO(heap-consolidation): `inst.heap` carries the heap this obligation
            // is checked in — once wired, consolidate it (materialise aliasing/
            // perm-sum facts into the e-graph) before `prove_under_pc`. Unused today.
            let id = state.get_val(ctx, val);
            let pc_lits: Vec<(egg::Id, Polarity)> = inst
                .pc
                .conds
                .iter()
                .map(|(v, p)| (state.get_val(ctx, v), *p))
                .collect();
            // For an opaque heap-free `#requires` token this proves the callee-cert
            // formula and releases the token (mirrors `eval_snap`); otherwise a
            // plain `prove_under_pc`.
            if !ctx.prove_under_pc(id, &pc_lits) {
                if std::env::var_os("SILVER_OXIDE_TRACE_ASSERT").is_some() {
                    eprintln!(
                        "[assert-fail] Assert inst, pc={} lits:\n{}",
                        pc_lits.len(),
                        crate::verify::viz::dump_term(ctx, id, 40),
                    );
                }
                return Err(VerifyError::AssertionFailed);
            }
            // An assert in a body **publishes nothing**. `ensures` is a
            // function's single export mechanism (`post_fact`, built from the
            // declaration link), and this assert — the exit `f#ensures(..)`, or a
            // heap-free callee's `g#requires(gargs)` — is a pure obligation:
            // discharged here and dropped. Silicon likewise hands a caller the
            // nested `g%precondition` *stamp* (our propagated `TokenStep`), never
            // the precondition formula.
        }
        InstKind::Refute(val) => {
            // `refute A` succeeds iff `A` is NOT provable in this state.
            let id = state.get_val(ctx, val);
            let pc_lits: Vec<(egg::Id, Polarity)> = inst
                .pc
                .conds
                .iter()
                .map(|(v, p)| (state.get_val(ctx, v), *p))
                .collect();
            if ctx.prove_under_pc(id, &pc_lits) {
                return Err(VerifyError::RefuteFailed);
            }
        }
    }
    Ok(())
}

/// Where a footprint slot's value comes from (see [`walk_footprint`]).
enum ValueSource {
    /// Read the chunk value at the slot's address from this heap (fresh if
    /// absent). Used by `fold`/`snap`/`exhale`, which read a heap they hold.
    /// A certificate walk propagates the hit chunk's recipe provenance.
    ReadHeap(Heap),
    /// Recover it from the snapshot `s` as `unwrap(proj_i(s))`. Used by
    /// `unfold`/`from_snap`, which reconstruct the footprint from a snapshot.
    /// The second field is `s`'s recipe term (certificate walks only) — each
    /// slot's recipe mirrors the projection: `unwrap(proj_i(s_recipe))`.
    ProjectSnap(egg::Id, Option<Val>),
    /// A fresh unconstrained value per slot. Used by `inhale`, which produces a
    /// resource's footprint with unknown location values.
    Fresh,
    /// What stands for each (single) slot of a quantifier body's call on a
    /// binder: the slot's value is `lookup(map, addr)` or the captured value
    /// (see [`frame_snapshot`]).
    Stands(std::rc::Rc<[crate::verify::quant::ResolvedSlot]>),
}

/// The direction of a footprint walk: consume a held footprint and **assert** the
/// resource's boolean (`fold`/`snap`), or produce one and **assume** it
/// (`unfold`/`from_snap`). `Build` only builds the snapshot a frame-only exhale
/// yields, with no heap effect and no check (see [`frame_snapshot`]).
enum Direction {
    Consume,
    Produce,
    Build,
}

/// The result of a footprint walk: the accumulator heap after all slot effects,
/// the snapshot members `present ? Some(v) : None` per slot, and — during a
/// certificate walk — each slot value's recipe term. Members are filled for
/// **both** directions: the `members.push` in the slot loop is unconditional, so
/// a `Produce` walk conses a snapshot exactly as a `Consume` walk does.
struct FootprintResult {
    heap: Heap,
    members: Vec<egg::Id>,
    /// Recipe provenance of each slot value, in slot order (certificate walks
    /// only; all `None` otherwise).
    slot_recipes: Vec<Option<Val>>,
    /// Each single slot's location, in slot order; `None` for a quantified one.
    slot_locs: Vec<Option<Loc>>,
}

/// The single per-slot footprint loop behind `fold`, `unfold`, `snap` and
/// `from_snap`. For each footprint slot of `resource(args)`: graft the slot's
/// `(addr, perm)`, obtain the slot value from `source`, apply the slot's heap
/// effect to the `base` accumulator (subtract for `Consume`, union for
/// `Produce`; permission scaled by `scale` when `Some`), and thread the actual
/// value through `subst` so a value-dependent inner address (e.g. `P(this.next)`)
/// resolves. Finally graft the body boolean and discharge it per `direction`
/// (`Consume` asserts under `pc_lits`, `Produce` assumes it guarded by them).
///
/// The caller owns everything *around* the slots: the predicate-chunk add/remove
/// (bracketing differs — `fold` adds after, `unfold` removes before) and what to
/// do with `heap` (`snap` discards it — functions frame, they don't consume).
#[allow(clippy::too_many_arguments)]
fn walk_footprint(
    ctx: &mut VerifyContext<'_>,
    certs: &HashMap<MemberId, ResourceDefinition>,
    resource: MemberId,
    args: &[egg::Id],
    // `args` in recipe space (see `SlotWalk::arg_recipes`).
    arg_recipes: Option<&[Val]>,
    base: Heap,
    source: ValueSource,
    direction: Direction,
    scale: Option<Scale>,
    pc_lits: &[(egg::Id, Polarity)],
    // The guard under which the body boolean is discharged (asserted for
    // `Consume`, assumed for `Produce`). Usually `pc_lits`; an `inhale` passes
    // `[0 < scale]` since it carries no path condition.
    bool_guard: &[(egg::Id, Polarity)],
    // A **frame-only** exhale (a heap-dependent function call's implicit
    // precondition check): a bare-wildcard footprint slot skips building the
    // wildcard perm and subtracting — sufficiency becomes a "holds a positive
    // share" prove, presence is unconditional. Keeps the wildcard `ite` residue
    // (which `ite-reduce` churns on) out of the persistent e-graph. `false` for
    // the consuming/producing walks.
    //
    // This was `is_snap` while `PureInst::Snap` existed. Stage 3b turned that
    // into a frame-only exhale and the flag became a second name for the same
    // bit — its one `true` caller passes `frame_only` — so it is now that bit.
    frame_only: bool,
) -> Result<FootprintResult, VerifyError> {
    let def = certs.get(&resource).ok_or(VerifyError::DependencyFailed)?;

    let before = base.clone();
    let mut heap = base;
    let mut values: Vec<egg::Id> = Vec::with_capacity(def.footprint.len());
    let mut recipes: Vec<Option<Val>> = Vec::with_capacity(def.footprint.len());
    let mut locs: Vec<Option<Loc>> = Vec::with_capacity(def.footprint.len());
    let mut members = Vec::with_capacity(def.footprint.len());
    let mut changed = Vec::new();
    let walk = SlotWalk {
        resource,
        args,
        arg_recipes,
        source: &source,
        direction: &direction,
        scale,
        pc_lits,
        frame_only,
        before: &before,
    };
    for (i, slot) in def.footprint.iter().enumerate() {
        let SlotOut {
            value,
            recipe,
            member,
            loc,
        } = match &slot.shape {
            SlotShape::Single { addr, perm } => walk_single_slot(
                ctx,
                &walk,
                i,
                &slot.kind,
                &slot.elem,
                addr,
                perm,
                &values,
                &recipes,
                &mut heap,
                &mut changed,
            )?,
            SlotShape::Quant { q, caps } => {
                // The amount the produced set holds per binding is scaled by
                // `qp_scale`: `None` where the walk's scale is provably `1`.
                // Only a produce is scaled (`unfold acc(P(..), s)`, and an
                // `unfolding` in a function at `wildcard`); a wildcard body and
                // a consume at a scale (`fold`) are not supported yet.
                let qp_scale = match walk.scale {
                    None => None,
                    Some(s) => {
                        let pm = s.amount();
                        let is_one = expr!(ctx, { pm } == (1 / 1));
                        if ctx.prove_under_pc(is_one, walk.pc_lits) {
                            None
                        } else if matches!(walk.direction, Direction::Produce)
                            && !q.body.contains_wc_permission()
                        {
                            Some(s)
                        } else {
                            return Err(VerifyError::Unimplemented("scale not equal to 1/1 in QP"));
                        }
                    }
                };
                walk_qp_slot(
                    ctx,
                    &walk,
                    i,
                    &slot.elem,
                    q,
                    caps,
                    qp_scale,
                    &values,
                    &mut heap,
                    &mut changed,
                )?
            }
        };
        members.push(member);
        values.push(value);
        recipes.push(recipe);
        locs.push(loc);
    }
    // The body boolean over params ++ all slot values.
    // `bool_guard`, not `pc_lits`: it is the path condition at every call site
    // (an `inhale` extends it with `0 < scale`, the others pass `pc_lits`
    // itself), so it is the tighter of the two and always a superset.
    let bool_id = def.bool.build(
        &mut ctx.egraph,
        |r| r.resolve(args, &values),
        pc_lits,
        &mut changed,
    );
    match direction {
        // The precondition/predicate body must hold over the consumed values.
        Direction::Consume => {
            if !ctx.prove_under_pc(bool_id, bool_guard) {
                if std::env::var_os("SILVER_OXIDE_TRACE_ASSERT").is_some() {
                    eprintln!(
                        "[assert-fail] resource-bool consume, pc={} lits:\n{}",
                        bool_guard.len(),
                        crate::verify::viz::dump_term(ctx, bool_id, 40),
                    );
                }
                return Err(VerifyError::AssertionFailed);
            }
        }
        // The reconstructed body facts hold only where this walk is reached.
        Direction::Produce => {
            ctx.assume_guarded(bool_id, bool_guard.iter().rev().copied());
        }
        Direction::Build => {}
    }
    // Two slots that may hold one location need no fact here: the heap ties
    // their values itself, at a read that meets both or at an exhale that
    // takes from one (`Heap::consume`), as it does for any two holders.
    Ok(FootprintResult {
        heap,
        members,
        slot_recipes: recipes,
        slot_locs: locs,
    })
}

/// The call-site inputs shared by every slot of one [`walk_footprint`] (see
/// its parameters).
struct SlotWalk<'a> {
    resource: MemberId,
    args: &'a [egg::Id],
    /// `args` in recipe space: in a certificate walk where every argument has
    /// a recipe, else `None`.
    arg_recipes: Option<&'a [Val]>,
    source: &'a ValueSource,
    direction: &'a Direction,
    scale: Option<Scale>,
    pc_lits: &'a [(egg::Id, Polarity)],
    frame_only: bool,
    /// The heap the walk started from.
    before: &'a Heap,
}

/// What one walked footprint slot contributes: its value (seen by later slots
/// and the body boolean), that value's recipe (certificate walks only), and its
/// snapshot member `present ? Some(value) : None`.
struct SlotOut {
    value: egg::Id,
    recipe: Option<Val>,
    member: egg::Id,
    /// A single slot's location.
    loc: Option<Loc>,
}

/// Walk footprint slot `i`, a single `acc(addr, perm)`: rebuild its address and
/// permission from their recipes, take its value from `w.source`, and apply its
/// effect to `heap` (subtract for `Consume`, union for `Produce`, or only prove
/// a positive share for a wildcard slot at a frame-only exhale).
#[allow(clippy::too_many_arguments)]
fn walk_single_slot(
    ctx: &mut VerifyContext<'_>,
    w: &SlotWalk<'_>,
    i: usize,
    kind: &LocationKind,
    elem: &Type,
    addr_r: &BodyRecipe,
    perm_r: &PermRecipe,
    values: &[egg::Id],
    recipes: &[Option<Val>],
    heap: &mut Heap,
    changed: &mut Vec<egg::Id>,
) -> Result<SlotOut, VerifyError> {
    // Rebuild the slot's address and permission from their recipes, resolving
    // params to `args` and any earlier-slot value refs to `values` (so a
    // value-dependent inner address like `list(this.next)` resolves).
    // Heap framing is a *syntactic* e-class match (`heap_subtract`,
    // `ValueSource::ReadHeap`), so a slot address has to be in normal form
    // before we look it up: an address reaching through a snapshot
    // (`Pt(f(r, cons(Some(unwrap(proj_0(s))))))`) only becomes e-class-equal to
    // the held chunk's address once `proj∘cons` / `unwrap∘Some` have fired.
    //
    // Reduce only when the rebuild actually introduced something to reduce: the
    // recipe is add-only, so if `build` added no e-node then every term it named
    // was already present, hence already normalized. Reducing unconditionally
    // per slot re-runs the ADT rule set over the whole graph and costs ~3.5x.
    // A wildcard slot at a frame-only exhale (bare or gated): build its
    // **presence** indicator rather than the wildcard perm — the same shape with
    // the wildcard leaf built as full permission `1`, so `ite(guard, 1, 0)`
    // whose `0 < …` folds to the gating guard (see [`frame_wildcard_slot`]).
    let resolve = |r: &SeedRef| r.resolve(w.args, values);
    let wc_slot = w.frame_only && perm_r.has_wildcard();
    let before = ctx.egraph.total_number_of_nodes();
    let addr = addr_r.build(&mut ctx.egraph, resolve, w.pc_lits, changed);
    let wildcard_as = if wc_slot {
        WildcardAs::One
    } else {
        WildcardAs::Fresh
    };
    let perm = build_perm(ctx, perm_r, &resolve, w.pc_lits, changed, wildcard_as);
    if ctx.egraph.total_number_of_nodes() != before {
        ctx.reduce();
    }
    let addr = ctx.egraph.find(addr);
    // Values are read from the *original* heap (aliased slots agree); the hit
    // chunk's recipe provenance rides along. With a chunk set of this kind the
    // location is read through all its holders, as a `Deref` is, and the read
    // names a set's value by the slot address's recipe.
    let read_r = ReadRecipe {
        addr: slot_addr_recipe(ctx, w, addr_r, recipes),
    };
    let loc = Loc {
        kind: kind.clone(),
        addr,
    };
    let slot_loc = loc.clone();
    // A frame check's presence `0 < ite(guard, 1, 0)` (folds to the gating
    // guard). The slot's value matters only where it is present -- the member
    // is `present ? Some(v) : None` -- so it is read under the pc extended by
    // presence: a conditional precondition slot (`x != null ==> acc(P(x))`)
    // reached where the pc leaves `x != null` open is then still read from the
    // chunk that holds it wherever it is present.
    let presence = wc_slot.then(|| {
        let pp = perm.to_id(ctx);
        expr!(ctx, (0 / 1) < r { pp })
    });
    let read_pc: Vec<(egg::Id, Polarity)> = match presence {
        Some(g) => w
            .pc_lits
            .iter()
            .copied()
            .chain(std::iter::once((g, Polarity::Positive)))
            .collect(),
        None => w.pc_lits.to_vec(),
    };
    let (value, recipe) = match w.source {
        ValueSource::Stands(slots) => (stand_value(ctx, &slots[i], addr), None),
        _ => slot_value(ctx, w, i, elem, |ctx, h| {
            // The bodies of its chunk sets evaluated once already, at their inhales.
            let read = h
                .read(ctx, &loc, &read_pc, Some(&read_r))
                .unwrap_or_else(|e| panic!("quantified chunk body failed to re-evaluate: {e:?}"));
            match read {
                Some(v) => (v.id, v.recipe),
                None => (ctx.fresh_symbolic_value(elem.clone()), None),
            }
        }),
    };
    let present = if let Some(guard) = presence {
        // A frame check: no heap effect; presence is the gating guard
        // `0 < ite(guard, 1, 0)` (folds to `guard`, `true` when unconditional).
        if !matches!(w.direction, Direction::Build) {
            let access = Access::Single {
                loc,
                perm,
                held: None,
            };
            heap.consume(ctx, access, ConsumeMode::Frame, w.pc_lits)?;
        }
        guard
    } else {
        // A function's precondition is read-only: every amount is a wildcard.
        assert!(
            !matches!(w.direction, Direction::Build),
            "a snapshot is built only for a function precondition, whose slots are wildcards"
        );
        let value = Value {
            id: value,
            recipe: recipe.clone(),
        };
        let held = slot_held_recipe(ctx, w, perm_r, recipes);
        apply_slot_perm(ctx, w, loc, perm, held, value, heap)?
    };
    let member = ctx.option_member(elem.clone(), present, value);
    Ok(SlotOut {
        value,
        recipe,
        member,
        loc: Some(slot_loc),
    })
}

fn walk_qp_slot(
    ctx: &mut VerifyContext<'_>,
    w: &SlotWalk<'_>,
    i: usize,
    elem: &Type,
    q: &Rc<vmir::QuantChunks>,
    caps_r: &[(usize, BodyRecipe)],
    // The set's `Captures::scale`; `Some` only on a produce.
    scale: Option<Scale>,
    values: &[egg::Id],
    heap: &mut Heap,
    changed: &mut Vec<egg::Id>,
) -> Result<SlotOut, VerifyError> {
    debug_assert!(
        matches!(elem, Type::ValueMap(..)),
        "element type of a qp must be a value map"
    );
    let resolve = |r: &SeedRef| r.resolve(w.args, values);
    let before = ctx.egraph.total_number_of_nodes();
    let caps = caps_r
        .iter()
        .map(|(k, recipe)| {
            (
                *k,
                recipe.build(&mut ctx.egraph, resolve, w.pc_lits, changed),
            )
        })
        .collect::<Rc<_>>();

    if ctx.egraph.total_number_of_nodes() != before {
        ctx.reduce()
    }
    let caps = caps
        .iter()
        .map(|&(k, id)| (k, ctx.egraph.find(id)))
        .collect();
    // The body's heap reads, resolved against the caller's heap, which is the
    // one heap every footprint heap of the resource body stands for here: as
    // produced so far on a produce, as before the walk on a consume (an exhale
    // reads values in the state before it).
    let reads_from = match w.direction {
        Direction::Produce => heap.clone(),
        Direction::Consume | Direction::Build => w.before.clone(),
    };
    let mut heap_of = |_: &mut VerifyContext<'_>, _: &_| Ok(reads_from.clone());
    let mut caps = crate::verify::heap::resolve_reads(ctx, q, caps, &mut heap_of, w.pc_lits)?;
    caps.scale = scale;

    let value = match w.direction {
        Direction::Produce => {
            let (map, recipe) = slot_value(ctx, w, i, elem, |_, _| {
                unreachable!("ReadHeap only consumes")
            });
            // Well-formedness was proven by the certificate walk; the body
            // boolean assumed after the slots carries the facts it relied on.
            let spec = QuantSpec {
                q: q.clone(),
                caps,
                check: QpCheck::Established,
            };
            let value = Value { id: map, recipe };
            *heap = heap.produce(ctx, Access::Set(spec), value.clone(), w.pc_lits)?;
            value
        }
        Direction::Consume => {
            let spec = QuantSpec {
                q: q.clone(),
                caps,
                check: QpCheck::Injectivity,
            };
            let mode = if w.frame_only {
                ConsumeMode::Frame
            } else {
                ConsumeMode::Take
            };
            let (out, held) = heap.consume(ctx, Access::Set(spec), mode, w.pc_lits)?;
            *heap = out;
            held
        }
        // Its value is the covering set's map, found only by the containment
        // check this direction skips.
        Direction::Build => {
            return Err(VerifyError::Unimplemented(
                "heap-dependent call inside a quantifier whose precondition holds a quantified permission",
            ));
        }
    };
    let (map, recipe) = (value.id, value.recipe);

    let present = expr!(ctx, true);
    Ok(SlotOut {
        value: map,
        recipe,
        member: ctx.option_member(elem.clone(), present, map),
        loc: None,
    })
}

/// A footprint slot's value, by `w.source`: a fresh value for `Fresh`, and
/// `unwrap(proj_i(s))` for `ProjectSnap(s)` (mirrored into the recipe during a
/// certificate walk). `ReadHeap(h)` depends on the slot's shape, so the caller
/// supplies it as `read_heap`. Works for any `elem`, so a quantified slot (whose
/// value is its `ValueMap`) can share it.
fn slot_value(
    ctx: &mut VerifyContext<'_>,
    w: &SlotWalk<'_>,
    i: usize,
    elem: &Type,
    read_heap: impl FnOnce(&mut VerifyContext<'_>, &Heap) -> (egg::Id, Option<Val>),
) -> (egg::Id, Option<Val>) {
    match w.source {
        ValueSource::ReadHeap(h) => read_heap(ctx, h),
        // `proj_i(s)` recovers the optional member (collapsing to the `cons`
        // argument when `s` is concrete); `unwrap` peels to the field value.
        // A certificate walk mirrors the projection into the recipe.
        ValueSource::ProjectSnap(s, s_recipe) => {
            let proj_id = ctx.alloc.proj(w.resource, 0, i);
            let opt_ty = ctx.alloc.option_type(elem.clone());
            let opt = ctx.add_func_app_id(proj_id, Box::new([]), opt_ty, Box::new([*s]));
            let value = ctx.option_unwrap(elem.clone(), opt);
            let recipe = match (ctx.recipe.is_some(), s_recipe) {
                (true, Some(sr)) => {
                    use crate::verify::rewrite::AxiomPure;
                    let unwrap_id = ctx.alloc.option_value();
                    let rb = ctx.recipe.as_mut().unwrap();
                    let opt = rb.emit(AxiomPure::App {
                        func: proj_id,
                        type_args: Vec::new(),
                        args: vec![sr.clone()],
                    });
                    Some(rb.emit(AxiomPure::App {
                        func: unwrap_id,
                        type_args: vec![elem.clone()],
                        args: vec![opt],
                    }))
                }
                _ => None,
            };
            (value, recipe)
        }
        // Inhale: an unconstrained fresh value per slot.
        ValueSource::Fresh => (ctx.fresh_symbolic_value(elem.clone()), None),
        ValueSource::Stands(_) => unreachable!("stand-ins rebuild single slots only"),
    }
}

/// The recipe-space arguments of a resource call: in a certificate walk where
/// every argument has a recipe, else `None`.
fn arg_recipes(ctx: &VerifyContext<'_>, state: &EvalState, args: &[Val]) -> Option<Vec<Val>> {
    ctx.recipe.as_ref()?;
    args.iter().map(|v| state.recipe_of(v)).collect()
}

/// The recipe of `0 < perm` for a footprint slot produced by the walk `w`,
/// `perm_r` over the call's arguments and the earlier slots' values (whose
/// recipes are `recipes`), as `quant::presence` builds it for a snapshot
/// member. `None` outside a certificate walk and when an operand has no
/// recipe.
///
/// A walk that scales the slot by `s` holds `s · p`, which is positive exactly
/// where `p` is when `s` is positive (amounts are never negative): a
/// `wildcard` scale (an `unfolding` in a function body) or a positive literal.
/// Any other scale leaves the slot without a recipe, since naming the wrong
/// positivity would let a summed value name the wrong slot.
fn slot_held_recipe(
    ctx: &mut VerifyContext<'_>,
    w: &SlotWalk<'_>,
    perm_r: &PermRecipe,
    recipes: &[Option<Val>],
) -> Option<Val> {
    if !matches!(w.direction, Direction::Produce) {
        return None;
    }
    let args = w.arg_recipes?;
    let scale_positive = w.scale.is_none_or(|s| match s {
        Scale::Wildcard(_) => true,
        Scale::Amount(pm) => matches!(
            ctx.egraph[ctx.egraph.find(pm)].data.known(),
            Some(vmir::Literal::Real(r)) if *r > num::BigRational::from(num::BigInt::from(0))
        ),
    });
    if !scale_positive {
        return None;
    }
    let values: Vec<Val> = recipes.iter().cloned().collect::<Option<_>>()?;
    let rb = ctx.recipe.as_mut()?;
    let mut steps = Vec::new();
    let first = rb.next_index();
    // An operand the splice cannot carry leaves the slot without one: the sum
    // then names no value, as before.
    let present =
        crate::verify::quant::presence(&mut steps, first, perm_r, &perm_r.res, args, &values)
            .ok()?;
    rb.emit_steps(steps);
    // No condition: present wherever the slot is (a `wildcard`, or `1/1`).
    Some(present.unwrap_or(vmir::TRUE))
}

/// The recipe of a footprint slot's address, `addr_r` spliced over the call's
/// arguments and the earlier slots' values (whose recipes are `recipes`).
/// `None` outside a certificate walk and when an operand has no recipe.
fn slot_addr_recipe(
    ctx: &mut VerifyContext<'_>,
    w: &SlotWalk<'_>,
    addr_r: &BodyRecipe,
    recipes: &[Option<Val>],
) -> Option<Val> {
    let args = w.arg_recipes?;
    let values: Vec<Val> = recipes.iter().cloned().collect::<Option<_>>()?;
    let rb = ctx.recipe.as_mut()?;
    let mut steps = Vec::new();
    let first = rb.next_index();
    // An operand the splice cannot carry leaves the address without one: a
    // read from a chunk set then names no value, as before.
    let addr = crate::verify::quant::splice(&mut steps, first, addr_r, args, &values).ok()?;
    rb.emit_steps(steps);
    Some(addr)
}

/// Apply a single slot `loc ↦ value` to `heap` per `w.direction`: take it
/// (`Consume`) or add it (`Produce`), with `perm` scaled by `w.scale` when set.
/// Returns the slot's presence `0 < perm` (unscaled).
fn apply_slot_perm(
    ctx: &mut VerifyContext<'_>,
    w: &SlotWalk<'_>,
    loc: Loc,
    perm: ChunkPerm,
    held: Option<Val>,
    value: Value,
    heap: &mut Heap,
) -> Result<egg::Id, VerifyError> {
    let scaled = match w.scale {
        Some(s) => perm.clone().scaled(ctx, s),
        None => perm.clone(),
    };
    let access = Access::Single {
        loc,
        perm: scaled,
        held,
    };
    *heap = match w.direction {
        Direction::Consume => {
            let (out, taken) = heap.consume(ctx, access, ConsumeMode::Take, w.pc_lits)?;
            // The slot value was read before (from the walk's source heap) and
            // the consume read the same location of the same state: they agree.
            if ctx.egraph.find(taken.id) != ctx.egraph.find(value.id) {
                let eq = expr!(ctx, { taken.id } == { value.id });
                ctx.assume_guarded(eq, w.pc_lits.iter().rev().copied());
            }
            out
        }
        Direction::Produce => heap.produce(ctx, access, value, w.pc_lits)?,
        Direction::Build => unreachable!("a built snapshot has no heap effect"),
    };
    let perm_id = perm.to_id(ctx);
    Ok(expr!(ctx, (0 / 1) < r { perm_id }))
}

/// The `cons` of a resource's snapshot from its per-slot `members`
/// (`present ? Some(v) : None`). Predicate snapshots are single-variant,
/// non-generic ADTs headed by the resource id.
fn build_snapshot(
    ctx: &mut VerifyContext<'_>,
    resource: MemberId,
    members: Vec<egg::Id>,
) -> egg::Id {
    let cons = ctx.alloc.cons(resource, 0);
    ctx.add_func_app_id(cons, Box::new([]), Type::Snap(resource), members.into())
}

/// Evaluate a **value-yielding** `Sub`: remove the chunk at `loc` and hand back
/// what was there, as `Option<T>` (`None` iff nothing was removed).
///
/// This is the consume half of a desugared `unfold`. Unlike the plain `Sub` in
/// [`eval_heap_inst`], which subtracts a freshly-minted chunk value, this reads
/// the **held** chunk's value -- that value *is* the predicate's snapshot, and
/// sharing it with the paired `inhale` is what makes the fold/unfold round-trip
/// hold. In a certificate walk the hit chunk's recipe rides along, so the
/// reconstructed slots purify to `unwrap(proj_i(s))` over it.
fn eval_sub_yield(
    ctx: &mut VerifyContext<'_>,
    state: &mut EvalState,
    inst: &Inst,
    pc: &PathConds,
) -> Result<(), VerifyError> {
    let InstKind::Heap(HeapInst::Sub {
        base, loc, perm, ..
    }) = &inst.kind
    else {
        unreachable!("eval_sub_yield called on a non-yielding-Sub instruction");
    };
    let region = state.region(ctx);
    let base_h = get_heap(ctx, state, base, &region)?;
    let l = state
        .loc(ctx, loc)
        .expect("acc location must be Addr-typed");
    let kind = l.kind.clone();
    let cperm = eval_perm_structural(ctx, state, perm);
    let pc_lits = pc_lits_raw(ctx, state, pc);
    // The consume hands back the value it took, read from the holders whose
    // permission it took, and (certificate walks) its provenance.
    let access = Access::Single {
        loc: l,
        perm: cperm.clone(),
        held: None,
    };
    let (out, taken) = base_h.consume(ctx, access, ConsumeMode::Take, &pc_lits)?;
    if ctx.recipe.is_some() && taken.recipe.is_none() {
        return Err(VerifyError::Unimplemented(
            "purify: consume of an unheld location",
        ));
    }
    let (held, held_recipe) = (taken.id, taken.recipe);
    state.push_heap(out);

    // Presence is `0 < perm`, built from the permission the instruction names.
    // For a literal amount it const-folds to `true` and `option_member` collapses
    // to a bare `Some`, so the paired `inhale`'s unwrap peels with no proof goal.
    let perm_id = cperm.to_id(ctx);
    let present = expr!(ctx, (0 / 1) < r { perm_id });
    let elem = kind.value.clone();
    let opt = ctx.option_member(elem.clone(), present, held);
    // The recipe must purify the value that is *pushed*, which is the option --
    // not the held value inside it. Handing the held value's recipe straight
    // through would leave the paired `OptionUnwrap` emitting `unwrap(held)`, a
    // stray peel with no `Some` under it: sound in the e-graph, where the
    // instruction's own `option_member` reduces, but dangling in a certificate
    // replayed at a client, where it silently severs the fold/unfold identity.
    // Wrapping here keeps the pair `unwrap(Some(r))`, which reduces by the same
    // rule on both paths.
    let opt_recipe = match (ctx.recipe.is_some(), &held_recipe) {
        (true, Some(hr)) => {
            let some_id = ctx.alloc.option_some();
            let rb = ctx.recipe.as_mut().unwrap();
            Some(rb.emit(crate::verify::rewrite::AxiomPure::App {
                func: some_id,
                type_args: vec![elem.clone()],
                args: vec![hr.clone()],
            }))
        }
        _ => None,
    };
    state.push_val(opt, Type::Option(Box::new(elem)), opt_recipe);
    Ok(())
}

/// Evaluate a resource `inhale` / `exhale`. `base inhale R(args) p` produces the
/// resource's footprint into `base` and **assumes** its boolean; `base exhale ..`
/// consumes it and **asserts** it. A self-framed callee additionally yields the
/// snapshot of its footprint as a pure `Val`.
///
/// Shared by the method walk and the resource/function-body walks: a
/// **frame-only** exhale (a heap-dependent function call's implicit precondition
/// check) occurs inside contract and function bodies, where a *consuming*
/// resource op would rightly be rejected.
fn eval_resource_op(
    ctx: &mut VerifyContext<'_>,
    program: &vmir::Program,
    state: &mut EvalState,
    inst: &Inst,
    certs: &HashMap<MemberId, ResourceDefinition>,
) -> Result<(), VerifyError> {
    let InstKind::Heap(
        hi @ (HeapInst::Inhale {
            base, call, perm, ..
        }
        | HeapInst::Exhale {
            base, call, perm, ..
        }),
    ) = &inst.kind
    else {
        unreachable!("eval_resource_op called on a non-resource-op instruction");
    };

    let is_inhale = matches!(hi, HeapInst::Inhale { .. });
    // A frame-only exhale is the implicit precondition check of a
    // heap-dependent function call: sufficiency is proven on a scratch
    // subtraction chain (so aliased slots require their sum) whose result
    // is discarded, and a bare-wildcard slot takes the presence path
    // rather than materializing a wildcard permission.
    let frame_only = matches!(
        hi,
        HeapInst::Exhale {
            frame_only: true,
            ..
        }
    );
    let pc_lits = collect_pc_lits(ctx, state, &inst.pc);
    let region = state.region(ctx);
    let base_h = get_heap(ctx, state, base, &region)?;
    let args: Vec<egg::Id> = call.args.iter().map(|v| state.get_val(ctx, v)).collect();
    let scale = eval_perm(ctx, state, perm);
    // Inhale: produce fresh chunks, assume the bool guarded by `0 < scale`
    // (it carries no path condition — the branch lives in the perm scale).
    // Exhale: consume the held chunks, assert the bool under `pc`.
    let (source, direction, bool_guard) = if is_inhale {
        // The bind is the value source: `Fresh` havocs each slot, while
        // `Bound(s)` recovers it as `unwrap(proj_i(s))` so this heap and
        // any other reconstruction from `s` name the *same* terms.
        let source = match hi {
            HeapInst::Inhale {
                bind: Bind::Bound(v),
                ..
            } => {
                // Always a plain `Snap`: a desugared `unfold` names the
                // `PureInst::OptionUnwrap` temp, not the `Option` itself, so
                // the seam is explicit in the instruction stream.
                let sv = state.get_val(ctx, v);
                ValueSource::ProjectSnap(sv, state.recipe_of(v))
            }
            HeapInst::Inhale {
                bind: Bind::SelfSlot,
                ..
            } => {
                return Err(VerifyError::Unimplemented(
                    "`with self` on a resource inhale",
                ));
            }
            _ => ValueSource::Fresh,
        };
        let pos = expr!(ctx, (0 / 1) < r { scale });
        let mut guard = vec![(pos, Polarity::Positive)];
        // Fork model: arms run unguarded (the branch no longer rides in
        // the perm scale), so the block cube must guard the inhaled bool
        // — otherwise a conditional `inhale` on one arm leaks its fact
        // past the branch.
        guard.extend_from_slice(&pc_lits);
        (source, Direction::Produce, guard)
    } else {
        (
            ValueSource::ReadHeap(base_h.clone()),
            Direction::Consume,
            pc_lits.clone(),
        )
    };
    let FootprintResult {
        heap: out,
        members,
        slot_recipes,
        ..
    } = walk_footprint(
        ctx,
        certs,
        call.resource,
        &args,
        arg_recipes(ctx, state, &call.args).as_deref(),
        base_h,
        source,
        direction,
        // A frame check does not scale: it reads the footprint at the
        // callee's own slot permissions, exactly as the dedicated `Snap`
        // did. Scaling by the (unused) `1/1` operand would rebuild every
        // slot permission as `1 * p`, which is equal but not identical,
        // and identity is what the `old(f(x)) == f(x)` congruence needs.
        if frame_only {
            None
        } else if state.perm_is_wild(perm) {
            Some(Scale::Wildcard(scale))
        } else {
            Some(Scale::Amount(scale))
        },
        &pc_lits,
        &bool_guard,
        frame_only,
    )?;
    if hi.produces_heap() {
        state.push_heap(out);
    }
    if let Some(res_id) = hi.snap_yield(&program.decls) {
        let s = build_snapshot(ctx, res_id, members);
        // A certificate walk mirrors the snapshot member by member (see
        // [`snapshot_recipe`]). Only a frame-only exhale (a function's
        // precondition check) is ever reached during a recipe walk; a
        // consuming inhale/exhale is method-only, where no recipe is in
        // flight and this yields `None` as before.
        let arg_recipes: Vec<Option<Val>> = call.args.iter().map(|v| state.recipe_of(v)).collect();
        let recipe = snapshot_recipe(ctx, certs, res_id, &arg_recipes, &slot_recipes, &pc_lits)?;
        state.push_val(s, Type::Snap(res_id), recipe);
    }
    Ok(())
}

/// The certificate recipe of a frame-only exhale's snapshot (`None` outside a
/// certificate walk), mirroring the snapshot [`walk_single_slot`] builds: per
/// slot `present ? Some(v_i) : None` over the read slot value's recipe, where
/// `present` is the slot's permission recipe over the call's `arg_recipes` and
/// the earlier slot values (`wildcard` read as present, as
/// `quant::presence` does for a quantifier body's call). A conditional
/// precondition (`x != null ==> acc(P(x))`) thus purifies to the same member
/// a caller's snapshot holds on both sides of its condition. Only a
/// frame-only exhale (a function's precondition check) is ever reached during
/// a recipe walk; a consuming inhale/exhale is method-only, where no recipe is
/// in flight.
fn snapshot_recipe(
    ctx: &mut VerifyContext<'_>,
    certs: &HashMap<MemberId, ResourceDefinition>,
    res_id: MemberId,
    arg_recipes: &[Option<Val>],
    slot_recipes: &[Option<Val>],
    pc_lits: &[(egg::Id, Polarity)],
) -> Result<Option<Val>, VerifyError> {
    use crate::verify::rewrite::{AxiomInst, AxiomPure};
    if ctx.recipe.is_none() {
        return Ok(None);
    }
    let def = certs.get(&res_id).ok_or(VerifyError::DependencyFailed)?;
    let some_id = ctx.alloc.option_some();
    let cons_id = ctx.alloc.cons(res_id, 0);
    // A slot read without provenance on a path the walk proves dead
    // (e.g. under `unfolding` of a predicate whose body is `false` — a
    // Prusti enum variant that cannot occur) may take any value: the
    // definitional axiom's ite never selects this branch. `unwrap(None)`
    // is that unspecified value, as a term the certificate can carry.
    let dead_path = slot_recipes.iter().any(|r| r.is_none()) && {
        let f = expr!(ctx, false);
        ctx.is_inconsistent() || ctx.prove_under_pc(f, pc_lits)
    };
    let (none_id, value_id) = (ctx.alloc.option_none(), ctx.alloc.option_value());
    let rb = ctx.recipe.as_mut().unwrap();
    let mut values: Vec<Val> = Vec::with_capacity(slot_recipes.len());
    let mut members_r = Vec::with_capacity(slot_recipes.len());
    for (slot, r) in def.footprint.iter().zip(slot_recipes) {
        let elem = slot.elem.clone();
        let v = match r.clone() {
            Some(v) => v,
            None if dead_path => {
                let none = rb.emit(AxiomPure::App {
                    func: none_id,
                    type_args: vec![elem.clone()],
                    args: Vec::new(),
                });
                rb.emit(AxiomPure::App {
                    func: value_id,
                    type_args: vec![elem.clone()],
                    args: vec![none],
                })
            }
            None => {
                return Err(VerifyError::Unimplemented(
                    "purify: snap value outside footprint",
                ));
            }
        };
        let some = rb.emit(AxiomPure::App {
            func: some_id,
            type_args: vec![elem.clone()],
            args: vec![v.clone()],
        });
        // A quantified slot is always present (see `walk_qp_slot`).
        let present = match &slot.shape {
            SlotShape::Single { perm, .. } if !matches!(perm.res, vmir::PermVal::Wildcard) => {
                let args = arg_recipes
                    .iter()
                    .cloned()
                    .collect::<Option<Vec<Val>>>()
                    .ok_or(VerifyError::Unimplemented(
                        "purify: conditional precondition slot over an argument without recipe",
                    ))?;
                let mut steps: Vec<AxiomInst> = Vec::new();
                let first = rb.next_index();
                let p = crate::verify::quant::presence(
                    &mut steps, first, perm, &perm.res, &args, &values,
                )?;
                rb.emit_steps(steps);
                p
            }
            _ => None,
        };
        let member = match present {
            None => some,
            Some(present) => {
                let none = rb.emit(AxiomPure::App {
                    func: none_id,
                    type_args: vec![elem],
                    args: Vec::new(),
                });
                rb.emit(AxiomPure::Ternary(present, some, none))
            }
        };
        values.push(v);
        members_r.push(member);
    }
    Ok(Some(rb.emit(AxiomPure::App {
        func: cons_id,
        type_args: Vec::new(),
        args: members_r,
    })))
}

/// What [`frame_snapshot`] builds a frame-only exhale's snapshot from.
pub(crate) enum SnapshotFrom {
    /// The heap, checking framing and the precondition under the given pc,
    /// as at a call.
    Check(Heap),
    /// What stands for each footprint slot of a quantifier body's call (see
    /// [`ValueSource::Stands`]); nothing to check.
    Stands(std::rc::Rc<[crate::verify::quant::ResolvedSlot]>),
}

/// A frame-only exhale's snapshot (see [`frame_snapshot`]).
pub(crate) struct FrameSnapshot {
    pub(crate) snap: egg::Id,
    /// Each single footprint slot's location; `None` for a quantified one.
    pub(crate) locs: Vec<Option<Loc>>,
}

/// The snapshot a frame-only exhale of `resource(args)` yields (a
/// heap-dependent call's). A quantifier body's call is checked per binding,
/// and its snapshot rebuilt per binding from what stands for its slots (see
/// `heap::ReadResolver`).
pub(crate) fn frame_snapshot(
    ctx: &mut VerifyContext<'_>,
    resource: MemberId,
    args: &[egg::Id],
    from: SnapshotFrom,
    pc: &[(egg::Id, Polarity)],
) -> Result<FrameSnapshot, VerifyError> {
    let certs = ctx
        .certs
        .expect("a verification context carries the certificates");
    let (heap, source, direction) = match from {
        SnapshotFrom::Check(h) => (h.clone(), ValueSource::ReadHeap(h), Direction::Consume),
        SnapshotFrom::Stands(slots) => {
            (Heap::empty(), ValueSource::Stands(slots), Direction::Build)
        }
    };
    let FootprintResult {
        members, slot_locs, ..
    } = walk_footprint(
        ctx, certs, resource, args, None, heap, source, direction, None, pc, pc, true,
    )?;
    let snap = build_snapshot(ctx, resource, members);
    Ok(FrameSnapshot {
        snap,
        locs: slot_locs,
    })
}

/// A footprint slot's value at `addr` from what stands for it.
fn stand_value(
    ctx: &mut VerifyContext<'_>,
    slot: &crate::verify::quant::ResolvedSlot,
    addr: egg::Id,
) -> egg::Id {
    use crate::verify::quant::Stand;
    match slot.stand {
        Stand::Value(v) => v,
        Stand::Map(m) => {
            let lookup = ctx.alloc.value_map(&slot.kind).lookup;
            ctx.add(Symbolic::FuncApp(lookup, Box::new([]), Box::new([m, addr])))
        }
    }
}

/// Evaluate a a bound `inhale`: widen a snapshot value back into a heap — the entry
/// of a heap-dependent function body reconstructing its precondition heap from
/// the snapshot parameter. Inverse of [`eval_snap`], inhale-shaped: one chunk
/// per footprint slot at the grafted address with the footprint permission and
/// value `unwrap(proj_i(s))` (as in `unfold`), and the resource's boolean is
/// **assumed** over the projected values. Returns the reconstructed heap.
fn eval_from_snap(
    ctx: &mut VerifyContext<'_>,
    state: &EvalState,
    inst: &Inst,
    certs: &HashMap<MemberId, ResourceDefinition>,
) -> Result<Heap, VerifyError> {
    let InstKind::Heap(HeapInst::Inhale {
        bind: Bind::Bound(snap),
        call,
        ..
    }) = &inst.kind
    else {
        unreachable!("eval_from_snap called on a non-bound-inhale instruction");
    };
    let (resource, args) = (&call.resource, &call.args);
    let arg_ids: Vec<egg::Id> = args.iter().map(|v| state.get_val(ctx, v)).collect();
    let s = state.get_val(ctx, snap);
    // The snapshot parameter's recipe — each reconstructed slot's provenance
    // becomes `unwrap(proj_i(s))` over it (certificate walks only).
    let s_recipe = if ctx.recipe.is_some() {
        Some(state.require_recipe(snap, OPERAND_RECIPE)?)
    } else {
        None
    };
    let pc_lits = collect_pc_lits(ctx, state, &inst.pc);

    // Reconstruct the precondition heap: produce one chunk per footprint slot
    // valued `unwrap(proj_i(s))` into the empty heap, assuming the resource body
    // (guarded by the path condition — a a bound `inhale` may sit under a branch).
    let FootprintResult { heap: out, .. } = walk_footprint(
        ctx,
        certs,
        *resource,
        &arg_ids,
        arg_recipes(ctx, state, args).as_deref(),
        Heap::empty(),
        ValueSource::ProjectSnap(s, s_recipe),
        Direction::Produce,
        None,
        &pc_lits,
        &pc_lits,
        false,
    )?;
    ctx.reduce();
    Ok(out)
}

/// Whether the program has a collection of `kind` anywhere: a value of a type
/// that mentions one (a parameter, a local, a field read, an operation's
/// result) or an operation on one (`|s|` is an `Int` over one). Every
/// collection the program can build or compare is one of these.
/// The location kinds of every quantified permission in the program, in first
/// appearance order and without duplicates: the kinds whose value map needs
/// its axiom.
fn decls_qp_kinds(
    decls: &typed_index_collections::TiVec<MemberId, vmir::Declaration>,
) -> Vec<LocationKind> {
    let mut out: Vec<LocationKind> = Vec::new();
    let mut scan = |insts: &[vmir::Inst]| {
        for i in insts {
            if let InstKind::Heap(HeapInst::AddForall { q, .. } | HeapInst::SubForall { q, .. }) =
                &i.kind
            {
                let kind = LocationKind::from_addr_type(&q.loc_ty).expect("QP loc_ty must be Addr");
                if !out.contains(&kind) {
                    out.push(kind);
                }
            }
        }
    };
    for d in decls.iter() {
        match d {
            vmir::Declaration::Method(m) => scan(&m.flatten()),
            vmir::Declaration::Resource(r) => scan(&r.body.insts),
            vmir::Declaration::Function(f) => {
                if let Some(b) = &f.body {
                    scan(&b.insts)
                }
            }
            _ => {}
        }
    }
    out
}

/// The address functions (field and predicate locations, applied with an
/// address result type) anywhere in the program: method, resource and function
/// bodies and the bodies of quantified permissions. Only these are injective;
/// a resource's member also names its boolean contract function, which is why
/// the result type decides.
fn decls_addr_funcs(
    decls: &typed_index_collections::TiVec<MemberId, vmir::Declaration>,
) -> crate::dhash::HashSet<crate::verify::lang::FuncId> {
    fn scan(insts: &[vmir::Inst], out: &mut crate::dhash::HashSet<crate::verify::lang::FuncId>) {
        for i in insts {
            match &i.kind {
                InstKind::Pure(Type::Addr { .. }, PureInst::FunctionCall(fc)) => {
                    out.insert(crate::verify::func_registry::func_id_for_member(
                        fc.function,
                    ));
                }
                InstKind::Heap(HeapInst::AddForall { q, .. } | HeapInst::SubForall { q, .. }) => {
                    scan(&q.body.insts, out)
                }
                _ => {}
            }
        }
    }
    let mut out = crate::dhash::HashSet::default();
    for d in decls.iter() {
        match d {
            vmir::Declaration::Method(m) => scan(&m.flatten(), &mut out),
            vmir::Declaration::Resource(r) => scan(&r.body.insts, &mut out),
            vmir::Declaration::Function(f) => {
                if let Some(b) = &f.body {
                    scan(&b.insts, &mut out)
                }
            }
            _ => {}
        }
    }
    out
}

fn decls_use_coll(
    decls: &typed_index_collections::TiVec<MemberId, vmir::Declaration>,
    kind: CollKind,
) -> bool {
    fn mentions(ty: &Type, kind: CollKind) -> bool {
        match ty {
            Type::Coll(k, tys) => *k == kind || tys.iter().any(|t| mentions(t, kind)),
            Type::Domain(_, args) => args.iter().any(|a| mentions(a, kind)),
            Type::Option(t) => mentions(t, kind),
            Type::Addr { value, .. } => mentions(value, kind),
            _ => false,
        }
    }
    // Recurses into quantifier bodies (a pure `forall`'s, a quantified
    // permission's) and their binder types: a collection used only there still
    // needs its axioms.
    fn insts_use(insts: &[vmir::Inst], kind: CollKind) -> bool {
        insts.iter().any(|i| match &i.kind {
            InstKind::Pure(ty, pi) => {
                mentions(ty, kind)
                    || match pi {
                        PureInst::Coll { op, tys, .. } => {
                            op.kind() == kind || tys.iter().any(|t| mentions(t, kind))
                        }
                        PureInst::CollLit {
                            kind: k, elem_ty, ..
                        } => *k == kind || mentions(elem_ty, kind),
                        PureInst::Forall(q) => {
                            q.bound.iter().any(|t| mentions(t, kind))
                                || insts_use(&q.body.insts, kind)
                        }
                        _ => false,
                    }
            }
            InstKind::Heap(HeapInst::AddForall { q, .. } | HeapInst::SubForall { q, .. }) => {
                q.bound.iter().any(|t| mentions(t, kind)) || insts_use(&q.body.insts, kind)
            }
            _ => false,
        })
    }
    let insts_use = |insts: &[vmir::Inst]| insts_use(insts, kind);
    decls.iter().any(|d| match d {
        vmir::Declaration::Method(m) => insts_use(&m.flatten()),
        vmir::Declaration::Function(f) => {
            f.params.iter().any(|t| mentions(t, kind))
                || mentions(&f.ret, kind)
                || f.body.as_ref().is_some_and(|b| insts_use(&b.insts))
        }
        vmir::Declaration::Resource(r) => {
            r.params.iter().any(|t| mentions(t, kind)) || insts_use(&r.body.insts)
        }
        vmir::Declaration::Axiom(a) => insts_use(&a.body.insts),
        _ => false,
    })
}

/// Axiomatize the state: make every domain axiom's fact available to the unit
/// about to be verified. An axiom body is evaluated into the e-graph and its
/// boolean merged with `true` up front; quantification over *values* is a
/// `forall` in the body, which becomes its own lazy-instantiation rule below.
/// A **generic** axiom is evaluated the same way: its body is one root `forall`
/// binding its types ([`vmir::Forall::ty_binders`]; a wrapper binding only types
/// when translation inferred its triggers), stated with them open in the node's
/// payload and fixed per instance by the trigger terms it matches
/// (`rewrite::forall_rule`) — there is no up-front type instantiation.
/// Axiom bodies are trusted — no obligations (div-by-zero, deref permission) are
/// checked on them.
///
/// Invariant: the eager ground-axiom evaluation below only ever sees **nullary**
/// quantifier occurrences — axioms are closed and `let` is rejected in pure
/// lowering, so an axiom's top-level `forall`s capture nothing. Occurrences with
/// captures enter the e-graph when an outer instance's body is built
/// (`rewrite::build_instance`) or while evaluating a hosting method/resource body,
/// never eagerly here.
fn assume_axioms(ctx: &mut VerifyContext<'_>, program: &vmir::Program) -> Result<(), VerifyError> {
    // One rule instantiates every `forall` in the program: quantifiers are
    // e-nodes (data), not rules, so one generic rule suffices — and a `forall`
    // materialized *during* saturation (by an outer instantiation, or by a
    // certificate graft) is picked up on the next iteration, which per-quantifier
    // rules structurally could not do (egg forbids mid-run rule injection).
    // Registered unconditionally: the table is filled lazily, so "empty right now"
    // says nothing about whether this unit will state a quantifier. The rule costs
    // nothing until a `Forall` node exists — its searcher scans one
    // `classes_by_op` bucket.
    let table = std::sync::Arc::clone(ctx.alloc.quant_table());
    ctx.axiom_rules
        .push(crate::verify::rewrite::forall_rule(table));
    for decl in program.decls.iter() {
        let vmir::Declaration::Axiom(ax) = decl else {
            continue;
        };
        let mut state = EvalState::new();
        for inst in &ax.body.insts {
            match &inst.kind {
                InstKind::Pure(ty, pi) => {
                    // Axiom bodies are heap-free and trusted — no `Deref`, so the
                    // path condition is irrelevant here.
                    let (id, _) = eval_pure_inst(ctx, &state, ty, pi, &[], None)?;
                    state.push_val(id, ty.clone(), None);
                }
                InstKind::Assume(val) => {
                    let id = state.get_val(ctx, val);
                    let true_ = expr!(ctx, true);
                    ctx.union(id, true_);
                    ctx.egraph.rebuild();
                }
                // An axiom body is never verified — a stray obligation
                // (there are none today: callees are precondition-free)
                // would be skipped, and heap insts cannot occur.
                InstKind::Assert(_) => {}
                _ => return Err(VerifyError::Unimplemented("non-pure inst in axiom body")),
            }
        }
        let res = state.get_val(ctx, &ax.body.res);
        let true_ = expr!(ctx, true);
        ctx.union(res, true_);
        ctx.egraph.rebuild();
    }
    // The built-in collection axioms (`verify::builtin_axioms`): each a closed
    // `forall` node, true, whose element type stays open for its trigger
    // matches to fix. Only a program that has collections of a kind gets that
    // kind's axioms, so others pay nothing. A map brings the set axioms too:
    // its axioms are about the sets `domain(m)` and `range(m)`.
    let uses = |kind| decls_use_coll(&program.decls, kind);
    for kind in CollKind::ALL {
        let needed = uses(kind) || (kind == CollKind::Set && uses(CollKind::Map));
        if !needed {
            continue;
        }
        let ids = ctx.alloc.coll_axioms(kind);
        // The witnesses' type, for type inference (`types::infer_type`).
        for (f, ret) in ctx.alloc.skolem_ret_types() {
            ctx.func_ret_types.entry(f).or_insert(ret);
        }
        for rid in ids {
            let tys: Box<[Type]> = ctx
                .alloc
                .quant_table()
                .read()
                .expect("recipe table lock")
                .get(rid)
                .open_types()
                .into();
            let q = ctx.add(Symbolic::Forall(rid, tys, Box::new([])));
            let true_ = expr!(ctx, true);
            ctx.union(q, true_);
        }
        ctx.egraph.rebuild();
    }
    // The axioms of each value map a quantified permission uses
    // (`verify::builtin_axioms::value_map`): closed and ground, true on
    // every path, so assumed unguarded like the collection axioms.
    let qp_kinds = decls_qp_kinds(&program.decls);
    for kind in &qp_kinds {
        for rid in ctx.alloc.value_map_axioms(kind) {
            let q = ctx.add(Symbolic::Forall(rid, Box::new([]), Box::new([])));
            let true_ = expr!(ctx, true);
            ctx.union(q, true_);
        }
    }
    if !qp_kinds.is_empty() {
        for (f, ret) in ctx.alloc.skolem_ret_types() {
            ctx.func_ret_types.entry(f).or_insert(ret);
        }
        ctx.egraph.rebuild();
    }
    // Field and predicate addresses are injective
    // (`rewrite::address_injectivity_rule`): `a != b` decides `f(a) != f(b)`,
    // which value maps (a read after a write), single takes from a chunk set
    // and the aliasing gates of single chunks compare.
    let addr_funcs = decls_addr_funcs(&program.decls);
    if !addr_funcs.is_empty() {
        ctx.axiom_rules
            .push(crate::verify::rewrite::address_injectivity_rule(addr_funcs));
    }
    // One lazy unfold rule per already-verified function certificate. `analyze`
    // guarantees a function only calls functions verified earlier, so every
    // function this unit could reference already has a cert in `fn_certs`. Each
    // rule rebuilds the recipe's body lazily (add-only) when a `FuncApp(f, ..)`
    // occurrence is seen during saturation (see `rewrite::function_rule`).
    if let Some(fn_certs) = ctx.fn_certs {
        // Sorted: `fn_certs` is a `HashMap`, whose iteration order varies per
        // process, and rule order decides which of several equivalent unfoldings
        // saturation reaches first — hence the peak-node count. Without this the
        // `verification_cost_matches_baseline` gate reads a different number on
        // every run.
        let mut certs: Vec<_> = fn_certs.iter().collect();
        certs.sort_unstable_by_key(|&(&id, _)| id);
        for (&id, def) in certs {
            let name = ctx.member_name(id);
            let func = crate::verify::func_registry::func_id_for_member(id);
            // The uniform `f%pre` presence trigger gates every body-unfold: the
            // rule fires only where a `FuncApp(f%pre, args)` node was minted (a
            // value-position call, or a propagation step in an unfolded body).
            //
            // A **contract** function (`#requires` / `#ensures`) is no different —
            // what distinguishes it is only *where it occurs*. A `#requires` occurs
            // as a call (the entry `assume`, a call-site obligation), so the
            // `FunctionCall` arm mints its token like any other callee's. A
            // `#ensures` occurs only inside a post, which reaches a caller through
            // [`post_fact`] and never through a call — so its token is released by
            // the export instead ([`PostRecipe::token`]).
            let pre_token = ctx.alloc.fn_pre_token(id, &name);
            ctx.axiom_rules.push(crate::verify::rewrite::function_rule(
                &name,
                func,
                std::sync::Arc::clone(def),
                pre_token,
            ));
            // A recursive function's post fact also triggers on its limited
            // twin `f'` — that is what delivers the postcondition at a
            // recursive unroll (the twin has no unfold rule by design). Behind
            // the same `f%pre` token, which the unrolled body propagates for its
            // recursive call.
            if def.limited.is_some() && def.post.is_some() {
                ctx.axiom_rules
                    .push(crate::verify::rewrite::function_post_rule(
                        &name,
                        std::sync::Arc::clone(def),
                        pre_token,
                    ));
            }
        }
    }
    Ok(())
}

/// Resolve a trigger pattern term's heads to what the e-graph speaks — the same
/// mapping [`prepare_body`] applies to the corresponding `PureInst`, so a pattern
/// matches exactly the nodes a body would build — and canonicalize its variables
/// into recipe space: at or above `binder_base` a variable is a binder (`Bound`),
/// below it a free enclosing value whose capture slot is `slot`.
pub(crate) fn prepare_trig_term(
    alloc: &mut crate::verify::func_registry::FuncRegistry,
    term: &vmir::TrigTerm,
    binder_base: usize,
    slot: &crate::dhash::HashMap<usize, usize>,
) -> Result<crate::verify::rewrite::PreparedTerm, VerifyError> {
    use crate::verify::rewrite::{PreparedTerm, TermHead};
    Ok(match term {
        vmir::TrigTerm::Var(k) if *k >= binder_base => PreparedTerm::Bound(k - binder_base),
        vmir::TrigTerm::Var(k) => PreparedTerm::Capture(slot[k]),
        vmir::TrigTerm::Lit(lit) => PreparedTerm::Lit(lit.clone()),
        vmir::TrigTerm::Wild => PreparedTerm::Wild,
        vmir::TrigTerm::App {
            head,
            type_args,
            args,
        } => {
            let func = |func| TermHead::Func {
                func,
                type_args: type_args.iter().cloned().collect(),
            };
            let head = match head {
                vmir::TrigHead::Func(id) | vmir::TrigHead::Loc(id) => {
                    func(crate::verify::func_registry::func_id_for_member(*id))
                }
                vmir::TrigHead::AdtCons { adt, variant } => func(alloc.cons(*adt, *variant)),
                vmir::TrigHead::AdtProj {
                    adt,
                    variant,
                    field,
                } => func(alloc.proj(*adt, *variant, *field)),
                vmir::TrigHead::AdtIs { adt, variant } => func(alloc.adt_is(*adt, *variant)),
                vmir::TrigHead::Coll(op) => {
                    assert_eq!(
                        type_args.len(),
                        op.kind().arity(),
                        "a `{op:?}` trigger names its collection's type arguments: {type_args:?}"
                    );
                    TermHead::Coll(type_args.iter().cloned().collect(), *op)
                }
                vmir::TrigHead::CollLit(kind) => {
                    let [elem_ty] = &type_args[..] else {
                        panic!(
                            "a `{kind:?}` literal trigger names its element type: {type_args:?}"
                        );
                    };
                    TermHead::CollLit(*kind, elem_ty.clone())
                }
            };
            PreparedTerm::App {
                head,
                args: args
                    .iter()
                    .map(|a| prepare_trig_term(alloc, a, binder_base, slot))
                    .collect::<Result<_, _>>()?,
            }
        }
    })
}

/// Lower an axiom/quantifier body's inst stream into registry-resolved pure
/// steps (every callee down to its verifier `FuncId`), ready for the applier —
/// which has no registry access at rule-application time. `names` resolves a
/// member to its source name, needed for the display label of a minted `f%pre`
/// token (the registry holds no interner). Every operand is passed through
/// `rename` (into the recipe's temp space). A `Deref` that `read` resolves
/// compiles as its [`ReadStep`](crate::verify::quant::ReadStep) says; any other
/// `Deref` is refused.
pub(crate) fn prepare_body(
    alloc: &mut crate::verify::func_registry::FuncRegistry,
    quants: &crate::verify::quant::RecipeTable,
    names: &dyn Fn(MemberId) -> String,
    insts: &[vmir::Inst],
    rename: &dyn Fn(&Val) -> Val,
    read: &dyn Fn(&vmir::HeapVal, &Val) -> Option<crate::verify::quant::ReadStep>,
    spliced: &[Vec<crate::verify::rewrite::AxiomInst>],
) -> Result<Vec<crate::verify::rewrite::AxiomInst>, VerifyError> {
    use crate::verify::rewrite::{AxiomInst, AxiomPure};
    let mut out = Vec::with_capacity(insts.len());
    let push = |out: &mut Vec<AxiomInst>, i: AxiomInst| {
        out.push(crate::verify::cert::map_operands(&i, rename))
    };
    let mut spliced = spliced.iter();
    for inst in insts {
        if let InstKind::Pure(_, PureInst::Deref(heap, addr)) = &inst.kind
            && let Some(step) = read(heap, addr)
        {
            if let crate::verify::quant::ReadStep::Lookup(kind, map) = step {
                out.push(AxiomInst::Val(AxiomPure::App {
                    func: alloc.value_map(&kind).lookup,
                    type_args: vec![],
                    args: vec![map, rename(addr)],
                }));
            }
            continue;
        }
        // A nested `forall` is a step like any other: it materializes the inner
        // e-node with the enclosing instance's values as capture children. Its
        // capture list is derived (`Forall::free_temps`) and cached on the table by
        // the innermost-first intern that ran just before this one.
        if let InstKind::Pure(_, PureInst::Forall(q)) = &inst.kind {
            let (recipe, free) = quants.entry_of(q)?;
            push(
                &mut out,
                AxiomInst::Forall {
                    recipe: *recipe,
                    type_args: quants.get(*recipe).open_types(),
                    caps: free.iter().map(|&k| Val::Temp(k)).collect(),
                },
            );
            continue;
        }
        // A call is two steps, exactly as on the eval walk (`eval_pure_inst`'s
        // `FunctionCall` arm): the application, plus the callee's `f%pre(args)`
        // presence token that lets `rewrite::function_rule` unfold its body *here*.
        // Without the token an instance materializes `f(σ)` opaquely and the callee
        // never unfolds under the quantifier. The token step occupies no temp slot,
        // so the application keeps this inst's slot.
        if let InstKind::Pure(_, PureInst::FunctionCall(fc)) = &inst.kind {
            let args: Vec<Val> = fc.args.iter().cloned().collect();
            push(
                &mut out,
                AxiomInst::Val(AxiomPure::App {
                    func: crate::verify::func_registry::func_id_for_member(fc.function),
                    type_args: fc.type_args.clone(),
                    args: args.clone(),
                }),
            );
            // The token's truth is released under this inst's own path condition —
            // a call under `i > 0 ==> ..` inside the body must not fire the
            // callee's axioms at a σ where `i > 0` fails. No `recipe_pc` needed:
            // a prepared body's `Val`s *are* the host body's temps, used
            // identity-wise, so `inst.pc`'s literals transfer directly.
            push(
                &mut out,
                AxiomInst::Token {
                    func: alloc.fn_pre_token(fc.function, &names(fc.function)),
                    type_args: fc.type_args.clone(),
                    args,
                    guards: inst.pc.conds.clone(),
                },
            );
            continue;
        }
        // A heap-dependent call's snapshot: captured, which `Canonical` names
        // in place of the step's temp, or rebuilt by the steps it spliced,
        // already in the recipe's space.
        if let InstKind::Heap(hi) = &inst.kind
            && hi.frame_exhale().is_some()
        {
            out.extend(spliced.next().expect("a splice per call").iter().cloned());
            continue;
        }
        let prepared = match &inst.kind {
            InstKind::Pure(_, pi) => AxiomInst::Val(match pi {
                PureInst::Binary(op, l, r) => AxiomPure::Binary(*op, l.clone(), r.clone()),
                PureInst::Ternary(c, t, e) => AxiomPure::Ternary(c.clone(), t.clone(), e.clone()),
                PureInst::RealCast(v) => AxiomPure::RealCast(v.clone()),
                PureInst::FunctionCall(_) => unreachable!("handled above"),
                PureInst::AdtCons {
                    adt,
                    type_args,
                    variant,
                    args,
                } => AxiomPure::App {
                    func: alloc.cons(*adt, *variant),
                    type_args: type_args.clone(),
                    args: args.clone(),
                },
                PureInst::AdtProj {
                    adt,
                    type_args,
                    variant,
                    field,
                    base,
                } => AxiomPure::App {
                    func: alloc.proj(*adt, *variant, *field),
                    type_args: type_args.clone(),
                    args: vec![base.clone()],
                },
                PureInst::AdtIs {
                    adt,
                    type_args,
                    variant,
                    base,
                } => AxiomPure::App {
                    func: alloc.adt_is(*adt, *variant),
                    type_args: type_args.clone(),
                    args: vec![base.clone()],
                },
                PureInst::Fresh
                | PureInst::Deref(..)
                | PureInst::Perm(..)
                | PureInst::OptionUnwrap(_)
                | PureInst::Forall(_) => {
                    return Err(VerifyError::Unimplemented("impure inst in axiom body"));
                }
                PureInst::Coll { tys, op, args } => AxiomPure::Coll {
                    tys: tys.clone(),
                    op: *op,
                    args: args.clone(),
                },
                PureInst::CollLit {
                    kind,
                    elem_ty,
                    elem,
                } => AxiomPure::CollLit {
                    kind: *kind,
                    elem_ty: elem_ty.clone(),
                    elem: elem.clone(),
                },
            }),
            InstKind::Assume(v) => AxiomInst::Assume(v.clone()),
            // Never verified; nothing to record.
            InstKind::Assert(_) => continue,
            // A permission step defines no value (only a quantified permission's
            // body has them; its inverse axiom reads them itself).
            InstKind::Perm(_) => continue,
            _ => return Err(VerifyError::Unimplemented("non-pure inst in axiom body")),
        };
        push(&mut out, prepared);
    }
    Ok(out)
}

/// The chunk set a quantified permission denotes at this program point: its
/// body plus the values its captured temps have in the live state and its heap
/// reads resolved against the live heaps (`heap::resolve_reads`), to be
/// checked whole when it is added or taken ([`QpCheck::Body`]).
fn quant_spec_of(
    ctx: &mut VerifyContext<'_>,
    state: &EvalState,
    q: &vmir::QuantChunks,
    pc: &PathConds,
) -> Result<QuantSpec, VerifyError> {
    let temps = q.free_temps().into_iter().map(|k| (k, state.vals[k]));
    let pc_lits = pc_lits_raw(ctx, state, pc);
    let mut heap_of = |ctx: &mut VerifyContext<'_>, h: &_| get_heap(ctx, state, h, &pc_lits);
    let caps = crate::verify::heap::resolve_reads(ctx, q, temps.collect(), &mut heap_of, &pc_lits)?;
    Ok(QuantSpec {
        q: Rc::new(q.clone()),
        caps,
        // Stated here, so nothing about it is established yet.
        check: QpCheck::Body,
    })
}

/// Per-instruction evaluator: the shared signature of [`eval_method_inst`] and
/// [`eval_resource_body_inst`], so [`walk_body`] can be parameterized by which
/// one the body kind uses.
type EvalFn = fn(
    &mut VerifyContext<'_>,
    &vmir::Program,
    &mut EvalState,
    &Inst,
    &HashMap<MemberId, ResourceDefinition>,
) -> Result<(), VerifyError>;

/// The single body-walk shared by all three drivers: for each instruction,
/// discharge its side-condition [`inst_obligations`] under the path condition,
/// then evaluate it, snapshotting for the visualizer throughout. One place, so an
/// obligation added to `inst_obligations` cannot be skipped by one driver.
///
/// `eval` selects the per-inst semantics (method/function vs resource body).
/// `footprint_ops`, when `Some`, collects each `acc`'s `(loc, perm)` operand in
/// body order — the resource driver's one extra responsibility (drives the
/// fold/unfold snapshot layout); `None` for method and function bodies.
#[allow(clippy::too_many_arguments)]
fn walk_body(
    ctx: &mut VerifyContext<'_>,
    program: &vmir::Program,
    state: &mut EvalState,
    snap: &mut Snapshotter,
    insts: &[Inst],
    certs: &HashMap<MemberId, ResourceDefinition>,
    eval: EvalFn,
    mut footprint_ops: Option<&mut Vec<(Val, vmir::PermVal)>>,
) -> Result<(), VerifyError> {
    for (inst_idx, inst) in insts.iter().enumerate() {
        assert_statement_pc_is_block_cube(ctx, inst);
        if let (
            Some(ops),
            InstKind::Heap(HeapInst::Add { loc, perm, .. } | HeapInst::Sub { loc, perm, .. }),
        ) = (&mut footprint_ops, &inst.kind)
        {
            ops.push((loc.clone(), perm.clone()));
        }
        let vals_before = state.vals.len();
        let heaps_before = state.heaps.len();
        // Rendering the instruction (and cloning heaps for the viz snapshot) is
        // pure diagnostics — deferred to the failure paths and the (env-gated)
        // snapshotter so the hot path pays nothing for it.
        let inst_text = |program: &vmir::Program| {
            format_inst(
                inst,
                &program.decls,
                &program.interner,
                &program.groups,
                vals_before,
                heaps_before,
            )
        };
        crate::verify::qp_trace::at_inst(inst_idx, insts.len(), || inst_text(program));
        let pc_lits = collect_pc_lits(ctx, state, &inst.pc);
        for (goal, err) in inst_obligations(ctx, state, &inst.kind, &pc_lits) {
            if !ctx.prove_under_pc(goal, &pc_lits) {
                if crate::verify::viz::dump_perm_enabled() {
                    eprintln!(
                        "[perm-dump] failed obligation goal:\n{}",
                        crate::verify::viz::dump_term(ctx, goal, 64),
                    );
                }
                crate::verify::qp_trace::flush_instances(ctx);
                let inst_text = inst_text(program);
                if snap.enabled() {
                    let heaps = display_heaps(state, &inst.kind, heaps_before);
                    snap.snapshot(ctx, &heaps, &format!("FAIL: {}", inst_text), Some(goal));
                }
                return Err(err.with_inst(inst_text, inst_idx, insts.len()));
            }
        }
        // A quantifier's own side conditions, discharged once per syntactic
        // occurrence against fresh binders in a scratch clone (see
        // [`check_forall_wd`]). It happens here, not in the evaluator, because it
        // needs the path condition — and here it cannot be forgotten by one of the
        // three drivers. Axiom bodies never reach this walk: they stay trusted.
        // It also resolves the quantifier's heap reads, which its step captures.
        if let InstKind::Pure(_, PureInst::Forall(q)) = &inst.kind {
            match check_forall_wd(ctx, q, state, &pc_lits) {
                Ok(reads) => {
                    crate::verify::qp_trace::forall_resolved(ctx, q, &reads);
                    state.forall_reads = reads;
                }
                Err(err) => {
                    crate::verify::qp_trace::flush_instances(ctx);
                    let inst_text = inst_text(program);
                    if snap.enabled() {
                        let heaps = display_heaps(state, &inst.kind, heaps_before);
                        snap.snapshot(ctx, &heaps, &format!("FAIL: {}", inst_text), None);
                    }
                    return Err(err.with_inst(inst_text, inst_idx, insts.len()));
                }
            }
        }
        let evaluated = eval(ctx, program, state, inst, certs);
        crate::verify::qp_trace::flush_instances(ctx);
        // A framing decided for this instruction never serves a later one.
        state.framed_read.take();
        if let Err(err) = evaluated {
            let inst_text = inst_text(program);
            if snap.enabled() {
                let heaps = display_heaps(state, &inst.kind, heaps_before);
                snap.snapshot(ctx, &heaps, &format!("FAIL: {}", inst_text), None);
            }
            return Err(err.with_inst(inst_text, inst_idx, insts.len()));
        }
        stats::bump(|s| s.insts_processed += 1);
        if snap.enabled() {
            let highlight =
                (state.vals.len() > vals_before).then(|| state.vals[state.vals.len() - 1]);
            let heaps = display_heaps(state, &inst.kind, heaps_before);
            snap.snapshot(ctx, &heaps, &inst_text(program), highlight);
        }
    }
    Ok(())
}

pub(crate) fn verify_method(
    program: &vmir::Program,
    method_name: &str,
    method: &Method,
    certs: &HashMap<MemberId, ResourceDefinition>,
    fn_certs: &HashMap<MemberId, std::sync::Arc<FunctionDefinition>>,
    alloc: &mut crate::verify::func_registry::FuncRegistry,
) -> Result<(), VerifyError> {
    let mut ctx = VerifyContext::new(&program.interner, &program.decls, &program.groups, alloc);
    ctx.fn_certs = Some(fn_certs);
    ctx.certs = Some(certs);
    assume_axioms(&mut ctx, program)?;
    let mut state = EvalState::new();
    crate::verify::qp_trace::begin_unit("method", method_name);
    let mut snap = Snapshotter::from_env(method_name);
    snap.snapshot(&ctx, &[], "init", None);

    // Block walker (Stage 3): evaluate blocks in stored order — which is
    // topological *and* the original emission order, so positional `EvalState`
    // temps (`vals[n]` == `Temp(n)`) stay valid — running each block's `join`
    // phase then its `body` phase through the same per-inst engine (`walk_body`)
    // the flat verifier used. The heap threads linearly via each inst's explicit
    // `base: HeapVal`; the per-predecessor structural merge is a later stage.
    for (bid, block) in method.blocks.iter_enumerated() {
        // Stored order is topological: a block's predecessors have smaller ids.
        // (A future lowering bug that broke this would corrupt positional eval.)
        debug_assert!(
            match &block.preds {
                vmir::Preds::Entry => true,
                vmir::Preds::From(p) => p.0 < bid.0,
                vmir::Preds::Join { then_, els, .. } => then_.0 < bid.0 && els.0 < bid.0,
            },
            "blocks must be stored in topological order (preds precede)"
        );
        // ── Stage-4 heap hook ──────────────────────────────────────────────
        // Stage 4 will derive this block's `h_in` from `block.preds` here (a
        // `HeapInst::Merge` for a `Join`). Stage 3 threads the heap linearly via
        // the insts' explicit `base: HeapVal` refs, so there is nothing to do.
        // ───────────────────────────────────────────────────────────────────
        // No scratch during the join phase: the cube's reach boolean is
        // materialized *by* the join, so it isn't resolvable until join runs.
        ctx.end_block();
        state.join_arms_dead = match &block.preds {
            vmir::Preds::Join { then_, els, .. } => Some((
                state.dead_blocks.contains(&then_.0),
                state.dead_blocks.contains(&els.0),
            )),
            _ => None,
        };
        state.block_cube = Some(block.cube.clone());
        walk_body(
            &mut ctx,
            program,
            &mut state,
            &mut snap,
            &block.join,
            certs,
            eval_method_inst,
            None,
        )?;
        // Record this block's cube for the (experimental) per-block scratch. All
        // body insts share it (vmir::Block::cube), so the scratch assumes it once.
        let cube = pc_lits_raw(&mut ctx, &state, &block.cube);
        ctx.begin_block(cube);
        walk_body(
            &mut ctx,
            program,
            &mut state,
            &mut snap,
            &block.body,
            certs,
            eval_method_inst,
            None,
        )?;
        // Dead-arm tagging: if the block's cube is now refuted in the ground
        // graph (an unreachable arm — e.g. after `inhale false` unioned a reach
        // flag with `false`), tag the block so a downstream join merge drops its
        // heap instead of forming a `0`-leaf that would need a case split. Cheap:
        // just consults folded literals, no clone. Reachability propagates too: a
        // block whose only predecessor is dead, or whose join has both arms dead,
        // is dead.
        let unreached = match &block.preds {
            vmir::Preds::From(p) => state.dead_blocks.contains(&p.0),
            vmir::Preds::Join { .. } => state.join_arms_dead == Some((true, true)),
            vmir::Preds::Entry => false,
        };
        // Raw: this runs *inside* the block, so `collect_pc_lits` would
        // prepend the very cube being lowered.
        let lits = pc_lits_raw(&mut ctx, &state, &block.cube);
        let refuted = unreached
            || lits.iter().any(|(id, pol)| {
                matches!(
                    ctx.egraph[ctx.egraph.find(*id)].data.known(),
                    Some(Literal::Bool(b)) if *b != matches!(pol, Polarity::Positive)
                )
            });
        if refuted {
            state.dead_blocks.insert(bid.0);
        }
    }
    ctx.end_block();
    if let Some(t) = crate::verify::heap::MergeTrace::take() {
        eprintln!(
            "[merge-trace] {method_name}: selects_built={} zero_leaves={} max_depth={}",
            t.selects_built, t.zero_leaves, t.max_depth
        );
    }
    Ok(())
}

/// Verify a resource self-contained: run its body in a fresh egraph with fresh
/// symbolic params (a two-state resource's pre-state snapshot is an ordinary
/// trailing param), discharging each instruction's side-condition obligations
/// under its path condition. A resource always has a body, so this always
/// yields a definition. This establishes well-formedness **once**; method call
/// sites reuse it without re-checking (see [`eval_resource_call`]).
pub(crate) fn verify_resource(
    program: &vmir::Program,
    resource_name: &str,
    resource: &Resource,
    certs: &HashMap<MemberId, ResourceDefinition>,
    fn_certs: &HashMap<MemberId, std::sync::Arc<FunctionDefinition>>,
    alloc: &mut crate::verify::func_registry::FuncRegistry,
) -> Result<ResourceDefinition, VerifyError> {
    let body = &resource.body;

    let mut ctx = VerifyContext::new(&program.interner, &program.decls, &program.groups, alloc);
    ctx.fn_certs = Some(fn_certs);
    ctx.certs = Some(certs);
    assume_axioms(&mut ctx, program)?;
    // The certificate recipe is built by the walk itself: each `acc` records a
    // footprint slot, `Deref`s resolve through chunk provenance. Installed
    // after `assume_axioms` so axiom bodies don't mirror into the recipe.
    ctx.recipe = Some(crate::verify::cert::RecipeBuilder::new(
        resource.params.len(),
        None,
    ));
    let params: Vec<egg::Id> = resource
        .params
        .iter()
        .map(|ty| ctx.fresh_symbolic_value(ty.clone()))
        .collect();
    // A two-state (`Ctx`) resource needs no special seeding: its pre-state
    // arrives as the trailing snapshot parameter (a fresh symbolic like any
    // other param) and its body's entry bound `inhale` reconstructs the pre-state
    // heap, implicitly assuming the precondition resource's boolean.
    let mut state = EvalState::with_args(params, resource.params.clone());

    crate::verify::qp_trace::begin_unit("resource", resource_name);
    let mut snap = Snapshotter::from_env(resource_name);
    snap.snapshot(&ctx, &[], "init", None);

    walk_body(
        &mut ctx,
        program,
        &mut state,
        &mut snap,
        &body.insts,
        certs,
        eval_resource_body_inst,
        None,
    )?;

    // Capture the body as a pure recipe (add-only; imports no e-classes). Each
    // call site rebuilds the footprint addresses/permissions and the body boolean
    // from the recipe and re-derives any merge it needs itself (Finding C).
    // Slot address/permission recipes and the body boolean are sliced out of
    // the shared step stream the walk built.
    use crate::verify::cert::SlotRecipe;
    let rb = ctx.recipe.take().expect("resource walk builds a recipe");
    let footprint = rb
        .pending_slots
        .iter()
        .map(|(kind, elem, shape)| {
            let shape = match shape {
                SlotShape::Single { addr, perm } => SlotShape::Single {
                    addr: rb.slice(addr)?,
                    // One `slice` per operand, shape preserved. Permission step
                    // lists are short (a `PermInst::Ite` only arises from a
                    // branch gate or a read-only weakening — see
                    // `Sink::gate_perm`).
                    perm: perm.try_map(&mut |v: &Val| rb.slice(v))?,
                },
                SlotShape::Quant { q, caps } => SlotShape::Quant {
                    q: q.clone(),
                    caps: caps
                        .iter()
                        .map(|(k, v)| Ok((*k, rb.slice(v)?)))
                        .collect::<Result<_, VerifyError>>()?,
                },
            };
            Ok(SlotRecipe {
                kind: kind.clone(),
                elem: elem.clone(),
                shape,
            })
        })
        .collect::<Result<Vec<_>, VerifyError>>()?;
    let bool_r = state
        .recipe_of(&body.res.1)
        .ok_or(VerifyError::Unimplemented(
            "purify: resource bool without a recipe",
        ))?;
    Ok(ResourceDefinition {
        footprint,
        bool: rb.slice_with_tokens(&bool_r)?,
    })
}

/// Verify a non-recursive function and capture its body as a **pure term
/// recipe** ([`FunctionDefinition`]) for other units to unfold lazily at call
/// sites (`rewrite::function_rule`). Abstract functions (no body) have nothing
/// to verify.
///
/// The walk runs the ordinary live eval (obligations, a bound `inhale`/`Unfold` heap
/// reconstruction, `Deref` values), logging each heap-reconstruction event into
/// `ctx.heap_events`. Then [`purify_function`] re-walks the body once, turning it
/// into an add-only recipe over the params (and the snapshot param, for
/// heap-dependent functions): every `Deref` becomes the pure term the snapshot
/// projects to (`unwrap(proj_i(snap))`, possibly nested through `unfolding`), and
/// the entry `assume f#requires` is **dropped** so that no precondition-derived
/// merge can ride into a call site (a recipe imports no e-classes).
///
/// Callees — including the function's own `f#requires`/`f#ensures` contract
/// functions — are ordinary `Function` decls verified earlier in dependency
/// order, so their recipes are already in `fn_certs` and their unfold rules
/// installed; saturation discharges the contract obligations lazily.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_function(
    program: &vmir::Program,
    function_name: &str,
    self_id: MemberId,
    function: &Function,
    certs: &HashMap<MemberId, ResourceDefinition>,
    fn_certs: &HashMap<MemberId, std::sync::Arc<FunctionDefinition>>,
    // The members of `self_id`'s SCC iff it is a genuine recursion cycle (else
    // `None`). In-SCC callees are retargeted to their limited twin in the recipe,
    // and this function's own limited twin is recorded so its unfold rule frames
    // `f(x) == f'(x)`.
    recursive_scc: Option<&crate::dhash::HashSet<MemberId>>,
    alloc: &mut crate::verify::func_registry::FuncRegistry,
) -> Result<Option<std::sync::Arc<FunctionDefinition>>, VerifyError> {
    let Some(body) = function.body.as_ref() else {
        // Abstract/uninterpreted function: no body to verify. Its contract
        // decls are verified as ordinary Functions (spec WF); synthesize the
        // guarded post axiom from the contract links, if any.
        return Ok(contract_post_definition(
            alloc,
            program,
            self_id,
            function,
            recursive_scc.is_some(),
        ));
    };

    let mut ctx = VerifyContext::new(&program.interner, &program.decls, &program.groups, alloc);
    ctx.fn_certs = Some(fn_certs);
    ctx.certs = Some(certs);
    assume_axioms(&mut ctx, program)?;
    // The certificate recipe is built by the walk itself (single pass): only the
    // limited-twin substitution set is fixed up front. The exported post is not
    // recognized out of the walk at all -- it is built from the declaration's
    // `ensures` link once the walk is done (`post_fact`). No precondition guard
    // is prepared: a function's facts are gated by its `f%pre` token, released
    // at the call.
    let recipe =
        crate::verify::cert::RecipeBuilder::new(function.params.len(), recursive_scc.cloned());
    // A postcondition is the `ensures` link of some function. Only matters for
    // recursive calls, so only looked for in a recursion cycle.
    let is_post_contract = recursive_scc.is_some()
        && program.decls.iter().any(|d| {
            matches!(d, vmir::Declaration::Function(g)
                if g.ensures.as_ref().is_some_and(|e| e.member == self_id))
        });
    ctx.recipe = Some(match is_post_contract {
        true => recipe.post_contract(),
        false => recipe,
    });
    // Recursive batch: every SCC member's spec-derived post axiom is available
    // while this body is checked (Silicon emits `post` in phase 1, before the
    // phase-2 body check) — this is what lets the exit assert use a recursive
    // call's postcondition (induction; termination is not checked, same as
    // Silicon without `decreases`). Like any post, it is gated by the call's
    // `f%pre` token, which the `FunctionCall` arm mints under the call's path
    // condition.
    if let Some(scc) = recursive_scc {
        for &m in scc {
            let vmir::Declaration::Function(mf) = &program.decls[m] else {
                continue;
            };
            if let Some(post) = contract_post_definition(ctx.alloc, program, m, mf, false) {
                let name = ctx.member_name(m);
                let func = crate::verify::func_registry::func_id_for_member(m);
                let pre_token = ctx.alloc.fn_pre_token(m, &name);
                ctx.axiom_rules.push(crate::verify::rewrite::post_rule(
                    &name, func, post, pre_token,
                ));
            }
        }
    }
    // Params seed the initial `Val::Temp(0..n_params)` slots (heap-free: no ctx heap).
    let params: Vec<egg::Id> = function
        .params
        .iter()
        .map(|ty| ctx.fresh_symbolic_value(ty.clone()))
        .collect();
    let param_types: Vec<Type> = function.params.iter().cloned().collect();
    let mut state = EvalState::with_args(params.clone(), param_types);

    crate::verify::qp_trace::begin_unit("function", function_name);
    let mut snap = Snapshotter::from_env(function_name);
    snap.snapshot(&ctx, &[], "init", None);
    // The method-body eval path handles every inst a function body can contain
    // (pure ops, the entry `assume`, `Snap`/a bound `inhale`/`Unfold` for heap-dependent
    // functions).
    walk_body(
        &mut ctx,
        program,
        &mut state,
        &mut snap,
        &body.insts,
        certs,
        eval_method_inst,
        None,
    )?;

    // The walk mirrored the body into a pure, add-only recipe (params → args
    // at call sites via `build_instance`); `Deref` values became the pure
    // terms their chunks carry as provenance (`unwrap(proj_i(snap))`), and the
    // entry `assume f#requires` was dropped (Finding B — `Assume` emits no
    // recipe step, so the precondition cannot leak into callers).
    let rb = ctx.recipe.take().expect("function walk builds a recipe");
    let (steps, token_steps) = rb.into_function_parts()?;
    // `ensures` is the single export mechanism: the post is built from the
    // declaration link, identically to an abstract function's, and kept as its
    // own recipe over the params. The body's own exit `assert f#ensures(..)`
    // stayed a pure obligation — it proved the link, it does not publish it.
    let post = post_fact(
        ctx.alloc,
        program,
        self_id,
        function,
        recursive_scc.is_some(),
    );
    let res = state
        .recipe_of(&body.res)
        .ok_or(VerifyError::Unimplemented(
            "purify: function result without a recipe",
        ))?;
    // A recursive function records its limited twin so the unfold rule frames
    // `f(x) == f'(x)`. Minted here (not in the recipe) so the id exists even if
    // the body has no reachable recursive call under some path.
    let limited = recursive_scc.map(|_| {
        let name = ctx.member_name(self_id);
        ctx.alloc.limited(self_id, &name)
    });
    Ok(Some(std::sync::Arc::new(FunctionDefinition {
        n_params: function.params.len(),
        steps,
        res: Some(res),
        limited,
        token_steps,
        post,
    })))
}

/// The **single** fact a function publishes to its callers: `f#ensures(params,
/// f(params))`, built from the declaration's `ensures` link — never from
/// anything the body contains. Identical for abstract and bodied functions,
/// heap-free and heap-dependent ones (a heap-dependent function's snapshot is a
/// trailing param, so it is just another `ContractArg::Val`).
///
/// Returned as a **standalone** [`PostRecipe`] over the params rather than as a
/// slice of the body's stream, so that replaying it needs no body instance.
///
/// The fact carries **no guards of its own**: it is gated by the call-site
/// `f%pre` token, which is minted exactly where the precondition was checked.
/// It expresses the result as the application
/// `f(params)` — not a rebuilt body — so at a recursive unroll's `f'(smaller)`
/// occurrence it talks about that very node, which is Silicon's `post` axiom.
fn post_fact(
    alloc: &mut crate::verify::func_registry::FuncRegistry,
    program: &vmir::Program,
    self_id: MemberId,
    function: &Function,
    // Set when this is the function's *own* definition and it sits in a
    // recursion cycle: the fact then expresses the result as the limited twin
    // `f'(params)`, matching the `f(x) == f'(x)` frame its unfold rule installs.
    // Without the frame an abstract SCC member would be unreachable from a
    // sibling's recipe, which lowers it to `f'`. Cleared for the in-batch
    // pre-seed (rules keyed on the full ids, no frames installed yet).
    recursive: bool,
) -> Option<crate::verify::cert::PostRecipe> {
    use crate::verify::cert::PostRecipe;
    use crate::verify::func_registry::func_id_for_member;
    use crate::verify::rewrite::{AxiomInst, AxiomPure};

    let en = function.ensures.as_ref()?;
    let n_params = function.params.len();
    let mut steps: Vec<AxiomInst> = Vec::new();
    let emit = |steps: &mut Vec<AxiomInst>, pure: AxiomPure| -> Val {
        let v = Val::Temp(n_params + steps.len());
        steps.push(AxiomInst::Val(pure));
        v
    };
    // Link args are over the params (`Temp(0..n_params)`) — identity in recipe
    // space, so they can be used verbatim.
    let self_app = emit(
        &mut steps,
        AxiomPure::App {
            func: match recursive {
                true => alloc.limited(self_id, program.name(self_id)),
                false => func_id_for_member(self_id),
            },
            type_args: Vec::new(),
            args: (0..n_params).map(Val::Temp).collect(),
        },
    );
    let args: Vec<Val> = en
        .args
        .iter()
        .map(|a| match a {
            vmir::ContractArg::Val(v) => v.clone(),
            vmir::ContractArg::Result => self_app.clone(),
        })
        .collect();
    let res = emit(
        &mut steps,
        AxiomPure::App {
            func: func_id_for_member(en.member),
            type_args: Vec::new(),
            args: args.clone(),
        },
    );
    // The `#ensures` member's own presence token. It is an ordinary function, and
    // this is the only position it ever occurs in — no call site mints it — so the
    // export carries it alongside the truth (`PostRecipe::token`).
    let token = emit(
        &mut steps,
        AxiomPure::App {
            func: alloc.fn_pre_token(en.member, program.name(en.member)),
            type_args: Vec::new(),
            args,
        },
    );
    Some(PostRecipe {
        steps,
        res,
        token: Some(token),
    })
}

/// Synthesize an **abstract** function's definition: no body, nothing to
/// verify — just [`post_fact`], the guarded post axiom `pre-token ⟹
/// f#ensures(params, f(params))` built from the contract links. (Silicon's
/// phase 1 emits `post`/`postProp` for abstract functions once the spec is
/// well-defined; our contract decls get that WF check as ordinary `Function`
/// verification, scheduled first by the link edges in `analyze`.) `None` when
/// there is nothing to export: no ensures.
fn contract_post_definition(
    alloc: &mut crate::verify::func_registry::FuncRegistry,
    program: &vmir::Program,
    self_id: MemberId,
    function: &Function,
    recursive: bool,
) -> Option<std::sync::Arc<FunctionDefinition>> {
    let post = post_fact(alloc, program, self_id, function, recursive)?;
    Some(std::sync::Arc::new(FunctionDefinition {
        n_params: function.params.len(),
        steps: Vec::new(),
        res: None,
        limited: recursive.then(|| alloc.limited(self_id, program.name(self_id))),
        // An abstract function has no body, hence no propagated callee tokens.
        token_steps: Vec::new(),
        post: Some(post),
    }))
}

/// Proof obligations implied by an instruction's kind, as `(goal, error)` pairs
/// that must each be proven `true` under the instruction's path condition. A
/// `Deref` requires a positive permission for the location it reads; `acc`
/// requires a non-negative permission; division requires a non-zero divisor.
///
/// This is the **single** source of per-instruction obligations: every body
/// driver (`verify_method`, `verify_resource`, `verify_function`) discharges
/// exactly this list, so an obligation added here is checked everywhere.
fn inst_obligations(
    ctx: &mut VerifyContext<'_>,
    state: &EvalState,
    kind: &InstKind,
    pc_lits: &[(egg::Id, Polarity)],
) -> Vec<(egg::Id, VerifyError)> {
    match kind {
        // `0 < perm(heap, loc)` — the location must be framed by the heap being
        // read. In a function body that heap is the one the entry bound `inhale`
        // reconstructs from the snapshot parameter, so this failing means a read
        // outside the declared precondition. `chunk_under_pc` lets an address
        // that only aliases a held chunk under this instruction's branch (e.g.
        // `y.f` where `x == y` holds here) still frame.
        InstKind::Pure(_, PureInst::Deref(hv, loc)) => {
            let heap = match get_heap(ctx, state, hv, pc_lits) {
                Ok(heap) => heap,
                Err(e) => return vec![(expr!(ctx, false), e)],
            };
            let Some(l) = state.loc(ctx, loc) else {
                return vec![(expr!(ctx, false), VerifyError::InsufficientPermission)];
            };
            // A chunk set's body evaluated once already, at its inhale.
            let framing = heap
                .framing(ctx, &l, pc_lits)
                .unwrap_or_else(|e| panic!("quantified chunk body failed to re-evaluate: {e:?}"));
            // Decided per leaf (a branch-structured held perm is never
            // materialized); on failure a trivially-false goal carries the error.
            if !framing.frames(ctx) {
                return vec![(expr!(ctx, false), VerifyError::InsufficientPermission)];
            }
            // The read takes its value from these same holders.
            state.framed_read.set(Some((*hv, loc.clone(), framing)));
            vec![]
        }
        // `not(perm < 0)` desugared to an `Ite`. Applies to a location combine, a
        // resource inhale/exhale, and fold/unfold (a consuming op with a negative
        // scale would flip `heap_subtract` into permission fabrication).
        //
        // A wildcard is positive by construction (assumed `0 < w` at creation) and
        // the e-graph has no real-order reasoning to *prove* `¬(w < 0)`; Silicon
        // likewise skips the assertion for a constrainable ARP, so a
        // wildcard-bearing permission carries no `perm ≥ 0` obligation.
        InstKind::Heap(
            HeapInst::Add { perm, .. }
            | HeapInst::Sub { perm, .. }
            | HeapInst::Inhale { perm, .. }
            | HeapInst::Exhale { perm, .. },
        ) if state.perm_is_wild(perm) => vec![],
        // A resource op carries the resource's **boolean**, so at zero permission
        // it would assume or assert facts about a footprint it transferred no
        // share of — a vacuous operation. Viper rejects `fold`/`unfold` at a
        // possibly-zero amount too, so this mirrors it rather than diverging.
        // Free today: every resource op carries a literal `1/1`, and `0 < 1/1`
        // const-folds.
        InstKind::Heap(HeapInst::Inhale { perm, .. } | HeapInst::Exhale { perm, .. }) => {
            let perm = eval_perm(ctx, state, perm);
            let goal = expr!(ctx, (0 / 1) < r { perm });
            vec![(
                goal,
                VerifyError::SideCondition("permission must be positive"),
            )]
        }
        // Slot ops carry no boolean, so moving zero permission says nothing and
        // is legal — and they carry every gated amount (`b ? 1/1 : 0` is exactly
        // `0` on the `!b` path), so a strict rule here would reject every
        // conditional `acc`.
        InstKind::Heap(HeapInst::Add { perm, .. } | HeapInst::Sub { perm, .. }) => {
            let perm = eval_perm(ctx, state, perm);
            let zero = expr!(ctx, 0 / 1);
            // Permission must not be negative: not (perm < 0).
            let goal = expr!(ctx, not({ perm } < r { zero }));
            vec![(
                goal,
                VerifyError::SideCondition("permission may be negative"),
            )]
        }
        // A pure expression's own side conditions: a divisor, a map key, a
        // sequence index (shared with the quantifier bodies' checked walk).
        InstKind::Pure(ty, pi) => pure_obligations(ctx, state, ty, pi),
        _ => vec![],
    }
}

#[cfg(test)]
mod e2e;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::lang::Symbolic;
    use crate::verify::test_support::fresh_ctx;
    use crate::vmir::BinOp;

    #[test]
    fn prove_under_empty_pc_proves_known_goal() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let g = ctx.add(Symbolic::Fresh(0));
        let true_ = ctx.add(Symbolic::Lit(Literal::Bool(true)));
        ctx.union(g, true_);
        ctx.egraph.rebuild();

        // Goal already in the `true` eclass → proven under the empty PC.
        assert!(ctx.prove_under_pc(g, &[]));
    }

    #[test]
    fn prove_unknown_goal_fails() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        // A free boolean never driven to `true` is not provable.
        let g = ctx.add(Symbolic::Fresh(0));
        assert!(!ctx.prove_under_pc(g, &[]));
    }

    #[test]
    fn prove_under_false_pc_is_vacuous() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        // Unprovable goal, but the path is unsatisfiable (`false`).
        let g = ctx.add(Symbolic::Fresh(0));
        let true_ = ctx.add(Symbolic::Lit(Literal::Bool(true)));
        let false_lit = ctx.add(Symbolic::Lit(Literal::Bool(false)));

        assert!(ctx.prove_under_pc(g, &[(false_lit, Polarity::Positive)]));
        // The vacuous proof must NOT fuse the goal into `true` unconditionally.
        ctx.saturate();
        assert_ne!(ctx.egraph.find(g), ctx.egraph.find(true_));
    }

    #[test]
    fn prove_commits_conditional_implication() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let c = ctx.add(Symbolic::Fresh(0));
        let x = ctx.add(Symbolic::Fresh(1));
        let true_ = ctx.add(Symbolic::Lit(Literal::Bool(true)));
        // goal = `c ? true : x` — true under hypothesis `c`, unknown otherwise.
        let goal = ctx.add(Symbolic::Ite([c, true_, x]));

        // Provable under PC `<c>`; commits `c ==> goal` into the live graph.
        assert!(ctx.prove_under_pc(goal, &[(c, Polarity::Positive)]));

        // Not leaked unconditionally: `c` still unknown ⇒ goal not yet true.
        ctx.saturate();
        assert_ne!(ctx.egraph.find(goal), ctx.egraph.find(true_));

        // Once `c` is established, the goal collapses to `true`.
        ctx.union(c, true_);
        ctx.saturate();
        assert_eq!(ctx.egraph.find(goal), ctx.egraph.find(true_));
    }

    #[test]
    fn const_fold_folds_subtraction() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let one = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let diff = ctx.add(Symbolic::Binary(BinOp::SubR, [one, one]));
        ctx.egraph.rebuild();

        let zero = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(0).into())));
        assert_eq!(ctx.egraph.find(diff), ctx.egraph.find(zero));
    }

    #[test]
    fn const_fold_folds_ternary() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let cond = ctx.add(Symbolic::Lit(Literal::Bool(true)));
        let t = ctx.add(Symbolic::Fresh(0));
        let e = ctx.add(Symbolic::Fresh(1));
        let ite = ctx.add(Symbolic::Ite([cond, t, e]));
        ctx.saturate();

        // `true ? t : e` collapses to the symbolic `t`.
        assert_eq!(ctx.egraph.find(ite), ctx.egraph.find(t));
        assert_ne!(ctx.egraph.find(ite), ctx.egraph.find(e));
    }

    #[test]
    fn rewrite_ite_false_picks_else() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let cond = ctx.add(Symbolic::Lit(Literal::Bool(false)));
        let t = ctx.add(Symbolic::Fresh(0));
        let e = ctx.add(Symbolic::Fresh(1));
        let ite = ctx.add(Symbolic::Ite([cond, t, e]));
        ctx.saturate();

        assert_eq!(ctx.egraph.find(ite), ctx.egraph.find(e));
        assert_ne!(ctx.egraph.find(ite), ctx.egraph.find(t));
    }

    #[test]
    fn rewrite_add_zero_int() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let x = ctx.add(Symbolic::Fresh(0));
        let zero = ctx.add(Symbolic::Lit(Literal::Int(num::BigInt::from(0))));
        let sum = ctx.add(Symbolic::Binary(BinOp::AddI, [x, zero]));
        ctx.saturate();

        assert_eq!(ctx.egraph.find(sum), ctx.egraph.find(x));
    }

    #[test]
    fn rewrite_add_zero_real_commuted() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let x = ctx.add(Symbolic::Fresh(0));
        let zero = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(0).into())));
        // `0 + x` (commuted) must also fold to `x`.
        let sum = ctx.add(Symbolic::Binary(BinOp::AddR, [zero, x]));
        ctx.saturate();

        assert_eq!(ctx.egraph.find(sum), ctx.egraph.find(x));
    }

    #[test]
    fn eq_true_unions_args() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let b = ctx.add(Symbolic::Fresh(1));
        let eq = ctx.add(Symbolic::Binary(BinOp::Eq, [a, b]));
        let true_ = ctx.add(Symbolic::Lit(Literal::Bool(true)));
        // `assume a == b` is modelled as unioning the equality with `true`.
        ctx.union(eq, true_);
        ctx.saturate();

        assert_eq!(ctx.egraph.find(a), ctx.egraph.find(b));
    }

    #[test]
    fn eq_unknown_does_not_union() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let b = ctx.add(Symbolic::Fresh(1));
        // Build the equality but never prove it true.
        let _eq = ctx.add(Symbolic::Binary(BinOp::Eq, [a, b]));
        ctx.saturate();

        assert_ne!(ctx.egraph.find(a), ctx.egraph.find(b));
    }

    #[test]
    fn eq_true_propagates_through_congruence() {
        let mut interner = lasso::Rodeo::new();
        let f = crate::verify::lang::FuncId(lasso::Key::into_usize(interner.get_or_intern("f")));
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let b = ctx.add(Symbolic::Fresh(1));
        let fa = ctx.add(Symbolic::FuncApp(f, Box::from([]), Box::from([a])));
        let fb = ctx.add(Symbolic::FuncApp(f, Box::from([]), Box::from([b])));
        let eq = ctx.add(Symbolic::Binary(BinOp::Eq, [a, b]));
        let true_ = ctx.add(Symbolic::Lit(Literal::Bool(true)));
        ctx.union(eq, true_);
        ctx.saturate();

        // Unioning the args lets congruence close `f(a) == f(b)`.
        assert_eq!(ctx.egraph.find(fa), ctx.egraph.find(fb));
    }

    #[test]
    fn realcast_folds_int_to_real() {
        // real(2) const-folds to the Real literal 2.
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);
        let two = ctx.add(Symbolic::Lit(Literal::Int(num::BigInt::from(2))));
        let cast = ctx.add(Symbolic::RealCast(two));
        let real_two = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(2).into())));
        ctx.saturate();
        assert_eq!(ctx.egraph.find(cast), ctx.egraph.find(real_two));
    }

    #[test]
    fn rewrite_and_true_collapses() {
        // b && true  =  ite(b, true, false)  =>  b
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);
        let b = ctx.add(Symbolic::Fresh(0));
        let t = ctx.add(Symbolic::Lit(Literal::Bool(true)));
        let f = ctx.add(Symbolic::Lit(Literal::Bool(false)));
        let ite = ctx.add(Symbolic::Ite([b, t, f]));
        ctx.saturate();
        assert_eq!(ctx.egraph.find(ite), ctx.egraph.find(b));
    }

    #[test]
    fn rewrite_and_self_collapses() {
        // b && b  =  ite(b, b, false)  =>  b
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);
        let b = ctx.add(Symbolic::Fresh(0));
        let f = ctx.add(Symbolic::Lit(Literal::Bool(false)));
        let ite = ctx.add(Symbolic::Ite([b, b, f]));
        ctx.saturate();
        assert_eq!(ctx.egraph.find(ite), ctx.egraph.find(b));
    }
}
