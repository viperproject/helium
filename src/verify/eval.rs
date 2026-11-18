//! Evaluation of VMIR values: the per-walk value/heap/permission state
//! ([`EvalState`]) and the evaluator for pure and permission instructions.
//! Shared by the body walks in `declaration` and by the heap, which evaluates
//! a quantified permission's body at given binder values.

use crate::dhash::HashMap;
use crate::verify::heap::Loc;

use crate::verify::cert::PermRecipe;
use crate::vmir::display::VmirDisplay;
use crate::{
    verify::{
        context::VerifyContext,
        error::VerifyError,
        heap::{ChunkPerm, Heap, LocationKind},
        lang::Symbolic,
    },
    vmir::{
        self, BinOp, Declaration, HeapInst, HeapVal, Inst, InstKind, MemberId, PathConds, Polarity, PureInst, Type, Val,
    },
};

pub(crate) struct EvalState {
    pub(crate) vals: Vec<egg::Id>,
    /// The VMIR `Type` of each `vals` entry, kept in lockstep. The location kind
    /// of an address operand is read straight from here (`loc_kind`) — no e-graph
    /// inference. Params seed the initial slots; every `Pure` inst pushes its type.
    pub(crate) val_types: Vec<Type>,
    /// Recipe provenance of each `vals` entry, kept in lockstep: the
    /// recipe-space `Val` a certificate walk associates with the body temp
    /// (`None` when no recipe is built — method bodies — or the value has no
    /// pure recipe). One counter with `vals`, so the recipe column cannot
    /// desync from the eval column.
    pub(crate) recipes: Vec<Option<Val>>,
    pub(crate) heaps: Vec<Heap>,
    /// Permission temps (`p`), evaluated at their defining instruction. A `p`
    /// temp denotes ONE permission: a `wildcard` under it is minted once here, so
    /// reading the temp twice reads the same share.
    pub(crate) perms: Vec<ChunkPerm>,
    /// The defining instruction of each `perms` entry, kept in lockstep. The
    /// certificate walk needs the permission's *shape* (not its e-classes) to
    /// slice a [`PermRecipe`], and a `ChunkPerm` has already lost which operands
    /// it came from.
    pub(crate) perm_defs: Vec<vmir::PermInst>,
    /// Blocks proven **unreachable** (method bodies): the block's cube is
    /// refuted in the ground graph — e.g. after `bb_unreach`'s `inhale false` —
    /// or every predecessor is. The Stage-4 join merge drops such an arm outright
    /// instead of selecting a `0`-leaf against it (which would need an
    /// exhaustiveness split).
    ///
    /// Deadness is a property of a BLOCK, not of its exit heap temp: an empty
    /// block's exit heap is its predecessor's temp, so tagging the temp also
    /// tagged every live block that passes the same heap through, and a later
    /// join dropped a live arm.
    pub(crate) dead_blocks: crate::dhash::HashSet<usize>,
    /// For the join block being evaluated: whether its (then, else) predecessor
    /// blocks are dead. The `Merge` reads this; `None` outside a method join.
    pub(crate) join_arms_dead: Option<(bool, bool)>,
}

impl EvalState {
    pub(crate) fn new() -> Self {
        Self {
            vals: Vec::new(),
            val_types: Vec::new(),
            recipes: Vec::new(),
            heaps: Vec::new(),
            perms: Vec::new(),
            perm_defs: Vec::new(),
            dead_blocks: crate::dhash::HashSet::default(),
            join_arms_dead: None,
        }
    }

    pub(crate) fn with_args(args: Vec<egg::Id>, arg_types: Vec<Type>) -> Self {
        debug_assert_eq!(args.len(), arg_types.len());
        // Params are the identity in recipe space.
        let recipes = (0..args.len()).map(|i| Some(Val::Temp(i))).collect();
        Self {
            vals: args,
            val_types: arg_types,
            recipes,
            heaps: Vec::new(),
            perms: Vec::new(),
            perm_defs: Vec::new(),
            dead_blocks: crate::dhash::HashSet::default(),
            join_arms_dead: None,
        }
    }

    pub(crate) fn get_val(&self, ctx: &mut VerifyContext<'_>, val: &Val) -> egg::Id {
        match val {
            Val::Temp(n) => self.vals[*n],
            Val::Literal(lit) => ctx.add(Symbolic::Lit(lit.clone())),
        }
    }

    /// The recipe-space term of a body operand: a temp's tracked provenance
    /// (`None` if it has none), a literal as itself.
    pub(crate) fn recipe_of(&self, val: &Val) -> Option<Val> {
        match val {
            Val::Temp(n) => self.recipes[*n].clone(),
            Val::Literal(lit) => Some(Val::Literal(lit.clone())),
        }
    }

    /// `recipe_of` for a certificate-walk operand that must have a recipe.
    pub(crate) fn require_recipe(&self, val: &Val, what: &'static str) -> Result<Val, VerifyError> {
        self.recipe_of(val).ok_or(VerifyError::Unimplemented(what))
    }

    /// The path condition translated into recipe space (guards of an exported
    /// fact). Errors if a guard value has no recipe.
    fn recipe_pc(&self, pc: &PathConds) -> Result<Vec<(Val, Polarity)>, VerifyError> {
        pc.conds
            .iter()
            .map(|(v, p)| {
                self.require_recipe(v, "purify: path-condition value without a recipe")
                    .map(|r| (r, *p))
            })
            .collect()
    }

    /// The location kind of an address operand, from its tracked VMIR type. A
    /// non-`Temp` or non-`Addr`-typed operand has no kind (`None`).
    /// The location an address operand names: its kind (from its VMIR type)
    /// and its address. `None` for a non-address operand.
    pub(crate) fn loc(&self, ctx: &mut VerifyContext<'_>, val: &Val) -> Option<Loc> {
        let kind = self.loc_kind(val)?;
        Some(Loc {
            kind,
            addr: self.get_val(ctx, val),
        })
    }

    fn loc_kind(&self, val: &Val) -> Option<LocationKind> {
        match val {
            Val::Temp(n) => LocationKind::from_addr_type(&self.val_types[*n]),
            Val::Literal(_) => None,
        }
    }

    /// The evaluation state a quantifier body starts from. Capture is implicit —
    /// the body indexes *this* table directly (see [`vmir::Forall`]) — so the frame
    /// is this state cut to `binder_base`, which is the `forall` step's own temp:
    /// the first binder takes that slot, so the step's own boolean is unnameable
    /// from inside. Heaps are dropped; a quantifier body is heap-free by
    /// construction.
    pub(crate) fn frame_for(&self, q: &vmir::Forall) -> Self {
        self.frame_at(q.binder_base)
    }

    /// [`Self::frame_for`] at an explicit `binder_base`. A quantified
    /// permission's body uses [`Self::frame_from_caps`] instead.
    fn frame_at(&self, binder_base: usize) -> Self {
        let mut frame = Self {
            vals: self.vals.clone(),
            val_types: self.val_types.clone(),
            recipes: self.recipes.clone(),
            heaps: Vec::new(),
            // A quantifier body is permission-free, like it is heap-free.
            perms: Vec::new(),
            perm_defs: Vec::new(),
            dead_blocks: crate::dhash::HashSet::default(),
            join_arms_dead: None,
        };
        // The step itself is not evaluated yet, so the table ends exactly at its
        // temp; the truncation is a no-op guard, not a cut.
        debug_assert!(frame.vals.len() >= binder_base);
        frame.vals.truncate(binder_base);
        frame.val_types.truncate(binder_base);
        frame.recipes.truncate(binder_base);
        frame
    }

    /// The evaluation state a quantified permission's body starts from, built
    /// from the captures its chunk stored at the inhale instead of a live state:
    /// the chunk is a closure, `caps` its environment.
    ///
    /// The table is exactly `binder_base` long, so binders and body steps land on
    /// the temps the body was lowered with. Only the captured slots are filled;
    /// `caps` is [`vmir::QuantChunks::free_temps`], so no other slot below
    /// `binder_base` is read. The rest hold an id that is not an e-class, so a
    /// read of one (a `free_temps` that missed an operand) fails loudly instead
    /// of silently resolving against an unrelated value.
    ///
    /// Types and recipes are placeholders: pure evaluation reads types only to
    /// classify `Deref`/`Perm` addresses and recipes only in a certificate walk,
    /// and a quantified permission's body has neither.
    pub(crate) fn frame_from_caps(binder_base: usize, caps: &[(usize, egg::Id)]) -> Self {
        const UNCAPTURED: usize = usize::MAX;
        let mut vals = vec![egg::Id::from(UNCAPTURED); binder_base];
        for &(k, id) in caps {
            assert!(
                k < binder_base,
                "capture {k} is not below binder_base {binder_base}"
            );
            vals[k] = id;
        }
        Self {
            vals,
            val_types: vec![Type::Bool; binder_base],
            recipes: vec![None; binder_base],
            heaps: Vec::new(),
            perms: Vec::new(),
            perm_defs: Vec::new(),
            dead_blocks: crate::dhash::HashSet::default(),
            join_arms_dead: None,
        }
    }

    pub(crate) fn push_val(&mut self, id: egg::Id, ty: Type, recipe: Option<Val>) {
        self.vals.push(id);
        self.val_types.push(ty);
        self.recipes.push(recipe);
    }
    pub(crate) fn push_heap(&mut self, heap: Heap) {
        self.heaps.push(heap);
    }
    pub(crate) fn push_perm(&mut self, perm: ChunkPerm, def: vmir::PermInst) {
        self.perms.push(perm);
        self.perm_defs.push(def);
    }

    /// Whether a permission operand is wildcard-derived, without evaluating it.
    /// A `Temp` answers from its already-evaluated `ChunkPerm`, whose `wild` flag
    /// carries the provenance the e-graph cannot.
    pub(crate) fn perm_is_wild(&self, pv: &vmir::PermVal) -> bool {
        match pv {
            vmir::PermVal::Amount(_) => false,
            vmir::PermVal::Wildcard => true,
            vmir::PermVal::Temp(i) => self.perms[*i].has_wild(),
        }
    }
}

/// Render a single instruction (method or resource body) for error context.
pub(crate) fn format_inst(
    inst: &Inst,
    decls: &typed_index_collections::TiVec<MemberId, Declaration>,
    interner: &lasso::Rodeo,
    groups: &lasso::Rodeo<lasso::Spur>,
    val_base: usize,
    heap_base: usize,
) -> String {
    VmirDisplay::new(
        (val_base, heap_base, 0usize, std::slice::from_ref(inst)),
        decls,
        interner,
        groups,
    )
    .to_string()
    .trim()
    .to_string()
}

/// Heap-fetch is monomorphic — `HeapVal` carries no ctx-heap variant. The
/// caller-supplied ctx heap of a resource body lives at
/// `state.heaps[0]` by convention (mirroring how params occupy the
/// initial `vals` slots).
pub(crate) fn get_heap(state: &EvalState, hv: &HeapVal) -> Heap {
    match hv {
        HeapVal::Empty => Heap::empty(),
        HeapVal::Temp(n) => state.heaps[*n].clone(),
    }
}

/// Heaps to visualize for an instruction, labeled as in VMIR (`h0`, `h1`, …).
/// For a heap `combine` this is the base operand plus the result; for any other
/// instruction it is the current working heap (if any). Called after the
/// instruction has been evaluated, so the result heap sits at `heaps_before`.
pub(crate) fn display_heaps(state: &EvalState, kind: &InstKind, heaps_before: usize) -> Vec<(String, Heap)> {
    match kind {
        InstKind::Heap(
            HeapInst::Add { base, .. }
            | HeapInst::Sub { base, .. }
            | HeapInst::Inhale { base, .. }
            | HeapInst::Exhale { base, .. },
        ) => {
            let mut res = vec![(base.to_string(), get_heap(state, base))];
            if heaps_before < state.heaps.len() {
                res.push((
                    format!("h{heaps_before}"),
                    state.heaps[heaps_before].clone(),
                ));
            }
            res
        }
        _ => state
            .heaps
            .last()
            .map(|h| (format!("h{}", state.heaps.len() - 1), h.clone()))
            .into_iter()
            .collect(),
    }
}

/// Lower `pc.conds` to live ids, and nothing else. Use this only where the
/// ambient cube is already accounted for by the caller; every ordinary inst
/// wants [`collect_pc_lits`].
pub(crate) fn pc_lits_raw(
    ctx: &mut VerifyContext<'_>,
    state: &EvalState,
    pc: &PathConds,
) -> Vec<(egg::Id, Polarity)> {
    pc.conds
        .iter()
        .map(|(v, p)| (state.get_val(ctx, v), *p))
        .collect()
}

/// The **effective** path condition of an instruction:
///
/// ```text
/// effective_pc(inst) = ambient_cube(phase) ++ inst.pc
/// ```
///
/// `Inst.pc` is a delta (see [`Sink::ambient`](crate::translate::sink)), so the
/// ambient cube has to be prepended here. `ctx.current_cube()` *is* that ambient:
/// it is set by `begin_block` before a block's body walk and cleared by
/// `end_block` before its join walk, which is exactly the phase split the delta
/// rule names — a join materializes its own cube literals, so there the ambient
/// is empty and the delta is the whole pc.
pub(crate) fn collect_pc_lits(
    ctx: &mut VerifyContext<'_>,
    state: &EvalState,
    pc: &PathConds,
) -> Vec<(egg::Id, Polarity)> {
    // Owned first: `state.get_val` needs `&mut ctx`.
    let mut lits = ctx.current_cube().to_vec();
    lits.extend(pc_lits_raw(ctx, state, pc));
    lits
}

/// Error tag for a certificate-walk operand whose recipe is missing — cannot
/// happen for a value produced by a purifiable inst, so hitting it means an
/// unsupported source leaked into a function/resource body.
pub(crate) const OPERAND_RECIPE: &str = "purify: operand without a recipe";

/// Evaluate a `PureInst` into its symbolic e-class id, plus — when a
/// certificate recipe is being built (`ctx.recipe`) — the recipe-space `Val`
/// mirroring it (the single-walk replacement of the old purify pass). `pc_lits`
/// is the path condition of the owning instruction. A `Deref` consults it to
/// resolve an address that only aliases a held chunk under the branch (e.g.
/// `y.f` where `x == y` holds on this path); a `FunctionCall` assumes the
/// callee's `f%pre` token under it (the token's truth is what releases the
/// callee's body equality and exported facts). Every other variant ignores it.
///
/// `recipe_pc` is the *same* path condition before lowering, needed only when a
/// recipe is being built: a propagated `g%pre` token records it in recipe space
/// (via [`EvalState::recipe_pc`]) so the token is released only under the
/// body-internal condition guarding its call. `None` where no recipe-space pc can
/// be recovered — a trusted axiom body, or a `forall` WD check whose pc is the
/// concatenation `host_pc ++ inst.pc` with the host half already lowered to ids.
/// `None` records empty guards, i.e. exactly the pre-guard behavior: it can only
/// release a token too eagerly, never too late, so it cannot mask a real failure.
pub(crate) fn eval_pure_inst(
    ctx: &mut VerifyContext<'_>,
    state: &EvalState,
    ty: &Type,
    pi: &PureInst,
    pc_lits: &[(egg::Id, Polarity)],
    recipe_pc: Option<&PathConds>,
) -> Result<(egg::Id, Option<Val>), VerifyError> {
    use crate::verify::rewrite::AxiomPure;
    Ok(match pi {
        PureInst::Fresh => (ctx.fresh_symbolic_value(ty.clone()), None),
        PureInst::Binary(op, l, r) => {
            let lhs = state.get_val(ctx, l);
            let rhs = state.get_val(ctx, r);
            let id = ctx.add(Symbolic::Binary(*op, [lhs, rhs]));
            let recipe = if ctx.recipe.is_some() {
                let l = state.require_recipe(l, OPERAND_RECIPE)?;
                let r = state.require_recipe(r, OPERAND_RECIPE)?;
                let rb = ctx.recipe.as_mut().unwrap();
                Some(rb.emit(AxiomPure::Binary(*op, l, r)))
            } else {
                None
            };
            (id, recipe)
        }
        PureInst::Ternary(c, t, e) => {
            let cond = state.get_val(ctx, c);
            let then_ = state.get_val(ctx, t);
            let else_ = state.get_val(ctx, e);
            let id = expr!(ctx, if {cond} then {then_} else {else_});
            let recipe = if ctx.recipe.is_some() {
                let c = state.require_recipe(c, OPERAND_RECIPE)?;
                let t = state.require_recipe(t, OPERAND_RECIPE)?;
                let e = state.require_recipe(e, OPERAND_RECIPE)?;
                let rb = ctx.recipe.as_mut().unwrap();
                Some(rb.emit(AxiomPure::Ternary(c, t, e)))
            } else {
                None
            };
            (id, recipe)
        }
        PureInst::RealCast(v) => {
            let inner = state.get_val(ctx, v);
            let id = ctx.add(Symbolic::RealCast(inner));
            let recipe = if ctx.recipe.is_some() {
                let v = state.require_recipe(v, OPERAND_RECIPE)?;
                let rb = ctx.recipe.as_mut().unwrap();
                Some(rb.emit(AxiomPure::RealCast(v)))
            } else {
                None
            };
            (id, recipe)
        }
        PureInst::Deref(hv, loc) => {
            let heap = get_heap(state, hv);
            // In a certificate walk the read names its value by the address
            // recipe and records holder agreements under the recipe-space pc.
            // A pc guard without a recipe leaves the pc `None`: the read then
            // records no agreement, which only loses facts.
            let read_recipe = ctx.recipe.is_some().then(|| crate::verify::heap::ReadRecipe {
                addr: state.recipe_of(loc),
                pc: recipe_pc.and_then(|pc| state.recipe_pc(pc).ok()),
            });
            let held = match state.loc(ctx, loc) {
                Some(l) => heap.read(ctx, &l, pc_lits, read_recipe.as_ref())?,
                None => None,
            };
            match held {
                // In a certificate walk the held value's provenance is the
                // deref's pure term -- per heap state, so a two-state resource
                // reading the same address in `h0` and `h1` stays distinct.
                Some(v) if ctx.recipe.is_some() => match v.recipe {
                    Some(r) => (v.id, Some(r)),
                    None => {
                        return Err(VerifyError::Unimplemented(
                            "purify: deref outside footprint",
                        ));
                    }
                },
                Some(v) => (v.id, None),
                None if ctx.recipe.is_some() => {
                    return Err(VerifyError::Unimplemented(
                        "purify: deref outside footprint",
                    ));
                }
                None => (ctx.fresh_symbolic_value(ty.clone()), None),
            }
        }
        PureInst::FunctionCall(fc) => {
            // A (possibly generic) Silver `function`: `type_args` are the
            // result-type vars, part of the `FuncApp` node identity (discriminant);
            // empty for a monomorphic call. Always heap-free: a heap-dependent
            // function receives its precondition snapshot as an ordinary arg.
            //
            // Just add the uninterpreted application, for abstract, heap-free and
            // heap-dependent callees alike. A verified callee's definitional
            // equality `f(args) == body` is installed lazily by its own
            // `rewrite::function_rule`, not eagerly here.
            let args: Vec<egg::Id> = fc.args.iter().map(|v| state.get_val(ctx, v)).collect();
            let func_id = crate::verify::func_registry::func_id_for_member(fc.function);
            let id = ctx.add_func_app_id(
                func_id,
                fc.type_args.clone().into(),
                ty.clone(),
                args.clone().into(),
            );
            // Silicon's `f%pre`, assumed only at call sites. Every genuine
            // value-position call mints the callee's `f%pre(args)` token node,
            // whose *presence* lets `rewrite::function_rule` materialize the
            // callee's body here, and whose *truth* is what releases the resulting
            // axioms (`tok ==> f == body`, `tok ==> each exported fact`). The one
            // non-value position is a **contract-function body** (a lowered
            // pre/post), handled below.
            //
            // The truth is assumed under this call's path condition, so a call
            // under a ternary/implication cannot release the callee's facts onto a
            // sibling intra-block path. This is the only place a pc enters the
            // function-unfold encoding; every axiom above inherits it transitively
            // through the token. `.rev()` because `pc_lits` is outermost-first and
            // `implication` folds innermost-first (cf. `InstKind::Assume`).
            //
            // A callee in the recursion cycle being verified has no cert yet, but
            // still gets its token: the spec-derived post rule of the batch
            // (`rewrite::post_rule`) replays the induction hypothesis only behind
            // it, i.e. only on the path that makes the call.
            let in_scc = ctx
                .recipe
                .as_ref()
                .is_some_and(|rb| rb.is_recursive_callee(fc.function));
            if in_scc
                || ctx
                    .fn_certs
                    .is_none_or(|certs| certs.contains_key(&fc.function))
            {
                let name = ctx.member_name(fc.function);
                let tok = ctx.alloc.fn_pre_token(fc.function, &name);
                let tok_id = ctx.add(Symbolic::FuncApp(tok, Box::new([]), args.into()));
                ctx.assume_token_guarded(tok_id, pc_lits.iter().rev().copied());
            }
            let recipe = if ctx.recipe.is_some() {
                let args: Vec<Val> = fc
                    .args
                    .iter()
                    .map(|a| state.require_recipe(a, OPERAND_RECIPE))
                    .collect::<Result<_, _>>()?;
                // A recursive call (callee in this function's SCC) targets the
                // limited twin `f'` — uninterpreted, so unfolding this recipe
                // at a call site stops after one level. Every other callee
                // keeps its full id and unfolds normally.
                let rb = ctx.recipe.as_ref().unwrap();
                let recursive = rb.is_recursive_callee(fc.function);
                let propagates_recursive = rb.propagates_recursive_token();
                let func = if recursive {
                    let name = ctx.member_name(fc.function);
                    ctx.alloc.limited(fc.function, &name)
                } else {
                    func_id
                };
                // Precondition propagation (Silicon's `bodyPreconditionPropagation`):
                // emit the callee's `g%pre(gargs)` token as an orphan recipe step so
                // that when *this* body is unfolded (its own token present), the
                // nested token re-materializes and `g` may unfold in turn — the
                // cascade proceeds down genuine value chains.
                //
                // Which calls those are is the **translator's** call, carried on the
                // inst as `FunctionCall::export`: a value-position use of a Silver
                // function exports, an obligation call / a call inside a contract
                // definition / an `@addr` or domain application does not.
                //
                // A recursive call propagates too, although it targets the limited
                // twin `f'`, which has no unfold rule for the token to trigger: the
                // token is what releases `f`'s post at `f'(gargs)`
                // (`rewrite::function_post_rule`), and nothing else may. Except in a
                // postcondition, which is a verifier fact (SCC membership) rather
                // than the translator's: see
                // `RecipeBuilder::propagates_recursive_token`.
                let propagate = fc.export && (!recursive || propagates_recursive);
                let g_pre = propagate.then(|| {
                    let name = ctx.member_name(fc.function);
                    ctx.alloc.fn_pre_token(fc.function, &name)
                });
                let rb = ctx.recipe.as_mut().unwrap();
                let call = rb.emit(AxiomPure::App {
                    func,
                    type_args: fc.type_args.clone(),
                    args: args.clone(),
                });
                if let Some(g_pre) = g_pre {
                    let tok = rb.emit(AxiomPure::App {
                        func: g_pre,
                        type_args: Vec::new(),
                        args,
                    });
                    // The callee-internal pc of *this* call, in recipe space: the
                    // release becomes `f%pre(a) ==> (pc[x:=a] ==> g%pre(gargs))`, so
                    // a conditionally-called callee's axioms do not fire at args the
                    // body never calls it at.
                    let guards = match recipe_pc {
                        Some(pc) => state.recipe_pc(pc)?,
                        None => Vec::new(),
                    };
                    // The token's value is never consumed, so `RecipeBuilder::slice`
                    // would prune it as unreachable from the result. Register it as
                    // a slice root.
                    rb.record_token_step(tok, guards);
                }
                Some(call)
            } else {
                None
            };
            (id, recipe)
        }
        // A `forall` is a single e-node: the compiled body (its `RecipeId`, interned
        // program-wide) as payload, the captured terms as children. Nothing is
        // instantiated here — the one generic rule (`rewrite::forall_rule`) does that
        // at the next saturation, guarded on this node merging `true`. In a recipe
        // it is a `Forall` step: replayed at each call site, it rebuilds the
        // quantifier e-node with the caller's arguments as capture children.
        PureInst::Forall(q) => {
            // Compile the quantifier the first time the walk reaches it. Its
            // capture list is derived, not stored — see `Forall::free_temps`.
            // Name resolution rides on copies of the two shared refs, so it does
            // not borrow `ctx` across the `&mut ctx.alloc`.
            let (interner, decls) = (ctx.interner, ctx.decls);
            let names = move |m| crate::verify::context::member_name_in(interner, decls, m);
            let (recipe_id, free) = crate::verify::quant::intern_forall(ctx.alloc, &names, q)?;
            let free: Vec<Val> = free.iter().map(|&k| Val::Temp(k)).collect();
            let caps: Box<[egg::Id]> = free.iter().map(|v| state.get_val(ctx, v)).collect();
            // Stated directly (not inside an instantiated quantifier), so every
            // type parameter of a generic axiom's quantifier is still open.
            let type_args = ctx
                .alloc
                .quant_table()
                .read()
                .expect("recipe table lock")
                .get(recipe_id)
                .open_types();
            let id = ctx.add(Symbolic::Forall(
                recipe_id,
                type_args.iter().cloned().collect(),
                caps,
            ));
            let recipe = if ctx.recipe.is_some() {
                let caps: Vec<Val> = free
                    .iter()
                    .map(|v| state.require_recipe(v, OPERAND_RECIPE))
                    .collect::<Result<_, _>>()?;
                let rb = ctx.recipe.as_mut().unwrap();
                Some(rb.emit_forall(recipe_id, type_args, caps))
            } else {
                None
            };
            (id, recipe)
        }
        // `unwrap(v)`: the `Some` payload of an `Option`. The result type IS the
        // element type, so it needs no type argument. A certificate walk mirrors
        // the projection, exactly as a footprint slot's `unwrap(proj_i(s))` does.
        PureInst::OptionUnwrap(v) => {
            let opt = state.get_val(ctx, v);
            let id = ctx.option_unwrap(ty.clone(), opt);
            let recipe = if ctx.recipe.is_some() {
                let vr = state.require_recipe(v, OPERAND_RECIPE)?;
                let unwrap_id = ctx.alloc.option_value();
                let rb = ctx.recipe.as_mut().unwrap();
                Some(rb.emit(crate::verify::rewrite::AxiomPure::App {
                    func: unwrap_id,
                    type_args: vec![ty.clone()],
                    args: vec![vr],
                }))
            } else {
                None
            };
            (id, recipe)
        }
        // perm(loc): permission amount held at `loc` in the given heap. Has no
        // pure recipe — a certificate walk rejects it.
        PureInst::Perm(hv, loc) => {
            if ctx.recipe.is_some() {
                return Err(VerifyError::Unimplemented(
                    "purify: perm in a function or resource body",
                ));
            }
            let heap = get_heap(state, hv);
            let id = match state.loc(ctx, loc) {
                Some(l) => heap.perm(ctx, &l, pc_lits)?,
                None => expr!(ctx, 0 / 1),
            };
            (id, None)
        }
        // Semantic ADT nodes. Each is a `FuncApp` over a verifier-minted **concept**
        // id (one per `(head, variant[, field])`, see `verify::func_registry`); the
        // ground `type_args` ride in the node's operator identity (the discriminant),
        // so distinct instantiations never merge and the ADT reductions fire
        // per concept regardless of instantiation.
        PureInst::AdtCons {
            adt,
            type_args,
            variant,
            args,
        } => {
            let cons = ctx.alloc.cons(*adt, *variant);
            let arg_ids: Vec<egg::Id> = args.iter().map(|v| state.get_val(ctx, v)).collect();
            let id =
                ctx.add_func_app_id(cons, type_args.clone().into(), ty.clone(), arg_ids.into());
            let recipe = if ctx.recipe.is_some() {
                let args: Vec<Val> = args
                    .iter()
                    .map(|a| state.require_recipe(a, OPERAND_RECIPE))
                    .collect::<Result<_, _>>()?;
                let rb = ctx.recipe.as_mut().unwrap();
                Some(rb.emit(AxiomPure::App {
                    func: cons,
                    type_args: type_args.clone(),
                    args,
                }))
            } else {
                None
            };
            (id, recipe)
        }
        PureInst::AdtProj {
            adt,
            type_args,
            variant,
            field,
            base,
        } => {
            let proj = ctx.alloc.proj(*adt, *variant, *field);
            let base_id = state.get_val(ctx, base);
            let id = ctx.add_func_app_id(
                proj,
                type_args.clone().into(),
                ty.clone(),
                Box::new([base_id]),
            );
            let recipe = if ctx.recipe.is_some() {
                let base = state.require_recipe(base, OPERAND_RECIPE)?;
                let rb = ctx.recipe.as_mut().unwrap();
                Some(rb.emit(AxiomPure::App {
                    func: proj,
                    type_args: type_args.clone(),
                    args: vec![base],
                }))
            } else {
                None
            };
            (id, recipe)
        }
        // A discriminator is the tester `Adt::Ctor?(base)`, a `Bool` application;
        // the `is` / `variant-narrow` rules give it its meaning.
        PureInst::AdtIs {
            adt,
            type_args,
            variant,
            base,
        } => {
            let tester = ctx.alloc.adt_is(*adt, *variant);
            let base_id = state.get_val(ctx, base);
            let id = ctx.add_func_app_id(
                tester,
                type_args.clone().into(),
                Type::Bool,
                Box::new([base_id]),
            );
            let recipe = if ctx.recipe.is_some() {
                let base = state.require_recipe(base, OPERAND_RECIPE)?;
                let rb = ctx.recipe.as_mut().unwrap();
                Some(rb.emit(AxiomPure::App {
                    func: tester,
                    type_args: type_args.clone(),
                    args: vec![base],
                }))
            } else {
                None
            };
            (id, recipe)
        }
        PureInst::Coll { tys, op, args } => {
            let ids: Box<[egg::Id]> = args.iter().map(|v| state.get_val(ctx, v)).collect();
            let recipe = if ctx.recipe.is_some() {
                let args: Vec<Val> = args
                    .iter()
                    .map(|v| state.require_recipe(v, OPERAND_RECIPE))
                    .collect::<Result<_, _>>()?;
                let rb = ctx.recipe.as_mut().unwrap();
                Some(rb.emit(AxiomPure::Coll {
                    tys: tys.clone(),
                    op: *op,
                    args,
                }))
            } else {
                None
            };
            let id = ctx.add(Symbolic::Coll(tys.clone(), *op, ids.clone()));
            // A collection equality (`CollOp::is_eq`) shares its class with the
            // untyped `l == r`, which the rest of equality reasoning reads.
            if op.is_eq() {
                let eq = ctx.add(Symbolic::Binary(BinOp::Eq, [ids[0], ids[1]]));
                ctx.union(id, eq);
            }
            (id, recipe)
        }
        PureInst::CollLit {
            kind,
            elem_ty,
            elem,
        } => {
            let id = state.get_val(ctx, elem);
            let recipe = if ctx.recipe.is_some() {
                let elem = state.require_recipe(elem, OPERAND_RECIPE)?;
                let rb = ctx.recipe.as_mut().unwrap();
                Some(rb.emit(AxiomPure::CollLit {
                    kind: *kind,
                    elem_ty: elem_ty.clone(),
                    elem,
                }))
            } else {
                None
            };
            (
                ctx.add(Symbolic::CollLit(*kind, Box::new(elem_ty.clone()), id)),
                recipe,
            )
        }
    })
}

/// Resolve a [`vmir::PermVal`] to its e-class id, minting a fresh positive share
/// for a bare `wildcard`. A concrete `Amount` is exactly the value it wraps, so a
/// non-wildcard program builds the same term it always did.
///
/// A `Temp` was evaluated at its defining instruction, so it costs a lookup and
/// mints nothing: a `p` temp denotes one permission however many times it is read.
pub(crate) fn eval_perm(ctx: &mut VerifyContext<'_>, state: &EvalState, perm: &vmir::PermVal) -> egg::Id {
    eval_perm_structural(ctx, state, perm).to_id(ctx)
}

/// [`eval_perm`], but keeping the permission's **branch structure** as a
/// [`ChunkPerm`] instead of flattening it into one `Ite` term — the method-body
/// twin of [`build_perm`], which does the same for a footprint slot.
///
/// The term is identical either way (`ChunkPerm::to_id` rebuilds exactly the `Ite`
/// this used to add); what differs is that the structure is visible ABOVE the
/// e-graph, so a consume can align the demand's arms against the held permission's
/// (`prove_sufficient_aligned`, `perm_sub_aligned`) rather than proving against an
/// opaque `ite`. Without this, `exhale c ==> acc(x.f)` in a *method* stays opaque
/// even though the same assertion inside a predicate does not.
///
/// `ChunkPerm::select` is the smart constructor, so an ungated permission still
/// builds exactly the `Leaf` it did before.
pub(crate) fn eval_perm_structural(
    ctx: &mut VerifyContext<'_>,
    state: &EvalState,
    perm: &vmir::PermVal,
) -> ChunkPerm {
    match perm {
        vmir::PermVal::Amount(v) => ChunkPerm::leaf(state.get_val(ctx, v)),
        vmir::PermVal::Wildcard => ChunkPerm::wild_leaf(ctx.fresh_wildcard()),
        vmir::PermVal::Temp(i) => state.perms[*i].clone(),
    }
}

/// Evaluate a permission *instruction*, the definition of a `p` temp. The one
/// place a permission gains branch structure in a method body.
pub(crate) fn eval_perm_inst(
    ctx: &mut VerifyContext<'_>,
    state: &EvalState,
    inst: &vmir::PermInst,
) -> ChunkPerm {
    match inst {
        vmir::PermInst::Ite(c, t, e) => {
            let c = state.get_val(ctx, c);
            let t = eval_perm_structural(ctx, state, t);
            let e = eval_perm_structural(ctx, state, e);
            ChunkPerm::select(ctx, c, t, e)
        }
    }
}

/// The recipe-space form of a footprint-slot permission (resource certificate
/// build): the `p` steps the slot's permission reaches, in dependency order, with
/// each operand replaced by its recipe temp.
///
/// Deliberately *not* flattened into pure steps — see [`PermRecipe`] for why the
/// `wildcard` must stay out of a rebuilt step stream.
///
/// The walk memoizes per body temp, so a permission read twice yields one step and
/// hence one wildcard at graft time, matching what the IR says: a `p` temp is one
/// permission.
pub(crate) fn perm_recipe(state: &EvalState, perm: &vmir::PermVal) -> Result<PermRecipe<Val>, VerifyError> {
    fn leaf(
        state: &EvalState,
        pv: &vmir::PermVal,
        steps: &mut Vec<vmir::PermInst<Val>>,
        memo: &mut HashMap<usize, usize>,
    ) -> Result<vmir::PermVal<Val>, VerifyError> {
        Ok(match pv {
            vmir::PermVal::Amount(v) => {
                vmir::PermVal::Amount(state.require_recipe(v, OPERAND_RECIPE)?)
            }
            vmir::PermVal::Wildcard => vmir::PermVal::Wildcard,
            vmir::PermVal::Temp(i) => {
                if let Some(j) = memo.get(i) {
                    return Ok(vmir::PermVal::Temp(*j));
                }
                let vmir::PermInst::Ite(c, t, e) = &state.perm_defs[*i];
                let t = leaf(state, t, steps, memo)?;
                let e = leaf(state, e, steps, memo)?;
                let c = state.require_recipe(c, OPERAND_RECIPE)?;
                steps.push(vmir::PermInst::Ite(c, t, e));
                let j = steps.len() - 1;
                memo.insert(*i, j);
                vmir::PermVal::Temp(j)
            }
        })
    }
    let mut steps = Vec::new();
    let mut memo = HashMap::default();
    let res = leaf(state, perm, &mut steps, &mut memo)?;
    Ok(PermRecipe { steps, res })
}
