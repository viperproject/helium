//! Evaluation of VMIR values: the per-walk value/heap/permission state
//! ([`EvalState`]) and the evaluator for pure and permission instructions.
//! Shared by the body walks in `declaration` and by the heap, which evaluates
//! a quantified permission's body at given binder values.

use crate::dhash::HashMap;
use crate::verify::heap::Loc;

use crate::verify::cert::PermRecipe;
use crate::verify::quant::{Resolved, ResolvedRead, Stand};

use crate::vmir::display::VmirDisplay;
use crate::{
    verify::{
        context::VerifyContext,
        error::VerifyError,
        heap::{ChunkPerm, Heap, LocationKind},
        lang::Symbolic,
    },
    vmir::{
        self, BinOp, CollOp, Declaration, HeapInst, HeapVal, Inst, InstKind, Literal, MapOp,
        MemberId, PathConds, Polarity, PureInst, SeqOp, Type, Val,
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
    /// The cube of the block being walked (`None` in a body without blocks):
    /// the region a heap built here gets ([`EvalState::region`]).
    pub(crate) block_cube: Option<PathConds>,
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
    /// The resolved heap reads and calls of the quantified-permission body
    /// being evaluated (see [`Resolved`]): its `Deref`s read these maps and its
    /// calls take these snapshots, never a heap (an instance frame has none).
    /// Empty in every other state.
    pub(crate) read_maps: Resolved,
    /// The resolved heap reads and calls of the pure `forall` about to be
    /// evaluated, set by the body walk from its [`check_forall_wd`] right
    /// before it: what the `Forall` step captures besides its temps.
    pub(crate) forall_reads: Resolved,
    /// In the quantified-permission instance being evaluated: the held set's
    /// wildcard function `W` and the body's location temp. A `wildcard` leaf
    /// is then `W(loc)` (see `heap::Captures::wildcard`). `None` mints a fresh
    /// wildcard, as everywhere else.
    pub(crate) wildcard: Option<(crate::verify::lang::FuncId, vmir::Val)>,
    /// The holders the framing obligation of the `Deref` being evaluated
    /// decided (see `inst_obligations`), keyed by its heap and address
    /// operands: its read takes the value from them instead of deciding the
    /// holders' positivity again. Set right before that one instruction's
    /// evaluation and taken by it; `None` everywhere else.
    pub(crate) framed_read: std::cell::Cell<Option<(HeapVal, Val, crate::verify::heap::Framing)>>,
}

impl EvalState {
    pub(crate) fn new() -> Self {
        Self {
            vals: Vec::new(),
            val_types: Vec::new(),
            recipes: Vec::new(),
            heaps: Vec::new(),
            block_cube: None,
            perms: Vec::new(),
            perm_defs: Vec::new(),
            dead_blocks: crate::dhash::HashSet::default(),
            join_arms_dead: None,
            read_maps: Resolved::default(),
            forall_reads: Resolved::default(),
            wildcard: None,
            framed_read: std::cell::Cell::new(None),
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
            block_cube: None,
            perms: Vec::new(),
            perm_defs: Vec::new(),
            dead_blocks: crate::dhash::HashSet::default(),
            join_arms_dead: None,
            read_maps: Resolved::default(),
            forall_reads: Resolved::default(),
            wildcard: None,
            framed_read: std::cell::Cell::new(None),
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
            block_cube: None,
            // A quantifier body is permission-free, like it is heap-free.
            perms: Vec::new(),
            perm_defs: Vec::new(),
            dead_blocks: crate::dhash::HashSet::default(),
            join_arms_dead: None,
            read_maps: Resolved::default(),
            forall_reads: Resolved::default(),
            wildcard: None,
            framed_read: std::cell::Cell::new(None),
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
            block_cube: None,
            perms: Vec::new(),
            perm_defs: Vec::new(),
            dead_blocks: crate::dhash::HashSet::default(),
            join_arms_dead: None,
            read_maps: Resolved::default(),
            forall_reads: Resolved::default(),
            wildcard: None,
            framed_read: std::cell::Cell::new(None),
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

    /// The region of a heap built here: the current block's cube, lowered.
    /// Never the instruction's own pc, whose suffix beyond the cube gates that
    /// one instruction only, not the heap the walk goes on with.
    ///
    /// In a join phase the cube is resolvable only once the join's reach
    /// values are defined. The lowering emits the `Merge` and everything after
    /// it behind them, so an undefined value here is a lowering bug, and
    /// `get_val` panics on it.
    pub(crate) fn region(&self, ctx: &mut VerifyContext<'_>) -> Vec<(egg::Id, Polarity)> {
        match &self.block_cube {
            Some(cube) => pc_lits_raw(ctx, self, cube),
            None => Vec::new(),
        }
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
///
/// The heap is used under `pc` (see [`Heap::at`]), which must establish the
/// region it was built in: an `old[L]` read fails where `L` may not have been
/// reached. A read passes its effective pc; an instruction that builds a new
/// heap passes [`EvalState::region`], so the result's region is the block's
/// cube. That holds for the empty heap too: a heap built on it in a block (a
/// loop head's invariant inhale) exists only where that block is reached.
pub(crate) fn get_heap(
    ctx: &mut VerifyContext<'_>,
    state: &EvalState,
    hv: &HeapVal,
    pc: &[(egg::Id, Polarity)],
) -> Result<Heap, VerifyError> {
    match hv {
        HeapVal::Empty => Heap::empty().at(ctx, pc),
        HeapVal::Temp(n) => state.heaps[*n].at(ctx, pc),
    }
}

/// The heap `hv` names exactly as it was built, its chunk guards relative to
/// its own region. Only for a consumer that rehomes it itself (a join merge,
/// which appends the arm literal) or does not reason with it (display).
pub(crate) fn get_heap_unrehomed(state: &EvalState, hv: &HeapVal) -> Heap {
    match hv {
        HeapVal::Empty => Heap::empty(),
        HeapVal::Temp(n) => state.heaps[*n].clone(),
    }
}

/// Heaps to visualize for an instruction, labeled as in VMIR (`h0`, `h1`, …).
/// For a heap `combine` this is the base operand plus the result; for any other
/// instruction it is the current working heap (if any). Called after the
/// instruction has been evaluated, so the result heap sits at `heaps_before`.
pub(crate) fn display_heaps(
    state: &EvalState,
    kind: &InstKind,
    heaps_before: usize,
) -> Vec<(String, Heap)> {
    match kind {
        InstKind::Heap(
            HeapInst::Add { base, .. }
            | HeapInst::Sub { base, .. }
            | HeapInst::Inhale { base, .. }
            | HeapInst::Exhale { base, .. },
        ) => {
            let mut res = vec![(base.to_string(), get_heap_unrehomed(state, base))];
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

/// The side conditions of a pure instruction, as `(goal, error)` pairs that
/// must each be proven under its path condition: a non-zero divisor, a map key
/// in the map's domain, a sequence index in range. Callees' preconditions are
/// not here: the lowering stitches them in as `Assert`s.
///
/// Shared by every body walk (the per-instruction obligations of
/// `declaration.rs`) and by the checked walk of a quantifier's body
/// ([`eval_body_checked`]), so the two cannot disagree on what a pure
/// expression needs.
pub(crate) fn pure_obligations(
    ctx: &mut VerifyContext<'_>,
    state: &EvalState,
    ty: &Type,
    pi: &PureInst,
) -> Vec<(egg::Id, VerifyError)> {
    match pi {
        // `not(divisor == 0)`. The divisor is homogeneous with the result
        // (casts), so the VMIR result type gives the zero's type.
        PureInst::Binary(op, _, r) if op.is_div_or_mod() => {
            let rv = state.get_val(ctx, r);
            let zero = match ty {
                Type::Int => ctx.add(Symbolic::Lit(Literal::Int(num::BigInt::from(0)))),
                _ => expr!(ctx, 0 / 1),
            };
            let goal = expr!(ctx, not({ rv } == { zero }));
            vec![(goal, VerifyError::SideCondition("divisor may be zero"))]
        }
        // A map lookup `m[k]` needs `k in domain(m)`.
        PureInst::Coll {
            tys,
            op: CollOp::Map(MapOp::Lookup),
            args: val,
        } => {
            let m = state.get_val(ctx, &val[0]);
            let k = state.get_val(ctx, &val[1]);
            let domain = ctx.add(Symbolic::Coll(
                tys.clone(),
                MapOp::Domain.into(),
                Box::new([m]),
            ));
            let present = ctx.add(Symbolic::Coll(
                Box::new([tys[0].clone()]),
                vmir::SetOp::Contains.into(),
                Box::new([domain, k]),
            ));
            vec![(
                present,
                VerifyError::SideCondition("map may not contain the key"),
            )]
        }
        PureInst::Coll {
            tys,
            op: op @ CollOp::Seq(_),
            args: val,
        } if op.has_side_condition() => {
            // `Index` is [s, i] and `Update` is [s, i, v], so both start with s and i.
            let s = state.get_val(ctx, &val[0]);
            let i = state.get_val(ctx, &val[1]);
            let len = ctx.add(Symbolic::Coll(
                tys.clone(),
                SeqOp::Len.into(),
                Box::new([s]),
            ));
            let lower = expr!(ctx, not({ i } <i 0));
            let upper = expr!(ctx, { i } < i { len });
            vec![
                (
                    lower,
                    VerifyError::SideCondition("sequence index may be negative"),
                ),
                (
                    upper,
                    VerifyError::SideCondition("sequence index may be out of bounds"),
                ),
            ]
        }
        _ => vec![],
    }
}

/// The path condition of one instruction of a quantifier's body, given the
/// body state it is evaluated in (see [`eval_body_checked`]).
pub(crate) type BodyPc<'a> =
    dyn FnMut(&mut VerifyContext<'_>, &EvalState, &Inst) -> Vec<(egg::Id, Polarity)> + 'a;

/// Evaluate a quantifier's body `insts` into `state` (whose binders are fresh
/// values) while discharging what makes it well-defined, so the body is shown
/// well-defined **for an arbitrary binding**: each instruction's
/// [`pure_obligations`], and the `assert f#requires(..)` stitched at a call to
/// a contract-bearing function. A callee's postcondition (`Assume`) is assumed
/// under the same pc. A nested `forall` is checked by [`check_forall_wd`].
///
/// `pc_of` gives each instruction's path condition: the host's plus the body's
/// own guards (short-circuit lowering emits them, so a guard proves its own
/// consequent's WD), and for a quantified permission also the binding's
/// condition. `Perm` instructions are evaluated; a pure `forall` has none.
///
/// The caller runs this on a scratch graph with the recipe paused: whatever
/// the fresh binders touch must not reach the live graph or a certificate.
/// How a walk at fresh binders hands each heap access of a quantifier body to
/// a [`crate::verify::heap::ReadResolver`] instead of evaluating it: a `Deref`
/// ([`Self::read`]) and a heap-dependent call's frame-only exhale
/// ([`Self::call`]), each given its pc, return their value at the walked
/// binding.
pub(crate) trait BodyReads {
    fn read(
        &mut self,
        ctx: &mut VerifyContext<'_>,
        state: &EvalState,
        heap: &HeapVal,
        addr: &Val,
        pc: &[(egg::Id, Polarity)],
    ) -> Result<egg::Id, VerifyError>;
    fn call(
        &mut self,
        ctx: &mut VerifyContext<'_>,
        state: &EvalState,
        heap: HeapVal,
        call: &vmir::ResourceCall,
        pc: &[(egg::Id, Polarity)],
    ) -> Result<egg::Id, VerifyError>;
}

/// A heap-dependent call's snapshot in a quantifier body walk: from `hook`
/// when one resolves the body, else the resolved call the state carries.
pub(crate) fn body_call_snapshot(
    ctx: &mut VerifyContext<'_>,
    state: &EvalState,
    hook: Option<&mut &mut dyn BodyReads>,
    heap: HeapVal,
    call: &vmir::ResourceCall,
    pc: &[(egg::Id, Polarity)],
) -> Result<egg::Id, VerifyError> {
    let Some(h) = hook else {
        let slots = state.read_maps.call_at(state.vals.len()).slots.clone();
        let args: Vec<egg::Id> = call.args.iter().map(|v| state.get_val(ctx, v)).collect();
        let from = crate::verify::declaration::SnapshotFrom::Stands(slots);
        return Ok(crate::verify::declaration::frame_snapshot(
            ctx,
            call.resource,
            &args,
            from,
            pc,
        )?
        .snap);
    };
    h.call(ctx, state, heap, call, pc)
}

pub(crate) fn eval_body_checked(
    ctx: &mut VerifyContext<'_>,
    state: &mut EvalState,
    insts: &[Inst],
    pc_of: &mut BodyPc<'_>,
    mut hook: Option<&mut dyn BodyReads>,
) -> Result<(), VerifyError> {
    for inst in insts {
        let pc_lits = pc_of(ctx, state, inst);
        match &inst.kind {
            InstKind::Pure(ty, PureInst::Deref(heap, addr)) if let Some(h) = &mut hook => {
                let id = h.read(ctx, state, heap, addr, &pc_lits)?;
                state.push_val(id, ty.clone(), None);
            }
            InstKind::Heap(hi) if let Some((heap, call)) = hi.frame_exhale() => {
                let id = body_call_snapshot(ctx, state, hook.as_mut(), heap, call, &pc_lits)?;
                state.push_val(id, Type::Snap(call.resource), None);
            }
            InstKind::Pure(ty, pi) => {
                for (goal, err) in pure_obligations(ctx, state, ty, pi) {
                    if !ctx.prove_under_pc(goal, &pc_lits) {
                        return Err(err);
                    }
                }
                // A nested quantifier: checked under this level's fresh
                // binders, then built like any other step.
                if let PureInst::Forall(inner) = pi {
                    // Its maps would have to be resolved against heaps this
                    // frame does not hold.
                    if !inner.heap_reads().is_empty() || !inner.heap_calls().is_empty() {
                        return Err(VerifyError::Unimplemented(
                            "heap read in a nested quantifier",
                        ));
                    }
                    check_forall_wd(ctx, inner, state, &pc_lits)?;
                }
                let (id, _) = eval_pure_inst(ctx, state, ty, pi, &pc_lits, None)?;
                state.push_val(id, ty.clone(), None);
            }
            InstKind::Perm(pi) => {
                let p = eval_perm_inst(ctx, state, pi);
                state.push_perm(p, pi.clone());
            }
            InstKind::Assume(val) => {
                let id = state.get_val(ctx, val);
                ctx.assume_guarded(id, pc_lits.iter().rev().copied());
            }
            // The reason a WD check needs the *ambient* facts: the guard that
            // establishes a callee's precondition may be the body's own
            // implication, or a fact the host already knows.
            InstKind::Assert(val) => {
                let id = state.get_val(ctx, val);
                if !ctx.prove_under_pc(id, &pc_lits) {
                    return Err(VerifyError::AssertionFailed);
                }
            }
            InstKind::Refute(_) | InstKind::Heap(_) => {
                return Err(VerifyError::Unimplemented(
                    "impure inst in a quantifier body",
                ));
            }
        }
    }
    Ok(())
}

/// Check the well-definedness of a `forall` **at the point it is encountered**,
/// in a throwaway clone of the live e-graph.
///
/// Cloning (rather than seeding a blank graph) makes every ambient fact — the
/// host's path condition, the assumed axioms, grafted certificates — available
/// for free, with no seeding policy and no pollution of the real graph: whatever
/// the binders' fresh values touch dies with the scratch graph.
///
/// The binders become fresh values while every enclosing value keeps its real
/// e-class, so the body's side conditions are discharged *for an arbitrary
/// binding* ([`eval_body_checked`]): `forall x :: {f(x)} x != 0 ==> f(10 / x)`
/// discharges the division under `<x != 0>`.
///
/// A nested `forall` is checked recursively, with its encloser's binders
/// already fresh, so no WD obligation survives into a recipe — instantiation runs
/// inside a rewrite rule, where nothing can be proven.
pub(crate) fn check_forall_wd(
    ctx: &mut VerifyContext<'_>,
    q: &vmir::Forall,
    host: &EvalState,
    host_pc: &[(egg::Id, Polarity)],
) -> Result<Resolved, VerifyError> {
    // A quantifier body holds no permission instruction.
    if q.body
        .insts
        .iter()
        .any(|i| matches!(i.kind, InstKind::Perm(_)))
    {
        return Err(VerifyError::Unimplemented(
            "permission instruction in a quantifier body",
        ));
    }
    // The same walk resolves the body's heap reads and calls against the
    // host's heaps.
    let heap_of = |ctx: &mut VerifyContext<'_>, h: &HeapVal| get_heap(ctx, host, h, host_pc);
    let mut resolver = crate::verify::heap::ReadResolver::new(heap_of);
    // `ctx.egraph` is a scratch clone for the duration; the live graph, its
    // fixpoint cache and the memo scope are handled by `with_scratch_graph`.
    // The certificate recipe (if one is being built) is paused: the WD walk's
    // scratch evaluation must not mirror steps into the host's recipe.
    ctx.pause_recipe(|ctx| {
        ctx.with_scratch_graph(|ctx| {
            // Capture is implicit: the body indexes the *enclosing* value table
            // directly, so the body's state is that table cut to the
            // quantifier's frame — which starts at the `forall` step's own temp,
            // shadowed by the first binder.
            let mut state = host.frame_for(q);
            for ty in q.bound.iter() {
                let fresh = ctx.fresh_symbolic_value(ty.clone());
                state.push_val(fresh, ty.clone(), None);
            }
            // The body's own guards sit *under* the host's. `host_pc` came from
            // a `collect_pc_lits` at the enclosing inst, so it already carries
            // the ambient cube — the body's ambient *is* `host_pc`. Prepending
            // the cube a second time would break `cube_matches`'s length test
            // and silently disable the block scratch.
            let mut pc_of = |ctx: &mut VerifyContext<'_>, state: &EvalState, inst: &Inst| {
                let mut pc = host_pc.to_vec();
                pc.extend(pc_lits_raw(ctx, state, &inst.pc));
                pc
            };
            eval_body_checked(
                ctx,
                &mut state,
                &q.body.insts,
                &mut pc_of,
                Some(&mut resolver),
            )
        })
    })?;
    Ok(resolver.finish())
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
        // A resolved read of a quantified permission's body: its stand-in.
        PureInst::Deref(hv, loc)
            if let Some((_, read)) = ResolvedRead::find(&state.read_maps.reads, hv, loc) =>
        {
            let id = match read.stand {
                Stand::Value(v) => v,
                Stand::Map(m) => {
                    let addr = state.get_val(ctx, loc);
                    let lookup = ctx.alloc.value_map(&read.kind).lookup;
                    ctx.add(Symbolic::FuncApp(lookup, Box::new([]), Box::new([m, addr])))
                }
            };
            (id, None)
        }
        PureInst::Deref(hv, loc) => {
            let heap = get_heap(ctx, state, hv, pc_lits)?;
            // In a certificate walk the read names its value by the address
            // recipe.
            let read_recipe = ctx
                .recipe
                .is_some()
                .then(|| crate::verify::heap::ReadRecipe {
                    addr: state.recipe_of(loc),
                });
            // The holders the framing obligation already decided, when this
            // walk checked one for this very read.
            let framed = state
                .framed_read
                .take()
                .filter(|(h, l, _)| h == hv && l == loc)
                .map(|(_, _, f)| f);
            let held = match (framed, state.loc(ctx, loc)) {
                (Some(f), _) => f.value(ctx, read_recipe.as_ref()),
                (None, Some(l)) => heap.read(ctx, &l, pc_lits, read_recipe.as_ref())?,
                (None, None) => None,
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
            // Its heap reads, resolved by this program point's `check_forall_wd`.
            let reads = &state.forall_reads;
            if reads.reads.len() != q.heap_reads().len()
                || reads.calls.len() != q.heap_calls().len()
            {
                return Err(VerifyError::Unimplemented(
                    "heap read in a quantifier evaluated outside a body walk",
                ));
            }
            // Compile the quantifier the first time the walk reaches it. Its
            // capture list is derived, not stored — see `Forall::free_temps`.
            // Name resolution rides on copies of the two shared refs, so it does
            // not borrow `ctx` across the `&mut ctx.alloc`.
            let (interner, decls) = (ctx.interner, ctx.decls);
            let names = move |m| crate::verify::context::member_name_in(interner, decls, m);
            let (recipe_id, free) = crate::verify::quant::intern_forall(
                ctx.alloc,
                &names,
                q,
                &reads.shape(),
                ctx.certs,
            )?;
            let free: Vec<Val> = free.iter().map(|&k| Val::Temp(k)).collect();
            // Children: the captured temps, then the read stand-ins and call
            // snapshots (`Canonical`).
            let temps = free.iter().map(|v| state.get_val(ctx, v));
            let caps: Box<[egg::Id]> = temps.chain(reads.cap_ids()).collect();
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
                let mut caps: Vec<Val> = free
                    .iter()
                    .map(|v| state.require_recipe(v, OPERAND_RECIPE))
                    .collect::<Result<_, _>>()?;
                // A stand-in with no recipe cannot be rebuilt where the
                // certificate is grafted.
                for r in reads.cap_recipes() {
                    caps.push(r.ok_or(VerifyError::Unimplemented(
                        "heap read in a quantifier of a resource body without a recipe",
                    ))?);
                }
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
            let heap = get_heap(ctx, state, hv, pc_lits)?;
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
pub(crate) fn eval_perm(
    ctx: &mut VerifyContext<'_>,
    state: &EvalState,
    perm: &vmir::PermVal,
) -> egg::Id {
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
        // `W(loc)`, positive: a fact about a fresh function at one argument,
        // so assuming it wherever an instance is evaluated is consistent. The
        // location precedes the amount in the body (`lower_quant_chunks`).
        vmir::PermVal::Wildcard => match &state.wildcard {
            Some((w, loc)) => {
                let addr = state.get_val(ctx, loc);
                let amount = ctx.add(Symbolic::FuncApp(*w, Box::new([]), Box::new([addr])));
                let pos = expr!(ctx, (0 / 1) < r { amount });
                let true_ = expr!(ctx, true);
                ctx.union(pos, true_);
                ChunkPerm::wild_leaf(amount)
            }
            None => ChunkPerm::wild_leaf(ctx.fresh_wildcard()),
        },
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
pub(crate) fn perm_recipe(
    state: &EvalState,
    perm: &vmir::PermVal,
) -> Result<PermRecipe<Val>, VerifyError> {
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
