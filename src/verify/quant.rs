//! The program-level **recipe table**: every syntactic `forall` in the program,
//! compiled once into registry-resolved pure steps + trigger patterns, and named
//! by a [`RecipeId`] that rides in the payload of a
//! [`Symbolic::Forall`](crate::verify::lang::Symbolic::Forall) e-node.
//!
//! A VMIR `forall` body shares its *enclosing* temp space (capture is implicit,
//! see [`vmir::Forall`]); a recipe lives in a canonical `caps ++ binders ++ steps`
//! space. [`intern`] does that renaming, which is what makes the same quantifier
//! stated at two program points one entry — and what turns the derived capture
//! list into the node's children.
//!
//! The table is built **before any verification unit runs** (`verify::verify`) and
//! frozen thereafter: instantiation can materialize new forall *nodes* mid-run
//! (an outer instance builds its inner quantifier with the outer σ as capture
//! children) but never a new *recipe* — a nested body is interned innermost-first
//! along with its encloser. That is what lets the single instantiation rule hold a
//! plain `Arc` snapshot of the table with no staleness risk: egg cannot inject
//! rules into a running `Runner`, but it does see a new *e-node* on the next
//! iteration.

use crate::dhash::HashMap;
use std::sync::Arc;

use typed_index_collections::TiVec;

use crate::verify::declaration::{prepare_body, prepare_trig_term};
use crate::verify::error::VerifyError;
use crate::verify::func_registry::FuncRegistry;
use crate::verify::lang::RecipeId;
use crate::verify::rewrite::{AxiomInst, AxiomPure, PreparedTerm};
use crate::vmir::{self, InstKind, PureInst, Type, Val};

/// A compiled `forall` body: the pure steps, the boolean result, the binder arity,
/// and the trigger groups (alternatives — a match of any one instantiates).
///
/// A recipe lives in a **canonical** temp space, independent of the program point
/// the quantifier was stated at: the captures first (they are the node's children,
/// so `n_caps` is the node's arity), then the binders (σ, see `bound`), then one per
/// step. [`intern`] renames the VMIR body — which shares its *enclosing* space, see
/// [`vmir::Forall`] — into it, which is what makes two structurally identical
/// quantifiers at different program points share one entry.
pub(crate) struct QuantRecipe {
    /// The capture arity: the number of enclosing values the body reads, hence the
    /// arity of every `Symbolic::Forall` node naming this recipe.
    pub(crate) n_caps: usize,
    /// The binder types, in binder order. Z3 reads them to declare its bound
    /// variables (`verify::z3`); a generic quantifier's mention its `Generic(i)`.
    pub(crate) bound: Box<[Type]>,
    /// The type arity: one past the highest `Generic(i)` the quantifier (binders,
    /// triggers, body, nested quantifiers) mentions, hence the length of every
    /// `Symbolic::Forall` node's type payload. `0` for a ground quantifier; only a
    /// generic axiom's quantifiers have one.
    pub(crate) n_tys: usize,
    /// The type parameters this quantifier's own level (binders, triggers, body
    /// outside nested quantifiers) mentions: what an instantiation must have
    /// fixed. A parameter only a nested quantifier mentions may stay open, for
    /// that quantifier's own trigger to fix.
    pub(crate) own_tys: Vec<usize>,
    /// Alternative trigger groups; each group is a conjunctive multi-pattern.
    pub(crate) groups: Vec<Vec<PreparedTerm>>,
    pub(crate) insts: Vec<AxiomInst>,
    pub(crate) res: Val,
    /// The condition an instance waits for (see [`Guard`]). Only the built-in
    /// collection axioms (`verify::coll_axioms`) have one today; a program's
    /// quantifier is instantiated at every match.
    pub(crate) guard: Option<Guard>,
}

/// When a recipe's instances are worth building, independent of its body and
/// not part of what the quantifier says: the applier builds the instance at a
/// σ only once every conjunct is already known `true` in the e-graph, and until
/// then skips it without memoizing it, so a later iteration builds it. For a
/// body `a ==> b`, a guard `a` skips the instances that would say nothing yet
/// but add terms the next round matches triggers against. The guard is
/// evaluated by lookup only and never added to the graph.
pub(crate) struct Guard {
    /// The steps that compute the conjuncts, in their own temp space: the seed
    /// (captures ++ binders, as for the body), then one temp per value-producing
    /// step. Shares no steps with the body.
    pub(crate) insts: Vec<AxiomInst>,
    /// The conjuncts, in that temp space.
    pub(crate) conds: Vec<Val>,
}

impl QuantRecipe {
    /// The type payload of a node whose type parameters are all still open:
    /// `Generic(0..n_tys)`. How a quantifier is stated in a generic axiom; an
    /// enclosing instantiation substitutes into it.
    pub(crate) fn open_types(&self) -> Vec<Type> {
        (0..self.n_tys).map(Type::Generic).collect()
    }
}

/// The identity of a recipe: the canonicalized body **and** its triggers. Two
/// `forall`s dedup onto one entry only when they denote the same proposition *and*
/// were written with the same patterns — a body match alone must not pool trigger
/// sets, or a quantifier silently instantiates on a trigger its author never wrote.
/// Alpha-equivalence and program-point independence are free: canonicalization has
/// already made binders and captures positional.
#[derive(PartialEq, Eq, Hash)]
struct RecipeKey {
    n_caps: usize,
    bound: Box<[Type]>,
    n_tys: usize,
    own_tys: Vec<usize>,
    groups: Vec<Vec<PreparedTerm>>,
    insts: Vec<AxiomInst>,
    res: Val,
}

#[derive(Default)]
pub(crate) struct RecipeTable {
    recipes: TiVec<RecipeId, QuantRecipe>,
    /// Dedup index: canonicalized recipe ⟹ id.
    by_key: HashMap<RecipeKey, RecipeId>,
    /// Lookup index for the eval walk: the syntactic `forall` ⟹ its recipe and its
    /// derived capture list ([`vmir::Forall::free_temps`], in the *enclosing* temp
    /// space). A `vmir::Forall` is `Hash + Eq`, so no id bookkeeping is needed on
    /// the IR, and caching the free list keeps a re-walked body from rescanning.
    by_forall: HashMap<vmir::Forall, (RecipeId, Arc<[usize]>)>,
    /// The built-in collection axioms' recipes (`verify::coll_axioms`), by
    /// `CollKind` (`CollKind::ALL` order), each compiled on first use.
    coll_axioms: [Option<Vec<RecipeId>>; 4],
    /// The axioms of each location kind's value map
    /// (`verify::builtin_axioms::value_map`), compiled on first use.
    value_map_axioms: HashMap<crate::verify::heap::LocationKind, Vec<RecipeId>>,
}

impl RecipeTable {
    pub(crate) fn get(&self, id: RecipeId) -> &QuantRecipe {
        &self.recipes[id]
    }

    /// The recipes of the built-in axioms of the collection `kind`, compiled
    /// into the table on the first call, their witnesses applications of
    /// `skolems`. Each is a closed quantifier (no captures) whose collection
    /// type arguments are open (`Generic(0)`, and `Generic(1)` for a map's
    /// value type), or `Int` for an axiom only about ranges.
    pub(crate) fn coll_axioms(
        &mut self,
        kind: crate::vmir::CollKind,
        skolems: &crate::verify::builtin_axioms::Skolems,
    ) -> Vec<RecipeId> {
        let slot = &mut self.coll_axioms[kind as usize];
        if slot.is_none() {
            let ids = crate::verify::builtin_axioms::recipes(kind, skolems)
                .into_iter()
                .map(|r| self.recipes.push_and_get_key(r))
                .collect();
            *slot = Some(ids);
        }
        slot.clone().expect("just compiled")
    }

    /// The axioms of `kind`'s value map, its functions `f`, compiled into the
    /// table on the first call.
    pub(crate) fn value_map_axioms(
        &mut self,
        kind: &crate::verify::heap::LocationKind,
        f: crate::verify::func_registry::ValueFunction,
    ) -> Vec<RecipeId> {
        if let Some(ids) = self.value_map_axioms.get(kind) {
            return ids.clone();
        }
        let ids: Vec<RecipeId> = crate::verify::builtin_axioms::value_map::recipes(kind, f)
            .into_iter()
            .map(|r| self.recipes.push_and_get_key(r))
            .collect();
        self.value_map_axioms.insert(kind.clone(), ids.clone());
        ids
    }

    /// Add a recipe built outside [`intern`], bypassing dedup: used for the
    /// inverse axiom of a quantified permission, which names freshly minted
    /// inverse functions and so can never equal an existing recipe.
    pub(crate) fn push_fresh(&mut self, recipe: QuantRecipe) -> RecipeId {
        self.recipes.push_and_get_key(recipe)
    }

    /// Whether `id` is one of the built-in collection axioms' recipes.
    pub(crate) fn is_coll_axiom(&self, id: RecipeId) -> bool {
        self.coll_axioms
            .iter()
            .flatten()
            .any(|ids| ids.contains(&id))
    }

    /// How many recipes have been compiled so far — i.e. how many distinct
    /// quantifiers the run has actually reached. Observes laziness; tests only.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.recipes.len()
    }

    /// A table of the one recipe `recipe`, for applier tests.
    #[cfg(test)]
    pub(crate) fn of(recipe: QuantRecipe) -> (Self, RecipeId) {
        let mut table = Self::default();
        let id = table.recipes.push_and_get_key(recipe);
        (table, id)
    }

    /// The recipe of an already-interned `forall`, with its capture list. A miss is
    /// a bug: every read is either preceded by [`intern_forall`] on the same
    /// quantifier, or reads a nested one that its encloser's innermost-first
    /// interning has already compiled.
    pub(crate) fn entry_of(
        &self,
        q: &vmir::Forall,
    ) -> Result<&(RecipeId, Arc<[usize]>), VerifyError> {
        self.by_forall
            .get(q)
            .ok_or(VerifyError::Unimplemented("forall recipe not interned"))
    }
}

/// Compile `q` if it has not been compiled yet, and hand back its recipe id and
/// derived capture list. This is the *only* way a recipe enters the table, and it
/// runs on the eval walk — so a quantifier a unit never reaches costs that unit
/// nothing, and a quantifier stated later in a body cannot be scanned for before
/// the walk gets there.
///
/// The write lock is taken once for the whole (recursive, innermost-first)
/// operation: `std::sync::RwLock` is not reentrant, and nothing under here touches
/// the e-graph, so no rule can be running.
pub(crate) fn intern_forall(
    alloc: &mut FuncRegistry,
    names: &dyn Fn(vmir::MemberId) -> String,
    q: &vmir::Forall,
    by_value: &[bool],
) -> Result<(RecipeId, Arc<[usize]>), VerifyError> {
    let table = Arc::clone(alloc.quant_table());
    let mut table = table.write().expect("recipe table lock");
    intern(alloc, &mut table, names, q, by_value)
}

fn intern_insts(
    alloc: &mut FuncRegistry,
    table: &mut RecipeTable,
    names: &dyn Fn(vmir::MemberId) -> String,
    insts: &[vmir::Inst],
) -> Result<(), VerifyError> {
    for inst in insts {
        if let InstKind::Pure(_, PureInst::Forall(q)) = &inst.kind {
            // Its maps would have to be capture children of the enclosing
            // recipe's `AxiomInst::Forall`, which carries only its temps.
            if !q.heap_reads().is_empty() {
                return Err(VerifyError::Unimplemented(
                    "heap read in a nested quantifier",
                ));
            }
            intern(alloc, table, names, q, &[])?;
        }
    }
    Ok(())
}

/// Compile one `forall` and intern it, **innermost-first**: its nested `forall`s
/// are interned first, so this body can name each of them by id (and read back its
/// capture list) while compiling. Idempotent.
///
/// The VMIR body lives in the *enclosing* temp space (capture is implicit); the
/// recipe lives in the canonical `caps ++ binders ++ steps` space the applier
/// replays. Canonicalization is the whole job here:
///
/// ```text
///   free:   e0, e2   (Forall::free_temps, triggers first)  -->  t0, t1
///   binder: e5                          (binder_base = 5)  -->  t2
///   steps:  e6, e7                                         -->  t3, t4
/// ```
///
/// A body that reads the heap compiles by how its reads were resolved
/// (`by_value`, see [`Stand`]), which can differ between uses: it is not
/// cached by occurrence, and its recipe is found again by [`RecipeKey`].
fn intern(
    alloc: &mut FuncRegistry,
    table: &mut RecipeTable,
    names: &dyn Fn(vmir::MemberId) -> String,
    q: &vmir::Forall,
    by_value: &[bool],
) -> Result<(RecipeId, Arc<[usize]>), VerifyError> {
    intern_insts(alloc, table, names, &q.body.insts)?;
    if let Some(entry) = table.by_forall.get(q) {
        return Ok(entry.clone());
    }

    let free = q.free_temps();
    let c = Canonical::new(
        alloc,
        table,
        names,
        q.binder_base,
        q.bound.len(),
        &free,
        &q.heap_reads(),
        by_value,
        &q.triggers,
        &q.body.insts,
    )?;
    let res = c.rename(&q.body.res);
    let (n_caps, insts, groups) = (c.n_caps(), c.insts, c.groups);
    let mut generics = crate::dhash::HashSet::default();
    forall_generics(q, true, &mut generics);
    let n_tys = generics.iter().max().map_or(0, |m| m + 1);
    let mut own = crate::dhash::HashSet::default();
    forall_generics(q, false, &mut own);
    let mut own_tys: Vec<usize> = own.into_iter().collect();
    own_tys.sort_unstable();

    let key = RecipeKey {
        n_caps,
        bound: q.bound.clone(),
        n_tys,
        own_tys: own_tys.clone(),
        groups: groups.clone(),
        insts: insts.clone(),
        res: res.clone(),
    };
    let id = match table.by_key.get(&key) {
        Some(&id) => id,
        None => {
            let id = table.recipes.push_and_get_key(QuantRecipe {
                n_caps,
                bound: q.bound.clone(),
                n_tys,
                own_tys,
                groups,
                insts,
                res,
                guard: None,
            });
            table.by_key.insert(key, id);
            id
        }
    };
    let entry = (id, Arc::from(free));
    if by_value.is_empty() {
        table.by_forall.insert(q.clone(), entry.clone());
    }
    Ok(entry)
}

/// One heap read of a quantifier body ([`vmir::HeapRead`]), resolved to what
/// stands for the enclosing heap there (`heap::ReadResolver`), so the body
/// stays heap-free: the stand-in is one more capture, in instances
/// ([`crate::verify::eval::EvalState::read_maps`]) and in the compiled body
/// ([`Canonical`], after the captured temps) alike.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedRead {
    /// The `Deref`'s heap and address operand: which read this resolves.
    pub(crate) heap: vmir::HeapVal,
    pub(crate) addr: Val,
    pub(crate) kind: crate::verify::heap::LocationKind,
    pub(crate) stand: Stand,
    /// The stand-in's recipe in a certificate walk (a footprint slot's seed),
    /// which a pure `forall` there captures it by; `None` otherwise.
    pub(crate) recipe: Option<Val>,
}

/// What stands for a resolved read.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Stand {
    /// A chunk set's value map: the read is `lookup(map, addr)`.
    Map(egg::Id),
    /// A single chunk's value, which the read is for every binding: one chunk
    /// frames it, so every address the read names is that chunk's.
    Value(egg::Id),
}

impl Stand {
    /// The captured e-class.
    pub(crate) fn id(self) -> egg::Id {
        match self {
            Stand::Map(id) | Stand::Value(id) => id,
        }
    }
}

/// How [`prepare_body`] compiles a resolved `Deref` (see [`Stand`]).
pub(crate) enum ReadStep {
    /// `lookup(map, addr)`, the map a capture already in the recipe's space.
    Lookup(Box<crate::verify::heap::LocationKind>, Val),
    /// No step: the read is its captured value, which [`Canonical::rename`]
    /// names in its place.
    Captured,
}

impl ResolvedRead {
    /// The resolved read `Deref(heap, addr)` among `reads`, with its index.
    pub(crate) fn find<'a>(
        reads: &'a [ResolvedRead],
        heap: &vmir::HeapVal,
        addr: &Val,
    ) -> Option<(usize, &'a ResolvedRead)> {
        reads
            .iter()
            .enumerate()
            .find(|(_, r)| r.heap == *heap && r.addr == *addr)
    }
}

/// A quantifier body moved into a recipe's canonical `caps ++ binders ++ steps`
/// temp space: its steps prepared ([`prepare_body`]) and renamed, its triggers
/// prepared. What [`intern`] does to a pure `forall`; also the starting point
/// of a quantified permission's inverse axiom
/// (`heap::quantified_permissions`), which [`Self::push`]es steps after it.
pub(crate) struct Canonical {
    binder_base: usize,
    /// Enclosing temp -> capture slot.
    slot: HashMap<usize, usize>,
    /// How many read stand-ins are captured after the temps (see [`ResolvedRead`]).
    n_reads: usize,
    /// Body temp of a read captured by value -> its capture slot. Its step is
    /// dropped ([`ReadStep::Captured`]), so [`Self::rename`] closes the gaps
    /// those steps leave (`dropped`, ascending).
    alias: HashMap<usize, usize>,
    dropped: Vec<usize>,
    /// The next free temp: one past the binders and every value step so far.
    next: usize,
    pub(crate) insts: Vec<AxiomInst>,
    pub(crate) groups: Vec<Vec<PreparedTerm>>,
}

impl Canonical {
    /// `free` is the scope's capture list (enclosing temps, in child order);
    /// the stand-ins of its heap `reads` ([`vmir::HeapRead`], resolved at each
    /// use into [`ResolvedRead`]s) are captured after them, in order, read `j`
    /// by value iff `by_value[j]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        alloc: &mut FuncRegistry,
        table: &RecipeTable,
        names: &dyn Fn(vmir::MemberId) -> String,
        binder_base: usize,
        n_bound: usize,
        free: &[usize],
        reads: &[vmir::HeapRead],
        by_value: &[bool],
        triggers: &[vmir::QuantTrigger],
        insts: &[vmir::Inst],
    ) -> Result<Self, VerifyError> {
        // Read `j`'s stand-in is the capture after the temps': slot `free.len() + j`.
        let read_of = |heap: &vmir::HeapVal, addr: &Val| {
            let j = reads
                .iter()
                .position(|r| r.heap == *heap && r.addr == *addr)?;
            Some((j, free.len() + j))
        };
        // The steps, as body temps from `binder_base + n_bound` on; a read
        // captured by value aliases its capture and drops its step.
        let mut alias = HashMap::default();
        let mut dropped = Vec::new();
        let steps = insts.iter().filter_map(|i| match &i.kind {
            InstKind::Pure(_, pi) => Some(pi),
            _ => None,
        });
        let mut n_vals = 0;
        for (t, pi) in (binder_base + n_bound..).zip(steps) {
            n_vals += 1;
            if let PureInst::Deref(heap, addr) = pi
                && let Some((j, slot)) = read_of(heap, addr)
                && by_value[j]
            {
                alias.insert(t, slot);
                dropped.push(t);
            }
        }
        let mut c = Self {
            binder_base,
            slot: free.iter().copied().zip(0..).collect(),
            n_reads: reads.len(),
            next: free.len() + reads.len() + n_bound + n_vals - dropped.len(),
            alias,
            dropped,
            insts: Vec::new(),
            groups: Vec::new(),
        };
        let read = |heap: &vmir::HeapVal, addr: &Val| {
            let (j, slot) = read_of(heap, addr)?;
            if by_value[j] {
                return Some(ReadStep::Captured);
            }
            let kind = crate::verify::heap::LocationKind::from_addr_type(&reads[j].addr_ty)
                .expect("a heap read's address is an address");
            Some(ReadStep::Lookup(Box::new(kind), Val::Temp(slot)))
        };
        c.insts = prepare_body(alloc, table, names, insts, &|v| c.rename(v), &read)?;
        c.groups = triggers
            .iter()
            .map(|g| {
                g.terms
                    .iter()
                    .map(|t| prepare_trig_term(alloc, t, binder_base, &c.slot))
                    .collect::<Result<_, _>>()
            })
            .collect::<Result<_, _>>()?;
        Ok(c)
    }

    /// The capture arity: the captured temps, then the read stand-ins.
    pub(crate) fn n_caps(&self) -> usize {
        self.slot.len() + self.n_reads
    }

    /// Canonicalize one operand. Below `binder_base` it is a capture, and its
    /// slot is its position in `free`; a read captured by value is its
    /// capture; otherwise it is a binder or a step, which keep their order
    /// after the captures, less the dropped steps before it.
    pub(crate) fn rename(&self, v: &Val) -> Val {
        match v {
            Val::Temp(k) if *k < self.binder_base => Val::Temp(self.slot[k]),
            Val::Temp(k) if let Some(&s) = self.alias.get(k) => Val::Temp(s),
            Val::Temp(k) => {
                let gaps = self.dropped.partition_point(|d| d < k);
                Val::Temp(self.n_caps() + (k - self.binder_base) - gaps)
            }
            lit => lit.clone(),
        }
    }

    /// The trigger pattern the canonical value `v` is built as, if it mentions
    /// every one of the `n_bound` binders: captures and binders become
    /// variables, a function application step an application. `None` if some
    /// part is not matchable (an interpreted operation) or a binder is missing.
    ///
    /// What a quantified permission's location `f(e(xs))` triggers on when the
    /// receiver reads the heap: the read is a `lookup` of a captured map here,
    /// which a VMIR trigger cannot name.
    pub(crate) fn pattern_of(&self, v: &Val, n_bound: usize) -> Option<PreparedTerm> {
        fn build(
            c: &Canonical,
            v: &Val,
            n_bound: usize,
            seen: &mut Vec<bool>,
        ) -> Option<PreparedTerm> {
            let n_caps = c.n_caps();
            Some(match v {
                Val::Literal(lit) => PreparedTerm::Lit(lit.clone()),
                Val::Temp(k) if *k < n_caps => PreparedTerm::Capture(*k),
                Val::Temp(k) if *k < n_caps + n_bound => {
                    seen[k - n_caps] = true;
                    PreparedTerm::Bound(k - n_caps)
                }
                // A step: the `k`-th instruction that holds a temp slot.
                Val::Temp(k) => {
                    let slotted =
                        |i: &&AxiomInst| matches!(i, AxiomInst::Val(_) | AxiomInst::Forall { .. });
                    let step = c.insts.iter().filter(slotted).nth(k - n_caps - n_bound)?;
                    let AxiomInst::Val(AxiomPure::App {
                        func,
                        type_args,
                        args,
                    }) = step
                    else {
                        return None;
                    };
                    let args = args.iter().map(|a| build(c, a, n_bound, seen));
                    PreparedTerm::App {
                        head: crate::verify::rewrite::TermHead::Func {
                            func: *func,
                            type_args: type_args.iter().cloned().collect(),
                        },
                        args: args.collect::<Option<_>>()?,
                    }
                }
            })
        }
        let mut seen = vec![false; n_bound];
        let pattern = build(self, v, n_bound, &mut seen)?;
        seen.iter().all(|&b| b).then_some(pattern)
    }

    /// Append a step (operands already canonical) and return its temp.
    pub(crate) fn push(&mut self, p: crate::verify::rewrite::AxiomPure) -> Val {
        self.insts.push(AxiomInst::Val(p));
        self.next += 1;
        Val::Temp(self.next - 1)
    }
}

/// Every `Generic(i)` index `q` mentions: its binder types, its triggers' type
/// arguments, and every type of its body — nested quantifiers included when
/// `deep`.
fn forall_generics(q: &vmir::Forall, deep: bool, out: &mut crate::dhash::HashSet<usize>) {
    fn trig(t: &vmir::TrigTerm, out: &mut crate::dhash::HashSet<usize>) {
        if let vmir::TrigTerm::App {
            type_args, args, ..
        } = t
        {
            type_args.iter().for_each(|ty| ty.collect_generics(out));
            args.iter().for_each(|a| trig(a, out));
        }
    }
    q.bound.iter().for_each(|ty| ty.collect_generics(out));
    q.triggers
        .iter()
        .flat_map(|g| g.terms.iter())
        .for_each(|t| trig(t, out));
    for inst in &q.body.insts {
        let InstKind::Pure(ty, pi) = &inst.kind else {
            continue;
        };
        ty.collect_generics(out);
        match pi {
            PureInst::FunctionCall(fc) => fc.type_args.iter().for_each(|t| t.collect_generics(out)),
            PureInst::AdtCons { type_args, .. }
            | PureInst::AdtProj { type_args, .. }
            | PureInst::AdtIs { type_args, .. } => {
                type_args.iter().for_each(|t| t.collect_generics(out))
            }
            PureInst::Coll { tys, .. } => tys.iter().for_each(|t| t.collect_generics(out)),
            PureInst::CollLit { elem_ty, .. } => elem_ty.collect_generics(out),
            PureInst::Forall(inner) if deep => forall_generics(inner, deep, out),
            _ => {}
        }
    }
}
