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
use crate::verify::rewrite::{AxiomInst, PreparedTerm};
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
    /// sequence axioms (`verify::seq_axioms`) have one today; a program's
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
    /// The built-in sequence axioms' recipes (`verify::seq_axioms`), compiled
    /// on first use.
    seq_axioms: Option<Vec<RecipeId>>,
}

impl RecipeTable {
    pub(crate) fn get(&self, id: RecipeId) -> &QuantRecipe {
        &self.recipes[id]
    }

    /// The recipes of the built-in sequence axioms, compiled into the table on
    /// the first call, their witnesses applications of `skolems`. Each is a
    /// closed quantifier (no captures) whose element type is open
    /// (`Generic(0)`), or `Int` for an axiom only about ranges.
    pub(crate) fn seq_axioms(
        &mut self,
        skolems: &crate::verify::seq_axioms::SeqSkolems,
    ) -> Vec<RecipeId> {
        if self.seq_axioms.is_none() {
            let ids = crate::verify::seq_axioms::recipes(skolems)
                .into_iter()
                .map(|r| self.recipes.push_and_get_key(r))
                .collect();
            self.seq_axioms = Some(ids);
        }
        self.seq_axioms.clone().expect("just compiled")
    }

    /// Whether `id` is one of the built-in sequence axioms' recipes.
    pub(crate) fn is_seq_axiom(&self, id: RecipeId) -> bool {
        self.seq_axioms
            .as_ref()
            .is_some_and(|ids| ids.contains(&id))
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
) -> Result<(RecipeId, Arc<[usize]>), VerifyError> {
    let table = Arc::clone(alloc.quant_table());
    let mut table = table.write().expect("recipe table lock");
    intern(alloc, &mut table, names, q)?;
    Ok(table.entry_of(q)?.clone())
}

fn intern_insts(
    alloc: &mut FuncRegistry,
    table: &mut RecipeTable,
    names: &dyn Fn(vmir::MemberId) -> String,
    insts: &[vmir::Inst],
) -> Result<(), VerifyError> {
    for inst in insts {
        if let InstKind::Pure(_, PureInst::Forall(q)) = &inst.kind {
            intern(alloc, table, names, q)?;
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
fn intern(
    alloc: &mut FuncRegistry,
    table: &mut RecipeTable,
    names: &dyn Fn(vmir::MemberId) -> String,
    q: &vmir::Forall,
) -> Result<RecipeId, VerifyError> {
    intern_insts(alloc, table, names, &q.body.insts)?;
    if let Some((id, _)) = table.by_forall.get(q) {
        return Ok(*id);
    }

    let free = q.free_temps();
    let n_caps = free.len();
    let slot: HashMap<usize, usize> = free.iter().copied().zip(0..).collect();
    // Canonicalize one operand. Below `binder_base` it is a capture, and its slot
    // is its position in `free`; at or above, it is a binder or a step, which keep
    // their order after the captures.
    let rename = |v: &Val| match v {
        Val::Temp(k) if *k < q.binder_base => Val::Temp(slot[k]),
        Val::Temp(k) => Val::Temp(n_caps + (k - q.binder_base)),
        lit => lit.clone(),
    };

    let insts: Vec<AxiomInst> = prepare_body(alloc, table, names, &q.body.insts)?
        .iter()
        .map(|i| crate::verify::cert::map_operands(i, rename))
        .collect();
    let groups: Vec<Vec<PreparedTerm>> = q
        .triggers
        .iter()
        .map(|g| {
            g.terms
                .iter()
                .map(|t| prepare_trig_term(alloc, t, q.binder_base, &slot))
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    let res = rename(&q.body.res);
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
    table.by_forall.insert(q.clone(), (id, free.into()));
    Ok(id)
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
            PureInst::Seq { elem_ty, .. } => elem_ty.collect_generics(out),
            PureInst::Forall(inner) if deep => forall_generics(inner, deep, out),
            _ => {}
        }
    }
}
