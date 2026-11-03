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
use crate::vmir::{self, InstKind, PureInst, Val};

/// A compiled `forall` body: the pure steps, the boolean result, the binder arity,
/// and the trigger groups (alternatives — a match of any one instantiates).
///
/// A recipe lives in a **canonical** temp space, independent of the program point
/// the quantifier was stated at: the captures first (they are the node's children,
/// so `n_caps` is the node's arity), then the `n_bound` binders (σ), then one per
/// step. [`intern`] renames the VMIR body — which shares its *enclosing* space, see
/// [`vmir::Forall`] — into it, which is what makes two structurally identical
/// quantifiers at different program points share one entry.
pub(crate) struct QuantRecipe {
    /// The capture arity: the number of enclosing values the body reads, hence the
    /// arity of every `Symbolic::Forall` node naming this recipe.
    pub(crate) n_caps: usize,
    pub(crate) n_bound: usize,
    /// Alternative trigger groups; each group is a conjunctive multi-pattern.
    pub(crate) groups: Vec<Vec<PreparedTerm>>,
    pub(crate) insts: Vec<AxiomInst>,
    pub(crate) res: Val,
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
    n_bound: usize,
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
}

impl RecipeTable {
    pub(crate) fn get(&self, id: RecipeId) -> &QuantRecipe {
        &self.recipes[id]
    }

    /// How many recipes have been compiled so far — i.e. how many distinct
    /// quantifiers the run has actually reached. Observes laziness; tests only.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.recipes.len()
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
                .collect()
        })
        .collect();
    let res = rename(&q.body.res);

    let key = RecipeKey {
        n_caps,
        n_bound: q.bound.len(),
        groups: groups.clone(),
        insts: insts.clone(),
        res: res.clone(),
    };
    let id = match table.by_key.get(&key) {
        Some(&id) => id,
        None => {
            let id = table.recipes.push_and_get_key(QuantRecipe {
                n_caps,
                n_bound: q.bound.len(),
                groups,
                insts,
                res,
            });
            table.by_key.insert(key, id);
            crate::trace::trace_event!(
                Quant,
                "recipe",
                recipe = id.0,
                bound = q.bound.len(),
                captures = n_caps,
                triggers = describe_triggers(alloc, names, &table.recipes[id].groups),
            );
            id
        }
    };
    table.by_forall.insert(q.clone(), (id, free.into()));
    Ok(id)
}

/// A recipe's trigger groups for the trace, e.g. `{f(x0), g(c0)} {h(x0)}`: binders
/// are `x<i>`, captures `c<i>`.
fn describe_triggers(
    alloc: &FuncRegistry,
    names: &dyn Fn(vmir::MemberId) -> String,
    groups: &[Vec<PreparedTerm>],
) -> String {
    fn term(
        t: &PreparedTerm,
        alloc: &FuncRegistry,
        names: &dyn Fn(vmir::MemberId) -> String,
    ) -> String {
        match t {
            PreparedTerm::Bound(i) => format!("x{i}"),
            PreparedTerm::Capture(i) => format!("c{i}"),
            PreparedTerm::Lit(l) => l.to_string(),
            PreparedTerm::App { func, args, .. } => {
                let head = alloc
                    .name(*func)
                    .map_or_else(|| names(vmir::MemberId::from(func.0)), str::to_string);
                let args: Vec<String> = args.iter().map(|a| term(a, alloc, names)).collect();
                format!("{head}({})", args.join(", "))
            }
        }
    }
    let groups: Vec<String> = groups
        .iter()
        .map(|g| {
            let terms: Vec<String> = g.iter().map(|t| term(t, alloc, names)).collect();
            format!("{{{}}}", terms.join(", "))
        })
        .collect();
    groups.join(" ")
}
