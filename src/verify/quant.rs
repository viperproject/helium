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
use crate::vmir::{self, InstKind, PureInst, SeqOp, TrigHead, TrigTerm, Val};

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
    check_triggers_fix_binder_types(q)?;

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
            id
        }
    };
    table.by_forall.insert(q.clone(), (id, free.into()));
    Ok(id)
}

/// Refuse a quantifier with a trigger group that does not fix the type of every
/// binder.
///
/// Matching is type-blind and relies on the head to say what its operands are: a
/// function has one signature, and an ADT head or a sequence literal carries its
/// type arguments. A `Symbolic::Seq` carries none, so `forall s: Seq[Int] :: {|s|}`
/// would also match `|t|` for a `t: Seq[Bool]`. Instantiating there states the
/// body about a value of the wrong type, which is unsound and can merge classes of
/// different types. Nothing at match time can tell the two apart, because an
/// e-class does not record its type.
fn check_triggers_fix_binder_types(q: &vmir::Forall) -> Result<(), VerifyError> {
    for group in q.triggers.iter() {
        let mut types = BinderTypes {
            binder_base: q.binder_base,
            fixed: vec![false; q.bound.len()],
            changed: true,
        };
        // A binder fixed in one place fixes its other occurrences, also across
        // the terms of a multi-pattern, so run until nothing changes.
        while types.changed {
            types.changed = false;
            for term in group.terms.iter() {
                types.visit(term, false);
            }
        }
        if types.fixed.contains(&false) {
            return Err(VerifyError::Unimplemented(
                "a sequence trigger that does not fix a binder's type",
            ));
        }
    }
    Ok(())
}

/// Which binders of one trigger group have a type the match itself determines.
struct BinderTypes {
    binder_base: usize,
    fixed: Vec<bool>,
    changed: bool,
}

impl BinderTypes {
    /// Walk `term`, where `known` says whether its position determines its type.
    /// Returns whether its type is determined.
    fn visit(&mut self, term: &TrigTerm, known: bool) -> bool {
        match term {
            TrigTerm::Lit(_) => true,
            // A capture matches one given e-class, of the type it was stated at.
            TrigTerm::Var(k) if *k < self.binder_base => true,
            TrigTerm::Var(k) => {
                let fixed = &mut self.fixed[k - self.binder_base];
                if known && !*fixed {
                    *fixed = true;
                    self.changed = true;
                }
                *fixed
            }
            TrigTerm::App {
                head: TrigHead::Seq(op),
                args,
                ..
            } => self.visit_seq(*op, args, known),
            TrigTerm::App { args, .. } => {
                for arg in args.iter() {
                    self.visit(arg, true);
                }
                true
            }
        }
    }

    fn visit_seq(&mut self, op: SeqOp, args: &[TrigTerm], known: bool) -> bool {
        match op {
            // The result says nothing about the operand.
            SeqOp::Len => {
                self.visit(&args[0], false);
                true
            }
            SeqOp::Range => {
                self.visit(&args[0], true);
                self.visit(&args[1], true);
                true
            }
            // The sequence and the element determine each other.
            SeqOp::Contains => {
                self.visit_linked(&[&args[0], &args[1]], false);
                true
            }
            // The result's type gives the sequence's and vice versa. The second
            // operand is an `Int` whatever the sequence holds.
            SeqOp::Index | SeqOp::Take | SeqOp::Drop => {
                self.visit(&args[1], true);
                self.visit(&args[0], known)
            }
            SeqOp::Concat => self.visit_linked(&[&args[0], &args[1]], known),
            SeqOp::Update => {
                self.visit(&args[1], true);
                self.visit_linked(&[&args[0], &args[2]], known)
            }
        }
    }

    /// Visit terms whose types determine each other: one known type fixes all.
    fn visit_linked(&mut self, terms: &[&TrigTerm], known: bool) -> bool {
        let known = terms.iter().fold(known, |k, t| self.visit(t, false) || k);
        for term in terms {
            self.visit(term, known);
        }
        known
    }
}
