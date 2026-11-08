//! Identifying magic wands.
//!
//! A wand `A --* B` is a resource, and Silicon identifies an instance of it by
//! its *structure* and the values of its *holes*: the maximal subexpressions
//! that neither read the current heap (a read under `old(..)` does not count)
//! nor mention a variable bound inside the wand (Silver's
//! `MagicWand.subexpressionsToEvaluate`). A hole is evaluated where the wand is
//! inhaled, exhaled or applied; every other part of the wand is evaluated in the
//! state that `apply` consumes `A` from and produces `B` into, as in Viper. Two
//! wands with the same structure and equal hole values are therefore the same
//! resource. The key is conservative: names bound inside the wand stay in it,
//! so two equal wands may fail to match (incompleteness), but two different
//! ones never do. One departure from Silicon, which keeps in the structure
//! whether an `acc` spells out its amount: `P(x)`, `acc(P(x))` and
//! `acc(P(x), write)` all assert the same resource, so they are one shape here
//! (the amount is a hole).

use crate::dhash::{HashMap, HashSet};
use lasso::Spur;

use crate::viper::{
    self,
    typed::{self, Type, WandId},
    walk::{AstWalkable, AstWalker, AstWalkerMut},
};

/// The wand structures of one program, interned in order of first occurrence.
pub(super) struct WandTable {
    /// Functions whose precondition reads the heap (Silver's
    /// `couldBeHeapDependent`): a call to one is evaluated in the current state,
    /// so it cannot be a hole.
    heap_fns: HashSet<Spur>,
    ids: HashMap<(viper::Exp, Vec<Type>), WandId>,
    pub(super) wands: Vec<typed::Wand>,
}

impl WandTable {
    pub(super) fn new(program: &viper::Program) -> Self {
        let pres: Vec<(Spur, &[viper::Exp])> = program
            .0
            .iter()
            .filter_map(|decl| match decl {
                viper::Declaration::Function(f) => {
                    Some((f.signature.name.0.id(), f.contract.precondition.as_slice()))
                }
                _ => None,
            })
            .collect();
        // Least fixpoint: a precondition reads the heap through a call too.
        let mut heap_fns = HashSet::default();
        loop {
            let found: Vec<Spur> = pres
                .iter()
                .filter(|(name, pres)| {
                    !heap_fns.contains(name) && pres.iter().any(|e| pinned(e, &heap_fns, &[]))
                })
                .map(|(name, _)| *name)
                .collect();
            if found.is_empty() {
                break;
            }
            heap_fns.extend(found);
        }
        Self {
            heap_fns,
            ids: HashMap::default(),
            wands: Vec::new(),
        }
    }

    /// Split `wand` (an `A --* B` that went through constraint generation) into
    /// its structure, with every hole replaced by a placeholder, and its holes in
    /// traversal order, each still carrying its type key.
    pub(super) fn split(&self, wand: &viper::Exp) -> (viper::Exp, Vec<viper::Exp>) {
        let mut structure = wand.clone();
        let mut holes = Holes {
            heap_fns: &self.heap_fns,
            bound: Vec::new(),
            holes: Vec::new(),
        };
        holes.walk_mut_exp(&mut structure);
        (structure, holes.holes)
    }

    /// The id of the structure `structure` with holes of types `holes`.
    pub(super) fn intern(&mut self, structure: viper::Exp, holes: Vec<Type>) -> WandId {
        let next = WandId(self.wands.len());
        *self
            .ids
            .entry((structure, holes))
            .or_insert_with_key(|(_, holes)| {
                self.wands.push(typed::Wand {
                    holes: holes.clone(),
                });
                next
            })
    }
}

/// Whether `exp` reads the current heap or mentions a name in `bound`: what
/// keeps a subexpression of a wand from being a hole.
fn pinned(exp: &viper::Exp, heap_fns: &HashSet<Spur>, bound: &[Spur]) -> bool {
    let mut p = Pinned {
        heap_fns,
        bound,
        under_old: false,
        found: false,
    };
    exp.walk(&mut p);
    p.found
}

/// Whether `exp` mentions a magic wand anywhere.
pub(super) fn mentions_wand(exp: &viper::Exp) -> bool {
    struct FindWand(bool);
    impl<'a> AstWalker<'a> for FindWand {
        fn walk_exp_kind(&mut self, exp: &'a viper::ExpKind) {
            self.0 |= matches!(exp, viper::ExpKind::MagicWand(..));
            exp.walk_children(self);
        }
    }
    let mut find = FindWand(false);
    exp.walk(&mut find);
    find.0
}

struct Pinned<'w> {
    heap_fns: &'w HashSet<Spur>,
    bound: &'w [Spur],
    under_old: bool,
    found: bool,
}

impl<'a> AstWalker<'a> for Pinned<'_> {
    fn walk_exp_kind(&mut self, exp: &'a viper::ExpKind) {
        use viper::ExpKind as K;
        let reads_heap = match exp {
            K::Old(_, inner) => {
                let outer = std::mem::replace(&mut self.under_old, true);
                inner.walk(self);
                self.under_old = outer;
                return;
            }
            K::Ident(id) => {
                self.found |= self.bound.contains(&id.id());
                false
            }
            K::Field(..)
            | K::Acc(_)
            | K::MagicWand(..)
            | K::HeapUpdate(..)
            | K::ForPerm(..)
            | K::UnOp(viper::UnOp::Perm, _) => true,
            K::Call(call) => match call.kind {
                Some(viper::ExpCallKind::Predicate) => true,
                Some(viper::ExpCallKind::Function) => self.heap_fns.contains(&call.name.id()),
                _ => false,
            },
            _ => false,
        };
        self.found |= reads_heap && !self.under_old;
        exp.walk_children(self);
    }
}

/// Replaces each hole of a wand by a placeholder, collecting the holes, and
/// erases every type key the rest of the structure carries.
struct Holes<'w> {
    heap_fns: &'w HashSet<Spur>,
    bound: Vec<Spur>,
    holes: Vec<viper::Exp>,
}

impl<'a> AstWalkerMut<'a> for Holes<'_> {
    fn walk_mut_exp(&mut self, exp: &'a mut viper::Exp) {
        use viper::ExpKind as K;
        // A binder is never a hole itself (as in Silver), whatever its body.
        let binds = matches!(*exp.kind, K::LetIn(..) | K::Quantifier(..));
        if !binds && !pinned(exp, self.heap_fns, &self.bound) {
            let placeholder = viper::Exp::unknown(K::Ident(viper::Ident::Raw("$hole".into())));
            self.holes.push(std::mem::replace(exp, placeholder));
            return;
        }
        exp.ty = viper::InferenceType::Unknown;
        match &mut *exp.kind {
            K::LetIn(binder, value, body) => {
                self.walk_mut_exp(value);
                self.bound.push(binder.0.id());
                self.walk_mut_exp(body);
                self.bound.pop();
            }
            K::Quantifier(_, vars, triggers, body) => {
                let outer = self.bound.len();
                self.bound.extend(vars.iter().map(|v| v.idn.0.id()));
                triggers.walk_mut(self);
                self.walk_mut_exp(body);
                self.bound.truncate(outer);
            }
            kind => kind.walk_mut_children(self),
        }
    }
}
