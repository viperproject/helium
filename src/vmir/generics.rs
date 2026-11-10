//! Generic-parameter bookkeeping for declarations whose bodies mention
//! `Type::Generic` (domain axioms): visiting every type a body mentions, and
//! compacting the parameters it actually uses.

use crate::vmir::{Forall, FunctionBody, InstKind, PureInst, TrigTerm, Type};

impl FunctionBody {
    /// Visit every type this body mentions, mutably: instruction result types,
    /// call / ADT `type_args`, and (recursively) every nested `forall`'s binder
    /// types and trigger `type_args`. Each callback sees a *whole* type, not its
    /// sub-types, so a rewrite is one `subst_generics`.
    pub fn for_each_type_mut(&mut self, f: &mut impl FnMut(&mut Type)) {
        for inst in &mut self.insts {
            if let InstKind::Pure(ty, pi) = &mut inst.kind {
                f(ty);
                match pi {
                    PureInst::FunctionCall(fc) => fc.type_args.iter_mut().for_each(&mut *f),
                    PureInst::AdtCons { type_args, .. }
                    | PureInst::AdtProj { type_args, .. }
                    | PureInst::AdtIs { type_args, .. } => type_args.iter_mut().for_each(&mut *f),
                    PureInst::Forall(q) => forall_types_mut(q, f),
                    _ => {}
                }
            }
        }
    }

    /// The generic parameters this body uses, in order of first occurrence.
    pub fn used_generics(&mut self) -> Vec<usize> {
        let mut order = Vec::new();
        self.for_each_type_mut(&mut |t| t.generics_in_order(&mut order));
        order
    }

    /// Renumber the generics used by this body to `0..k` in order of first
    /// occurrence and return `k`, the body's own type-parameter arity.
    pub fn compact_generics(&mut self, declared: usize) -> usize {
        let order = self.used_generics();
        self.for_each_type_mut(&mut |t| *t = t.subst_generics(&compaction(&order, declared)));
        order.len()
    }
}

/// Substitution arguments mapping each used generic `order[k]` to `Generic(k)`.
/// Unused indices never occur, so any filler is fine.
pub(crate) fn compaction(order: &[usize], declared: usize) -> Vec<Type> {
    let mut args = vec![Type::Generic(0); declared];
    for (new, &old) in order.iter().enumerate() {
        args[old] = Type::Generic(new);
    }
    args
}

fn forall_types_mut(q: &mut Forall, f: &mut impl FnMut(&mut Type)) {
    q.bound.iter_mut().for_each(&mut *f);
    for group in q.triggers.iter_mut() {
        for term in group.terms.iter_mut() {
            trig_types_mut(term, f);
        }
    }
    q.body.for_each_type_mut(f);
}

fn trig_types_mut(term: &mut TrigTerm, f: &mut impl FnMut(&mut Type)) {
    if let TrigTerm::App {
        type_args, args, ..
    } = term
    {
        type_args.iter_mut().for_each(&mut *f);
        args.iter_mut().for_each(|a| trig_types_mut(a, f));
    }
}
