//! Generic-parameter bookkeeping: renumbering the parameters a domain function's
//! signature mentions to `0..k`, and instantiating a generic axiom body at
//! concrete types.

use crate::vmir::{Forall, FunctionBody, InstKind, PureInst, TrigTerm, Type};

impl FunctionBody {
    /// This body with every `Generic(i)` replaced by `tys[i]`: the result type of
    /// each step, the type arguments of every application, and (recursively) each
    /// nested `forall`'s binder types and trigger type arguments. How a generic
    /// axiom is instantiated.
    pub fn subst_generics(&self, tys: &[Type]) -> FunctionBody {
        let mut body = self.clone();
        body.for_each_type_mut(&mut |t| *t = t.subst_generics(tys));
        body
    }

    /// The type of every value and every bound variable of this body, nested
    /// quantifiers included.
    pub fn value_types(&self, out: &mut Vec<Type>) {
        for inst in &self.insts {
            if let InstKind::Pure(ty, pi) = &inst.kind {
                out.push(ty.clone());
                if let PureInst::Forall(q) = pi {
                    out.extend(q.bound.iter().cloned());
                    q.body.value_types(out);
                }
            }
        }
    }

    /// Visit every type this body mentions, mutably. Each callback sees a *whole*
    /// type, not its sub-types, so a rewrite is one `subst_generics`.
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
                    PureInst::Seq { elem_ty, .. } | PureInst::SeqLit { elem_ty, .. } => f(elem_ty),
                    _ => {}
                }
            }
        }
    }
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

/// Substitution arguments mapping each used generic `order[k]` to `Generic(k)`.
/// Unused indices never occur, so any filler is fine.
pub(crate) fn compaction(order: &[usize], declared: usize) -> Vec<Type> {
    let mut args = vec![Type::Generic(0); declared];
    for (new, &old) in order.iter().enumerate() {
        args[old] = Type::Generic(new);
    }
    args
}
