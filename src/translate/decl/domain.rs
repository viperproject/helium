//! Lower a Silver `domain` declaration to its `vmir::Domain` stub, each of
//! its (bodyless) domain functions to a `vmir::Function`, and each of its
//! axioms to a `vmir::Axiom`. One `DomainTranslator` owns the domain
//! stub *and* every function/axiom slot the domain declares.
//!
//! Domains may be **generic**. A domain function is generic over only the
//! domain parameters its signature mentions, renumbered `0..k` in order of first
//! occurrence, so every call recovers its type arguments from its argument and
//! result types. An axiom is renumbered the same way over the parameters it
//! mentions, and binds them in a root `forall`
//! ([`vmir::Forall::ty_binders`]) whose triggers fix them: the axiom's own
//! top-level `forall` when each of its trigger groups does, otherwise a wrapper
//! that binds only types, with triggers inferred by [`generic_root`].

use crate::dhash::{HashMap, HashSet};
use std::marker::PhantomData;

use lasso::Spur;

use crate::translate::context::DomainFnSig;
use crate::translate::lower_type;
use crate::translate::{DeclSlot, Declarator, Definer};
use crate::translate::{Declared, Metaed, TranslationContext, TranslationError, pure_exp};
use crate::viper::typed;
use crate::vmir;
use crate::vmir::{InstKind, PureInst, QuantTrigger, TrigHead, TrigTerm, Val};

pub(crate) struct DomainTranslator<'a, P = Declared> {
    src: &'a typed::Domain,
    silver_name: Spur,
    slot: DeclSlot<vmir::Domain>,
    /// One `Function` slot per domain function, parallel to `src.functions`.
    fn_slots: Vec<DeclSlot<vmir::Function>>,
    /// One slot per axiom, parallel to `src.axioms`. An anonymous axiom's slot
    /// is registered under the generated name `{domain}#axiom{i}` (`#` marks a
    /// generated member); axioms are not callable, so no `name_map` entry.
    axiom_slots: Vec<DeclSlot<vmir::Axiom>>,
    _p: PhantomData<P>,
}

impl<'a> DomainTranslator<'a, Declared> {
    /// Reserve the domain stub + a `Function` slot per domain function, and
    /// publish every `name_map` entry. A domain function's param/ret types are
    /// *not* lowered here — they may reference any ADT/domain by id, so lowering
    /// waits for `define`.
    pub(crate) fn declare(
        d: &'a typed::Domain,
        ctx: &mut TranslationContext<'_>,
        decl: &mut impl Declarator,
    ) -> Self {
        let name_str = ctx.interner.resolve(&d.name.0).to_string();
        let (id, slot) = decl.alloc_slot::<vmir::Domain>(&name_str);
        ctx.name_map.insert(d.name.0, id);

        let mut fn_slots = Vec::with_capacity(d.functions.len());
        for df in &d.functions {
            let fn_name = ctx.interner.resolve(&df.name.0).to_string();
            let (fid, fslot) = decl.alloc_slot::<vmir::Function>(&fn_name);
            ctx.name_map.insert(df.name.0, fid);
            fn_slots.push(fslot);
        }

        let mut axiom_slots = Vec::with_capacity(d.axioms.len());
        for (i, ax) in d.axioms.iter().enumerate() {
            let ax_name = match &ax.name {
                Some(n) => ctx.interner.resolve(&n.0).to_string(),
                None => format!("{name_str}#axiom{i}"),
            };
            let (_, aslot) = decl.alloc_slot::<vmir::Axiom>(&ax_name);
            axiom_slots.push(aslot);
        }

        DomainTranslator {
            src: d,
            silver_name: d.name.0,
            slot,
            fn_slots,
            axiom_slots,
            _p: PhantomData,
        }
    }

    /// Publish each domain function's lowered signature, so a call site (in any
    /// body, whatever order `define` runs in) can recover the instantiation.
    pub(crate) fn meta(self, ctx: &mut TranslationContext<'_>) -> DomainTranslator<'a, Metaed> {
        let domain_params: Vec<Spur> = self.src.type_params.iter().map(|i| i.0).collect();
        for df in &self.src.functions {
            let params: Vec<vmir::Type> = df
                .params
                .iter()
                .map(|p| lower_type(&ctx.name_map, &domain_params, &p.ty))
                .collect();
            let ret = lower_type(&ctx.name_map, &domain_params, &df.ret);
            // Generic over only the parameters the signature mentions,
            // renumbered `0..k` in order of first occurrence.
            let mut order = Vec::new();
            params
                .iter()
                .chain([&ret])
                .for_each(|t| generics_in_order(t, &mut order));
            let compact = compaction(&order, domain_params.len());
            let id = ctx.name_map[&df.name.0];
            ctx.domain_fn_sigs.insert(
                id,
                DomainFnSig {
                    params: params.iter().map(|t| t.subst_generics(&compact)).collect(),
                    ret: ret.subst_generics(&compact),
                    ty_params: order.len(),
                },
            );
        }
        DomainTranslator {
            src: self.src,
            silver_name: self.silver_name,
            slot: self.slot,
            fn_slots: self.fn_slots,
            axiom_slots: self.axiom_slots,
            _p: PhantomData,
        }
    }
}

impl DomainTranslator<'_, Metaed> {
    pub(crate) fn define(
        self,
        ctx: &mut TranslationContext<'_>,
        definer: &mut impl Definer,
    ) -> Result<(), TranslationError> {
        let name_str = ctx.interner.resolve(&self.silver_name).to_string();
        let name = definer.intern_name(&name_str);
        let domain_params: Vec<Spur> = self.src.type_params.iter().map(|i| i.0).collect();
        let declared = domain_params.len();
        definer.define_domain(
            self.slot,
            vmir::Domain {
                name,
                ty_params: declared.into(),
            },
        );
        for (df, fslot) in self.src.functions.iter().zip(self.fn_slots) {
            let sig = ctx.domain_fn_sigs[&ctx.name_map[&df.name.0]].clone();
            // Silver only admits nullary `unique` functions (enforced by the
            // typechecker); a unique function with parameters would have no
            // single value to keep distinct.
            assert!(
                !df.unique || sig.params.is_empty(),
                "unique domain function `{}` has parameters; the typechecker must reject it",
                ctx.interner.resolve(&df.name.0)
            );
            let fn_name = definer.intern_name(ctx.interner.resolve(&df.name.0));
            definer.define_function(
                fslot,
                vmir::Function {
                    name: fn_name,
                    ty_params: sig.ty_params.into(),
                    params: sig.params.into(),
                    ret: sig.ret,
                    body: None,
                    requires: None,
                    ensures: None,
                    unique: df.unique,
                },
            );
        }
        // Axioms: the body is pure and heap-free — a callee is at most a
        // precondition-free Silver function (typecheck-enforced), so the inert
        // `Empty` heap is never read. Axiom bodies are never verified, only
        // assumed.
        let env = HashMap::default();
        // The domain's parameters are in scope for every axiom body.
        ctx.decl_generics = domain_params;
        let mut result = Ok(());
        for (i, (ax, aslot)) in self.src.axioms.iter().zip(self.axiom_slots).enumerate() {
            let body = match pure_exp::lower_axiom_body(ctx, &env, &ax.exp) {
                Ok(body) => body,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            };
            // The axiom's carried name matches its slot registration: the
            // Silver name when given, the generated `{domain}#axiom{i}` slot
            // name otherwise.
            let ax_name = match &ax.name {
                Some(n) => ctx.interner.resolve(&n.0).to_string(),
                None => format!("{name_str}#axiom{i}"),
            };
            // Bind only the parameters the axiom mentions, renumbered `0..k`
            // in order of first occurrence, as for a domain function.
            let mut order = Vec::new();
            body.for_each_type(&mut |t| generics_in_order(t, &mut order));
            let body = if order.is_empty() {
                body
            } else {
                let body = body.subst_generics(&compaction(&order, declared));
                match generic_root(body, order.len()) {
                    Some(body) => body,
                    None => {
                        result = Err(TranslationError::UntypedGenericAxiom(ax_name));
                        break;
                    }
                }
            };
            let name = definer.intern_name(&ax_name);
            definer.define_axiom(
                aslot,
                vmir::Axiom {
                    name: Some(name),
                    body,
                },
            );
        }
        ctx.decl_generics.clear();
        result
    }
}

/// Append each `Generic(i)` of `ty` not yet in `out`, in order of occurrence.
fn generics_in_order(ty: &vmir::Type, out: &mut Vec<usize>) {
    match ty {
        vmir::Type::Generic(i) if !out.contains(i) => out.push(*i),
        vmir::Type::Domain(_, args) => args.iter().for_each(|a| generics_in_order(a, out)),
        vmir::Type::Option(t) => generics_in_order(t, out),
        vmir::Type::Coll(_, tys) => tys.iter().for_each(|t| generics_in_order(t, out)),
        vmir::Type::Addr { value, .. } => generics_in_order(value, out),
        _ => {}
    }
}

/// The substitution renumbering the domain parameters `order` lists to
/// `Generic(0..order.len())`, over a domain of `arity` parameters. A parameter
/// not in `order` maps to `Generic(0)`: it does not occur, so it is never read.
fn compaction(order: &[usize], arity: usize) -> Vec<vmir::Type> {
    let mut compact = vec![vmir::Type::Generic(0); arity];
    for (new, &old) in order.iter().enumerate() {
        compact[old] = vmir::Type::Generic(new);
    }
    compact
}

/// The type parameters a trigger term fixes: those the type arguments of its
/// applications mention.
fn fixes(term: &TrigTerm) -> HashSet<usize> {
    let mut out = HashSet::default();
    term.collect_generics(&mut out);
    out
}

/// Bind the `n` type parameters a generic axiom's (renumbered) `body` mentions
/// in a root `forall` whose triggers fix all of them, or `None` when some
/// parameter is mentioned by no application (nothing could ever fix it).
///
/// * **Flatten.** A body that is one `forall`, each of whose trigger groups
///   fixes every parameter, binds them itself: its own triggers instantiate it
///   at every type they match.
/// * **Wrap** otherwise, in a `forall` binding only types, at temp 0 (the body
///   keeps its numbering: it starts at the wrapper's `binder_base`, 0). Its
///   triggers are inferred from the applications outside nested `forall`s and
///   from the trigger terms of those `forall`s (and of the `forall`s nested in
///   them, whose triggers may fix a type the outer ones leave open), which
///   stand in for their bodies. A candidate keeps its head and type arguments; its value
///   arguments become `Wild` (the wrapper binds no values, and an inner binder
///   means nothing out here).
///   - Each candidate whose type arguments alone fix every parameter, and that
///     contains no smaller such candidate, is a trigger on its own (the
///     triggers are alternatives).
///   - With no such candidate, one multi-pattern is picked greedily: again and
///     again the earliest candidate fixing the most parameters still open.
fn generic_root(mut body: vmir::FunctionBody, n: usize) -> Option<vmir::FunctionBody> {
    let all: HashSet<usize> = (0..n).collect();
    let fixes_all = |t: &TrigTerm| all.is_subset(&fixes(t));
    if let [inst] = &mut body.insts[..]
        && let InstKind::Pure(_, PureInst::Forall(q)) = &mut inst.kind
        && body.res == Val::Temp(q.binder_base)
        && q.triggers.iter().all(|g| {
            let mut fixed = HashSet::default();
            g.terms.iter().for_each(|t| t.collect_generics(&mut fixed));
            all.is_subset(&fixed)
        })
    {
        q.ty_binders = n.into();
        return Some(body);
    }

    // Each candidate, with whether it contains a smaller candidate fixing every
    // parameter (then it is not minimal).
    let mut cands: Vec<(TrigTerm, bool)> = Vec::new();
    // Per top-level temp: whether its value contains an application fixing
    // every parameter.
    let mut holds_full: Vec<bool> = Vec::new();
    for inst in &body.insts {
        let InstKind::Pure(_, pi) = &inst.kind else {
            continue;
        };
        let full = if let PureInst::Forall(q) = pi {
            forall_candidates(q, &fixes_all, &mut cands)
        } else {
            let mut inner = false;
            pi.for_each_operand(&mut |v| {
                inner |= matches!(v, Val::Temp(k) if holds_full[*k]);
            });
            match app_candidate(pi) {
                Some(term) => {
                    let full = fixes_all(&term);
                    cands.push((term, inner));
                    full || inner
                }
                None => inner,
            }
        };
        holds_full.push(full);
    }

    let mut minimal: Vec<TrigTerm> = Vec::new();
    for (term, inner) in &cands {
        if !inner && fixes_all(term) && !minimal.contains(term) {
            minimal.push(term.clone());
        }
    }
    let triggers: Vec<QuantTrigger> = if !minimal.is_empty() {
        minimal
            .into_iter()
            .map(|t| QuantTrigger {
                terms: Box::new([t]),
            })
            .collect()
    } else {
        let mut open = all.clone();
        let mut group: Vec<TrigTerm> = Vec::new();
        while !open.is_empty() {
            let mut best: Option<(&TrigTerm, usize)> = None;
            for (t, _) in &cands {
                let gain = fixes(t).intersection(&open).count();
                if gain > best.map_or(0, |(_, g)| g) {
                    best = Some((t, gain));
                }
            }
            let (best, _) = best?;
            fixes(best).iter().for_each(|i| {
                open.remove(i);
            });
            group.push(best.clone());
        }
        vec![QuantTrigger {
            terms: group.into(),
        }]
    };
    let wrapper = vmir::Forall {
        binder_base: 0,
        ty_binders: n.into(),
        bound: Box::new([]),
        triggers: triggers.into(),
        body,
    };
    Some(vmir::FunctionBody {
        insts: vec![vmir::Inst::new(
            Default::default(),
            InstKind::Pure(vmir::Type::Bool, PureInst::Forall(Box::new(wrapper))),
        )],
        res: Val::Temp(0),
    })
}

/// The trigger candidate of a body step that is a matchable application: its
/// head and type arguments, every value argument `Wild`.
fn app_candidate(pi: &PureInst) -> Option<TrigTerm> {
    let wild = |n: usize| vec![TrigTerm::Wild; n].into_boxed_slice();
    let (head, type_args, args) = match pi {
        PureInst::FunctionCall(fc) => (
            TrigHead::Func(fc.function),
            fc.type_args.clone(),
            wild(fc.args.iter().count()),
        ),
        PureInst::AdtCons {
            adt,
            type_args,
            variant,
            args,
        } => (
            TrigHead::AdtCons {
                adt: *adt,
                variant: *variant,
            },
            type_args.clone(),
            wild(args.len()),
        ),
        PureInst::AdtProj {
            adt,
            type_args,
            variant,
            field,
            ..
        } => (
            TrigHead::AdtProj {
                adt: *adt,
                variant: *variant,
                field: *field,
            },
            type_args.clone(),
            wild(1),
        ),
        PureInst::AdtIs {
            adt,
            type_args,
            variant,
            ..
        } => (
            TrigHead::AdtIs {
                adt: *adt,
                variant: *variant,
            },
            type_args.clone(),
            wild(1),
        ),
        PureInst::Coll { tys, op, args } => (TrigHead::Coll(*op), tys.to_vec(), wild(args.len())),
        PureInst::CollLit { kind, elem_ty, .. } => {
            (TrigHead::CollLit(*kind), vec![elem_ty.clone()], wild(1))
        }
        _ => return None,
    };
    Some(TrigTerm::App {
        head,
        type_args,
        args,
    })
}

/// Push the candidates of the triggers of `q` and of every `forall` nested in
/// its body (see [`trigger_candidates`]). Returns whether one of them
/// satisfies `fixes_all`.
fn forall_candidates(
    q: &vmir::Forall,
    fixes_all: &impl Fn(&TrigTerm) -> bool,
    out: &mut Vec<(TrigTerm, bool)>,
) -> bool {
    let mut full = false;
    for t in q.triggers.iter().flat_map(|g| g.terms.iter()) {
        full |= trigger_candidates(t, fixes_all, out);
    }
    for inst in &q.body.insts {
        if let InstKind::Pure(_, PureInst::Forall(inner)) = &inst.kind {
            full |= forall_candidates(inner, fixes_all, out);
        }
    }
    full
}

/// Push every application subterm of the trigger term `t` as a candidate, its
/// variables made `Wild`, marked non-minimal when a proper subterm satisfies
/// `fixes_all`. Returns whether `t` contains an application that does.
fn trigger_candidates(
    t: &TrigTerm,
    fixes_all: &impl Fn(&TrigTerm) -> bool,
    out: &mut Vec<(TrigTerm, bool)>,
) -> bool {
    let TrigTerm::App {
        head,
        type_args,
        args,
    } = t
    else {
        return false;
    };
    let mut inner = false;
    for a in args.iter() {
        inner |= trigger_candidates(a, fixes_all, out);
    }
    let term = TrigTerm::App {
        head: head.clone(),
        type_args: type_args.clone(),
        args: args.iter().map(wild_vars).collect(),
    };
    let full = fixes_all(&term);
    out.push((term, inner));
    full || inner
}

/// `t` with every variable `Wild`.
fn wild_vars(t: &TrigTerm) -> TrigTerm {
    match t {
        TrigTerm::Var(_) => TrigTerm::Wild,
        TrigTerm::App {
            head,
            type_args,
            args,
        } => TrigTerm::App {
            head: head.clone(),
            type_args: type_args.clone(),
            args: args.iter().map(wild_vars).collect(),
        },
        other => other.clone(),
    }
}
