//! Lower a Silver `domain` declaration to its `vmir::Domain` stub, each of
//! its (bodyless) domain functions to a `vmir::Function`, and each of its
//! axioms to a `vmir::Axiom`. One `DomainTranslator` owns the domain
//! stub *and* every function/axiom slot the domain declares.
//!
//! Domains may be **generic**. VMIR has no link between a domain and its
//! functions, so each function is generic over only the domain parameters its
//! signature mentions, renumbered `0..k` (`Generic(i)`). An axiom is likewise
//! generic over the parameters its body mentions, with its type triggers
//! inferred from the body (`vmir::Axiom::infer`).

use crate::dhash::HashMap;
use std::marker::PhantomData;

use lasso::Spur;

use crate::translate::context::DomainFnSig;
use crate::translate::lower_type;
use crate::translate::{DeclSlot, Declarator, Definer};
use crate::translate::{Declared, Metaed, TranslationContext, TranslationError, pure_exp};
use crate::viper::typed;
use crate::vmir;

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
            let mut params: Vec<vmir::Type> = df
                .params
                .iter()
                .map(|p| lower_type(&ctx.name_map, &domain_params, &p.ty))
                .collect();
            let mut ret = lower_type(&ctx.name_map, &domain_params, &df.ret);
            // Generic over only the parameters the signature mentions,
            // renumbered `0..k` in order of first occurrence.
            let mut order = Vec::new();
            params.iter().for_each(|t| t.generics_in_order(&mut order));
            ret.generics_in_order(&mut order);
            let compact = vmir::compaction(&order, domain_params.len());
            params
                .iter_mut()
                .for_each(|t| *t = t.subst_generics(&compact));
            ret = ret.subst_generics(&compact);
            let id = ctx.name_map[&df.name.0];
            ctx.domain_fn_sigs.insert(
                id,
                DomainFnSig {
                    params,
                    ret,
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
            let fn_name = definer.intern_name(ctx.interner.resolve(&df.name.0));
            definer.define_function(
                fslot,
                vmir::Function {
                    name: fn_name,
                    params: sig.params.into(),
                    ty_params: sig.ty_params.into(),
                    ret: sig.ret,
                    body: None,
                    requires: None,
                    ensures: None,
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
            let name = definer.intern_name(&ax_name);
            definer.define_axiom(aslot, vmir::Axiom::infer(Some(name), body, declared));
        }
        ctx.decl_generics.clear();
        result
    }
}
