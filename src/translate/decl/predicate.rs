//! Lower a Silver `predicate` to its VMIR declaration(s).
//!
//! A **concrete** predicate (one with a body) is a `vmir::Resource`, always
//! self-framed: its id serves as snapshot head, address `LocId`, and footprint
//! reference.
//!
//! An **abstract** (bodyless) predicate has no footprint to name, so it is not a
//! resource at all — it is exactly the two declarations a bodyless resource
//! used to construct on demand:
//!
//! ```text
//! function opaque(e0: Ref, e1: Int) -> &[opaque] opaque#snap @ *
//! domain opaque#snap
//! ```
//!
//! the location function (an ordinary `vmir::Function`, the same shape a field
//! lowers to) and the opaque `Domain` that is its value sort. `#snap` is a plain
//! interned name, following the `f#requires` / `f#ensures` convention. Because
//! the abstract case mints no `Resource`, a `fold`/`unfold`/`unfolding` on it is
//! not merely rejected but unrepresentable — see `pure_exp::lower_pred_call`.

use crate::dhash::HashMap;
use std::marker::PhantomData;

use lasso::Spur;

use crate::translate::{DeclSlot, Declarator, Definer};
use crate::translate::{Declared, Metaed, TranslationContext, TranslationError, spatial};
use crate::viper::typed;
use crate::vmir;

/// The slots a predicate reserves, one shape per body-presence case. Which arm
/// is taken is fixed in `declare`, because `alloc_slot` ordering fixes
/// `MemberId` order and hence dump order.
enum PredSlots {
    /// A concrete predicate: one resource under the bare name.
    Concrete(DeclSlot<vmir::Resource>),
    /// An abstract predicate: the location function under the bare name, then
    /// the opaque snapshot domain under `{name}#snap`.
    Abstract {
        loc: DeclSlot<vmir::Function>,
        snap: DeclSlot<vmir::Domain>,
    },
}

pub(crate) struct PredicateTranslator<'a, P = Declared> {
    src: &'a typed::Predicate,
    silver_name: Spur,
    slots: PredSlots,
    _p: PhantomData<P>,
}

impl<'a> PredicateTranslator<'a, Declared> {
    /// Reserve the predicate's slot(s) (filled by `define`) and publish its
    /// address type. Its address is grouped by the predicate name, not by the
    /// reserved id, and its permission cap is unbounded either way — that
    /// `Bound::Unbounded` is the only thing distinguishing a predicate location
    /// from a field's `1/1` downstream.
    pub(crate) fn declare(
        p: &'a typed::Predicate,
        ctx: &mut TranslationContext<'_>,
        d: &mut impl Declarator,
    ) -> Self {
        let name_str = ctx.interner.resolve(&p.name.0).to_owned();
        let group = d.intern_group(&name_str);
        let (slots, value) = if p.body.is_some() {
            let (id, slot) = d.alloc_slot::<vmir::Resource>(&name_str);
            ctx.name_map.insert(p.name.0, id);
            // Only a concrete predicate is foldable; the missing entry for an
            // abstract one is what `lower_pred_call` rejects on.
            ctx.pred_resources.insert(p.name.0, id);
            (PredSlots::Concrete(slot), vmir::Type::Snap(id))
        } else {
            // Function first, then domain — this fixes the dump order.
            let (loc_id, loc) = d.alloc_slot::<vmir::Function>(&name_str);
            ctx.name_map.insert(p.name.0, loc_id);
            let (snap_id, snap) = d.alloc_slot::<vmir::Domain>(&format!("{name_str}#snap"));
            (
                PredSlots::Abstract { loc, snap },
                vmir::Type::domain(snap_id),
            )
        };
        ctx.addr_types.insert(
            p.name.0,
            vmir::Type::addr(group, value, vmir::Bound::Unbounded),
        );
        PredicateTranslator {
            src: p,
            silver_name: p.name.0,
            slots,
            _p: PhantomData,
        }
    }

    /// No `name_map`-dependent metadata to publish.
    pub(crate) fn meta(self, _ctx: &mut TranslationContext<'_>) -> PredicateTranslator<'a, Metaed> {
        PredicateTranslator {
            src: self.src,
            silver_name: self.silver_name,
            slots: self.slots,
            _p: PhantomData,
        }
    }
}

impl PredicateTranslator<'_, Metaed> {
    pub(crate) fn define(
        self,
        ctx: &TranslationContext<'_>,
        definer: &mut impl Definer,
    ) -> Result<(), TranslationError> {
        let p = self.src;
        let params: Vec<vmir::Type> = p.params.iter().map(|pp| ctx.lower_type(&pp.ty)).collect();
        let name_str = ctx.interner.resolve(&self.silver_name).to_owned();
        let name = definer.intern_name(&name_str);
        match self.slots {
            PredSlots::Concrete(slot) => {
                let body_exp = p
                    .body
                    .as_ref()
                    .expect("declare took the concrete arm only for a predicate with a body");
                // Self-framed: params occupy `Val::Temp(0..n)`, heaps accumulate
                // from `Empty` starting at `HeapVal::Temp(0)`.
                let mut env: HashMap<Spur, vmir::Val> = HashMap::default();
                for (i, param) in p.params.iter().enumerate() {
                    env.insert(param.name.0, vmir::Val::Temp(i));
                }
                // On error the unfilled slot is simply dropped (no cleanup
                // needed) and the whole `Builder` discarded up the stack.
                let body = spatial::lower_spatial_never(
                    ctx,
                    &env,
                    body_exp,
                    params.len(),
                    vmir::HeapVal::Empty,
                    0,
                    // Predicate body: real permissions, not wildcards.
                    false,
                )?;
                definer.define_resource(
                    slot,
                    vmir::Resource {
                        name,
                        params,
                        precond: vmir::Precond::SelfFramed,
                        body,
                    },
                );
            }
            PredSlots::Abstract { loc, snap } => {
                // The location function: the predicate's arguments in, its
                // (unbounded) address out. The same shape a field's lowers to.
                let ret = ctx
                    .addr_types
                    .get(&self.silver_name)
                    .expect("declare publishes every predicate's address type")
                    .clone();
                definer.define_function(
                    loc,
                    vmir::Function {
                        name,
                        params: params.into(),
                        ret,
                        body: None,
                        requires: None,
                        ensures: None,
                    },
                );
                let snap_name = definer.intern_name(&format!("{name_str}#snap"));
                definer.define_domain(snap, vmir::Domain { name: snap_name });
            }
        }
        Ok(())
    }
}
