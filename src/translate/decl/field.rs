//! Lower a Silver `field` to its address function (`Ref -> Addr<T>`).
//!
//! A field is an ordinary [`vmir::Function`] interned under the field's **bare
//! name** (no `@`-suffix) — there is no distinct "address" declaration. It has
//! no body, so `meta` only publishes the field's address type into
//! `addr_types`, and `define` reuses that published type for the function's
//! `Addr<T>` return.

use std::marker::PhantomData;

use lasso::Spur;

use crate::translate::{DeclSlot, Declarator, Definer};
use crate::translate::{Declared, Metaed, TranslationContext, TranslationError};
use crate::viper::typed;
use crate::vmir;

pub(crate) struct FieldTranslator<'a, P = Declared> {
    src: &'a typed::Field,
    silver_name: Spur,
    group: Spur,
    slot: DeclSlot<vmir::Function>,
    _p: PhantomData<P>,
}

impl<'a> FieldTranslator<'a, Declared> {
    /// Reserve the field's `Function` slot and publish its `name_map` entry.
    /// The value type is *not* resolved here — it may reference an ADT declared
    /// later, so its lowering waits for a complete `name_map` (see `meta`).
    pub(crate) fn declare(
        f: &'a typed::Field,
        ctx: &mut TranslationContext<'_>,
        d: &mut impl Declarator,
    ) -> Self {
        let name_str = ctx.interner.resolve(&f.0.name.0).to_owned();
        let group = d.intern_group(&name_str);
        let (id, slot) = d.alloc_slot::<vmir::Function>(&name_str);
        ctx.name_map.insert(f.0.name.0, id);
        FieldTranslator {
            src: f,
            silver_name: f.0.name.0,
            group,
            slot,
            _p: PhantomData,
        }
    }

    /// Publish the field's complete address type (`name_map` is complete now,
    /// so an ADT-typed field resolves to its real `Addr<Adt>` value). A field's
    /// cap is full permission, `1/1`.
    pub(crate) fn meta(self, ctx: &mut TranslationContext<'_>) -> FieldTranslator<'a, Metaed> {
        let value = ctx.lower_type(&self.src.0.ty);
        let bound = vmir::Bound::Bounded(num::BigRational::from(num::BigInt::from(1)));
        let addr = vmir::Type::addr(self.group, value, bound);
        ctx.addr_types.insert(self.silver_name, addr);
        FieldTranslator {
            src: self.src,
            silver_name: self.silver_name,
            group: self.group,
            slot: self.slot,
            _p: PhantomData,
        }
    }
}

impl FieldTranslator<'_, Metaed> {
    pub(crate) fn define(
        self,
        ctx: &TranslationContext<'_>,
        definer: &mut impl Definer,
    ) -> Result<(), TranslationError> {
        // Reuse the address type `meta` already lowered into `addr_types`.
        let ret = ctx
            .addr_types
            .get(&self.silver_name)
            .expect("meta publishes every field's address type before define")
            .clone();
        let name = definer.intern_name(ctx.interner.resolve(&self.silver_name));
        definer.define_function(
            self.slot,
            vmir::Function {
                ty_params: 0.into(),
                name,
                params: vec![vmir::Type::Ref].into(),
                ret,
                body: None,
                requires: None,
                ensures: None,
                unique: false,
            },
        );
        Ok(())
    }
}
