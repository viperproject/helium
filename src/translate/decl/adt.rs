//! Lower a Silver `adt` (ADT) declaration to its `vmir::Adt` stub + variant
//! shapes, plus per-constructor/destructor metadata.

use std::marker::PhantomData;

use lasso::Spur;

use crate::translate::lower_type;
use crate::translate::{DeclSlot, Declarator, Definer};
use crate::translate::{Declared, Metaed, TranslationContext, TranslationError};
use crate::viper::typed;
use crate::vmir;

/// An ADT's `declare` reserves only the stub id (+ type-param count) — no
/// variant computation, since a variant field type or constructor may reference
/// *any* ADT by id (including one declared later in the source). `meta`
/// publishes the constructor (`ctor_tag`) / destructor (`dtor_sem`) metadata
/// once every ADT stub id is in `name_map`; `define` recomputes the variant
/// shapes and fills the `DeclSlot`.
pub(crate) struct AdtTranslator<'a, P = Declared> {
    src: &'a typed::Adt,
    silver_name: Spur,
    id: vmir::MemberId,
    slot: DeclSlot<vmir::Adt>,
    _p: PhantomData<P>,
}

impl<'a> AdtTranslator<'a, Declared> {
    pub(crate) fn declare(
        adt: &'a typed::Adt,
        ctx: &mut TranslationContext<'_>,
        d: &mut impl Declarator,
    ) -> Self {
        let name_str = ctx.interner.resolve(&adt.name.0).to_string();
        let (id, slot) = d.alloc_slot::<vmir::Adt>(&name_str);
        ctx.name_map.insert(adt.name.0, id);
        AdtTranslator {
            src: adt,
            silver_name: adt.name.0,
            id,
            slot,
            _p: PhantomData,
        }
    }

    /// Publish constructor/destructor metadata. A constructor/destructor is not
    /// itself a declaration: `ctor_tag` maps a constructor name to its
    /// `(owning ADT, tag)`, `dtor_sem` maps an `(ADT, field name)` pair to its
    /// `(adt id, variant, field)` projection.
    pub(crate) fn meta(self, ctx: &mut TranslationContext<'_>) -> AdtTranslator<'a, Metaed> {
        for (tag, v) in self.src.variants.iter().enumerate() {
            ctx.adt.ctor_tag.insert(v.name.0, (self.silver_name, tag));
            for (field, p) in v.params.iter().enumerate() {
                ctx.adt
                    .dtor_sem
                    .insert((self.silver_name, p.name.0), (self.id, tag, field));
            }
        }
        AdtTranslator {
            src: self.src,
            silver_name: self.silver_name,
            id: self.id,
            slot: self.slot,
            _p: PhantomData,
        }
    }
}

impl AdtTranslator<'_, Metaed> {
    pub(crate) fn define(
        self,
        ctx: &TranslationContext<'_>,
        definer: &mut impl Definer,
    ) -> Result<(), TranslationError> {
        // Variant field types may mention the ADT's type parameters (→ `Generic`).
        let type_params: Vec<Spur> = self.src.type_params.iter().map(|i| i.0).collect();
        let mut variants: Vec<vmir::AdtVariant> = Vec::new();
        for (tag, v) in self.src.variants.iter().enumerate() {
            let ctor_name = definer.intern_name(ctx.interner.resolve(&v.name.0));
            let field_types: Vec<vmir::Type> = v
                .params
                .iter()
                .map(|p| lower_type(&ctx.name_map, &type_params, &p.ty))
                .collect();
            if variants.len() <= tag {
                variants.resize(
                    tag + 1,
                    vmir::AdtVariant {
                        name: None,
                        field_types: Vec::new(),
                    },
                );
            }
            variants[tag] = vmir::AdtVariant {
                name: Some(ctor_name),
                field_types,
            };
        }
        let name = definer.intern_name(ctx.interner.resolve(&self.silver_name));
        definer.define_adt(
            self.slot,
            vmir::Adt {
                name,
                ty_params: type_params.len().into(),
                variants,
            },
        );
        Ok(())
    }
}
