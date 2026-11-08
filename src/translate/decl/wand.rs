//! Lower the program's magic-wand structures (`typed::Program::wands`).
//!
//! A wand structure is a resource with no body, identified by the values of its
//! holes, so it lowers exactly as an abstract predicate does (see `predicate`):
//!
//! ```text
//! function wand#0(e0: Ref, e1: Real) -> &[wand#0] wand#0#snap @ *
//! domain wand#0#snap
//! ```
//!
//! A wand chunk's value is never read — `apply` produces the right-hand side
//! with fresh values — so the snapshot domain stays opaque, and a join that
//! equates the values of two instances at one location is harmless.

use crate::translate::decl::predicate::OpaqueSlots;
use crate::translate::{Declarator, Definer, TranslationContext};
use crate::viper::typed;
use crate::vmir;

pub(crate) struct WandTranslator<'a> {
    src: &'a [typed::Wand],
    slots: Vec<OpaqueSlots>,
}

/// The name of structure `k`. `#` keeps it apart from every Silver name.
fn wand_name(k: usize) -> String {
    format!("wand#{k}")
}

impl<'a> WandTranslator<'a> {
    /// Reserve every structure's slots and publish its location function and
    /// address type (each structure is its own location group, with an
    /// unbounded permission cap: a state may hold several instances).
    pub(crate) fn declare(
        wands: &'a [typed::Wand],
        ctx: &mut TranslationContext<'_>,
        d: &mut impl Declarator,
    ) -> Self {
        let slots = (0..wands.len())
            .map(|k| {
                let name = wand_name(k);
                let group = d.intern_group(&name);
                let (slots, loc_id, value) = OpaqueSlots::declare(d, &name);
                let addr = vmir::Type::addr(group, value, vmir::Bound::Unbounded);
                ctx.wands.push((loc_id, addr));
                slots
            })
            .collect();
        WandTranslator { src: wands, slots }
    }

    pub(crate) fn define(self, ctx: &TranslationContext<'_>, definer: &mut impl Definer) {
        for (k, (wand, slots)) in self.src.iter().zip(self.slots).enumerate() {
            let params = wand.holes.iter().map(|ty| ctx.lower_type(ty)).collect();
            slots.define(definer, &wand_name(k), params, ctx.wands[k].1.clone());
        }
    }
}
