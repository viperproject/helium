//! Lower a typed Silver `Type` to a VMIR `Type`.

use crate::dhash::HashMap;

use lasso::Spur;

use crate::viper::typed;
use crate::vmir;

/// Lower a typed Silver type to a VMIR type.
///
/// `names` resolves a domain/ADT name `Spur` to its VMIR declaration id;
/// `generics` is the enclosing generic declaration's type-parameter list (used
/// to map a `Type::Generic` to its 0-based index). Only an **ADT declaration**
/// has type parameters — everywhere else (`TranslationContext::lower_type`)
/// `generics` is empty.
pub(crate) fn lower_type(
    names: &HashMap<Spur, vmir::MemberId>,
    generics: &[Spur],
    ty: &typed::Type,
) -> vmir::Type {
    match ty {
        typed::Type::Bool => vmir::Type::Bool,
        typed::Type::Int => vmir::Type::Int,
        typed::Type::Real => vmir::Type::Real,
        typed::Type::Ref => vmir::Type::Ref,
        typed::Type::Generic(id) => {
            let idx = generics
                .iter()
                .position(|p| *p == id.0)
                .expect("generic type parameter not in the enclosing declaration's scope");
            vmir::Type::Generic(idx)
        }
        typed::Type::Domain(id, args) => match names.get(&id.0) {
            // An ADT (or a modeled domain): keep the head + recurse on args so
            // the monomorphization key `(head, args)` is faithful.
            Some(&head) => {
                let args = args
                    .iter()
                    .map(|a| lower_type(names, generics, a))
                    .collect();
                vmir::Type::Domain(head, args)
            }
            // A domain with no VMIR declaration (not yet modeled). Fall back to
            // Ref, as before; ADTs are always present (declared in pass 1).
            None => vmir::Type::Ref,
        },
        // TODO: Seq/Set — modeled as builtin parametric types like Option.
        typed::Type::Collection(_) => vmir::Type::Ref,
    }
}
