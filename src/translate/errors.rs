use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranslationError {
    Unsupported(&'static str),
    UnknownIdent(String),
    /// `fold` / `unfold` / `unfolding` naming a bodyless (abstract) predicate,
    /// which lowers to a `function` + `domain` pair and so has no `Resource` to
    /// name. Typecheck rejects this first (`TypeError::AbstractPredicateNotFoldable`);
    /// this is the backstop that keeps the `ResourceCall` unbuildable.
    AbstractPredicateNotFoldable(String),
    /// A generic domain's axiom (named) mentioning a type parameter that no
    /// application in it does (say, only as a binder's type): no trigger can fix
    /// it, and the verifier instantiates a generic axiom only at the types its
    /// trigger matches fix, so the axiom could never be assumed.
    UntypedGenericAxiom(String),
}

impl fmt::Display for TranslationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TranslationError::Unsupported(what) => write!(f, "unsupported: {what}"),
            TranslationError::UnknownIdent(name) => write!(f, "unknown identifier: {name}"),
            TranslationError::AbstractPredicateNotFoldable(name) => write!(
                f,
                "predicate `{name}` has no body: it cannot be folded, unfolded, or used in `unfolding`"
            ),
            TranslationError::UntypedGenericAxiom(name) => write!(
                f,
                "axiom `{name}` uses a domain type parameter that no trigger fixes: no application in it has the parameter in its type arguments"
            ),
        }
    }
}

impl TranslationError {
    /// Whether this is "we do not implement that construct" rather than a
    /// genuine error in the input program. Mirrors
    /// `TypeError::is_unsupported`.
    pub fn is_unsupported(&self) -> bool {
        matches!(
            self,
            TranslationError::Unsupported(_) | TranslationError::UntypedGenericAxiom(_)
        )
    }
}

impl std::error::Error for TranslationError {}
