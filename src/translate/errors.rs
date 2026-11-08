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
    /// A `domain` declaring type parameters. Generics live on ADTs only: a
    /// generic domain would need a *type* trigger to instantiate its axioms, and
    /// Silver has no syntax to write one.
    GenericDomainUnsupported(String),
    /// A heap read inside a `forall` body that mentions a binder (the payload
    /// names what it reads). Its footprint differs per instance, so framing it
    /// needs quantified permissions, which are not supported. A read that
    /// mentions no binder is evaluated at the quantifier's position instead.
    BinderDependentHeapRead(String),
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
            TranslationError::GenericDomainUnsupported(name) => write!(
                f,
                "domain `{name}` declares type parameters: generic domains are not supported, declare an `adt` instead"
            ),
            TranslationError::BinderDependentHeapRead(what) => write!(
                f,
                "read of {what} inside a `forall` depends on a quantified variable (this needs quantified permissions)"
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
            TranslationError::Unsupported(_)
                | TranslationError::GenericDomainUnsupported(_)
                | TranslationError::BinderDependentHeapRead(_)
        )
    }
}

impl std::error::Error for TranslationError {}
