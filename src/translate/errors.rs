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
    /// A generic domain's axiom (named) with a part whose type parameters nothing
    /// fixes: a use of a parameter outside every `forall`, or a `forall` whose
    /// trigger groups (with the types an enclosing `forall` already fixes) do not
    /// bind every parameter it mentions. The verifier instantiates a generic
    /// axiom only at the types its trigger matches fix, so such a part could
    /// never be assumed.
    UntypedGenericAxiom(String),
    /// A heap-dependent function called inside a `forall` body. Its snapshot
    /// (a frame-only `exhale`) would have to be taken *per instance*, over a footprint
    /// that may mention the binders — quantified permissions, which are not
    /// supported. Even a binder-independent footprint is rejected for now: the
    /// quantifier's compiled body must stay pure and heap-free, since the
    /// verifier rebuilds it inside a rewrite rule, where the symbolic heap is out
    /// of reach and no obligation can be discharged.
    HeapDepFunctionInQuantifier(String),
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
                "axiom `{name}` uses a domain type parameter that no trigger fixes: every use must sit in a `forall` whose triggers (or an enclosing `forall`'s) bind it"
            ),
            TranslationError::HeapDepFunctionInQuantifier(name) => write!(
                f,
                "heap-dependent function `{name}` called inside a `forall`: a quantifier body must be heap-free (this needs quantified permissions)"
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
                | TranslationError::UntypedGenericAxiom(_)
                | TranslationError::HeapDepFunctionInQuantifier(_)
        )
    }
}

impl std::error::Error for TranslationError {}
