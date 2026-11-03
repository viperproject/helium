use rusttyc::TcErr;

use super::lattice::{TcTypeErr, ViperTcType};
use crate::viper::interner::Interner;

#[derive(Debug, Clone)]
pub enum TypeError {
    TypeMismatch {
        expected: ViperTcType,
        found: ViperTcType,
        context: &'static str,
    },
    UndefinedVariable(String),
    PredicateInPureContext(String),
    PermissionInPureContext,
    /// A heap-reading construct (`e.f`, a `function` call, `unfolding`) used in a
    /// pure context (e.g. a domain axiom).
    HeapInPureContext,
    WrongArgCount {
        name: String,
        expected: usize,
        found: usize,
    },
    FieldBaseNotRef,
    IllegalOldUsage,
    IllegalLabeledOldUsage,
    IllegalResultUsage,
    UndefinedLabel(String),
    ShadowedName(String),
    WrongReturnCount {
        expected: usize,
        found: usize,
    },
    /// A `Generic` type parameter occurred outside a scope that binds it.
    UnboundTypeParam(String),
    /// A type name that is no declared domain, ADT or type parameter in scope.
    UndeclaredType(String),
    /// A domain or ADT type applied to the wrong number of type arguments.
    WrongTypeArgCount {
        name: String,
        expected: usize,
        found: usize,
    },
    /// An assignment to a method parameter (only locals and return
    /// variables are assignable).
    NotAssignable(String),
    /// The same variable twice among a method call's targets.
    DuplicateTarget(String),
    /// `e.f` where several ADTs declare `f` and `e`'s type is not known to
    /// be one of them.
    AmbiguousAdtField(String),
    /// An `exists` quantifier (only pure `forall` is supported so far).
    ExistsUnsupported,
    /// A Viper construct this verifier does not implement. Carries the
    /// construct's name as it is written in the source (`magic wand `--*``,
    /// `Seq`, `package`, ...). Kept apart from the genuine type errors so a
    /// caller can tell "we do not support this" from "this program is
    /// ill-typed" — see [`TypeError::is_unsupported`].
    Unsupported(&'static str),
    /// A `forall` with no trigger group (or an empty one). Triggers are never
    /// inferred — every quantifier must state how it is instantiated.
    MissingTrigger,
    /// A trigger term whose root is not an application (a function, domain
    /// function, ADT constructor/destructor/discriminator call).
    TriggerNotAnApplication,
    /// A subterm of a trigger that is neither a variable, a literal, nor a
    /// nested application — interpreted operators (`+`, `!`, `?:`, `let`) and
    /// nested quantifiers cannot be matched on.
    TriggerBadSubterm,
    /// A trigger group that does not mention every bound variable: matching it
    /// would leave a binder uninstantiated. Names the missing variable.
    TriggerNotCovering(String),
    /// A field dereference in a domain axiom.
    FieldAccessInAxiom,
    /// An `unfolding` expression in a domain axiom.
    UnfoldingInAxiom,
    /// `fold` / `unfold` / `unfolding` naming a bodyless (abstract) predicate.
    /// An abstract predicate is a bare location with no body to exchange for.
    AbstractPredicateNotFoldable(String),
    /// An axiom calls a Silver `function` that has a precondition.
    PreconditionedFunctionInAxiom(String),
    /// A call in an axiom left a type parameter of a *foreign* domain's
    /// function unconstrained (an enclosing-domain parameter would default to
    /// itself, Silver's `ground()` rule).
    UnconstrainedTypeParamInAxiom(String),
    Tc(TcErr<ViperTcType>),
    /// A solver mismatch ([`TcTypeErr::Mismatch`]) with both types spelled as
    /// Viper types, made by [`TypeError::with_names`].
    Mismatch(String, String),
    Other(String),
}

impl TypeError {
    /// Whether this is "we do not implement that construct" rather than "this
    /// program is ill-typed". The distinction is what lets a caller report an
    /// unsupported declaration as such instead of as a type error.
    ///
    /// The test is what Viper itself would say. A program Silver accepts and we
    /// refuse is our limitation, whatever phase catches it: an inferred trigger,
    /// `perm()` in a contract, a labelled `old` in a postcondition, an axiom over
    /// a function with a precondition, a call in an axiom that leans on Silver's
    /// `ground()` rule. A program Silver also rejects — `old` where there is no
    /// pre-state, a field read in an axiom, `unfold` of a bodyless predicate — is
    /// an error in the input and stays one.
    pub fn is_unsupported(&self) -> bool {
        matches!(
            self,
            TypeError::Unsupported(_)
                | TypeError::ExistsUnsupported
                | TypeError::MissingTrigger
                | TypeError::TriggerNotAnApplication
                | TypeError::TriggerBadSubterm
                | TypeError::TriggerNotCovering(_)
                | TypeError::PermissionInPureContext
                | TypeError::IllegalLabeledOldUsage
                | TypeError::PreconditionedFunctionInAxiom(_)
                | TypeError::UnconstrainedTypeParamInAxiom(_)
        )
    }
}

impl TypeError {
    /// Replace a solver mismatch by a [`TypeError::Mismatch`] naming both
    /// types. The solver's own error carries only interned identifiers, so
    /// this runs where the interner is in scope. Anything else is unchanged.
    pub(super) fn with_names(self, interner: &Interner) -> Self {
        match self {
            TypeError::Tc(
                TcErr::Bound(_, _, TcTypeErr::Mismatch(t1, t2))
                | TcErr::KeyEquation(_, _, TcTypeErr::Mismatch(t1, t2)),
            ) => TypeError::Mismatch(viper_name(&t1, interner), viper_name(&t2, interner)),
            other => other,
        }
    }
}

/// A solver type variant as Viper spells it, in backticks. Only the head is
/// known (type arguments are separate solver keys), so a generic ADT or domain
/// shows its arguments as `_`.
fn viper_name(ty: &ViperTcType, interner: &Interner) -> String {
    match ty {
        ViperTcType::Bool => "`Bool`".to_string(),
        ViperTcType::Int => "`Int`".to_string(),
        ViperTcType::Real => "`Perm`".to_string(),
        ViperTcType::Ref => "`Ref`".to_string(),
        ViperTcType::Numeric => "a number (`Int` or `Perm`)".to_string(),
        ViperTcType::Domain(id, 0) => format!("`{}`", interner.resolve(&id.0)),
        ViperTcType::Domain(id, n) => format!(
            "`{}[{}]`",
            interner.resolve(&id.0),
            vec!["_"; *n].join(", ")
        ),
        ViperTcType::Generic(id) => format!("`{}`", interner.resolve(&id.0)),
        ViperTcType::Top => "`_`".to_string(),
    }
}

impl From<TcErr<ViperTcType>> for TypeError {
    fn from(e: TcErr<ViperTcType>) -> Self {
        TypeError::Tc(e)
    }
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeError::TypeMismatch {
                expected,
                found,
                context,
            } => {
                write!(
                    f,
                    "Type mismatch in {context}: expected {expected:?}, found {found:?}"
                )
            }
            TypeError::UndefinedVariable(name) => write!(f, "Undefined variable: {name}"),
            TypeError::PredicateInPureContext(name) => {
                write!(f, "Predicate `{name}` used in pure expression context")
            }
            TypeError::PermissionInPureContext => write!(f, "`perm` not allowed here"),
            TypeError::HeapInPureContext => {
                write!(f, "heap-dependent expression not allowed in a pure context")
            }
            TypeError::WrongArgCount {
                name,
                expected,
                found,
            } => {
                write!(f, "`{name}` expects {expected} args, got {found}")
            }
            TypeError::FieldBaseNotRef => write!(f, "Field access base must have type Ref"),
            TypeError::IllegalOldUsage => write!(f, "`old` not allowed in this context"),
            TypeError::IllegalLabeledOldUsage => {
                write!(f, "labeled `old` not allowed in this context")
            }
            TypeError::IllegalResultUsage => write!(f, "`result` not allowed in this context"),
            TypeError::UndefinedLabel(name) => {
                write!(f, "label `{name}` is not defined in this method")
            }
            TypeError::ShadowedName(name) => {
                write!(f, "name `{name}` already declared in this scope")
            }
            TypeError::WrongReturnCount { expected, found } => {
                write!(
                    f,
                    "assignment expects {expected} target(s) on LHS, found {found}"
                )
            }
            TypeError::UnboundTypeParam(name) => {
                write!(f, "unbound type parameter `{name}`")
            }
            TypeError::UndeclaredType(name) => write!(f, "undeclared type `{name}`"),
            TypeError::WrongTypeArgCount {
                name,
                expected,
                found,
            } => write!(
                f,
                "type `{name}` expects {expected} type argument(s), got {found}"
            ),
            TypeError::NotAssignable(name) => {
                write!(f, "`{name}` is a method parameter and cannot be assigned")
            }
            TypeError::AmbiguousAdtField(name) => write!(
                f,
                "ambiguous ADT field `{name}`: several ADTs declare it and the receiver's type is not known"
            ),
            TypeError::DuplicateTarget(name) => {
                write!(f, "`{name}` occurs more than once among the call's targets")
            }
            TypeError::ExistsUnsupported => {
                write!(f, "`exists` quantifiers are not supported yet")
            }
            TypeError::Unsupported(what) => {
                write!(f, "unsupported construct: {what}")
            }
            TypeError::MissingTrigger => write!(
                f,
                "`forall` needs an explicit trigger — triggers are never inferred, write `{{ f(x) }}`"
            ),
            TypeError::TriggerNotAnApplication => write!(
                f,
                "a trigger term must be a function, domain-function or ADT application"
            ),
            TypeError::TriggerBadSubterm => write!(
                f,
                "a trigger may only contain variables, literals and nested applications"
            ),
            TypeError::TriggerNotCovering(name) => write!(
                f,
                "trigger does not mention bound variable `{name}`; every trigger group must cover all binders"
            ),
            TypeError::FieldAccessInAxiom => {
                write!(f, "field access not allowed in a domain axiom")
            }
            TypeError::UnfoldingInAxiom => {
                write!(f, "`unfolding` not allowed in a domain axiom")
            }
            TypeError::AbstractPredicateNotFoldable(name) => {
                write!(
                    f,
                    "predicate `{name}` has no body: it cannot be folded, unfolded, or used in `unfolding`"
                )
            }
            TypeError::PreconditionedFunctionInAxiom(name) => {
                write!(
                    f,
                    "cannot use function `{name}`, which has preconditions, inside a domain axiom"
                )
            }
            TypeError::UnconstrainedTypeParamInAxiom(name) => {
                write!(
                    f,
                    "unconstrained type parameter `{name}` in a domain axiom; annotate the call"
                )
            }
            TypeError::Tc(e) => write!(f, "Constraint error: {e:?}"),
            TypeError::Mismatch(t1, t2) => {
                write!(f, "type mismatch: cannot unify {t1} and {t2}")
            }
            TypeError::Other(msg) => write!(f, "{msg}"),
        }
    }
}
