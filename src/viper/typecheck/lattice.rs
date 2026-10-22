use rusttyc::Constructable;
use rusttyc::types::{Arity, Partial, Variant};

use crate::viper::typed::{Ident, Type};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ViperTcType {
    Bool,
    Int,
    Real,
    Ref,
    Numeric, // supertype of Int and Real
    /// A domain/ADT type, identified by name and its **type arity** (number of
    /// type parameters). The type *arguments* are tracked as `rusttyc` children
    /// (see `arity`/`construct`), so `Option[Int]` and `Option[Bool]` are
    /// distinguished by their child types, not by the variant alone.
    Domain(Ident, usize),
    /// A **rigid** type parameter (of the enclosing domain, in an axiom): it
    /// unifies only with itself and constructs to `Type::Generic`. Distinct
    /// from an *instantiation* variable (a plain unconstrained key), which any
    /// concrete type may still pin.
    Generic(Ident),
    Top,
}

/// Why a meet or construction failed. Kept structural (not a message) so the
/// clashing types can be named once an interner is at hand — see
/// `TypeError::with_names`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TcTypeErr {
    /// Two variants that have no common subtype.
    Mismatch(ViperTcType, ViperTcType),
    /// A key nothing constrained, so no concrete type can be built for it.
    Abstract,
}

impl std::fmt::Display for TcTypeErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TcTypeErr::Mismatch(t1, t2) => write!(f, "Cannot unify {t1:?} and {t2:?}"),
            TcTypeErr::Abstract => write!(f, "Cannot construct abstract type"),
        }
    }
}

impl std::error::Error for TcTypeErr {}

impl Variant for ViperTcType {
    type Err = TcTypeErr;

    fn arity(&self) -> Arity {
        match self {
            ViperTcType::Domain(_, n) => Arity::Fixed(*n),
            _ => Arity::Fixed(0),
        }
    }

    fn top() -> Self {
        ViperTcType::Top
    }

    fn meet(lhs: Partial<Self>, rhs: Partial<Self>) -> Result<Partial<Self>, Self::Err> {
        use ViperTcType::*;
        let variant = match (lhs.variant, rhs.variant) {
            (Top, x) | (x, Top) => x,
            (Numeric, Numeric) => Numeric,
            (Numeric, x @ (Int | Real)) | (x @ (Int | Real), Numeric) => x,
            (Bool, Bool) => Bool,
            (Ref, Ref) => Ref,
            (Int, Int) => Int,
            (Real, Real) => Real,
            (Domain(a, n), Domain(b, m)) if a == b && n == m => Domain(a, n),
            (Generic(a), Generic(b)) if a == b => Generic(a),
            (t1, t2) => {
                return Err(TcTypeErr::Mismatch(t1, t2));
            }
        };
        // A `Domain(_, n)` has fixed arity `n`, so its `least_arity` must be `n`;
        // every other variant is 0-ary.
        let least_arity = match &variant {
            Domain(_, n) => *n,
            _ => 0,
        };
        Ok(Partial {
            variant,
            least_arity,
        })
    }
}

impl Constructable for ViperTcType {
    type Type = Type;

    fn construct(
        &self,
        children: &[Self::Type],
    ) -> Result<Self::Type, <Self as rusttyc::ContextSensitiveVariant>::Err> {
        Ok(match self {
            ViperTcType::Bool => Type::Bool,
            ViperTcType::Int => Type::Int,
            ViperTcType::Real | ViperTcType::Numeric => Type::Real,
            ViperTcType::Ref => Type::Ref,
            ViperTcType::Domain(id, _) => Type::Domain(*id, children.to_vec()),
            ViperTcType::Generic(id) => Type::Generic(*id),
            ViperTcType::Top => {
                return Err(TcTypeErr::Abstract);
            }
        })
    }
}

/// The top-level variant of a `Type`, **without** its type arguments. The
/// arguments are imposed separately as `rusttyc` children (see
/// `ConstraintCtx::impose_type`); a bare `Generic` has no top-level variant
/// (it is bound to a fresh type variable), so it maps to `Top`.
pub fn type_to_tc(ty: &Type) -> ViperTcType {
    match ty {
        Type::Bool => ViperTcType::Bool,
        Type::Int => ViperTcType::Int,
        Type::Real => ViperTcType::Real,
        Type::Ref => ViperTcType::Ref,
        Type::Domain(id, args) => ViperTcType::Domain(*id, args.len()),
        Type::Generic(_) | Type::Collection(_) => ViperTcType::Top,
    }
}
