use rusttyc::Constructable;
use rusttyc::types::{Arity, Partial, Variant};

use crate::viper::typed::{BuiltinCollection, Ident, Type};

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
    /// `Seq[E]`: children `[E, Int, E]` (see [`MEMBER`]).
    Seq,
    /// `Set[E]`: child `[E]`.
    Set,
    /// `Map[K, V]`: children `[K, K, V]` (see [`MEMBER`]).
    Map,
    /// A `Seq`, a `Set` or a `Map`, not yet known which: what `|c|` and
    /// `x in c` ask of `c`, `x` being its [`MEMBER`] child. Abstract, like
    /// `Numeric`, but never defaulted: a key left at `Collection` cannot be
    /// built.
    Collection,
    /// A `Seq` or a `Map`, not yet known which: what `c[i]` and `c[i := v]`
    /// ask of `c`, `i` being its [`INDEX`] child and `c[i]` its [`ELEMENT`]
    /// child. Abstract like `Collection`.
    Indexable,
    Top,
}

/// The child positions of the collection variants. Constraint generation
/// runs before the kind of an indexed or tested collection is known, and
/// `rusttyc` relates children by position, so one constraint has to fit every
/// kind it may turn out to be: each kind puts the type its operations share at
/// the same position, and repeats a type where two positions coincide.
///
/// - [`MEMBER`]: what `x in c` tests, the element of a `Seq` or `Set` and the
///   key of a `Map`.
/// - [`INDEX`]: what `c[i]` takes, `Int` for a `Seq` and the key of a `Map`.
/// - [`ELEMENT`]: what `c[i]` yields, the element of a `Seq` and the value of
///   a `Map`.
///
/// So a `Seq[E]` is `[E, Int, E]`, a `Set[E]` is `[E]` (it has no index), and
/// a `Map[K, V]` is `[K, K, V]`. The repeats are constraints, not lattice
/// facts: whoever makes a key a `Seq` or `Map` equates them (see
/// `ConstraintCtx::seq_key` and `ConstraintCtx::map_key`).
pub const MEMBER: usize = 0;
/// See [`MEMBER`].
pub const INDEX: usize = 1;
/// See [`MEMBER`].
pub const ELEMENT: usize = 2;

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
            ViperTcType::Set | ViperTcType::Collection => Arity::Fixed(1),
            ViperTcType::Seq | ViperTcType::Map | ViperTcType::Indexable => Arity::Fixed(3),
            ViperTcType::Bool
            | ViperTcType::Int
            | ViperTcType::Real
            | ViperTcType::Ref
            | ViperTcType::Numeric
            | ViperTcType::Generic(_)
            | ViperTcType::Top => Arity::Fixed(0),
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
            (Seq, Seq) => Seq,
            (Set, Set) => Set,
            (Map, Map) => Map,
            (Collection, Collection) => Collection,
            (Indexable, Indexable) => Indexable,
            (Collection, x @ (Seq | Set | Map | Indexable))
            | (x @ (Seq | Set | Map | Indexable), Collection) => x,
            (Indexable, x @ (Seq | Map)) | (x @ (Seq | Map), Indexable) => x,
            (t1, t2) => {
                return Err(TcTypeErr::Mismatch(t1, t2));
            }
        };
        // A `Domain(_, n)` has fixed arity `n`, so its `least_arity` must be `n`;
        // every other variant is 0-ary.
        let least_arity = match &variant {
            Domain(_, n) => *n,
            Set | Collection => 1,
            Seq | Map | Indexable => 3,
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
            ViperTcType::Top | ViperTcType::Collection | ViperTcType::Indexable => {
                return Err(TcTypeErr::Abstract);
            }
            ViperTcType::Seq => Type::Collection(BuiltinCollection::Seq(Box::new(
                children
                    .get(MEMBER)
                    .expect("sequence must have a child type")
                    .clone(),
            ))),
            ViperTcType::Set => Type::Collection(BuiltinCollection::Set(Box::new(
                children
                    .get(MEMBER)
                    .expect("set must have a child type")
                    .clone(),
            ))),
            ViperTcType::Map => Type::Collection(BuiltinCollection::Map(
                Box::new(
                    children
                        .get(MEMBER)
                        .expect("map must have a key type")
                        .clone(),
                ),
                Box::new(
                    children
                        .get(ELEMENT)
                        .expect("map must have a value type")
                        .clone(),
                ),
            )),
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
        Type::Collection(BuiltinCollection::Seq(_)) => ViperTcType::Seq,
        Type::Collection(BuiltinCollection::Set(_)) => ViperTcType::Set,
        Type::Collection(BuiltinCollection::Map(..)) => ViperTcType::Map,
        Type::Generic(_) | Type::Collection(_) => ViperTcType::Top,
    }
}
