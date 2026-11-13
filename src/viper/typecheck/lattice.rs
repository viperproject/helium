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
    /// `Seq[E]`: children `[E, Bool, Int, E]` (see [`MEMBER`]).
    Seq,
    /// `Set[E]`: children `[E, Bool]`.
    Set,
    /// `Map[K, V]`: children `[K, Bool, K, V]` (see [`MEMBER`]).
    Map,
    /// `Multiset[E]`: children `[E, Int]`: `x in m` is a count.
    Multiset,
    /// A `Seq`, a `Set`, a `Map` or a `Multiset`, not yet known which: what
    /// `|c|` and `x in c` ask of `c`, `x` being its [`MEMBER`] child and
    /// `x in c` its [`MEMBERSHIP`] child. Abstract, like `Numeric`, but never
    /// defaulted: a key left at `Collection` cannot be built.
    Collection,
    /// A `Set` or a `Multiset`, not yet known which: what `union`,
    /// `intersection`, `setminus` and `subset` ask of their operands. Abstract
    /// like `Collection`.
    AnySet,
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
/// - [`MEMBER`]: what `x in c` tests, the element of a `Seq`, `Set` or
///   `Multiset` and the key of a `Map`.
/// - [`MEMBERSHIP`]: what `x in c` yields, a `Bool`, except for a `Multiset`,
///   where it is the `Int` count of `x`.
/// - [`INDEX`]: what `c[i]` takes, `Int` for a `Seq` and the key of a `Map`.
/// - [`ELEMENT`]: what `c[i]` yields, the element of a `Seq` and the value of
///   a `Map`.
///
/// So a `Seq[E]` is `[E, Bool, Int, E]`, a `Set[E]` is `[E, Bool]` and a
/// `Multiset[E]` `[E, Int]` (they have no index), and a `Map[K, V]` is
/// `[K, Bool, K, V]`. The fixed and repeated children are constraints, not
/// lattice facts: whoever makes a key a collection imposes them (see
/// `ConstraintCtx::seq_key`, `set_key`, `map_key` and `multiset_key`).
pub const MEMBER: usize = 0;
/// See [`MEMBER`].
pub const MEMBERSHIP: usize = 1;
/// See [`MEMBER`].
pub const INDEX: usize = 2;
/// See [`MEMBER`].
pub const ELEMENT: usize = 3;

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
            ViperTcType::Set
            | ViperTcType::Multiset
            | ViperTcType::Collection
            | ViperTcType::AnySet => Arity::Fixed(2),
            ViperTcType::Seq | ViperTcType::Map | ViperTcType::Indexable => Arity::Fixed(4),
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
            (Multiset, Multiset) => Multiset,
            (Collection, Collection) => Collection,
            (AnySet, AnySet) => AnySet,
            (Indexable, Indexable) => Indexable,
            (Collection, x @ (Seq | Set | Map | Multiset | AnySet | Indexable))
            | (x @ (Seq | Set | Map | Multiset | AnySet | Indexable), Collection) => x,
            (AnySet, x @ (Set | Multiset)) | (x @ (Set | Multiset), AnySet) => x,
            (Indexable, x @ (Seq | Map)) | (x @ (Seq | Map), Indexable) => x,
            (t1, t2) => {
                return Err(TcTypeErr::Mismatch(t1, t2));
            }
        };
        // A `Domain(_, n)` has fixed arity `n`, so its `least_arity` must be `n`;
        // every other variant is 0-ary.
        let least_arity = match &variant {
            Domain(_, n) => *n,
            Set | Multiset | Collection | AnySet => 2,
            Seq | Map | Indexable => 4,
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
            ViperTcType::Top
            | ViperTcType::Collection
            | ViperTcType::AnySet
            | ViperTcType::Indexable => {
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
            ViperTcType::Multiset => Type::Collection(BuiltinCollection::MultiSet(Box::new(
                children
                    .get(MEMBER)
                    .expect("multiset must have a child type")
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
        Type::Collection(BuiltinCollection::MultiSet(_)) => ViperTcType::Multiset,
        Type::Generic(_) => ViperTcType::Top,
    }
}
