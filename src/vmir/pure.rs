use crate::vmir::display::VmirDisplay;
use crate::vmir::{FunctionCall, HeapVal, Type};
use std::fmt::{self, Display, Formatter};

/// A value can be either a literal or a temporary variable defined earlier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Val {
    Literal(Literal),
    Temp(usize),
}

/// A binary operator, **tagged with the sort of its operands** (`I` = `Int`,
/// `R` = `Real`, i.e. Viper's `Perm`).
///
/// Viper writes one `+` for both integer and permission addition; the sort is
/// recovered during lowering, where the typed AST still has it. Carrying it on the
/// operator is what lets an e-graph rewrite *produce* a literal: `x - x => 0` has
/// no operand literal to read a sort off, and the wrong choice puts an `Int` and a
/// `Real` literal in one e-class — a type error the analysis panics on.
///
/// `Eq` is deliberately **not** tagged: it also compares `Bool`, `Ref`, ADTs and
/// snapshots, so a two-sort tag does not fit it, and none of its rules need to
/// synthesise a typed literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BinOp {
    AddI,
    AddR,
    SubI,
    SubR,
    MulI,
    MulR,
    /// SIDECOND: The second operand must be non-zero.
    DivI,
    /// SIDECOND: The second operand must be non-zero.
    DivR,
    /// SIDECOND: The second operand must be non-zero. `Int`-only by typing, so
    /// it needs no sort tag.
    Mod,
    LtI,
    LtR,
    /// Polymorphic over every type — see the type note above.
    Eq,
}

impl BinOp {
    /// The operand sort, or `None` for the polymorphic `Eq`.
    pub fn sort(self) -> Option<Type> {
        use BinOp::*;
        match self {
            AddI | SubI | MulI | DivI | Mod | LtI => Some(Type::Int),
            AddR | SubR | MulR | DivR | LtR => Some(Type::Real),
            Eq => None,
        }
    }

    /// Whether this is the division-or-modulo family, whose non-zero-divisor
    /// side condition is checked without consulting a heap.
    pub fn is_div_or_mod(self) -> bool {
        matches!(self, BinOp::DivI | BinOp::DivR | BinOp::Mod)
    }

    /// Whether this is one of the additive operators, in either sort.
    pub fn is_additive(self) -> bool {
        matches!(self, BinOp::AddI | BinOp::AddR | BinOp::SubI | BinOp::SubR)
    }
}

/// Pick the sort-tagged operator for `ty`. Each panics on a non-numeric operand
/// type: the typed Viper AST has already rejected `+` on a `Bool` or an ADT, so
/// reaching one of these means the lowering handed over the wrong type, and
/// silently guessing a sort would plant a type error deep in the e-graph.
macro_rules! sorted_ctor {
    ($($name:ident => $int:ident / $real:ident),* $(,)?) => {
        impl BinOp {
            $(
                pub fn $name(ty: &Type) -> BinOp {
                    match ty {
                        Type::Int => BinOp::$int,
                        Type::Real => BinOp::$real,
                        other => panic!(
                            concat!("`", stringify!($name), "` on a non-numeric operand type: {:?}"),
                            other
                        ),
                    }
                }
            )*
        }
    };
}

sorted_ctor! {
    add => AddI / AddR,
    sub => SubI / SubR,
    mul => MulI / MulR,
    div => DivI / DivR,
    lt  => LtI  / LtR,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Literal {
    Null,
    Bool(bool),
    Int(num::BigInt),
    Real(num::BigRational),
}

pub const NULL: Val = Val::Literal(Literal::Null);
pub const TRUE: Val = Val::Literal(Literal::Bool(true));
pub const FALSE: Val = Val::Literal(Literal::Bool(false));

pub fn none() -> Val {
    Val::Literal(Literal::Real(num::BigInt::from(0).into()))
}
pub fn write() -> Val {
    Val::Literal(Literal::Real(num::BigInt::from(1).into()))
}

/// Which built-in collection a [`Type::Coll`] is. Each is a value over its
/// type arguments (an element type `E`, or a key and value type `K`, `V`),
/// axiomatized rather than modelled (see `verify::coll_axioms`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CollKind {
    Seq,
    Set,
    Map,
}

impl CollKind {
    /// Every collection kind, in `CollKind as usize` order.
    pub const ALL: [CollKind; 3] = [CollKind::Seq, CollKind::Set, CollKind::Map];

    /// The Viper name of the collection type, `Seq`, `Set` or `Map`.
    pub fn name(self) -> &'static str {
        match self {
            CollKind::Seq => "Seq",
            CollKind::Set => "Set",
            CollKind::Map => "Map",
        }
    }

    /// How many type arguments the collection type takes: `Seq[E]` and
    /// `Set[E]` one, `Map[K, V]` two.
    pub fn arity(self) -> usize {
        match self {
            CollKind::Seq | CollKind::Set => 1,
            CollKind::Map => 2,
        }
    }

    /// Whether the kind has a one-element literal, `Seq(e)` or `Set(e)`
    /// ([`PureInst::CollLit`]). A map literal is updates on the empty map.
    pub fn has_unit(self) -> bool {
        match self {
            CollKind::Seq | CollKind::Set => true,
            CollKind::Map => false,
        }
    }
}

/// An operation on a built-in collection. Shared by `PureInst::Coll`,
/// `TrigHead::Coll` and `Symbolic::Coll`, so the three layers cannot drift
/// apart. Every operation works on one collection kind, the one
/// [`CollOp::kind`] names; its operands follow that kind's own order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
pub enum CollOp {
    Seq(SeqOp),
    Set(SetOp),
    Map(MapOp),
}

impl CollOp {
    /// The collection kind this operation works on.
    pub fn kind(self) -> CollKind {
        match self {
            CollOp::Seq(_) => CollKind::Seq,
            CollOp::Set(_) => CollKind::Set,
            CollOp::Map(_) => CollKind::Map,
        }
    }

    /// The empty collection of `kind`.
    pub fn empty(kind: CollKind) -> CollOp {
        match kind {
            CollKind::Seq => CollOp::Seq(SeqOp::Empty),
            CollKind::Set => CollOp::Set(SetOp::Empty),
            CollKind::Map => CollOp::Map(MapOp::Empty),
        }
    }

    /// The program's typed equality of two collections of `kind`.
    pub fn eq(kind: CollKind) -> CollOp {
        match kind {
            CollKind::Seq => CollOp::Seq(SeqOp::Eq),
            CollKind::Set => CollOp::Set(SetOp::Eq),
            CollKind::Map => CollOp::Map(MapOp::Eq),
        }
    }

    /// `|c|`: a sequence's length, a set's or a map's cardinality.
    pub fn size(kind: CollKind) -> CollOp {
        match kind {
            CollKind::Seq => CollOp::Seq(SeqOp::Len),
            CollKind::Set => CollOp::Set(SetOp::Card),
            CollKind::Map => CollOp::Map(MapOp::Card),
        }
    }

    /// `x in c`: operands `[c, x]`, the collection first. Panics for a map,
    /// whose `k in m` is `k in domain(m)`: a set membership (see [`MapOp`]).
    pub fn contains(kind: CollKind) -> CollOp {
        match kind {
            CollKind::Seq => CollOp::Seq(SeqOp::Contains),
            CollKind::Set => CollOp::Set(SetOp::Contains),
            CollKind::Map => panic!("a map's `k in m` is `k in domain(m)`, a set membership"),
        }
    }

    /// Whether this is the typed equality of its kind, which the verifier keeps
    /// in one e-class with the untyped `==` (see [`SeqOp::Eq`]).
    pub fn is_eq(self) -> bool {
        matches!(
            self,
            CollOp::Seq(SeqOp::Eq) | CollOp::Set(SetOp::Eq) | CollOp::Map(MapOp::Eq)
        )
    }

    /// The type arguments of the collection this operation works on, given
    /// the ones written on the node: a `Range` is always on `Seq<Int>`.
    pub fn coll_tys(self, tys: &[Type]) -> Box<[Type]> {
        match self {
            CollOp::Seq(SeqOp::Range) => Box::new([Type::Int]),
            _ => tys.into(),
        }
    }

    /// How many operands `PureInst::Coll` of this operation takes.
    pub fn arity(self) -> usize {
        match self {
            CollOp::Seq(op) => op.arity(),
            CollOp::Set(op) => op.arity(),
            CollOp::Map(op) => op.arity(),
        }
    }

    /// Whether this op has a side condition on its operands (a sequence index
    /// in range, a map key present), so `emit_pure_guarded` must check it (the
    /// counterpart of `BinOp::is_div_or_mod`).
    pub fn has_side_condition(self) -> bool {
        match self {
            CollOp::Seq(op) => op.has_bound_check(),
            CollOp::Set(_) => false,
            CollOp::Map(op) => op.has_key_check(),
        }
    }

    /// The operand types, in operand order, for a collection of type
    /// arguments `tys`.
    pub fn operand_types(self, tys: &[Type]) -> Vec<Type> {
        match self {
            CollOp::Seq(op) => op.operand_types(elem_of(self, tys)),
            CollOp::Set(op) => op.operand_types(elem_of(self, tys)),
            CollOp::Map(op) => {
                let (k, v) = key_value_of(tys);
                op.operand_types(k, v)
            }
        }
    }

    /// The result type, for a collection of type arguments `tys`.
    pub fn result_type(self, tys: &[Type]) -> Type {
        match self {
            CollOp::Seq(op) => op.result_type(elem_of(self, tys)),
            CollOp::Set(op) => op.result_type(elem_of(self, tys)),
            CollOp::Map(op) => {
                let (k, v) = key_value_of(tys);
                op.result_type(k, v)
            }
        }
    }

    pub fn token(self) -> &'static str {
        match self {
            CollOp::Seq(op) => op.token(),
            CollOp::Set(op) => op.token(),
            CollOp::Map(op) => op.token(),
        }
    }
}

/// The element type of a `Seq` or `Set` operation's type arguments.
fn elem_of(op: CollOp, tys: &[Type]) -> &Type {
    match tys {
        [elem] => elem,
        _ => panic!("`{op:?}` works on one element type, got {tys:?}"),
    }
}

/// The key and value types of a `Map` operation's type arguments.
fn key_value_of(tys: &[Type]) -> (&Type, &Type) {
    match tys {
        [k, v] => (k, v),
        _ => panic!("a map operation works on a key and a value type, got {tys:?}"),
    }
}

/// A sequence operation (see [`CollOp`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SeqOp {
    /// `|s|` : Int
    Len,
    /// `s ++ t` : Seq<T>
    Concat,
    /// `s[i]` : T. SIDECOND: `0 <= i < |s|`.
    Index,
    /// `s[i := v]` : Seq<T>. SIDECOND: `0 <= i < |s|`.
    Update,
    /// `s[..n]` : Seq<T>. Total: Viper clamps `n`.
    Take,
    /// `s[n..]` : Seq<T>. Total: Viper clamps `n`.
    Drop,
    /// `x in s` : Bool. Operands are `[s, x]`: the sequence always comes first.
    Contains,
    /// `[a..b)` : Seq<Int>
    Range,
    /// `Seq[T]()` : Seq<T>. Takes no operands, so `elem_ty` is all that names `T`.
    Empty,
    /// `s == t` : Bool, a program's equality of two sequences. Typed, unlike
    /// `BinOp::Eq`, so the extensionality axioms can trigger on it: the
    /// verifier keeps it in one e-class with the untyped `s == t`, which every
    /// other part of equality reasoning reads.
    Eq,
}

impl SeqOp {
    /// How many operands `PureInst::Coll` of this operation takes.
    pub fn arity(self) -> usize {
        match self {
            SeqOp::Empty => 0,
            SeqOp::Len => 1,
            SeqOp::Concat
            | SeqOp::Index
            | SeqOp::Take
            | SeqOp::Drop
            | SeqOp::Contains
            | SeqOp::Range
            | SeqOp::Eq => 2,
            SeqOp::Update => 3,
        }
    }

    /// Whether this op has the in-range side condition, so `emit_pure_guarded`
    /// must check it (the counterpart of `BinOp::is_div_or_mod`).
    pub fn has_bound_check(self) -> bool {
        matches!(self, SeqOp::Index | SeqOp::Update)
    }

    /// The operand types, in operand order, for a sequence of element type
    /// `elem_ty` (for a `Range`, `Int`).
    pub fn operand_types(self, elem_ty: &Type) -> Vec<Type> {
        let seq = || Type::seq(elem_ty.clone());
        match self {
            SeqOp::Empty => vec![],
            SeqOp::Len => vec![seq()],
            SeqOp::Concat | SeqOp::Eq => vec![seq(), seq()],
            SeqOp::Index | SeqOp::Take | SeqOp::Drop => vec![seq(), Type::Int],
            SeqOp::Update => vec![seq(), Type::Int, elem_ty.clone()],
            SeqOp::Contains => vec![seq(), elem_ty.clone()],
            SeqOp::Range => vec![Type::Int, Type::Int],
        }
    }

    /// The result type, for a sequence of element type `elem_ty`.
    pub fn result_type(self, elem_ty: &Type) -> Type {
        match self {
            SeqOp::Len => Type::Int,
            SeqOp::Contains | SeqOp::Eq => Type::Bool,
            SeqOp::Index => elem_ty.clone(),
            SeqOp::Concat
            | SeqOp::Update
            | SeqOp::Take
            | SeqOp::Drop
            | SeqOp::Range
            | SeqOp::Empty => Type::seq(elem_ty.clone()),
        }
    }

    pub fn token(self) -> &'static str {
        match self {
            SeqOp::Len => "seq.len",
            SeqOp::Index => "seq.at",
            SeqOp::Update => "seq.upd",
            SeqOp::Concat => "seq.++",
            SeqOp::Take => "seq.take",
            SeqOp::Drop => "seq.drop",
            SeqOp::Range => "seq.range",
            SeqOp::Contains => "seq.in",
            SeqOp::Empty => "seq.empty",
            SeqOp::Eq => "seq.eq",
        }
    }
}

/// A map operation (see [`CollOp`]). A map literal `Map(k := v, ..)` has no
/// node of its own: it is `Update`s on `Empty`, one per entry, in order, so a
/// repeated key keeps its last value. Nor does `k in m`: as in Silicon, it is
/// `k in domain(m)`, a `SetOp::Contains` on `Domain`, so a trigger on either
/// form is the same term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MapOp {
    /// `|m|` : Int, the number of keys.
    Card,
    /// `m[k]` : V. SIDECOND: `k in domain(m)`.
    Lookup,
    /// `m[k := v]` : Map<K, V>. Total: adds `k` if it is not a key yet.
    Update,
    /// `domain(m)` : Set<K>
    Domain,
    /// `range(m)` : Set<V>
    Range,
    /// `Map[K, V]()` : Map<K, V>. Takes no operands, so the node's type
    /// arguments are all that name `K` and `V`.
    Empty,
    /// `m == n` : Bool, a program's equality of two maps (see [`SeqOp::Eq`]).
    Eq,
}

impl MapOp {
    /// How many operands `PureInst::Coll` of this operation takes.
    pub fn arity(self) -> usize {
        match self {
            MapOp::Empty => 0,
            MapOp::Card | MapOp::Domain | MapOp::Range => 1,
            MapOp::Lookup | MapOp::Eq => 2,
            MapOp::Update => 3,
        }
    }

    /// Whether this op needs its key in the map, so `emit_pure_guarded` must
    /// check it.
    pub fn has_key_check(self) -> bool {
        matches!(self, MapOp::Lookup)
    }

    /// The operand types, in operand order, for a map from `k` to `v`.
    pub fn operand_types(self, k: &Type, v: &Type) -> Vec<Type> {
        let map = || Type::map(k.clone(), v.clone());
        match self {
            MapOp::Empty => vec![],
            MapOp::Card | MapOp::Domain | MapOp::Range => vec![map()],
            MapOp::Lookup => vec![map(), k.clone()],
            MapOp::Update => vec![map(), k.clone(), v.clone()],
            MapOp::Eq => vec![map(), map()],
        }
    }

    /// The result type, for a map from `k` to `v`.
    pub fn result_type(self, k: &Type, v: &Type) -> Type {
        match self {
            MapOp::Card => Type::Int,
            MapOp::Eq => Type::Bool,
            MapOp::Lookup => v.clone(),
            MapOp::Update | MapOp::Empty => Type::map(k.clone(), v.clone()),
            MapOp::Domain => Type::set(k.clone()),
            MapOp::Range => Type::set(v.clone()),
        }
    }

    pub fn token(self) -> &'static str {
        match self {
            MapOp::Card => "map.card",
            MapOp::Lookup => "map.at",
            MapOp::Update => "map.upd",
            MapOp::Domain => "map.domain",
            MapOp::Range => "map.range",
            MapOp::Empty => "map.empty",
            MapOp::Eq => "map.eq",
        }
    }
}

/// A set operation (see [`CollOp`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SetOp {
    /// `|s|` : Int
    Card,
    /// `x in s` : Bool. Operands are `[s, x]`: the set always comes first.
    Contains,
    /// `s union t` : Set<T>
    Union,
    /// `s intersection t` : Set<T>
    Intersection,
    /// `s setminus t` : Set<T>
    Difference,
    /// `s subset t` : Bool
    Subset,
    /// `s union Set(x)` : Set<T>, Silicon's `Set_unionone`. No surface syntax:
    /// a literal `Set(a, b, c)` is `Set(a)` with `b`, then `c`, added, which is
    /// what lets its cardinality follow without an extensionality argument.
    Add,
    /// `Set[T]()` : Set<T>. Takes no operands, so `elem_ty` is all that names `T`.
    Empty,
    /// `s == t` : Bool, a program's equality of two sets (see [`SeqOp::Eq`]).
    Eq,
}

impl SetOp {
    /// How many operands `PureInst::Coll` of this operation takes.
    pub fn arity(self) -> usize {
        match self {
            SetOp::Empty => 0,
            SetOp::Card => 1,
            SetOp::Contains
            | SetOp::Union
            | SetOp::Intersection
            | SetOp::Difference
            | SetOp::Subset
            | SetOp::Add
            | SetOp::Eq => 2,
        }
    }

    /// The operand types, in operand order, for a set of element type `elem_ty`.
    pub fn operand_types(self, elem_ty: &Type) -> Vec<Type> {
        let set = || Type::set(elem_ty.clone());
        match self {
            SetOp::Empty => vec![],
            SetOp::Card => vec![set()],
            SetOp::Contains | SetOp::Add => vec![set(), elem_ty.clone()],
            SetOp::Union | SetOp::Intersection | SetOp::Difference | SetOp::Subset | SetOp::Eq => {
                vec![set(), set()]
            }
        }
    }

    /// The result type, for a set of element type `elem_ty`.
    pub fn result_type(self, elem_ty: &Type) -> Type {
        match self {
            SetOp::Card => Type::Int,
            SetOp::Contains | SetOp::Subset | SetOp::Eq => Type::Bool,
            SetOp::Union | SetOp::Intersection | SetOp::Difference | SetOp::Add | SetOp::Empty => {
                Type::set(elem_ty.clone())
            }
        }
    }

    pub fn token(self) -> &'static str {
        match self {
            SetOp::Card => "set.card",
            SetOp::Contains => "set.in",
            SetOp::Union => "set.union",
            SetOp::Intersection => "set.inter",
            SetOp::Difference => "set.diff",
            SetOp::Subset => "set.subset",
            SetOp::Add => "set.add",
            SetOp::Empty => "set.empty",
            SetOp::Eq => "set.eq",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PureInst {
    Fresh,
    Binary(BinOp, Val, Val),
    Ternary(Val, Val, Val),
    /// Cast an Int value to a Real
    RealCast(Val),
    /// Read a value from a heap at a given location.
    Deref(HeapVal, Val),
    /// Query the permission amount of an address in a heap.
    Perm(HeapVal, Val),
    /// A (possibly generic) Silver `function` application. Always pure and
    /// heap-free: a heap-dependent function takes its precondition **snapshot**
    /// (yielded by the frame-only `exhale` of its `#requires` at the call site)
    /// as an ordinary trailing argument. Generics (`type_args`) live inside the `FunctionCall`.
    FunctionCall(FunctionCall),
    /// `unwrap(v)` -- the `Some` payload of an `Option`. The result type is the
    /// option's element type, so no type argument is stored.
    ///
    /// This is the **seam** of a desugared `unfold`: the paired `Sub` yields
    /// `Option<Snap(P)>` because it is the one instruction that *discovers*
    /// whether the location held anything, and the resource `inhale` needs a
    /// plain `Snap(P)`. Naming the coercion here rather than performing it
    /// implicitly keeps `Bind::Bound` plainly typed.
    ///
    /// Not a proof obligation: presence is `0 < perm`, which const-folds to
    /// `true` for a literal amount, so the yield is concretely `Some(v)` and
    /// this peels by the existing `unwrap o Some` reduction.
    OptionUnwrap(Val),
    /// Construct ADT value: variant `variant` of the ADT `adt` instantiated at
    /// `type_args`, over `args`. The ADT is named by its (possibly generic)
    /// declaration `MemberId`; `type_args` is its monomorphization (empty for a
    /// non-generic ADT). Replaces a synthetic-constructor `FunctionCall`.
    AdtCons {
        adt: crate::vmir::MemberId,
        type_args: Vec<crate::vmir::Type>,
        variant: usize,
        args: Vec<Val>,
    },
    /// Project field `field` of variant `variant` of the ADT `adt` instantiated
    /// at `type_args`, from `base` (`field`-th constructor argument). Replaces a
    /// synthetic-destructor `FunctionCall`.
    AdtProj {
        adt: crate::vmir::MemberId,
        type_args: Vec<crate::vmir::Type>,
        variant: usize,
        field: usize,
        base: Val,
    },
    /// The discriminator `base.isC`: whether `base : adt[type_args]` was built
    /// by variant `variant`. A `Bool`, printed `Adt::C?(base)`; a negative test
    /// is this tester being `false`, so it is an equality the verifier can use.
    AdtIs {
        adt: crate::vmir::MemberId,
        type_args: Vec<crate::vmir::Type>,
        variant: usize,
        base: Val,
    },

    /// The singleton collection `Seq(elem)` or `Set(elem)`. A longer sequence
    /// literal is a right-nested concatenation of these, `Seq(a, b, c)` being
    /// `Seq(a) ++ (Seq(b) ++ Seq(c))`; a longer set literal adds each further
    /// element (`SetOp::Add`); an empty one is the kind's `Empty`. Never a
    /// `Map` (see [`CollKind::has_unit`]).
    CollLit {
        kind: CollKind,
        elem_ty: Type,
        elem: Val,
    },
    /// A collection operation. Build it with `PureInst::coll`, which checks the
    /// arity. `tys` are the type arguments of the collection it works on
    /// (`[Int]` for `Range`): the result does not always say, as `|s|` shows.
    Coll {
        tys: Box<[Type]>,
        op: CollOp,
        args: Vec<Val>,
    },

    /// An inline, nestable pure `forall` (see [`Forall`](crate::vmir::Forall)).
    /// Produces the quantified `Bool`. Its body shares *this* temp space —
    /// capture is implicit, so a free occurrence is just the enclosing temp.
    Forall(Box<crate::vmir::Forall>),
}

impl PureInst {
    /// Visit every `Val` operand, in source order. A `Forall` yields the enclosing
    /// temps its triggers and body mention (its own binders and steps stay
    /// inside), which is exactly its implicit capture list.
    pub fn for_each_operand(&self, f: &mut impl FnMut(&Val)) {
        match self {
            PureInst::Fresh => {}
            PureInst::Binary(_, l, r) => {
                f(l);
                f(r);
            }
            PureInst::Ternary(c, t, e) => {
                f(c);
                f(t);
                f(e);
            }
            PureInst::RealCast(v)
            | PureInst::Deref(_, v)
            | PureInst::Perm(_, v)
            | PureInst::AdtProj { base: v, .. }
            | PureInst::AdtIs { base: v, .. }
            | PureInst::CollLit { elem: v, .. } => f(v),
            PureInst::FunctionCall(fc) => fc.args.iter().for_each(&mut *f),
            PureInst::OptionUnwrap(v) => f(v),
            PureInst::AdtCons { args, .. } => args.iter().for_each(&mut *f),
            PureInst::Forall(q) => q.free_temps().iter().for_each(|k| f(&Val::Temp(*k))),
            PureInst::Coll { args, .. } => args.iter().for_each(&mut *f),
        }
    }
    /// The collection operation `op` on a collection of type arguments `tys`.
    /// Panics if `args` or `tys` do not have the lengths `op` takes.
    pub fn coll(tys: impl Into<Box<[Type]>>, op: impl Into<CollOp>, args: Vec<Val>) -> PureInst {
        let op = op.into();
        let tys = tys.into();
        assert_eq!(
            args.len(),
            op.arity(),
            "`{op:?}` takes {} operands, got {args:?}",
            op.arity()
        );
        assert_eq!(
            tys.len(),
            op.kind().arity(),
            "`{op:?}` takes {} type arguments, got {tys:?}",
            op.kind().arity()
        );
        PureInst::Coll { tys, op, args }
    }

    /// The singleton `kind(elem)`. Panics for a kind without one (a `Map`).
    pub fn coll_lit(kind: CollKind, elem_ty: Type, elem: Val) -> PureInst {
        assert!(
            kind.has_unit(),
            "`{}` has no one-element literal",
            kind.name()
        );
        PureInst::CollLit {
            kind,
            elem_ty,
            elem,
        }
    }
}

// ======================
// DISPLAY INFRASTRUCTURE
// ======================

impl Display for Val {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Val::Literal(lit) => write!(f, "{lit}"),
            Val::Temp(i) => write!(f, "e{i}"),
        }
    }
}

impl Display for BinOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // The sort suffix is part of the token (`+i` / `+r`), so a VMIR dump and
        // an e-graph rewrite pattern name the same operator the same way.
        let op = match self {
            BinOp::AddI => "+i",
            BinOp::AddR => "+r",
            BinOp::SubI => "-i",
            BinOp::SubR => "-r",
            BinOp::MulI => "*i",
            BinOp::MulR => "*r",
            BinOp::DivI => "/i",
            BinOp::DivR => "/r",
            BinOp::Mod => "%",
            BinOp::LtI => "<i",
            BinOp::LtR => "<r",
            BinOp::Eq => "==",
        };
        write!(f, "{op}")
    }
}

impl Display for Literal {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Null => write!(f, "null"),
            Literal::Bool(b) => write!(f, "{b}"),
            Literal::Int(v) => write!(f, "{v}"),
            // Reals are permission amounts: always show as a fraction (no space),
            // e.g. `1/1`, `1/2`, `0/1` — never the reduced integer form.
            Literal::Real(v) => write!(f, "{}/{}", v.numer(), v.denom()),
        }
    }
}

impl<'a> Display for VmirDisplay<'a, &'a PureInst> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.item {
            PureInst::Fresh => write!(f, "fresh"),
            PureInst::Binary(op, lhs, rhs) => write!(f, "{lhs} {op} {rhs}"),
            PureInst::RealCast(v) => write!(f, "real({v})"),
            PureInst::Ternary(cond, then_val, else_val) => {
                write!(f, "{cond} ? {then_val} : {else_val}")
            }
            PureInst::Deref(heap, loc) => write!(f, "*[{heap}] {loc}"),
            PureInst::OptionUnwrap(v) => write!(f, "unwrap({v})"),
            // `FunctionCall` renders itself (`name[heap](args)`) via its own
            // `VmirDisplay` impl, which resolves the callee `MemberId` -> name.
            PureInst::FunctionCall(call) => write!(f, "{}", self.with(call)),
            PureInst::Perm(heap, loc) => write!(f, "perm[{heap}] {loc}"),
            PureInst::AdtCons {
                adt, variant, args, ..
            } => {
                write!(f, "{}(", self.adt_variant(*adt, *variant))?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            PureInst::AdtProj {
                adt,
                variant,
                field,
                base,
                ..
            } => write!(f, "{}.{field}({base})", self.adt_variant(*adt, *variant)),
            PureInst::AdtIs {
                adt, variant, base, ..
            } => write!(f, "{}?({base})", self.adt_variant(*adt, *variant)),
            PureInst::Forall(q) => write!(f, "{}", self.with(&**q)),
            PureInst::CollLit {
                kind,
                elem_ty,
                elem,
            } => {
                write!(f, "{}<{}>({elem})", kind.name(), self.with(elem_ty))
            }
            PureInst::Coll { tys, op, args } => {
                use CollOp::{Map, Seq, Set};
                match (op, args.as_slice()) {
                    (Seq(SeqOp::Empty) | Set(SetOp::Empty) | Map(MapOp::Empty), []) => {
                        let tys: Vec<String> =
                            tys.iter().map(|t| self.with(t).to_string()).collect();
                        write!(f, "{}<{}>()", op.kind().name(), tys.join(", "))
                    }
                    (Seq(SeqOp::Len) | Set(SetOp::Card) | Map(MapOp::Card), [s]) => {
                        write!(f, "|{s}|")
                    }
                    (Seq(SeqOp::Concat), [s, t]) => write!(f, "{s} ++ {t}"),
                    (Seq(SeqOp::Index), [s, i]) => write!(f, "{s}[{i}]"),
                    (Seq(SeqOp::Update), [s, i, v]) => write!(f, "{s}[{i} := {v}]"),
                    (Seq(SeqOp::Take), [s, n]) => write!(f, "{s}[..{n}]"),
                    (Seq(SeqOp::Drop), [s, n]) => write!(f, "{s}[{n}..]"),
                    (Seq(SeqOp::Contains) | Set(SetOp::Contains), [s, x]) => {
                        write!(f, "{x} in {s}")
                    }
                    (Map(MapOp::Lookup), [m, k]) => write!(f, "{m}[{k}]"),
                    (Map(MapOp::Update), [m, k, v]) => write!(f, "{m}[{k} := {v}]"),
                    (Map(MapOp::Domain), [m]) => write!(f, "domain({m})"),
                    (Map(MapOp::Range), [m]) => write!(f, "range({m})"),
                    (Map(MapOp::Eq), [m, n]) => write!(f, "{m} ==map {n}"),
                    (Seq(SeqOp::Range), [a, b]) => write!(f, "[{a}..{b})"),
                    (Seq(SeqOp::Eq), [s, t]) => write!(f, "{s} ==seq {t}"),
                    (Set(SetOp::Union), [s, t]) => write!(f, "{s} union {t}"),
                    (Set(SetOp::Intersection), [s, t]) => write!(f, "{s} intersection {t}"),
                    (Set(SetOp::Difference), [s, t]) => write!(f, "{s} setminus {t}"),
                    (Set(SetOp::Subset), [s, t]) => write!(f, "{s} subset {t}"),
                    (Set(SetOp::Add), [s, x]) => write!(f, "{s} add {x}"),
                    (Set(SetOp::Eq), [s, t]) => write!(f, "{s} ==set {t}"),
                    (op, args) => unreachable!(
                        "`{op:?}` with {} operands; `PureInst::coll` checks arity",
                        args.len()
                    ),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SET_OPS: [SetOp; 9] = [
        SetOp::Card,
        SetOp::Contains,
        SetOp::Union,
        SetOp::Intersection,
        SetOp::Difference,
        SetOp::Subset,
        SetOp::Add,
        SetOp::Empty,
        SetOp::Eq,
    ];

    /// Every set operation's operand types agree with its arity, it works on
    /// a set, and its token is its own.
    #[test]
    fn set_op_tables_agree() {
        let mut tokens = std::collections::HashSet::new();
        for op in SET_OPS {
            let c = CollOp::from(op);
            assert_eq!(c.kind(), CollKind::Set);
            assert_eq!(c.operand_types(&[Type::Int]).len(), c.arity(), "{op:?}");
            assert!(!c.has_side_condition(), "{op:?}");
            assert!(tokens.insert(c.token()), "{op:?} reuses a token");
        }
        assert_eq!(CollOp::size(CollKind::Set), SetOp::Card.into());
        assert_eq!(CollOp::contains(CollKind::Set), SetOp::Contains.into());
        assert_eq!(CollOp::empty(CollKind::Set), SetOp::Empty.into());
        assert!(CollOp::eq(CollKind::Set).is_eq() && CollOp::eq(CollKind::Seq).is_eq());
        assert!(!CollOp::from(SetOp::Subset).is_eq());
    }

    #[test]
    #[should_panic(expected = "takes 2 operands")]
    fn coll_checks_set_arity() {
        PureInst::coll([Type::Int], SetOp::Union, vec![Val::Temp(0)]);
    }

    const MAP_OPS: [MapOp; 7] = [
        MapOp::Card,
        MapOp::Lookup,
        MapOp::Update,
        MapOp::Domain,
        MapOp::Range,
        MapOp::Empty,
        MapOp::Eq,
    ];

    /// Every map operation's operand types agree with its arity, it works on a
    /// map, and its token is its own; only a lookup has a side condition.
    #[test]
    fn map_op_tables_agree() {
        let tys = [Type::Ref, Type::Bool];
        let mut tokens = std::collections::HashSet::new();
        for op in MAP_OPS {
            let c = CollOp::from(op);
            assert_eq!(c.kind(), CollKind::Map);
            assert_eq!(c.operand_types(&tys).len(), c.arity(), "{op:?}");
            assert_eq!(c.has_side_condition(), op == MapOp::Lookup, "{op:?}");
            assert!(tokens.insert(c.token()), "{op:?} reuses a token");
        }
        assert_eq!(CollOp::size(CollKind::Map), MapOp::Card.into());
        assert_eq!(CollOp::empty(CollKind::Map), MapOp::Empty.into());
        assert!(CollOp::eq(CollKind::Map).is_eq());
        assert!(!CollKind::Map.has_unit());
    }

    /// A lookup yields a value, `domain` a set of keys, `range` a set of
    /// values, and an update takes a key and a value in that order.
    #[test]
    fn map_op_types() {
        let tys = [Type::Ref, Type::Bool];
        let c = |op: MapOp| CollOp::from(op);
        assert_eq!(c(MapOp::Lookup).result_type(&tys), Type::Bool);
        assert_eq!(c(MapOp::Domain).result_type(&tys), Type::set(Type::Ref));
        assert_eq!(c(MapOp::Range).result_type(&tys), Type::set(Type::Bool));
        assert_eq!(
            c(MapOp::Update).operand_types(&tys),
            [Type::map(Type::Ref, Type::Bool), Type::Ref, Type::Bool]
        );
        assert_eq!(c(MapOp::Card).result_type(&tys), Type::Int);
    }

    #[test]
    #[should_panic(expected = "takes 2 type arguments")]
    fn coll_checks_map_type_arity() {
        PureInst::coll([Type::Int], MapOp::Card, vec![Val::Temp(0)]);
    }

    #[test]
    #[should_panic(expected = "has no one-element literal")]
    fn map_has_no_unit() {
        PureInst::coll_lit(CollKind::Map, Type::Int, Val::Temp(0));
    }
}
