use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PrePostDec {
    Pre(Exp),
    Post(Exp),
    Decreases(Decreases),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Decreases {
    pub kind: Option<DecreasesKind>,
    pub guard: Option<Exp>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DecreasesKind {
    Star,
    Underscore,
    Exp(Vec<Exp>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InferenceType {
    Unknown,
    Infer(rusttyc::TcKey),
    Computed(Type),
    Impure,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Exp {
    pub ty: InferenceType,
    pub kind: Box<ExpKind>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExpKind {
    Const(ConstKind),
    /// x
    Ident(Ident),
    /// result keyword
    Result,
    // old(e) or old[label](e)
    Old(Option<Ident>, Exp),
    // e : Type
    Ascribe(Exp, Type),

    /// op e
    UnOp(UnOp, Exp),
    /// e1 op e2
    BinOp(BinOp, Exp, Exp),
    /// c ? e1 : e2
    Ternary(Exp, Exp, Exp),
    /// let x == (e1) in e2
    LetIn(IdnDecl, Exp, Exp),
    /// e[e1]
    Index(Exp, IndexOp),

    /// acc(e1, 1/1)
    Acc(AccExp),
    /// e.f
    Field(Exp, Ident),

    /// Application of a function, predicate, macro, or adt constructor.
    /// ident(e1, ..., en)
    Call(Call<ExpCallKind>),

    /// adt.member
    AdtDestructor(Exp, Ident),
    /// adt.isCons
    AdtDiscriminator(Exp, Ident),

    /// unfolding(e) in E, and similarly for folding, applying, and packaging.
    HeapUpdate(HeapUpdateOp, AccExp, Exp),
    /// forall/exists x: T, y: U, ... :: { trigger } e
    Quantifier(QuantifierKind, Vec<IdnDeclTyped>, Vec<Trigger>, Exp),
    /// Quantified permissions. forperm x: T, y: U, ... [Perm] :: e1
    ForPerm(Vec<IdnDeclTyped>, ResAccess, Exp),
    /// e1 --* e2
    MagicWand(Exp, Exp),

    /// A sequence literal: `Seq[T]()`, `Seq()`, `Seq(e1, ..., en)` or
    /// `Seq[T](e1, ..., en)`. The element type is `Some` only when written;
    /// an empty literal without one gets its type from inference.
    SeqLit(Option<Type>, Vec<Exp>),
    /// A set literal: `Set[T]()`, `Set()`, `Set(e1, ..., en)` or
    /// `Set[T](e1, ..., en)`, element type as for [`ExpKind::SeqLit`].
    SetLit(Option<Type>, Vec<Exp>),

    /// A construct the verifier does not implement, parsed only so that the
    /// declaration containing it can be reported as unsupported instead of
    /// taking the whole file down with a parse error. Carries the construct's
    /// name (`Set`, `lhs`, `folding`, ...). Never lowered: typecheck turns it
    /// into `TypeError::Unsupported`.
    Unsupported(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExpCallKind {
    Function,
    DomainFunction,
    Predicate,
    AdtConstructor,
    Macro,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StmtCallKind {
    Method,
    Macro,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Call<KnownCallKind> {
    pub kind: Option<KnownCallKind>,
    pub name: Ident,
    pub args: Vec<Exp>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstKind {
    Bool(bool),
    Int(num::BigInt),
    Real(num::BigRational),
    Null,
    Epsilon,
    Wildcard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeapUpdateOp {
    Unfold,
    Fold,
    Apply,
    Package,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantifierKind {
    Forall,
    Exists,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AccExp {
    /// Either a field or a predicate call
    pub loc: Exp,
    /// The permission amount
    pub perm: Exp,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BinOp {
    Implies,
    Or,
    And,
    Iff,
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    Plus,
    Minus,
    Mult,
    Div,
    /// Integer division (`\`). Silver reserves `/` for permission division, so
    /// `Int` division has its own token.
    IntDiv,
    Mod,
    Union,
    SetMinus,
    Intersection,
    Subset,
    Concat,
    Range,
    InhaleExhale,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnOp {
    Not,
    Neg,
    Perm,
    /// `|e|`: the length of a sequence (the cardinality of a collection).
    Cardinality,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Trigger {
    pub exp: Vec<Exp>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResAccess {
    Loc(Exp),
    Exp(AccExp),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexOp {
    Index(Exp),
    LowerBound(Exp),
    UpperBound(Exp),
    Range(Exp, Exp),
    Assign(Exp, Exp),
}
