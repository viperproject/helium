use lasso::Spur;

use crate::viper::Interner;

#[derive(Debug, PartialEq)]
pub struct Program {
    pub decls: Vec<Declaration>,
    /// The symbol alphabet every `Spur` in this program resolves through.
    pub interner: Interner,
}

/// A resolved identifier string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ident(pub Spur);

/// An identifier bundled with its explicit type (e.g., `x: Int`).
#[derive(Debug, Clone, PartialEq)]
pub struct TypedIdent {
    pub name: Ident,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BuiltinCollection {
    Seq(Box<Type>),
    Set(Box<Type>),
    MultiSet(Box<Type>),
    Map(Box<Type>, Box<Type>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Bool,
    Int,
    Real,
    Ref,
    Generic(Ident),
    Collection(BuiltinCollection),
    Domain(Ident, Vec<Type>),
}

impl From<&crate::viper::parsed::ast::Type> for Type {
    fn from(t: &crate::viper::parsed::ast::Type) -> Self {
        use crate::viper::parsed::ast::Type as A;
        match t {
            A::Bool => Type::Bool,
            A::Int => Type::Int,
            A::Real => Type::Real,
            A::Ref => Type::Ref,
            A::Generic(id) => Type::Generic(Ident(id.id())),
            A::Domain(id, args) => {
                Type::Domain(Ident(id.id()), args.iter().map(Self::from).collect())
            }
            A::Seq(ty) => Type::Collection(BuiltinCollection::Seq(Box::new(Self::from(&**ty)))),        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnOp {
    Not,
    Neg,
    Cardinality,
}
#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Or,
    And,
    Implies,
    Iff,
    Eq,
    Neq,

    Lt,
    Le,
    Gt,
    Ge,

    Plus,
    Minus,
    Mult,
    Div,
    /// Integer division (`\`) — `Int` operands only, unlike `Div`.
    IntDiv,
    Mod,

    In,
    Union,
    SetMinus,
    Intersection,
    Subset,
    Concat,
    Range,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Bool(bool),
    Int(num::BigInt),
    Real(num::BigRational),
    Null,
    Wildcard,
}

/// A pure expression bundled with its synthesized type.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedPureExp<Ext> {
    pub ty: Type,
    pub exp: Box<PureExpKind<Ext>>,
}

/// A generic function or predicate call.
#[derive(Debug, Clone, PartialEq)]
pub struct Call<Ext> {
    pub name: Ident,
    pub args: Vec<TypedPureExp<Ext>>,
}

/// The **heap-free** core expression variants — exactly those legal in a domain
/// axiom. Heap-dependent constructs (`e.f`, `unfolding`, calls to Silver
/// `function`s, `old`/`perm`/`result`) are supplied per context through `Ext`
/// (see [`HeapNode`]), so `PureExpKind<!>` is provably heap-free.
#[derive(Debug, Clone, PartialEq)]
pub enum PureExpKind<Ext> {
    Ident(Ident),
    Const(Literal),
    Unary(UnOp, TypedPureExp<Ext>),
    Binary(BinOp, TypedPureExp<Ext>, TypedPureExp<Ext>),
    Ternary {
        if_: TypedPureExp<Ext>,
        then: TypedPureExp<Ext>,
        else_: TypedPureExp<Ext>,
    },
    LetIn {
        binder: Ident,
        value: TypedPureExp<Ext>,
        exp: TypedPureExp<Ext>,
    },
    /// Call to a domain function (pure).
    DomainFunctionCall(Call<Ext>),
    /// Constructor call of an ADT variant.
    AdtConstructor(Call<Ext>),
    /// A projection from an ADT.
    AdtDestructor(TypedPureExp<Ext>, Ident),
    /// A variant check on an ADT (e.g., `e.isCons(list)`).
    AdtDiscriminator(TypedPureExp<Ext>, Ident),
    /// `Seq(e1, ..., en)`; empty for `Seq[T]()`. The element type is the
    /// node's own `Seq[T]`.
    SeqLit(Vec<TypedPureExp<Ext>>),
    /// `s[..]`: an index, a take, a drop, a slice, or an update of `s`.
    SeqIndex(TypedPureExp<Ext>, SeqIndexOp<Ext>),
    /// The context-specific extension — heap nodes (`Ext::Heap`), `old`, `perm`,
    /// `result`. Uninhabited (`!`) in a pure context, so none are constructible.
    Ext(Ext),
}

/// What `s[..]` does to the sequence `s`.
#[derive(Debug, Clone, PartialEq)]
pub enum SeqIndexOp<Ext> {
    /// `s[i]`
    Index(TypedPureExp<Ext>),
    /// `s[..n]`
    Take(TypedPureExp<Ext>),
    /// `s[n..]`
    Drop(TypedPureExp<Ext>),
    /// `s[a..b]`, i.e. `s[..b][a..]`
    Slice(TypedPureExp<Ext>, TypedPureExp<Ext>),
    /// `s[i := v]`
    Update(TypedPureExp<Ext>, TypedPureExp<Ext>),
}

/// The heap-reading expression constructs, shared by every heap-bearing context
/// (`Ext`): a field dereference, a Silver `function` call, and `unfolding`. A
/// pure context omits these by construction (its `Ext` has no `Heap` variant).
#[derive(Debug, Clone, PartialEq)]
pub enum HeapNode<Ext> {
    /// Heap field access: `e.f` where `f` is a Silver `field` declaration.
    Field(TypedPureExp<Ext>, Ident),
    /// Call to a (heap-dependent) Silver `function`.
    FunctionCall(Call<Ext>),
    /// Evaluates the inner expression under a temporary unfolding of the predicate.
    Unfolding(PredicateWithPerm<Ext>, TypedPureExp<Ext>),
}

/// An expression that asserts or transfers heap resources.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialExp<PureExt>(pub Box<SpatialExpKind<PureExt>>);

#[derive(Debug, Clone, PartialEq)]
pub enum SpatialExpKind<PureExt> {
    Implies(TypedPureExp<PureExt>, SpatialExp<PureExt>),
    Conj(SpatialExp<PureExt>, SpatialExp<PureExt>),
    Ternary {
        if_: TypedPureExp<PureExt>,
        then: SpatialExp<PureExt>,
        else_: SpatialExp<PureExt>,
    },
    /// Represents `acc(resource, perm)`.
    Acc(ResourceExp<PureExt>, TypedPureExp<PureExt>),
    /// Lifts a pure boolean expression into the spatial context.
    Pure(TypedPureExp<PureExt>),
}

/// The valid targets for acc or perm queries.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceExp<PureExt>(pub Box<ResourceExpKind<PureExt>>);

#[derive(Debug, Clone, PartialEq)]
pub enum ResourceExpKind<PureExt> {
    /// A mutable heap location: `e.f`
    Field(TypedPureExp<PureExt>, Ident),
    /// A spatial predicate: `P(args)`
    PredicateCall(Call<PureExt>),
}

/// A predicate call bundled with an explicit permission amount.
#[derive(Debug, Clone, PartialEq)]
pub struct PredicateWithPerm<PureExt> {
    pub pred_call: Call<PureExt>,
    pub perm: TypedPureExp<PureExt>,
}

/// Heap access with **no** state extension: predicate bodies, function
/// preconditions and bodies, method preconditions. (`old`/`perm`/`result` are
/// not available here.)
#[derive(Debug, Clone, PartialEq)]
pub enum HeapExt {
    Heap(HeapNode<HeapExt>),
    /// A pure `forall` quantifier. Its innards are always [`AxiomExt`]-typed:
    /// quantifier bodies are pure and heap-free regardless of host position.
    Forall(Box<Forall>),
}

/// The extensions allowed in a domain axiom: a call to a Silver `function`
/// (Viper's only restriction on such calls is that the callee has **no
/// precondition**, which also makes it heap-free), and a pure `forall`
/// quantifier. Field access, `unfolding`, `old`, `perm`, and `result` remain
/// illegal.
///
/// The no-precondition rule is enforced during **typechecking**, by
/// `check_axiom_function_calls` (`viper::typecheck`, raising
/// `TypeError::PreconditionedFunctionInAxiom`) — not after lowering. That check
/// runs *only* over domain axioms, and it bars **any** precondition, heap-free
/// ones included. A `forall` hosted in a contract, a statement, or a predicate
/// body is `AxiomExt`-typed too but never reaches it; there the only gate on a
/// heap-dependent callee is at lowering (see `translate::pure_exp`'s
/// `in_quantifier` rejection).
#[derive(Debug, Clone, PartialEq)]
pub enum AxiomExt {
    FunctionCall(Call<AxiomExt>),
    Forall(Box<Forall>),
}

/// A pure universal quantifier: `forall x: T, ... :: { trig } body`. The `body`
/// is boolean; `triggers` is a disjunction of trigger *groups*, each a
/// conjunction of trigger terms (Silver's `{ .. }{ .. }` syntax). Only
/// `forall` reaches here — `exists` is rejected at typechecking.
///
/// Triggers and body are [`AxiomExt`]-typed in **every** host position
/// (axioms, contracts, predicate bodies, method statements): quantifier
/// bodies are pure and heap-free, so heap derefs, `unfolding`, `old`,
/// `result`, and `perm` inside a `forall` are rejected by `AxiomExt`'s rules.
/// Free variables of the enclosing scope are permitted — translation turns
/// them into capture parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Forall {
    pub bound: Vec<TypedIdent>,
    pub triggers: Vec<Vec<TypedPureExp<AxiomExt>>>,
    pub body: TypedPureExp<AxiomExt>,
}

/// Extensions allowed in function postconditions: heap access + `result`.
#[derive(Debug, Clone, PartialEq)]
pub enum FuncEnsuresExt {
    Heap(HeapNode<FuncEnsuresExt>),
    Result,
    Old(TypedPureExp<FuncEnsuresExt>),
    /// A pure `forall` quantifier (innards [`AxiomExt`]-typed; see [`Forall`]).
    Forall(Box<Forall>),
}

/// Extensions allowed in method postconditions: heap access + `old`.
#[derive(Debug, Clone, PartialEq)]
pub enum MethodEnsuresExt {
    Heap(HeapNode<MethodEnsuresExt>),
    Old(TypedPureExp<MethodEnsuresExt>),
    /// A pure `forall` quantifier (innards [`AxiomExt`]-typed; see [`Forall`]).
    Forall(Box<Forall>),
}

/// Extensions allowed in imperative method bodies: heap access + labelled `old`
/// + `perm`.
#[derive(Debug, Clone, PartialEq)]
pub enum MethodBodyExt {
    Heap(HeapNode<MethodBodyExt>),
    Old(Option<Spur>, TypedPureExp<MethodBodyExt>),
    Perm(ResourceExp<MethodBodyExt>),
    /// A pure `forall` quantifier (innards [`AxiomExt`]-typed; see [`Forall`]).
    Forall(Box<Forall>),
}

pub type PureMethodExp = TypedPureExp<MethodBodyExt>;
pub type SpatialMethodExp = SpatialExp<MethodBodyExt>;

#[derive(Debug, Clone, PartialEq)]
pub struct StmtBlock(pub Vec<Statement>);

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Assume(SpatialMethodExp),
    Assert(SpatialMethodExp),
    /// `refute A`: succeeds iff `A` is **not** provable in this state.
    Refute(SpatialMethodExp),
    Inhale(SpatialMethodExp),
    Exhale(SpatialMethodExp),
    If(PureMethodExp, StmtBlock, Option<StmtBlock>),
    /// Variable declaration (e.g., `var x: Int := 5`)
    Var(Vec<TypedIdent>, Option<AssignRhs>),
    /// Imperative assignment (e.g., `x, y.f := 1, 2`)
    Assign(Vec<AssignLhs>, AssignRhs),
    Block(StmtBlock),
    Fold(PredicateWithPerm<MethodBodyExt>),
    Unfold(PredicateWithPerm<MethodBodyExt>),
    /// Label marking a heap state for `old[L](...)` to refer back to; also a
    /// jump target for `goto`.
    ///
    /// A label may carry loop invariants (`label L invariant A invariant B`) —
    /// Silver's `Label(name, invs)`. They are the invariants of the loop whose
    /// head this label is, and every `goto L` reaching it must re-establish
    /// them. This is how Prusti transmits loop invariants, since it emits
    /// goto-CFGs with no `while` at all.
    Label(Spur, Vec<SpatialMethodExp>),
    /// `goto L`: unconditional jump to the block labelled `L`.
    Goto(Spur),
    /// `while (c) invariant A { .. }`.
    ///
    /// Kept structured all the way to CFG construction: a `while` head needs no
    /// name, so an earlier `label`+`goto` rewrite would mint a synthetic identifier
    /// for nothing and lose the source shape diagnostics want. `viper::cfg` turns it
    /// into the same head/body/back-edge structure a hand-written `goto` loop
    /// produces, so there is one loop shape downstream of the CFG.
    ///
    /// `decreases` clauses are dropped by typechecking.
    // TODO(loops): termination — `decreases` is parsed and ignored.
    While(PureMethodExp, Vec<SpatialMethodExp>, StmtBlock),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignLhs {
    /// Field assignment to `e.f` where `e` is of type `Ref`.
    Field(PureMethodExp, Ident),
    /// Local variable assignment.
    Var(Ident),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignRhs {
    /// E.g., `new(*)` or `new(f1, f2)`
    New(StarOrFields),
    MethodCall(Call<MethodBodyExt>),
    Exp(PureMethodExp),
}

#[derive(Debug, Clone, PartialEq)]
pub enum StarOrFields {
    Star,
    Fields(Vec<Ident>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Function(Function),
    Predicate(Predicate),
    Method(Method),
    Field(Field),
    Adt(Adt),
    Domain(Domain),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Domain {
    pub name: Ident,
    pub type_params: Vec<Ident>,
    pub functions: Vec<DomainFunction>,
    pub axioms: Vec<Axiom>,
}

/// A ground (quantifier-free) domain axiom: a closed boolean expression over
/// domain functions, ADT operations, and precondition-free Silver `function`s.
/// Implicitly generic over any of the owning domain's type parameters it
/// mentions (an unconstrained instantiation defaults to the parameter itself,
/// mirroring Silver's `ground()` rule).
#[derive(Debug, Clone, PartialEq)]
pub struct Axiom {
    pub name: Option<Ident>,
    pub exp: TypedPureExp<AxiomExt>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Adt {
    pub name: Ident,
    pub type_params: Vec<Ident>,
    pub variants: Vec<AdtVariant>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdtVariant {
    pub name: Ident,
    pub params: Vec<TypedIdent>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field(pub TypedIdent);

#[derive(Debug, Clone, PartialEq)]
pub struct DomainFunction {
    pub name: Ident,
    pub params: Vec<TypedIdent>,
    pub ret: Type,
}

/// A purely mathematical function that cannot mutate state.
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: Ident,
    pub params: Vec<TypedIdent>,
    pub ret: Type,
    pub requires: Option<SpatialExp<HeapExt>>,
    pub ensures: Option<TypedPureExp<FuncEnsuresExt>>,
    pub body: Option<TypedPureExp<HeapExt>>,
}

/// An imperative sub-routine that can mutate the heap.
#[derive(Debug, Clone, PartialEq)]
pub struct Method {
    pub name: Ident,
    pub params: Vec<TypedIdent>,
    pub rets: Vec<TypedIdent>,
    pub requires: Option<SpatialExp<HeapExt>>,
    pub ensures: Option<SpatialExp<MethodEnsuresExt>>,
    pub body: Option<StmtBlock>,
}

/// A spatial macro representing a fraction of the heap.
#[derive(Debug, Clone, PartialEq)]
pub struct Predicate {
    pub name: Ident,
    pub params: Vec<TypedIdent>,
    pub body: Option<SpatialExp<HeapExt>>,
}

impl<Ext> SeqIndexOp<Ext> {
    /// The operands besides the sequence itself, in source order.
    pub fn operands(&self) -> Vec<&TypedPureExp<Ext>> {
        match self {
            SeqIndexOp::Index(e) | SeqIndexOp::Take(e) | SeqIndexOp::Drop(e) => vec![e],
            SeqIndexOp::Slice(a, b) | SeqIndexOp::Update(a, b) => vec![a, b],
        }
    }
}
