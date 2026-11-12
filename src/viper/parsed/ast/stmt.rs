use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Statement {
    Assume(Exp),
    Assert(Exp),
    Refute(Exp),
    Inhale(Exp),
    Exhale(Exp),
    Fold(AccExp),
    Unfold(AccExp),
    Goto(Ident),
    Label(IdnDecl, Vec<Invariant>),
    Var(Vec<IdnDeclTyped>, Option<AssignRhs>),
    While(Exp, Vec<Invariant>, Vec<Decreases>, StmtBlock),
    If(Exp, StmtBlock, Option<StmtBlock>),
    /// `package A --* B { proof script }`
    Package(Exp, Option<StmtBlock>),
    /// `apply A --* B`
    Apply(Exp),
    Assign(Vec<AssignLhs>, AssignRhs),
    Block(StmtBlock),
    /// A statement the verifier does not implement (`quasihavoc`, `fresh`,
    /// ...), parsed only so the enclosing declaration can be
    /// reported as unsupported. See [`ExpKind::Unsupported`].
    Unsupported(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AssignLhs {
    Ident(Ident),
    Field(Exp, Ident),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AssignRhs {
    Exp(Exp),
    Call(Call<StmtCallKind>),
    New(StarOrNames),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StarOrNames {
    Star,
    Names(Vec<Ident>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Invariant(pub Exp);
