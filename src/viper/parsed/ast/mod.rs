use lasso::Spur;

mod exp;
mod stmt;

pub use exp::*;
pub use stmt::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Program(pub Vec<Declaration>);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ident {
    Raw(String),
    Interned(Spur),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IdnDecl(pub Ident);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Declaration {
    Import(Import),
    Define(Define),
    Domain(Domain),
    DomainElement(DomainElement),
    Field(Field),
    Function(Function),
    Predicate(Predicate),
    Method(Method),
    Adt(Adt),
    AdtConstructor(AdtConstructor),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DomainElement {
    pub domain: Ident,
    pub kind: DomainElementKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DomainElementKind {
    Function(DomainFunction),
    Axiom(Axiom),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Axiom {
    pub name: Option<IdnDecl>,
    pub exp: ExpBlock,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Import {
    pub path: String,
    pub local: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Define {
    pub name: IdnDecl,
    pub args: Vec<IdnDecl>,
    pub body: ExpOrBlock,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExpOrBlock {
    Exp(Exp),
    Block(StmtBlock),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IdnDeclTyped {
    pub idn: IdnDecl,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArgOrType {
    Arg(IdnDeclTyped),
    Type(Type),
}

impl ArgOrType {
    pub fn idn(&self) -> Option<&IdnDecl> {
        match self {
            ArgOrType::Arg(id) => Some(&id.idn),
            ArgOrType::Type(_) => None,
        }
    }

    pub fn ty(&self) -> &Type {
        match self {
            ArgOrType::Arg(id) => &id.ty,
            ArgOrType::Type(ty) => ty,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Block<T>(pub T);

pub type ExpBlock = Block<Exp>;
pub type StmtBlock = Block<Vec<Statement>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Field(pub IdnDeclTyped);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Domain {
    pub name: IdnDecl,
    pub params: Vec<IdnDecl>,
    pub interpretation: Vec<(Ident, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Function {
    pub signature: Signature,
    pub contract: Contract,
    pub body: Option<ExpBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Contract {
    pub precondition: Vec<Exp>,
    pub postcondition: Vec<Exp>,
    pub decreases: Vec<Decreases>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DomainFunction {
    pub unique: bool,
    pub signature: Signature,
    pub interpretation: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Signature {
    pub name: IdnDecl,
    pub args: Vec<ArgOrType>,
    pub ret: Vec<ArgOrType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Bool,
    Int,
    Real,
    Ref,
    Generic(Ident),
    Domain(Ident, Vec<Type>),
    Seq(Box<Type>),
    Set(Box<Type>),
    /// `Map[K, V]`, key type first.
    Map(Box<Type>, Box<Type>),
    Multiset(Box<Type>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Predicate {
    pub signature: Signature,
    pub body: Option<ExpBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Method {
    pub signature: Signature,
    pub contract: Contract,
    pub body: Option<StmtBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Adt {
    pub name: IdnDecl,
    pub params: Vec<IdnDecl>,
    pub variants: Vec<Variant>,
    pub derives: Vec<String>,
}

/// Duplicated data with `AdtConstructor`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variant {
    pub name: IdnDecl,
    pub fields: Vec<ArgOrType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdtConstructor {
    pub signature: Signature,
}
