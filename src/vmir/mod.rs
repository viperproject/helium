mod adt;
mod analyze;
mod domain;
mod function;
mod heap;
mod inst;
mod method;
mod pure;
mod quant;
mod resource;
mod ty;

pub mod display;

pub use ty::Type;

pub use heap::{Assign, Bind, HeapInst, HeapVal, PermInst, PermVal};
pub use pure::{
    BinOp, CollKind, CollOp, FALSE, Literal, MapOp, NULL, PureInst, SeqOp, SetOp, TRUE, Val, none,
    write,
};

pub use adt::TyParams;
pub use adt::{Adt, AdtVariant};
pub use analyze::{AnalysisError, AnalyzedProgram, DepGraph, analyze};
pub use domain::{Axiom, Domain};
pub use function::{
    Args, ContractArg, ContractCall, Function, FunctionBody, FunctionCall, Params, Requires,
};
pub use inst::{Inst, InstKind, PathConds, Polarity};
pub use method::{Block, BlockId, Method, Preds};
pub use quant::{Forall, QuantTrigger, TrigHead, TrigTerm};
pub use resource::{Precond, Resource, ResourceBody, ResourceCall};
pub use ty::Bound;

use derive_more::{From, Into};
use lasso::{Rodeo, Spur};
use typed_index_collections::TiVec;

/// A dense index into [`Program::decls`]. Purely positional — it is **not** an
/// interner key; names are resolved through [`Program::name`].
#[derive(Debug, From, Into, Eq, PartialEq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct MemberId(pub usize);

#[derive(Debug, Clone)]
pub struct Program {
    pub decls: TiVec<MemberId, Declaration>,
    /// Cheap string repr for member (and constructor) names. Its `Spur` keys are
    /// independent of `MemberId`.
    pub interner: Rodeo,
    /// Location **group** tags (`Type::Addr.group`), interned separately — a group
    /// is just a name, never a declaration.
    pub groups: Rodeo<Spur>,
}

impl Program {
    /// The display name of a member.
    pub fn name(&self, id: MemberId) -> &str {
        self.interner.resolve(&self.decls[id].name())
    }

    /// The member with the given name, if any. A linear scan — for tests/debug
    /// only; VMIR proper never looks a member up by string.
    pub fn id(&self, name: &str) -> Option<MemberId> {
        let s = self.interner.get(name)?;
        self.decls
            .iter_enumerated()
            .find(|(_, d)| d.name() == s)
            .map(|(id, _)| id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Declaration {
    Domain(Domain),
    Axiom(Axiom),
    Function(Function),
    Method(Method),
    Resource(Resource),
    Adt(Adt),
}

impl Declaration {
    pub fn name(&self) -> Spur {
        match self {
            Self::Domain(d) => d.name,
            Self::Axiom(a) => a.name.unwrap_or_default(),
            Self::Function(f) => f.name,
            Self::Method(m) => m.name,
            Self::Resource(r) => r.name,
            Self::Adt(a) => a.name,
        }
    }
}
