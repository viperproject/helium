use crate::vmir::display::VmirDisplay;
use crate::vmir::{FunctionBody, TyParams};
use std::fmt::{self, Display, Formatter};

use lasso::Spur;

/// A domain — a namespace of uninterpreted functions, generic over `ty_params`
/// types. Its functions and axioms are free-standing declarations: a function is
/// generic over the domain parameters its signature mentions (renumbered, see
/// [`Function::ty_params`](crate::vmir::Function::ty_params)); an axiom binds
/// the ones it mentions itself (see [`Axiom`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Domain {
    pub name: Spur,
    pub ty_params: TyParams,
}

/// An axiom: a closed boolean fact the verifier **assumes** in every
/// verification unit. In VMIR an axiom is a free-standing declaration, not bound
/// to a domain (Silver domains only supply the source syntax). The body is a
/// pure, heap-free inst stream (`Pure` + the `Assume`s stitched from a callee's
/// `#ensures`; no params, so `Val::Temp` counts from 0) whose `res` is the
/// axiom's boolean — merged with `true` before verification. Axiom bodies are
/// **never verified**: no well-definedness obligations (div-by-zero etc.) are
/// checked on them. Quantification over *values* is an inline
/// [`Forall`](crate::vmir::Forall) step in the body.
///
/// A **generic** axiom quantifies over its types explicitly: its body is a
/// single `forall` step whose [`ty_binders`](crate::vmir::Forall::ty_binders)
/// bind every `Generic(i)` the axiom mentions, and whose triggers fix them —
/// either the axiom's own top-level `forall` (when each of its trigger groups
/// fixes every type), or a wrapper binding only types, with inferred triggers
/// (`translate::decl::domain`). Outside that root the body is ground.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Axiom {
    pub name: Option<Spur>,
    pub body: FunctionBody,
}

impl<'a> Display for VmirDisplay<'a, &'a Domain> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let name = self.interner.resolve(&self.item.name);
        writeln!(f, "domain {name}{}", self.item.ty_params)
    }
}

impl<'a> Display for VmirDisplay<'a, &'a Axiom> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "axiom")?;
        if let Some(n) = &self.item.name {
            write!(f, " {}", self.interner.resolve(n))?;
        }
        writeln!(f, " {{")?;
        write!(
            f,
            "{}",
            self.with((0usize, 0usize, 0usize, &self.item.body.insts[..]))
        )?;
        writeln!(f, "  result: {}", self.item.body.res)?;
        write!(f, "}}")
    }
}
