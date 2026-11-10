use crate::vmir::{FunctionBody, TyParams};
use crate::vmir::display::VmirDisplay;
use std::fmt::{self, Display, Formatter};

use lasso::Spur;

/// A domain — a namespace of uninterpreted functions, possibly generic
/// (`ty_params` is its declared arity). Its functions and axioms are separate
/// declarations, each generic over only the domain parameters it mentions.
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
/// [`Forall`](crate::vmir::Forall) step in the body; there is no quantification
/// over types. An axiom is generic over `ty_params` types (`Generic(0..n)`),
/// only those its body mentions: it holds at every instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Axiom {
    pub name: Option<Spur>,
    pub body: FunctionBody,
    pub ty_params: TyParams,
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
        write!(f, "{}", self.item.ty_params)?;
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
