use crate::vmir::display::VmirDisplay;
use crate::vmir::{FunctionBody, TyParams};
use std::fmt::{self, Display, Formatter};

use lasso::Spur;

/// A domain — a namespace of uninterpreted functions, generic over `ty_params`
/// types. Its functions and axioms are free-standing declarations: a function is
/// generic over the domain parameters its signature mentions (renumbered, see
/// [`Function::ty_params`](crate::vmir::Function::ty_params)), an axiom over all
/// of them (the domain's own numbering).
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
/// A **generic** axiom (`ty_params > 0`, the parameters of the domain it came
/// from, as `Generic(i)`) holds at every instantiation. There is no type
/// quantifier: every `Generic` in the body occurs inside a `forall`, and the
/// types are fixed when that `forall` is instantiated, by the type arguments of
/// the trigger terms it matched (or by an enclosing instantiation). Outside any
/// `forall` the body is ground (translation-enforced).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Axiom {
    pub name: Option<Spur>,
    pub ty_params: TyParams,
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
        writeln!(f, "{} {{", self.item.ty_params)?;
        write!(
            f,
            "{}",
            self.with((0usize, 0usize, 0usize, &self.item.body.insts[..]))
        )?;
        writeln!(f, "  result: {}", self.item.body.res)?;
        write!(f, "}}")
    }
}
