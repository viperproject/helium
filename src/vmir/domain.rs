use crate::vmir::display::VmirDisplay;
use crate::vmir::{FunctionBody, TyParams, Type, compaction};
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
/// [`Forall`](crate::vmir::Forall) step in the body. An axiom is generic over
/// `ty_params` types (`Generic(0..n)`): the type parameters of the domain it came
/// from, restricted to those its body mentions. It holds at every instance. There
/// is no type quantifier in the IR: the verifier instantiates the axiom at every
/// match of its `ty_triggers` among the types of a verification unit. Each trigger
/// is a conjunction of type patterns that must bind all of `ty_params`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Axiom {
    pub name: Option<Spur>,
    pub body: FunctionBody,
    pub ty_params: TyParams,
    pub ty_triggers: Vec<Vec<Type>>,
}

impl Axiom {
    /// The axiom over `body`, which mentions the `declared` parameters of its domain
    /// as `Generic(i)`. It is generic over the ones it mentions, renumbered `0..k`
    /// in order of first occurrence, with the type triggers inferred from `body`.
    /// Let `P` be the parameters mentioned, and `C` the non-bare types of the
    /// values and bound variables of `body`:
    /// 1. `P` empty: closed, no parameters and no trigger.
    /// 2. Some types of `C` mention all of `P`: each is a trigger of its own.
    /// 3. Otherwise one trigger: repeatedly the type of `C` that covers most
    ///    uncovered parameters, then a bare pattern per parameter still uncovered.
    pub fn infer(name: Option<Spur>, mut body: FunctionBody, declared: usize) -> Axiom {
        let mut order = Vec::new();
        body.for_each_type_mut(&mut |t| t.generics_in_order(&mut order));
        let mut c = Vec::new();
        body.value_types(&mut c);
        c.retain(|t| !matches!(t, Type::Generic(_)));
        c.sort_unstable();
        c.dedup();
        let gens: Vec<Vec<usize>> = c
            .iter()
            .map(|t| {
                let mut g = Vec::new();
                t.generics_in_order(&mut g);
                g
            })
            .collect();
        let mut triggers: Vec<Vec<Type>> = Vec::new();
        if !order.is_empty() {
            triggers.extend(
                (0..c.len())
                    .filter(|&k| gens[k].len() == order.len())
                    .map(|k| vec![c[k].clone()]),
            );
            if triggers.is_empty() {
                let mut uncovered = order.clone();
                let gain =
                    |k: usize, un: &[usize]| gens[k].iter().filter(|i| un.contains(i)).count();
                let mut group = Vec::new();
                while let Some(k) = (0..c.len())
                    .max_by_key(|&k| gain(k, &uncovered))
                    .filter(|&k| gain(k, &uncovered) > 0)
                {
                    group.push(c[k].clone());
                    uncovered.retain(|i| !gens[k].contains(i));
                }
                group.extend(uncovered.into_iter().map(Type::Generic));
                triggers.push(group);
            }
        }
        let compact = compaction(&order, declared);
        body.for_each_type_mut(&mut |t| *t = t.subst_generics(&compact));
        triggers
            .iter_mut()
            .flatten()
            .for_each(|t| *t = t.subst_generics(&compact));
        Axiom {
            name,
            body,
            ty_params: order.len().into(),
            ty_triggers: triggers,
        }
    }
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
        // Type-trigger groups, alternatives side by side, as a `forall` prints its
        // term triggers: `axiom ax<2> :: {Pair<?0, ?1>}{Other<?0>} {`. Within a
        // group the types are a conjunction, comma-separated.
        if !self.item.ty_triggers.is_empty() {
            write!(f, " ::")?;
            for (g, group) in self.item.ty_triggers.iter().enumerate() {
                write!(f, "{}{{", if g == 0 { " " } else { "" })?;
                for (k, ty) in group.iter().enumerate() {
                    if k > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", self.with(ty))?;
                }
                write!(f, "}}")?;
            }
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
