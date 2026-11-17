//! The read-over-write axiom of a quantified chunk set's value map (see
//! `FuncRegistry::value_map`), built with the collection axioms' term language:
//!
//! ```text
//! forall m, a, v, r :: {lookup(store(m, a, v), r)}
//!     lookup(store(m, a, v), r) == (r == a ? v : lookup(m, r))
//! ```
//!
//! One unconditional equation: the e-graph's `ite` rule picks the branch once
//! `r == a` is decided either way. Ground per location kind (the map, address
//! and value types are fixed by it), so the recipe has no type parameters.

use super::{STerm, Skolems, Steps, eq, ite, trigger, v};
use crate::verify::func_registry::ValueFunction;
use crate::verify::heap::LocationKind;
use crate::verify::quant::QuantRecipe;
use crate::vmir::{CollKind, Type};

/// The read-over-write axiom of `kind`'s value map, its functions `f`.
pub(crate) fn recipe(kind: &LocationKind, f: ValueFunction) -> QuantRecipe {
    let addr = Type::Addr {
        group: kind.group,
        value: Box::new(kind.value.clone()),
        bound: kind.bound.clone(),
    };
    let map = Type::ValueMap(Box::new(addr.clone()));
    // Binders: m, a, v, r.
    let bound: Box<[Type]> = Box::new([map, addr.clone(), kind.value.clone(), addr]);
    let (m, a, val, r) = (v(0), v(1), v(2), v(3));
    let lookup = |m, r| STerm::App(f.lookup, vec![m, r]);
    let stored = STerm::App(f.store, vec![m.clone(), a.clone(), val.clone()]);
    let read = lookup(stored, r.clone());
    let body = eq(read.clone(), ite(eq(r.clone(), a), val, lookup(m, r)));

    // The terms are all `App`s, `Bin`s and `Ite`s, so the collection kind,
    // element type and witnesses the builder carries are never consulted.
    let unused = CollKind::Seq;
    let skolems = Skolems { funcs: vec![] };
    let mut steps = Steps {
        next: bound.len(),
        insts: Vec::new(),
        kind: unused,
        tys: Box::new([kind.value.clone()]),
        skolems: &skolems,
    };
    let res = steps.emit(&body);
    QuantRecipe {
        n_caps: 0,
        bound,
        n_tys: 0,
        own_tys: vec![],
        groups: vec![vec![trigger(&read, unused, std::slice::from_ref(&kind.value))]],
        insts: steps.insts,
        res,
        guard: None,
    }
}
