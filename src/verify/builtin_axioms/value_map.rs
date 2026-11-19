//! The axioms of a quantified chunk set's value map (see
//! `FuncRegistry::value_map`), built with the collection axioms' term language:
//!
//! ```text
//! forall m, a, v, r :: {lookup(store(m, a, v), r)}
//!     lookup(store(m, a, v), r) == (r == a ? v : lookup(m, r))
//! forall c, m1, m2, r :: {lookup(join(c, m1, m2), r)}
//!     lookup(join(c, m1, m2), r) == (c ? lookup(m1, r) : lookup(m2, r))
//! ```
//!
//! Read-over-write and read-over-join. Each is one unconditional equation: the
//! e-graph's `ite` rule picks the branch once its condition is decided either
//! way. Ground per location kind (the map, address and value types are fixed
//! by it), so the recipes have no type parameters.

use super::{STerm, Skolems, Steps, eq, ite, trigger, v};
use crate::verify::func_registry::ValueFunction;
use crate::verify::heap::LocationKind;
use crate::verify::quant::QuantRecipe;
use crate::vmir::{CollKind, Type};

/// The axioms of `kind`'s value map, its functions `f`.
pub(crate) fn recipes(kind: &LocationKind, f: ValueFunction) -> Vec<QuantRecipe> {
    let addr = Type::Addr {
        group: kind.group,
        value: Box::new(kind.value.clone()),
        bound: kind.bound.clone(),
    };
    let map = Type::ValueMap(Box::new(addr.clone()));
    let lookup = |m, r| STerm::App(f.lookup, vec![m, r]);

    // Read-over-write. Binders: m, a, v, r.
    let (m, a, val, r) = (v(0), v(1), v(2), v(3));
    let stored = STerm::App(f.store, vec![m.clone(), a.clone(), val.clone()]);
    let read = lookup(stored, r.clone());
    let body = eq(read.clone(), ite(eq(r.clone(), a), val, lookup(m, r)));
    let over_write = axiom(
        kind,
        Box::new([map.clone(), addr.clone(), kind.value.clone(), addr.clone()]),
        &read,
        &body,
    );

    // Read-over-join. Binders: c, m1, m2, r.
    let (c, m1, m2, r) = (v(0), v(1), v(2), v(3));
    let joined = STerm::App(f.join, vec![c.clone(), m1.clone(), m2.clone()]);
    let read = lookup(joined, r.clone());
    let body = eq(read.clone(), ite(c, lookup(m1, r.clone()), lookup(m2, r)));
    let over_join = axiom(kind, Box::new([Type::Bool, map.clone(), map, addr]), &read, &body);

    vec![over_write, over_join]
}

/// The recipe of `forall bound :: {read} body`.
fn axiom(kind: &LocationKind, bound: Box<[Type]>, read: &STerm, body: &STerm) -> QuantRecipe {
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
    let res = steps.emit(body);
    QuantRecipe {
        n_caps: 0,
        bound,
        n_tys: 0,
        own_tys: vec![],
        groups: vec![vec![trigger(read, unused, std::slice::from_ref(&kind.value))]],
        insts: steps.insts,
        res,
        guard: None,
    }
}
