//! Type reconstruction for the **type-free** e-graph. `Symbolic` nodes carry no
//! type; when the visualizer or the certificate transplanter needs one, it is
//! recovered here from the node shape plus the irreducible `Fresh`/`FuncApp`
//! sources recorded in `VerifyContext`'s side-oracle maps.

use crate::dhash::HashMap;

use egg::{EGraph, Id};

use crate::verify::analysis::ConstFold;
use crate::verify::lang::{FuncId, Symbolic};
use crate::vmir::{BinOp, Literal, Type};

/// Reconstruct an e-class's type from the type-free e-graph (memoized,
/// cycle-safe, best-effort): `Lit`→literal type, `RealCast`→Real,
/// `Binary`→`Bool` for comparisons else operand type, `Ite`→branch type, and
/// the irreducible `Fresh`/`FuncApp` sources from the supplied side-oracle maps.
/// Shared by grafting and the visualization.
pub(crate) fn infer_type(
    egraph: &EGraph<Symbolic, ConstFold>,
    fresh_types: &HashMap<u32, Type>,
    func_ret_types: &HashMap<FuncId, Type>,
    id: Id,
    memo: &mut HashMap<Id, Option<Type>>,
) -> Option<Type> {
    let canon = egraph.find(id);
    if let Some(t) = memo.get(&canon) {
        return t.clone();
    }
    memo.insert(canon, None); // seed for cycles
    let nodes = egraph[canon].nodes.clone();
    let mut result = None;
    for node in &nodes {
        let t = match node {
            Symbolic::Lit(l) => Some(lit_type(l)),
            Symbolic::RealCast(_) => Some(Type::Real),
            Symbolic::Fresh(u) => fresh_types.get(u).cloned(),
            // A wildcard is always a permission (`Real`).
            // Addresses are ordinary func apps: a field/predicate address function
            // records its `Addr{..}` return type in `func_ret_types` like any other.
            Symbolic::FuncApp(f, _, _) => func_ret_types.get(f).cloned(),
            Symbolic::Binary(op, [l, _]) => match op {
                BinOp::Eq | BinOp::LtI | BinOp::LtR => Some(Type::Bool),
                _ => infer_type(egraph, fresh_types, func_ret_types, *l, memo),
            },
            Symbolic::Ite([_, then, _]) => {
                infer_type(egraph, fresh_types, func_ret_types, *then, memo)
            }
            // A quantifier is a proposition.
            Symbolic::Forall(..) => Some(Type::Bool),
        };
        if t.is_some() {
            result = t;
            break;
        }
    }
    memo.insert(canon, result.clone());
    result
}

/// Type of a literal value.
pub(crate) fn lit_type(lit: &Literal) -> Type {
    match lit {
        Literal::Null => Type::Ref,
        Literal::Bool(_) => Type::Bool,
        Literal::Int(_) => Type::Int,
        Literal::Real(_) => Type::Real,
    }
}
