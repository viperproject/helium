use crate::vmir::Literal;
use crate::vmir::Type;
use crate::vmir::{BinOp, CollKind, CollOp};
use egg::*;
use std::fmt::{Display, Formatter};

/// A verifier-allocated function-application id in the e-graph. **Disconnected
/// from VMIR `MemberId`**: the verifier assigns these (see `verify::func_registry`) — a
/// plain function reuses its declaration's index, ADT constructor/projection/tester
/// ops get one freshly-minted index per *concept* (polymorphic; the type
/// instantiation rides in the `FuncApp` discriminant, not the id).
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct FuncId(pub usize);

/// A compiled quantifier body, interned in the program-level
/// [`RecipeTable`](crate::verify::quant::RecipeTable). It is the *payload* of a
/// [`Symbolic::Forall`] node — the quantifier's "code", with the capture
/// children as its environment.
#[derive(
    Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, derive_more::From, derive_more::Into,
)]
pub struct RecipeId(pub usize);

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Symbolic {
    Fresh(u32),
    Lit(Literal),
    Binary(BinOp, [Id; 2]),
    Ite([Id; 3]),
    /// A function application `f[type_args](value_args)`. The e-graph is
    /// **polymorphic**: one `FuncId` per concept, with the **ground** type
    /// instantiation carried in the enode payload. It is *not* a child (ground
    /// types never merge, so they want no e-class) and *not* in the discriminant
    /// (which would fragment egg's `classes_by_op` op-index per instantiation).
    /// Distinctness — `mk[Int]` ≠ `mk[Bool]` — comes from the enode's derived
    /// `Eq`/`Hash` via egg's congruence `memo`. `children()` returns only the
    /// value args; the [`Discriminant`] is the concept `FuncId` alone.
    /// Addresses are ordinary function applications too: a field/predicate's
    /// address function (its own `FuncId`) over its args, with the rich
    /// `Type::Addr{group,value,bound}` as its return type (recorded in
    /// `func_ret_types` and recoverable by `infer_type`). No dedicated location
    /// sort or sentinel id. (No rewrite rule matches an address `FuncId`.)
    FuncApp(FuncId, Box<[Type]>, Box<[Id]>),
    RealCast(Id),
    /// A pure `forall`: the compiled body (`RecipeId`, interned program-wide) as
    /// payload, the **captured** outer terms as children. The node *is* the
    /// occurrence — no opaque occurrence function, no capture arity to track: a
    /// capture is child `c`, canonicalized by congruence and deduped by hashcons
    /// (so two alpha-equivalent `forall`s with the same captures are one
    /// e-class).
    ///
    /// Triggers *are* part of a recipe's identity: they are validated, never
    /// inferred, so two `forall`s that denote the same proposition but were written
    /// with different patterns stay separate recipes — pooling their trigger sets
    /// would instantiate one on a pattern its author never wrote.
    /// Instantiation adds the guarded clause
    /// `Ite(forall, body[caps, σ], true) == true`, so the instance is released only
    /// once this node merges `true`.
    ///
    /// The type payload instantiates the recipe's type parameters (a generic
    /// axiom's `Generic(i)`), one entry per parameter: entry `i` is either ground
    /// (fixed by an enclosing instantiation) or `Generic(i)` itself (still open, to
    /// be fixed by the type arguments of the trigger terms an instantiation
    /// matches). Empty for a ground quantifier. Like `FuncApp`'s, it is payload,
    /// not a child, and not in the discriminant.
    Forall(RecipeId, Box<[Type]>, Box<[Id]>),

    /// A collection operation on a collection of this element type (`Int` for
    /// a `Range`), operands in `PureInst::Coll` order.
    Coll(Box<Type>, CollOp, Box<[Id]>),

    /// The singleton collection of one element, of this kind and element type
    /// (`PureInst::CollLit`).
    CollLit(CollKind, Box<Type>, Id),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Discriminant {
    Fresh(u32),
    Lit(Literal),
    Binary(BinOp),
    Ite,
    /// The concept id alone (no type instantiation). Keeping the type args out of
    /// the discriminant means egg's `classes_by_op` indexes one bucket per concept
    /// (not per instantiation), and `discriminant()` stays a cheap `Copy`.
    /// Distinctness across instantiations does *not* rely on this — it comes from
    /// the full-enode `Eq`/`Hash` in the congruence memo (see [`Symbolic::FuncApp`]).
    FuncApp(FuncId),
    RealCast,
    /// Every quantifier, in one `classes_by_op` bucket. The recipe is deliberately
    /// *not* in the discriminant: the single instantiation rule re-reads each
    /// matched node for its recipe and captures anyway, so per-recipe bucketing
    /// bought nothing — and it made the searcher enumerate every recipe in the
    /// program, including the quantifiers of methods this unit will never touch.
    Forall,
    Coll(CollOp),
    CollLit(CollKind),
}

impl Language for Symbolic {
    type Discriminant = Discriminant;

    fn discriminant(&self) -> Self::Discriminant {
        use Discriminant as D;
        use Symbolic as S;
        match self {
            S::Fresh(s) => D::Fresh(*s),
            S::Lit(l) => D::Lit(l.clone()),
            S::Binary(op, _) => D::Binary(*op),
            S::Ite(_) => D::Ite,
            S::FuncApp(id, _, _) => D::FuncApp(*id),
            S::RealCast(_) => D::RealCast,
            S::Forall(..) => D::Forall,
            S::Coll(_, op, _) => D::Coll(*op),
            S::CollLit(kind, ..) => D::CollLit(*kind),
        }
    }

    fn matches(&self, other: &Self) -> bool {
        use Symbolic::*;
        match (self, other) {
            (Fresh(s1), Fresh(s2)) => s1 == s2,
            (Lit(l1), Lit(l2)) => l1 == l2,
            (Binary(op1, _), Binary(op2, _)) => op1 == op2,
            (Ite(_), Ite(_)) => true,
            (RealCast(_), RealCast(_)) => true,
            // Operator identity is the concept id + value arity, consistent with
            // the type-blind discriminant. (Distinctness across instantiations is
            // the memo's job via full-enode `Eq`, not `matches`.)
            (FuncApp(id1, _, args1), FuncApp(id2, _, args2)) => {
                id1 == id2 && args1.len() == args2.len()
            }
            (Forall(r1, _, caps1), Forall(r2, _, caps2)) => r1 == r2 && caps1.len() == caps2.len(),
            (Coll(_, op1, _), Coll(_, op2, _)) => op1 == op2,
            (CollLit(k1, ..), CollLit(k2, ..)) => k1 == k2,
            _ => false,
        }
    }

    fn children(&self) -> &[Id] {
        use Symbolic::*;
        match self {
            Fresh(..) | Lit(..) => &[],
            Binary(_, ids) => ids,
            Ite(ids) => ids,
            RealCast(id) => std::slice::from_ref(id),
            // Type args are in the payload, not children — only value args.
            FuncApp(_, _, ids) => ids,
            // The recipe and types are payload; the captures are the children.
            Forall(_, _, caps) => caps,
            Coll(_, _, ids) => ids,
            CollLit(_, _, id) => std::slice::from_ref(id),
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        use Symbolic::*;
        match self {
            Fresh(..) | Lit(..) => &mut [],
            Binary(_, ids) => ids,
            Ite(ids) => ids,
            RealCast(id) => std::slice::from_mut(id),
            FuncApp(_, _, ids) => ids,
            Forall(_, _, caps) => caps,
            Coll(_, _, ids) => ids,
            CollLit(_, _, id) => std::slice::from_mut(id),
        }
    }
}

impl Display for Symbolic {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Symbolic::Fresh(id) => write!(f, "fresh{id}"),
            Symbolic::Lit(l) => write!(f, "{l}"),
            Symbolic::Binary(op, _) => write!(f, "{op}"),
            Symbolic::Ite(_) => write!(f, "ITE"),
            Symbolic::RealCast(_) => write!(f, "real"),
            // Id-only label, with the ground type instantiation folded in (so each
            // instantiation is a distinct, self-describing node). The viz resolves
            // the `fn{id}` token to the concept's source name when it renders the
            // dot (it holds the interner).
            Symbolic::FuncApp(id, tys, _) => {
                if tys.is_empty() {
                    write!(f, "fn{}", id.0)
                } else {
                    // Type-argument instantiation is rendered in angle brackets
                    // (`fn3<Int>`), matching VMIR Display — `[..]` is reserved for a
                    // generic binder's arity and for heaps.
                    let args: Vec<String> = tys.iter().map(|t| t.to_string()).collect();
                    write!(f, "fn{}<{}>", id.0, args.join(", "))
                }
            }
            Symbolic::Forall(r, tys, _) => {
                if tys.is_empty() {
                    write!(f, "forall#{}", r.0)
                } else {
                    let args: Vec<String> = tys.iter().map(|t| t.to_string()).collect();
                    write!(f, "forall#{}<{}>", r.0, args.join(", "))
                }
            }
            Symbolic::Coll(elem_ty, op, _) => write!(f, "{}<{elem_ty}>", op.token()),
            Symbolic::CollLit(kind, ty, _) => write!(f, "{}<{ty}>", kind.name()),
        }
    }
}

/// Error from [`Symbolic::from_op`] when a rewrite-pattern token is unknown.
#[derive(Debug)]
pub struct FromOpError(String);

impl Display for FromOpError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for FromOpError {}

impl FromOp for Symbolic {
    type Error = FromOpError;

    /// Parse a node from an s-expression operator + already-parsed children.
    /// Enables egg's string `rewrite!` macro (no type tags to encode now).
    /// Pattern variables (`?x`) are handled by egg before this is called.
    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        use Symbolic::*;
        let bin = |o: BinOp| {
            if children.len() == 2 {
                Ok(Binary(o, [children[0], children[1]]))
            } else {
                Err(FromOpError(format!("`{op}` expects 2 children")))
            }
        };
        match (op, children.len()) {
            ("true", 0) => Ok(Lit(Literal::Bool(true))),
            ("false", 0) => Ok(Lit(Literal::Bool(false))),
            ("null", 0) => Ok(Lit(Literal::Null)),
            ("ite", 3) => Ok(Ite([children[0], children[1], children[2]])),
            ("real", 1) => Ok(RealCast(children[0])),
            // Sort-tagged arithmetic and comparison (`i` = Int, `r` = Real). The
            // bare forms are rejected below: a pattern like `(- ?x ?x)` that does
            // not say which sort it means cannot decide whether to produce `0` or
            // `0/1`, and guessing merges an `Int` literal with a `Real` one.
            ("+i", _) => bin(BinOp::AddI),
            ("+r", _) => bin(BinOp::AddR),
            ("-i", _) => bin(BinOp::SubI),
            ("-r", _) => bin(BinOp::SubR),
            ("*i", _) => bin(BinOp::MulI),
            ("*r", _) => bin(BinOp::MulR),
            ("/i", _) => bin(BinOp::DivI),
            ("/r", _) => bin(BinOp::DivR),
            ("mod", _) => bin(BinOp::Mod),
            ("<i", _) => bin(BinOp::LtI),
            ("<r", _) => bin(BinOp::LtR),
            // Polymorphic: no sort to name.
            ("==", _) => bin(BinOp::Eq),
            ("+" | "-" | "*" | "/" | "<", _) => Err(FromOpError(format!(
                "`{op}` is sort-ambiguous — write `{op}i` (Int) or `{op}r` (Real)"
            ))),
            // A bare integer is `Lit(Int)`; a fraction (`0/1`, `1/2`) is `Lit(Real)`.
            (lit, 0) => {
                if let Ok(n) = lit.parse::<num::BigInt>() {
                    Ok(Lit(Literal::Int(n)))
                } else if let Ok(r) = lit.parse::<num::BigRational>() {
                    Ok(Lit(Literal::Real(r)))
                } else {
                    Err(FromOpError(format!("unknown leaf `{lit}`")))
                }
            }
            // `Symbolic::Coll` is deliberately absent: an operator name alone has no
            // element type, and a placeholder would build a wrong node.
            _ => Err(FromOpError(format!("unknown op `{op}`/{}", children.len()))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dhash::HashMap;
    use crate::verify::analysis::ConstFold;
    use crate::verify::types::infer_type;
    use crate::vmir::{SeqOp, SetOp};

    type Graph = EGraph<Symbolic, ConstFold>;

    fn coll(g: &mut Graph, elem: Type, op: impl Into<CollOp>, args: &[Id]) -> Id {
        g.add(Symbolic::Coll(Box::new(elem), op.into(), args.into()))
    }

    fn ty_of(g: &Graph, id: Id) -> Option<Type> {
        infer_type(
            g,
            &HashMap::default(),
            &HashMap::default(),
            id,
            &mut HashMap::default(),
        )
    }

    /// The same operands under a sequence and a set operation, or in a
    /// sequence and a set literal, are different terms: the kind is part of
    /// the node, so congruence never merges them.
    #[test]
    fn seq_and_set_nodes_of_one_shape_stay_apart() {
        let mut g = Graph::default();
        let (s, x) = (g.add(Symbolic::Fresh(0)), g.add(Symbolic::Fresh(1)));
        let seq_in = coll(&mut g, Type::Int, SeqOp::Contains, &[s, x]);
        let set_in = coll(&mut g, Type::Int, SetOp::Contains, &[s, x]);
        let seq_lit = g.add(Symbolic::CollLit(CollKind::Seq, Box::new(Type::Int), x));
        let set_lit = g.add(Symbolic::CollLit(CollKind::Set, Box::new(Type::Int), x));
        g.rebuild();
        assert_ne!(g.find(seq_in), g.find(set_in));
        assert_ne!(g.find(seq_lit), g.find(set_lit));
        // The discriminant keeps them in different `classes_by_op` buckets.
        assert_ne!(
            g.id_to_node(seq_in).discriminant(),
            g.id_to_node(set_in).discriminant()
        );
    }

    /// Set operations are ordinary congruent nodes: equal operands make equal
    /// unions.
    #[test]
    fn set_operations_are_congruent() {
        let mut g = Graph::default();
        let (a, a2, b) = (
            g.add(Symbolic::Fresh(0)),
            g.add(Symbolic::Fresh(1)),
            g.add(Symbolic::Fresh(2)),
        );
        let u1 = coll(&mut g, Type::Int, SetOp::Union, &[a, b]);
        let u2 = coll(&mut g, Type::Int, SetOp::Union, &[a2, b]);
        let i1 = coll(&mut g, Type::Int, SetOp::Intersection, &[a, b]);
        g.rebuild();
        assert_ne!(g.find(u1), g.find(u2));
        g.union(a, a2);
        g.rebuild();
        assert_eq!(g.find(u1), g.find(u2));
        assert_ne!(g.find(u1), g.find(i1), "union is not intersection");
    }

    /// Each set node's type is recovered from the node alone.
    #[test]
    fn set_node_types_are_inferred() {
        let mut g = Graph::default();
        let (a, b, x) = (
            g.add(Symbolic::Fresh(0)),
            g.add(Symbolic::Fresh(1)),
            g.add(Symbolic::Fresh(2)),
        );
        let set_int = Type::set(Type::Int);
        let cases = [
            (coll(&mut g, Type::Int, SetOp::Empty, &[]), set_int.clone()),
            (coll(&mut g, Type::Int, SetOp::Card, &[a]), Type::Int),
            (
                coll(&mut g, Type::Int, SetOp::Contains, &[a, x]),
                Type::Bool,
            ),
            (
                coll(&mut g, Type::Int, SetOp::Union, &[a, b]),
                set_int.clone(),
            ),
            (
                coll(&mut g, Type::Int, SetOp::Intersection, &[a, b]),
                set_int.clone(),
            ),
            (
                coll(&mut g, Type::Int, SetOp::Difference, &[a, b]),
                set_int.clone(),
            ),
            (coll(&mut g, Type::Int, SetOp::Subset, &[a, b]), Type::Bool),
            (
                coll(&mut g, Type::Int, SetOp::Add, &[a, x]),
                set_int.clone(),
            ),
            (coll(&mut g, Type::Int, SetOp::Eq, &[a, b]), Type::Bool),
            (
                g.add(Symbolic::CollLit(CollKind::Set, Box::new(Type::Int), x)),
                set_int.clone(),
            ),
            (
                coll(&mut g, Type::set(Type::Bool), SetOp::Union, &[a, b]),
                Type::set(Type::set(Type::Bool)),
            ),
        ];
        g.rebuild();
        for (id, want) in cases {
            assert_eq!(ty_of(&g, id), Some(want), "{}", g.id_to_node(id));
        }
    }
}
