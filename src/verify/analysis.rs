use crate::dhash::HashMap;
use std::collections::BTreeSet;
use std::sync::Arc;

use egg::{Analysis, DidMerge, EGraph, Id, Justification, Language};
use num::BigRational;

use crate::verify::lang::{FuncId, Symbolic};
use crate::vmir::{BinOp, Literal, MemberId, Type};

/// What [`Data::Adt`] knows about an ADT value: the variants it may still be.
#[derive(Debug, Clone, PartialEq)]
pub struct AdtData {
    /// The ADT, e.g. `List`.
    pub head: MemberId,
    /// The ADT's generic instantiation.
    pub tys: Box<[Type]>,
    /// The variants still possible, by constructor id. Never empty: an empty set
    /// is [`Data::Inconsistent`].
    pub possible: BTreeSet<FuncId>,
}

/// An e-class's analysis data: its folded value and its instantiation
/// [`generation`](ClassData::generation).
#[derive(Debug, Clone, PartialEq)]
pub struct ClassData {
    pub value: Data,
    /// How many quantifier instantiations deep the class's oldest term is (Z3's
    /// instance generation): 0 for a term the program built, `g` for one built
    /// by a generation-`g` instance. A node is created at the larger of its
    /// children's generations and the generation of the instance being built
    /// ([`ConstFold::building`]), and keeps it. A class holds the least of its
    /// nodes', so only a union changes it, and only downwards.
    pub generation: u32,
}

/// Const-fold analysis data: a lattice over what is known about an e-class's
/// value. Only literals and ADT variant sets are tracked; other types are
/// reconstructed for visualization from the side oracle in `verify::context`.
///
/// - `Unknown`: nothing known yet (for an ADT value: every variant possible).
/// - `Known(lit)`: folds to `lit`.
/// - `Adt(data)`: the value is one of `data.possible`. A constructor application
///   `C(..)` starts at `{C}`; merging two classes intersects their sets, and the
///   variant-narrow rule shrinks a set from decided testers. This gives
///   **constructor distinctness** with no tester term and no O(variants) axioms
///   — a variant clash is an empty intersection, detected even when the program
///   never mentions a discriminator.
/// - `Inconsistent`: two **same-typed** literals of differing value were merged
///   (`true == false`, `5 == 6`), or two variant sets of one ADT head at one
///   instantiation with no variant in common (constructors are free, so
///   `Cons(..) == Nil` is a contradiction) — the e-class, and thus the whole
///   verification unit, is contradictory. Merging across *different* types is
///   instead a verifier panic: a type error, not a fact about the program.
#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    Unknown,
    Known(Literal),
    Adt(AdtData),
    Inconsistent,
}

/// What `ConstFold` knows about an e-class's *value*, as a comparison key: a
/// folded literal, or the ADT variants the value may still be. Two classes
/// carrying fingerprints that [`Fingerprint::differs_from`] separates cannot
/// denote the same value, so a rule may use the pair to refute one of them.
///
/// The variant half rests on **free constructors** — distinct variants of one
/// ADT head at one instantiation are disjoint, so values with disjoint variant
/// sets differ. That is the same premise [`ConstFold::make`] uses to fold an
/// `==` between such values to `false` and [`ConstFold::merge`] uses to declare
/// a class [`Data::Inconsistent`] (see the [`Data`] doc). If the ADT encoding
/// ever gains non-free constructors, all three sites break together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fingerprint<'a> {
    Lit(&'a Literal),
    /// The variants the value may be, at one instantiation of one head.
    Adt {
        head: MemberId,
        tys: &'a [Type],
        possible: &'a BTreeSet<FuncId>,
    },
}

impl Fingerprint<'_> {
    /// Whether these two values are provably different: different literals, or
    /// variant sets with no variant in common.
    ///
    /// Conservative in every direction it is not sure about:
    ///
    /// * Variant sets are separated only **within one head and instantiation**.
    ///   Constructor ids are unique across ADTs, so two heads always look
    ///   disjoint, and the type args are part of the operator's identity in the
    ///   polymorphic e-graph, so `List[Int]::Nil` vs `List[Bool]::Nil` is a type
    ///   error, not a fact about the program — report "not different" and let
    ///   the merge panic in [`ConstFold::merge`] catch a real violation.
    /// * Overlapping sets are **not** separated, so the same constructor at
    ///   different arguments is not either: the fingerprint is the variant, not
    ///   the whole term, so `c(x)` vs `c(y)` reports "not different" even when
    ///   `x ≢ y`. Losing that is incompleteness, never unsoundness.
    /// * Mixing a literal with a constructor cannot arise (that merge is a type
    ///   error panic) and reports "not different" if it somehow does.
    pub fn differs_from(&self, other: &Self) -> bool {
        match (self, other) {
            (Fingerprint::Lit(a), Fingerprint::Lit(b)) => a != b,
            (
                Fingerprint::Adt {
                    head: h1,
                    tys: t1,
                    possible: p1,
                },
                Fingerprint::Adt {
                    head: h2,
                    tys: t2,
                    possible: p2,
                },
            ) => h1 == h2 && t1 == t2 && p1.is_disjoint(p2),
            _ => false,
        }
    }
}

impl Data {
    /// The folded literal, if this e-class is a known constant.
    pub fn known(&self) -> Option<&Literal> {
        match self {
            Data::Known(lit) => Some(lit),
            _ => None,
        }
    }

    /// What is known about this e-class's value, as a comparison key. `None`
    /// for `Unknown` (nothing known) and for `Inconsistent` (the graph is
    /// already contradictory; other machinery handles it).
    pub fn fingerprint(&self) -> Option<Fingerprint<'_>> {
        match self {
            Data::Known(lit) => Some(Fingerprint::Lit(lit)),
            Data::Adt(adt) => {
                assert!(
                    !adt.possible.is_empty(),
                    "empty variant set must be `Inconsistent`"
                );
                Some(Fingerprint::Adt {
                    head: adt.head,
                    tys: &adt.tys,
                    possible: &adt.possible,
                })
            }
            Data::Unknown | Data::Inconsistent => None,
        }
    }

    /// Whether this e-class merged conflicting same-typed literals.
    pub fn is_inconsistent(&self) -> bool {
        matches!(self, Data::Inconsistent)
    }
}

impl ClassData {
    /// See [`Data::known`].
    pub fn known(&self) -> Option<&Literal> {
        self.value.known()
    }

    /// See [`Data::fingerprint`].
    pub fn fingerprint(&self) -> Option<Fingerprint<'_>> {
        self.value.fingerprint()
    }

    /// See [`Data::is_inconsistent`].
    pub fn is_inconsistent(&self) -> bool {
        self.value.is_inconsistent()
    }
}

/// Whether two literals are of the same VMIR type (so a value conflict is an
/// inconsistency rather than a type error).
fn same_type(a: &Literal, b: &Literal) -> bool {
    use Literal::*;
    matches!(
        (a, b),
        (Bool(_), Bool(_)) | (Int(_), Int(_)) | (Real(_), Real(_)) | (Null, Null)
    )
}

#[derive(Default, Debug, Clone)]
pub struct ConstFold {
    /// Constructor id → the ADT head it belongs to (from
    /// `FuncRegistry::ctor_table`). Any `FuncApp` whose id is absent is an
    /// ordinary function, not a constructor. Empty by default (tests without
    /// ADTs), which simply disables the variant sets.
    ctors: Arc<HashMap<FuncId, MemberId>>,
    /// While a bucket rule is applied (`rewrite::PerClass`), the classes whose
    /// nodes read differently after each union; see [`Analysis::pre_union`].
    pub(crate) union_log: std::cell::RefCell<Option<Vec<Id>>>,
    /// The generation of the quantifier instance whose body is being added, 0
    /// outside one: the least generation a node added now is created at (see
    /// [`ClassData::generation`]).
    pub(crate) building: std::cell::Cell<u32>,
}

impl ConstFold {
    pub fn new(ctors: Arc<HashMap<FuncId, MemberId>>) -> Self {
        Self {
            ctors,
            union_log: Default::default(),
            building: Default::default(),
        }
    }

    /// The ADT head of `f`, if `f` is a constructor.
    fn head_of(&self, f: FuncId) -> Option<MemberId> {
        self.ctors.get(&f).copied()
    }
}

impl ConstFold {
    fn make_value(egraph: &EGraph<Symbolic, Self>, enode: &Symbolic) -> Data {
        use Data::{Adt, Inconsistent, Known, Unknown};
        let value = |c: &Id| &egraph[*c].data.value;
        match enode {
            Symbolic::Lit(lit) => Known(lit.clone()),

            // A constructor application is exactly its own variant.
            Symbolic::FuncApp(f, tys, _) if let Some(head) = egraph.analysis.head_of(*f) => {
                Adt(AdtData {
                    head,
                    tys: tys.clone(),
                    possible: BTreeSet::from([*f]),
                })
            }

            // A quantifier is opaque to constant folding: its truth is decided by
            // the instantiation rule (guarded merges with `true`), never by its
            // payload or its capture children.
            Symbolic::Fresh(_) | Symbolic::FuncApp(..) | Symbolic::Forall(..) => Unknown,

            Symbolic::RealCast(c) => match value(c) {
                Known(Literal::Int(n)) => Known(Literal::Real(BigRational::from(n.clone()))),
                Known(_) | Adt { .. } => {
                    unreachable!("RealCast operand must be an integer literal")
                }
                Inconsistent => Inconsistent,
                Unknown => Unknown,
            },

            Symbolic::Binary(op, [l, r]) => match (value(l), value(r)) {
                (Inconsistent, _) | (_, Inconsistent) => Inconsistent,
                // Disequality of ADT values: values whose variant sets share no
                // variant are distinct (free constructors), so an `==` between
                // them folds to `false` outright. No tester term and no
                // distinctness axiom needed — the sets are in the operands' data.
                (Adt(lhs), Adt(rhs))
                    if *op == BinOp::Eq
                        && lhs.head == rhs.head
                        && lhs.tys == rhs.tys
                        && lhs.possible.is_disjoint(&rhs.possible) =>
                {
                    Known(Literal::Bool(false))
                }
                (Known(lv), Known(rv)) => eval_binary(*op, lv, rv).map_or(Unknown, Known),
                _ => Unknown,
            },

            Symbolic::Ite([c, t, e]) => match value(c) {
                Known(Literal::Bool(true)) => value(t).clone(),
                Known(Literal::Bool(false)) => value(e).clone(),
                Known(_) | Adt(..) => unreachable!("Condition of ITE must be a boolean literal"),
                Inconsistent => Inconsistent,
                // Undecided condition: the value is one of the two arms, so it
                // may be any variant either arm may be — the *union* of their
                // sets, a sound over-approximation. (Merging, by contrast,
                // intersects: there both sides describe the same value.) egg
                // re-runs `make` when an arm's data changes, so this stays
                // current as the arms narrow.
                Unknown => match (value(t), value(e)) {
                    (Adt(a), Adt(b)) => {
                        assert!(
                            a.head == b.head && a.tys == b.tys,
                            "type error: ite branches of different ADTs or instantiations"
                        );
                        Adt(AdtData {
                            head: a.head,
                            tys: a.tys.clone(),
                            possible: a.possible.union(&b.possible).copied().collect(),
                        })
                    }
                    _ => Unknown,
                },
            },
            Symbolic::Seq(_, _) => Unknown,
            Symbolic::SeqLit(_,_) => Unknown,
        }
    }

    fn merge_value(&self, a: &mut Data, b: Data) -> DidMerge {
        use Data::{Adt, Inconsistent, Known, Unknown};
        match (&mut *a, &b) {
            (Inconsistent, Inconsistent) => DidMerge(false, false),
            (Inconsistent, _) => DidMerge(false, true),
            (_, Inconsistent) => {
                *a = Inconsistent;
                DidMerge(true, false)
            }
            (Adt(a_adt), Adt(b_adt)) => {
                // Both sides describe one value, so it is a variant both allow:
                // intersect. Only comparable within one instantiation: the type
                // args are part of the operator's identity (the polymorphic
                // e-graph keeps `List[Int]::Nil` and `List[Bool]::Nil` apart by
                // discriminant), so a clash across instantiations is a type error,
                // not a contradiction.
                if a_adt.head != b_adt.head || a_adt.tys != b_adt.tys {
                    panic!(
                        "type error: merged constructors of different ADTs or \
                         instantiations: {:?}{:?} vs {:?}{:?}",
                        a_adt.head, a_adt.tys, b_adt.head, b_adt.tys
                    );
                }
                let before = a_adt.possible.len();
                a_adt.possible.retain(|f| b_adt.possible.contains(f));
                let after = a_adt.possible.len();
                if after == 0 {
                    // No variant satisfies both sides: free constructors, contradiction.
                    *a = Inconsistent;
                    return DidMerge(true, true);
                }
                // a changed iff it lost a variant; b would change iff the result is
                // smaller than b (the result is a subset of b, so lengths suffice).
                DidMerge(after != before, after != b_adt.possible.len())
            }
            (Adt(..), Known(lit)) => {
                panic!("type error: merged an ADT constructor with a literal: {lit:?}")
            }
            (Known(lit), Adt(..)) => {
                panic!("type error: merged an ADT constructor with a literal: {lit:?}")
            }
            (Unknown, Adt(..)) => {
                *a = b;
                DidMerge(true, false)
            }
            (Adt(..), Unknown) => DidMerge(false, true),
            (Known(x), Known(y)) => {
                if x == y {
                    DidMerge(false, false)
                } else if same_type(x, y) {
                    // Same-typed conflict ⇒ contradiction (not a panic).
                    *a = Inconsistent;
                    DidMerge(true, true)
                } else {
                    panic!("type error: merged literals of different types: {x:?} vs {y:?}");
                }
            }
            (Unknown, Known(y)) => {
                *a = Known(y.clone());
                DidMerge(true, false)
            }
            (Known(_), Unknown) => DidMerge(false, true),
            (Unknown, Unknown) => DidMerge(false, false),
        }
    }
}

impl Analysis<Symbolic> for ConstFold {
    type Data = ClassData;

    fn make(egraph: &mut EGraph<Symbolic, Self>, enode: &Symbolic, _id: Id) -> Self::Data {
        // A term is at least as deep as its subterms, whichever rule built it:
        // otherwise a rule that builds new terms out of an instance's terms would
        // hand the next instance generation-0 bindings again.
        let generation = enode
            .children()
            .iter()
            .map(|&c| egraph[c].data.generation)
            .fold(egraph.analysis.building.get(), u32::max);
        ClassData {
            value: Self::make_value(egraph, enode),
            generation,
        }
    }

    /// A node keeps the generation it was created at: re-making it (a child's
    /// value changed) must not lower it to its children's current ones, so it
    /// reports its class's.
    fn remake(egraph: &mut EGraph<Symbolic, Self>, enode: &Symbolic, id: Id) -> Self::Data {
        ClassData {
            value: Self::make_value(egraph, enode),
            generation: egraph[id].data.generation,
        }
    }

    /// The value merges as a lattice; the generation is the least of the two.
    /// Only a value change is reported: a parent was created at its own
    /// generation and does not depend on its children's later ones.
    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        a.generation = a.generation.min(b.generation);
        self.merge_value(&mut a.value, b.value)
    }

    /// When `union_log` is on, record the classes whose nodes will read
    /// differently after this union: the root that survives (it gains nodes),
    /// the parents of the class absorbed (a child's identity changes), and the
    /// parents of the survivor if its data changes. egg keeps the class with more
    /// parents as the root, the first on a tie; were that to change, only which
    /// classes get walked again would, never what a walk may derive.
    fn pre_union(egraph: &EGraph<Symbolic, Self>, id1: Id, id2: Id, _: &Option<Justification>) {
        let mut log = egraph.analysis.union_log.borrow_mut();
        let Some(log) = log.as_mut() else {
            return;
        };
        let (a, b) = (egraph.find(id1), egraph.find(id2));
        if a == b {
            return;
        }
        let (keep, gone) = if egraph[a].parents().len() < egraph[b].parents().len() {
            (b, a)
        } else {
            (a, b)
        };
        log.push(keep);
        log.extend(egraph[gone].parents());
        let (kept, absorbed) = (&egraph[keep].data.value, &egraph[gone].data.value);
        if kept != absorbed && *absorbed != Data::Unknown && *kept != Data::Inconsistent {
            log.extend(egraph[keep].parents());
        }
    }

    fn modify(egraph: &mut EGraph<Symbolic, Self>, id: Id) {
        match egraph[id].data.value.clone() {
            Data::Known(lit) => {
                // The literal a class folds to is derived from the class's terms,
                // so a new one is as deep as they are.
                let at = egraph[id].data.generation;
                let outer = egraph.analysis.building.replace(at);
                let lit_id = egraph.add(Symbolic::Lit(lit));
                egraph.analysis.building.set(outer);
                egraph.union(id, lit_id);
            }
            // One contradictory class makes the whole graph contradictory, so
            // record that *in the graph* — `true == false` — and callers decide
            // it with two `find`s instead of scanning every class. The merged
            // class is itself `Inconsistent` (a same-typed `Bool` conflict), so
            // this re-fires on it once and then unions an already-merged pair.
            Data::Inconsistent => {
                let t = egraph.add(Symbolic::Lit(Literal::Bool(true)));
                let f = egraph.add(Symbolic::Lit(Literal::Bool(false)));
                egraph.union(t, f);
            }
            Data::Unknown | Data::Adt(..) => {}
        }
    }
}

/// Fold a binary op over two literals. The operator names its own operand sort,
/// so each arm matches exactly one literal pair; a mismatch means the operand
/// does not have the sort the operator claims, which is a lowering bug rather
/// than something to fold.
///
/// `None` for a literal division by zero: the term is unspecified (an
/// uninterpreted value, matching SMT semantics), not a fold-time panic —
/// well-definedness is a separate obligation, and never checked at all inside
/// an axiom body.
pub fn eval_binary(op: BinOp, l: &Literal, r: &Literal) -> Option<Literal> {
    use Literal::{Int, Real};
    /// Destructure the operands at the sort the operator declares, or panic.
    macro_rules! operands {
        ($variant:ident) => {
            match (l, r) {
                ($variant(a), $variant(b)) => (a, b),
                _ => unreachable!(
                    "operands are not {} for {op:?}: {l:?}, {r:?}",
                    stringify!($variant)
                ),
            }
        };
    }
    Some(match op {
        BinOp::AddI => {
            let (a, b) = operands!(Int);
            Int(a + b)
        }
        BinOp::AddR => {
            let (a, b) = operands!(Real);
            Real(a + b)
        }
        BinOp::SubI => {
            let (a, b) = operands!(Int);
            Int(a - b)
        }
        BinOp::SubR => {
            let (a, b) = operands!(Real);
            Real(a - b)
        }
        BinOp::MulI => {
            let (a, b) = operands!(Int);
            Int(a * b)
        }
        BinOp::MulR => {
            let (a, b) = operands!(Real);
            Real(a * b)
        }
        // Viper's `%` and `\` are SMT-LIB's `mod` and `div`: Euclidean, the
        // remainder always in `[0, |b|)`. Rust's (and BigInt's) operators
        // truncate toward zero instead, which differs on a negative dividend
        // (`-7 % 3`: 2 in Viper, -1 truncated).
        BinOp::Mod => {
            let (a, b) = operands!(Int);
            if *b == num::BigInt::ZERO {
                return None;
            }
            Int(euclid_mod(a, b))
        }
        BinOp::DivI => {
            let (a, b) = operands!(Int);
            if *b == num::BigInt::ZERO {
                return None;
            }
            Int((a - euclid_mod(a, b)) / b)
        }
        BinOp::DivR => {
            let (a, b) = operands!(Real);
            if *b == num::BigRational::from(num::BigInt::ZERO) {
                return None;
            }
            Real(a / b)
        }
        BinOp::LtI => {
            let (a, b) = operands!(Int);
            Literal::Bool(a < b)
        }
        BinOp::LtR => {
            let (a, b) = operands!(Real);
            Literal::Bool(a < b)
        }
        BinOp::Eq => Literal::Bool(l == r),
    })
}

/// SMT-LIB `mod`: the remainder of `a` by `b` (nonzero) in `[0, |b|)`.
fn euclid_mod(a: &num::BigInt, b: &num::BigInt) -> num::BigInt {
    use num::Signed as _;
    let r = a % b;
    if r.is_negative() { r + b.abs() } else { r }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num::BigInt;

    type Graph = EGraph<Symbolic, ConstFold>;

    fn generation(g: &Graph, id: Id) -> u32 {
        g[id].data.generation
    }

    /// Adds `node` as if a generation-`at` instance built it.
    fn add_at(g: &mut Graph, at: u32, node: Symbolic) -> Id {
        let outer = g.analysis.building.replace(at);
        let id = g.add(node);
        g.analysis.building.set(outer);
        id
    }

    fn int(n: i64) -> Symbolic {
        Symbolic::Lit(Literal::Int(BigInt::from(n)))
    }

    fn app(f: usize, arg: Id) -> Symbolic {
        Symbolic::FuncApp(FuncId(f), Box::new([]), Box::new([arg]))
    }

    /// A node is created at least as deep as its children and as the instance
    /// building it; a union keeps the shallower class's generation.
    #[test]
    fn generation_is_max_at_creation_and_min_at_union() {
        let mut g = Graph::default();
        let a = g.add(Symbolic::Fresh(0));
        let fa = add_at(&mut g, 3, app(0, a));
        assert_eq!(generation(&g, fa), 3);
        // A rule outside any instance builds on the generation-3 term.
        let gfa = g.add(app(1, fa));
        assert_eq!(generation(&g, gfa), 3);
        g.union(gfa, a);
        g.rebuild();
        assert_eq!(generation(&g, gfa), 0);
    }

    /// Re-making a parent after its child's value changed keeps the parent's
    /// creation generation, even though the child is now shallower.
    #[test]
    fn remake_keeps_the_creation_generation() {
        let mut g = Graph::default();
        let x = add_at(&mut g, 2, Symbolic::Fresh(0));
        let one = g.add(int(1));
        let sum = add_at(&mut g, 2, Symbolic::Binary(BinOp::AddI, [x, one]));
        let five = g.add(int(5));
        g.union(x, five);
        g.rebuild();
        assert_eq!(generation(&g, x), 0);
        assert_eq!(g[sum].data.known(), Some(&Literal::Int(BigInt::from(6))));
        assert_eq!(generation(&g, sum), 2);
    }

    /// A literal a class folds to is new at the class's generation, so folding
    /// does not hand the next instance a shallower binding than the terms it was
    /// folded from.
    #[test]
    fn folded_literal_inherits_its_class_generation() {
        let mut g = Graph::default();
        let x = add_at(&mut g, 4, Symbolic::Fresh(0));
        let one = g.add(int(1));
        let sum = add_at(&mut g, 4, Symbolic::Binary(BinOp::AddI, [x, one]));
        // `x := 5` comes from a generation-4 term; `x + 1` folds to a new `6`
        // during the rebuild, outside any instance.
        let five = add_at(&mut g, 4, int(5));
        g.union(x, five);
        g.rebuild();
        let six = g.lookup(int(6)).expect("folded");
        assert_eq!(g.find(six), g.find(sum));
        assert_eq!(generation(&g, six), 4);
    }
}
