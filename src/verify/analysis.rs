use crate::dhash::{HashMap, HashSet};
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

/// A **unique constant**: an application of a `unique` domain function
/// (`vmir::Function::unique`) at one type instantiation. Two different unique
/// constants of one type denote different values (Silicon's `distinct`); that
/// is all `unique` promises. A unique constant may still equal a literal, a
/// constructor value or any other non-unique term.
#[derive(Debug, Clone, PartialEq)]
pub struct UniqueConst {
    pub func: FuncId,
    /// The instantiation. Part of the constant's identity: a generic unique
    /// function is a different constant at each instantiation, and since it is
    /// nullary its type args are fixed by its return type, so two instances of
    /// one function never share a type.
    pub tys: Box<[Type]>,
}

/// An e-class's analysis data: its folded value, the unique constant it holds,
/// its instantiation [`generation`](ClassData::generation), and whether it
/// reaches a quantifier.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassData {
    pub value: Data,
    /// The unique constant this class holds, if any. Kept **beside** `value`
    /// rather than as a [`Data`] variant, because it is independent of it: a
    /// class may hold both `k()` and `5`, and folding to `5` must not forget
    /// `k()` (or a later merge with `k2()`, also `5`, would miss that `k() ==
    /// k2()` is a contradiction). Merging two classes with *different* unique
    /// constants makes `value` [`Data::Inconsistent`].
    pub unique: Option<UniqueConst>,
    /// How many quantifier instantiations deep the class's oldest term is (Z3's
    /// instance generation): 0 for a term the program built, `g` for one built
    /// by a generation-`g` instance. A node is created at the larger of its
    /// children's generations and the generation of the instance being built
    /// ([`ConstFold::building`]), and keeps it. A class holds the least of its
    /// nodes', so only a union changes it, and only downwards.
    pub generation: u32,
    /// The class holds a [`Symbolic::Forall`] node, or a node with a child
    /// class that [reaches](ClassData::reaches_forall) one. An
    /// over-approximation: it is OR-ed on merge and never cleared, so a parent
    /// keeps it after the child folds to a literal. Read by the Z3 rung to decide
    /// whether a goal needs the quantifiers it otherwise leaves out.
    pub has_forall: bool,
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

/// What `ConstFold` knows about an e-class's *value*, as a comparison key: what
/// [`Data`] folded it to, and the unique constant it holds. Two classes carrying
/// fingerprints that [`Fingerprint::differs_from`] separates cannot denote the
/// same value, so a rule may use the pair to refute one of them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fingerprint<'a> {
    pub value: Option<ValueFingerprint<'a>>,
    pub unique: Option<&'a UniqueConst>,
}

impl Fingerprint<'_> {
    /// Whether these two values are provably different: their folded values
    /// differ ([`ValueFingerprint::differs_from`]), or they are two different
    /// unique constants.
    ///
    /// The unique half assumes the two classes have one type, as every caller's
    /// do (an `ite` and its arms; two applications of one function at one
    /// argument position). Unique constants of different types are not
    /// distinct, but comparing values of different types is a type error the
    /// caller would have made first.
    pub fn differs_from(&self, other: &Self) -> bool {
        let value = match (&self.value, &other.value) {
            (Some(a), Some(b)) => a.differs_from(b),
            _ => false,
        };
        let unique = match (self.unique, other.unique) {
            (Some(a), Some(b)) => a != b,
            _ => false,
        };
        value || unique
    }
}

/// The [`Data`] half of a [`Fingerprint`]: a folded literal, or the ADT
/// variants the value may still be.
///
/// The variant half rests on **free constructors** — distinct variants of one
/// ADT head at one instantiation are disjoint, so values with disjoint variant
/// sets differ. That is the same premise [`ConstFold::make`] uses to fold an
/// `==` between such values to `false` and [`ConstFold::merge`] uses to declare
/// a class [`Data::Inconsistent`] (see the [`Data`] doc). If the ADT encoding
/// ever gains non-free constructors, all three sites break together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValueFingerprint<'a> {
    Lit(&'a Literal),
    /// The variants the value may be, at one instantiation of one head.
    Adt {
        head: MemberId,
        tys: &'a [Type],
        possible: &'a BTreeSet<FuncId>,
    },
}

impl ValueFingerprint<'_> {
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
            (ValueFingerprint::Lit(a), ValueFingerprint::Lit(b)) => a != b,
            (
                ValueFingerprint::Adt {
                    head: h1,
                    tys: t1,
                    possible: p1,
                },
                ValueFingerprint::Adt {
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
    pub fn fingerprint(&self) -> Option<ValueFingerprint<'_>> {
        match self {
            Data::Known(lit) => Some(ValueFingerprint::Lit(lit)),
            Data::Adt(adt) => {
                assert!(
                    !adt.possible.is_empty(),
                    "empty variant set must be `Inconsistent`"
                );
                Some(ValueFingerprint::Adt {
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

    /// What is known about this class's value: [`Data::fingerprint`] and the
    /// unique constant it holds. `None` when neither is known.
    pub fn fingerprint(&self) -> Option<Fingerprint<'_>> {
        let fp = Fingerprint {
            value: self.value.fingerprint(),
            unique: self.unique.as_ref(),
        };
        (fp.value.is_some() || fp.unique.is_some()).then_some(fp)
    }

    /// See [`Data::is_inconsistent`].
    pub fn is_inconsistent(&self) -> bool {
        self.value.is_inconsistent()
    }

    /// Whether a quantifier may still decide this class's value. A class that
    /// folded to a literal does not count: its value is a fact of its own. That
    /// also keeps the bit from spreading through `true`, which every assumed
    /// quantifier and every proven goal merges with.
    pub fn reaches_forall(&self) -> bool {
        self.has_forall && self.known().is_none()
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
    /// The `unique` function ids (from `FuncRegistry::unique_table`). An
    /// application of one is a [`UniqueConst`]. Empty by default, which
    /// disables unique-constant distinctness.
    uniques: Arc<HashSet<FuncId>>,
    /// While a bucket rule is applied (`rewrite::PerClass`), the classes whose
    /// nodes read differently after each union; see [`Analysis::pre_union`].
    pub(crate) union_log: std::cell::RefCell<Option<Vec<Id>>>,
    /// The generation of the quantifier instance whose body is being added, 0
    /// outside one: the least generation a node added now is created at (see
    /// [`ClassData::generation`]).
    pub(crate) building: std::cell::Cell<u32>,
}

impl ConstFold {
    pub fn new(ctors: Arc<HashMap<FuncId, MemberId>>, uniques: Arc<HashSet<FuncId>>) -> Self {
        Self {
            ctors,
            uniques,
            union_log: Default::default(),
            building: Default::default(),
        }
    }

    /// The ADT head of `f`, if `f` is a constructor.
    fn head_of(&self, f: FuncId) -> Option<MemberId> {
        self.ctors.get(&f).copied()
    }

    /// The unique constant `enode` is, if it applies a `unique` function.
    fn unique_of(&self, enode: &Symbolic) -> Option<UniqueConst> {
        match enode {
            Symbolic::FuncApp(f, tys, args) if self.uniques.contains(f) => {
                assert!(
                    args.is_empty(),
                    "unique function {f:?} applied to {} args; it must be nullary",
                    args.len()
                );
                Some(UniqueConst {
                    func: *f,
                    tys: tys.clone(),
                })
            }
            _ => None,
        }
    }
}

impl ConstFold {
    /// Whether `enode` is a quantifier or has a child class that
    /// [reaches](ClassData::reaches_forall) one.
    fn reaches_forall(egraph: &EGraph<Symbolic, Self>, enode: &Symbolic) -> bool {
        matches!(enode, Symbolic::Forall(..))
            || enode
                .children()
                .iter()
                .any(|&c| egraph[c].data.reaches_forall())
    }

    fn make_value(egraph: &EGraph<Symbolic, Self>, enode: &Symbolic) -> Data {
        let value = egraph
            .analysis
            .fold_value(enode, |c: &Id| &egraph[*c].data.value);
        // Two different unique constants are different values, so an `==`
        // between them is `false`. Decided here rather than in `fold_value`,
        // which sees only the operands' `Data`: the unique constant is kept
        // beside it (see `ClassData::unique`). A quantifier-guard probe, which
        // folds through `fold_value` alone, does not see this; that loses
        // completeness, never soundness.
        if let (Data::Unknown, Symbolic::Binary(BinOp::Eq, [l, r])) = (&value, enode)
            && let (Some(a), Some(b)) = (&egraph[*l].data.unique, &egraph[*r].data.unique)
            && a != b
        {
            return Data::Known(Literal::Bool(false));
        }
        value
    }

    /// The value `enode` folds to, reading each child's value through `value`.
    /// [`Self::make_value`] reads the children's e-classes; a caller probing a
    /// node that is *not* in the e-graph (the quantifier guard check) passes
    /// its own child values instead, under whatever ids it numbered them by.
    pub(crate) fn fold_value<'a>(&self, enode: &Symbolic, value: impl Fn(&Id) -> &'a Data) -> Data {
        use Data::{Adt, Inconsistent, Known, Unknown};
        match enode {
            Symbolic::Lit(lit) => Known(lit.clone()),

            // A constructor application is exactly its own variant.
            Symbolic::FuncApp(f, tys, _) if let Some(head) = self.head_of(*f) => Adt(AdtData {
                head,
                tys: tys.clone(),
                possible: BTreeSet::from([*f]),
            }),

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
            Symbolic::Coll(..) => Unknown,
            Symbolic::CollLit(..) => Unknown,
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
            unique: egraph.analysis.unique_of(enode),
            generation,
            has_forall: Self::reaches_forall(egraph, enode),
        }
    }

    /// A node keeps the generation it was created at: re-making it (a child's
    /// value changed) must not lower it to its children's current ones, so it
    /// reports its class's.
    fn remake(egraph: &mut EGraph<Symbolic, Self>, enode: &Symbolic, id: Id) -> Self::Data {
        ClassData {
            value: Self::make_value(egraph, enode),
            unique: egraph.analysis.unique_of(enode),
            generation: egraph[id].data.generation,
            has_forall: Self::reaches_forall(egraph, enode),
        }
    }

    /// The value merges as a lattice; the unique constant too (none, one, or
    /// two different ones, which makes the value `Inconsistent`); the generation
    /// is the least of the two; `has_forall` is OR-ed. A value, unique or
    /// `has_forall` change is reported, as a parent reads all three; a
    /// generation change is not: a parent was created at its own generation and
    /// does not depend on its children's later ones.
    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        a.generation = a.generation.min(b.generation);
        // Reported, so the bit reaches the parents.
        let forall = DidMerge(!a.has_forall && b.has_forall, a.has_forall && !b.has_forall);
        a.has_forall |= b.has_forall;
        let value = self.merge_value(&mut a.value, b.value);
        let unique = match (&a.unique, b.unique) {
            (_, None) => DidMerge(false, a.unique.is_some()),
            (None, Some(u)) => {
                a.unique = Some(u);
                DidMerge(true, false)
            }
            (Some(x), Some(y)) if *x == y => DidMerge(false, false),
            // Two different unique constants in one class: they are distinct by
            // declaration, so the class is contradictory. `a` keeps its own
            // constant; `Inconsistent` is the top of the value lattice, so the
            // result does not depend on which of the two it keeps.
            (Some(_), Some(_)) => {
                let changed = a.value != Data::Inconsistent;
                a.value = Data::Inconsistent;
                DidMerge(changed, true)
            }
        };
        value | unique | forall
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
        let (kept, absorbed) = (&egraph[keep].data, &egraph[gone].data);
        let value_changes = kept.value != absorbed.value
            && absorbed.value != Data::Unknown
            && kept.value != Data::Inconsistent;
        // The survivor gains a unique constant, or meets a second one (which
        // makes its value `Inconsistent`).
        let unique_changes = absorbed.unique.is_some()
            && kept.unique != absorbed.unique
            && kept.value != Data::Inconsistent;
        if value_changes || unique_changes {
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

    /// Unique functions `10`, `11`, `12`; `20` is a plain nullary function.
    fn unique_graph() -> Graph {
        let uniques = [10, 11, 12].into_iter().map(FuncId).collect();
        Graph::new(ConstFold::new(Default::default(), Arc::new(uniques)))
    }

    fn konst(f: usize, tys: &[Type]) -> Symbolic {
        Symbolic::FuncApp(FuncId(f), tys.into(), Box::new([]))
    }

    fn eq(l: Id, r: Id) -> Symbolic {
        Symbolic::Binary(BinOp::Eq, [l, r])
    }

    /// Whether the graph has recorded a contradiction (`true == false`).
    fn contradictory(g: &Graph) -> bool {
        let t = g.lookup(Symbolic::Lit(Literal::Bool(true)));
        let f = g.lookup(Symbolic::Lit(Literal::Bool(false)));
        matches!((t, f), (Some(t), Some(f)) if g.find(t) == g.find(f))
    }

    /// `==` between two different unique constants folds to `false`; between
    /// a unique and a plain constant, or a unique and itself, it does not.
    #[test]
    fn eq_of_distinct_unique_constants_folds_false() {
        let mut g = unique_graph();
        let a = g.add(konst(10, &[]));
        let b = g.add(konst(11, &[]));
        let p = g.add(konst(20, &[]));
        let ab = g.add(eq(a, b));
        let ap = g.add(eq(a, p));
        g.rebuild();
        assert_eq!(g[ab].data.known(), Some(&Literal::Bool(false)));
        assert_eq!(g[ap].data.known(), None);
        assert_eq!(
            g[a].data
                .fingerprint()
                .map(|fp| fp.differs_from(&g[b].data.fingerprint().unwrap())),
            Some(true)
        );
        assert!(g[p].data.fingerprint().is_none());
    }

    /// One generic unique function at two instantiations is two constants.
    #[test]
    fn instantiations_are_distinct_constants() {
        let mut g = unique_graph();
        let int = g.add(konst(10, &[Type::Int]));
        let boolean = g.add(konst(10, &[Type::Bool]));
        let same = g.add(konst(10, &[Type::Int]));
        assert_eq!(int, same);
        let e = g.add(eq(int, boolean));
        g.rebuild();
        assert_eq!(g[e].data.known(), Some(&Literal::Bool(false)));
    }

    /// Merging two different unique constants is a contradiction.
    #[test]
    fn merging_distinct_unique_constants_is_inconsistent() {
        let mut g = unique_graph();
        let a = g.add(konst(10, &[]));
        let b = g.add(konst(11, &[]));
        let x = g.add(Symbolic::Fresh(0));
        g.union(x, a);
        g.rebuild();
        assert!(!contradictory(&g));
        g.union(x, b);
        g.rebuild();
        assert!(g[x].data.is_inconsistent());
        assert!(contradictory(&g));
    }

    /// Merging a unique constant with a plain one, or with a literal, is not.
    #[test]
    fn unique_constant_may_equal_plain_terms() {
        let mut g = unique_graph();
        let a = g.add(konst(10, &[]));
        let p = g.add(konst(20, &[]));
        let five = g.add(int(5));
        g.union(a, p);
        g.union(a, five);
        g.rebuild();
        assert!(!contradictory(&g));
        assert_eq!(g[a].data.known(), Some(&Literal::Int(BigInt::from(5))));
        assert_eq!(g[a].data.unique.as_ref().map(|u| u.func), Some(FuncId(10)));
    }

    /// Folding to a literal keeps the unique constant: two unique constants
    /// both equal to `5` are a contradiction, whichever class absorbs which.
    #[test]
    fn literal_does_not_hide_the_unique_constant() {
        for order in [false, true] {
            let mut g = unique_graph();
            let a = g.add(konst(10, &[]));
            let b = g.add(konst(11, &[]));
            let five = g.add(int(5));
            let (first, second) = if order { (a, b) } else { (b, a) };
            g.union(first, five);
            g.rebuild();
            assert!(!contradictory(&g));
            g.union(second, five);
            g.rebuild();
            assert!(contradictory(&g), "order {order}");
        }
    }

    /// An `==` built before its operands became unique constants is
    /// re-folded once they are.
    #[test]
    fn eq_refolds_when_operands_learn_unique_constants() {
        let mut g = unique_graph();
        let x = g.add(Symbolic::Fresh(0));
        let y = g.add(Symbolic::Fresh(1));
        let xy = g.add(eq(x, y));
        g.rebuild();
        assert_eq!(g[xy].data.known(), None);
        let a = g.add(konst(10, &[]));
        let b = g.add(konst(11, &[]));
        g.union(x, a);
        g.union(y, b);
        g.rebuild();
        assert_eq!(g[xy].data.known(), Some(&Literal::Bool(false)));
        assert!(!contradictory(&g));
    }
}
