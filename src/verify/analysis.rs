use crate::dhash::HashMap;
use std::cell::{Cell, RefCell};
use std::sync::Arc;

use egg::{Analysis, DidMerge, EGraph, Id, Justification};
use num::{BigInt, BigRational};

use crate::verify::interval::Interval;
use crate::verify::lang::{FuncId, Symbolic};
use crate::verify::stats;
use crate::vmir::{BinOp, Literal, MemberId, Type};

/// Const-fold analysis data: a lattice over an e-class's folded value.
/// **Type-free**: only the literal is tracked (types are reconstructed for
/// visualization from the side oracle in `verify::context`).
///
/// - `Unknown`: not (yet) a constant.
/// - `Range(iv, depth)`: an `Int` class whose value lies in `iv` — bounded on
///   at least one side, and never a single point (that is `Known`). `depth` is
///   how the bounds were derived (see [`Depth`]); it is not part of the value,
///   so two ranges compare equal by their intervals alone. See
///   [`crate::verify::interval`].
/// - `Known(lit)`: folds to `lit`.
/// - `Ctor(f, tys)`: the e-class holds an application of ADT constructor `f` at
///   instantiation `tys`. This gives **constructor distinctness** with no `tag`
///   term and no O(variants) axioms — a variant clash is a lattice conflict, and
///   it is detected even when the program never mentions a discriminator.
/// - `Inconsistent`: two **same-typed** literals of differing value were merged
///   (`true == false`, `5 == 6`), or two **different constructors of one ADT head
///   at one instantiation** (constructors are free, so `Cons(..) == Nil` is a
///   contradiction) — the e-class, and thus the whole verification unit, is
///   contradictory. Merging across *different* types is instead a verifier panic:
///   a type error, not a fact about the program.
#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    Unknown,
    Range(Interval, Depth),
    Known(Literal),
    Ctor(FuncId, Box<[Type]>),
    Inconsistent,
}

/// The length of the longest chain of derived steps — an arithmetic node
/// re-evaluated over its operands' ranges, or a comparison narrowing one operand
/// by the other's range — behind a range's bounds. A literal is at depth `0`,
/// so a comparison against one narrows at depth `1`; a union adds no step.
///
/// It is what makes narrowing terminate without costing precision where none
/// is at stake. Narrowing is a descending chain, and on a cyclic e-graph that
/// chain need not end: in a class holding both `x` and `x - 1` every bound
/// narrows the next. But a chain of derived steps longer than the number of
/// e-node ids must pass through some class twice, i.e. it runs round a cycle —
/// the Bellman-Ford argument for negative cycles. Up to that length a derived
/// step narrows exactly; beyond it, only the standard narrowing operator
/// applies (an unbounded side may become bounded, a bounded one stays; see
/// [`Interval::narrow`]). Over the integers a bound that keeps moving leaves
/// every value behind, so such a cycle only exists in a class with no value at
/// all: what stops is the discovery of a contradiction, never a verdict that
/// does not already follow from one.
///
/// Compares equal to every other depth, so a change of depth alone is not a
/// change of the data (it re-evaluates no parents).
#[derive(Debug, Clone, Copy, Default)]
pub struct Depth(u32);

impl PartialEq for Depth {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Depth {
    /// One derived step past the deepest of `inputs`.
    fn after(inputs: &[&Data]) -> Depth {
        let deepest = inputs.iter().map(|d| d.depth().0).max().unwrap_or(0);
        Depth(deepest.saturating_add(1))
    }
}

/// What `ConstFold` has pinned an e-class's *value* to, when it has pinned one:
/// a folded literal, or an ADT constructor identity. Two classes carrying
/// fingerprints that [`Fingerprint::differs_from`] separates cannot denote the
/// same value, so a rule may use the pair to refute one of them.
///
/// The constructor half rests on **free constructors** — distinct variants of one
/// ADT head at one instantiation are disjoint. That is the same premise
/// [`ConstFold::make`] uses to fold a `Ctor == Ctor` comparison to `false` and
/// [`ConstFold::merge`] uses to declare a class [`Data::Inconsistent`] (see the
/// [`Data`] doc). If the ADT encoding ever gains non-free constructors, all three
/// sites break together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fingerprint<'a> {
    Lit(&'a Literal),
    Ctor(FuncId, &'a [Type]),
}

impl Fingerprint<'_> {
    /// Whether these two pinned values are provably different.
    ///
    /// Conservative in every direction it is not sure about:
    ///
    /// * Two constructors are separated only **within one instantiation**. The
    ///   type args are part of the operator's identity in the polymorphic
    ///   e-graph, so `List[Int]::Nil` vs `List[Bool]::Nil` is a type error, not a
    ///   fact about the program — report "not different" and let the merge
    ///   panic in [`ConstFold::merge`] catch a real violation.
    /// * The same constructor at different arguments is **not** separated:
    ///   the fingerprint is the constructor's identity, not the whole term, so
    ///   `c(x)` vs `c(y)` reports "not different" even when `x ≢ y`. Losing that
    ///   is incompleteness, never unsoundness.
    /// * Mixing a literal with a constructor cannot arise (that merge is a type
    ///   error panic) and reports "not different" if it somehow does.
    pub fn differs_from(&self, other: &Self) -> bool {
        match (self, other) {
            (Fingerprint::Lit(a), Fingerprint::Lit(b)) => a != b,
            (Fingerprint::Ctor(f, ftys), Fingerprint::Ctor(g, gtys)) => f != g && ftys == gtys,
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

    /// This e-class's pinned value, as a comparison key. `None` for `Unknown`
    /// (nothing pinned) and for `Inconsistent` (the graph is already
    /// contradictory; other machinery handles it).
    pub fn fingerprint(&self) -> Option<Fingerprint<'_>> {
        match self {
            Data::Known(lit) => Some(Fingerprint::Lit(lit)),
            Data::Ctor(f, tys) => Some(Fingerprint::Ctor(*f, tys)),
            Data::Unknown | Data::Range(..) | Data::Inconsistent => None,
        }
    }

    /// Whether this e-class merged conflicting same-typed literals.
    pub fn is_inconsistent(&self) -> bool {
        matches!(self, Data::Inconsistent)
    }

    fn known_bool(&self) -> Option<bool> {
        match self {
            Data::Known(Literal::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    /// The interval of an `Int` class: its range, or the point it folds to.
    /// `None` for anything else, including `Unknown` — which may be an `Int`
    /// class not bounded yet, so a caller that knows the sort reads it as
    /// [`Interval::TOP`].
    pub fn interval(&self) -> Option<Interval> {
        match self {
            Data::Range(iv, _) => Some(*iv),
            Data::Known(Literal::Int(n)) => Some(Interval::of(n)),
            _ => None,
        }
    }

    fn depth(&self) -> Depth {
        match self {
            Data::Range(_, depth) => *depth,
            _ => Depth(0),
        }
    }

    /// The data of an `Int` class known to lie in `iv`.
    fn of_interval(iv: Interval, depth: Depth) -> Data {
        match iv.as_point() {
            Some(v) => Data::Known(Literal::Int(BigInt::from(v))),
            None if iv.is_top() => Data::Unknown,
            None => Data::Range(iv, depth),
        }
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
    /// ADTs), which simply disables the distinctness lattice.
    ctors: Arc<HashMap<FuncId, MemberId>>,
    /// While a bucket rule is applied (`rewrite::PerClass`), the classes whose
    /// nodes read differently after each union; see [`Analysis::pre_union`].
    pub(crate) union_log: RefCell<Option<Vec<Id>>>,
    /// The number of e-node ids minted so far: the longest a chain of derived
    /// steps can be without running round a cycle (see [`Depth`]). Kept by
    /// `make`, which sees every new node.
    ids: Cell<usize>,
    /// Set by `pre_union`, cleared by the `modify` that ends the union. A union
    /// meets the two classes' facts exactly; a node re-evaluated inside
    /// `rebuild` is a derived step (see `merge`).
    in_union: Cell<bool>,
    /// Set by `merge` when a re-evaluation inside `rebuild` decides a boolean
    /// class, so its `modify` queues the comparisons it holds. A union that
    /// decides one is seen by `pre_union` instead.
    decided_by_remake: Cell<bool>,
    /// Comparisons decided since their operands were last narrowed by them:
    /// `(op, lhs, rhs, holds)`. Drained by [`ConstFold::narrow_decided`].
    decided: RefCell<Vec<(BinOp, Id, Id, bool)>>,
    /// Whether `decided` is being drained: narrowing an operand re-enters
    /// `modify`, which must leave the queue to the loop already draining it.
    draining: Cell<bool>,
}

impl ConstFold {
    pub fn new(ctors: Arc<HashMap<FuncId, MemberId>>) -> Self {
        Self {
            ctors,
            ..Default::default()
        }
    }

    /// The ADT head of `f`, if `f` is a constructor.
    fn head_of(&self, f: FuncId) -> Option<MemberId> {
        self.ctors.get(&f).copied()
    }

    /// The meet of two classes' data: what is known of a class holding both.
    /// `a` is the class's current data. Unless `exact`, `b` is derived, and
    /// past the cycle bound of [`Depth`] narrows `a` only by the standard
    /// narrowing operator.
    fn meet(&self, a: &Data, b: &Data, exact: bool) -> Data {
        use Data::{Ctor, Inconsistent, Known, Range, Unknown};
        match (a, b) {
            (Inconsistent, _) | (_, Inconsistent) => Inconsistent,
            (Unknown, x) | (x, Unknown) => x.clone(),
            (Ctor(f, ftys), Ctor(g, gtys)) => {
                // Distinctness. Only comparable within one instantiation: the type
                // args are part of the operator's identity (the polymorphic
                // e-graph keeps `List[Int]::Nil` and `List[Bool]::Nil` apart by
                // discriminant), so a clash across instantiations is a type error,
                // not a contradiction.
                let (fh, gh) = (self.head_of(*f), self.head_of(*g));
                if fh != gh || ftys != gtys {
                    panic!(
                        "type error: merged constructors of different ADTs or \
                         instantiations: {f:?}{ftys:?} vs {g:?}{gtys:?}"
                    );
                }
                if f == g {
                    a.clone()
                } else {
                    // Distinct variants of one ADT, same instantiation: free
                    // constructors are disjoint, so this class is contradictory.
                    Inconsistent
                }
            }
            (Ctor(..), Known(lit)) | (Known(lit), Ctor(..)) => {
                panic!("type error: merged an ADT constructor with a literal: {lit:?}")
            }
            (Ctor(..), Range(iv, _)) | (Range(iv, _), Ctor(..)) => {
                panic!("type error: merged an ADT constructor with an integer range {iv:?}")
            }
            (Known(x), Known(y)) => {
                if x == y {
                    a.clone()
                } else if same_type(x, y) {
                    // Same-typed conflict ⇒ contradiction (not a panic).
                    Inconsistent
                } else {
                    panic!("type error: merged literals of different types: {x:?} vs {y:?}");
                }
            }
            (Range(iv, _), Known(Literal::Int(n))) | (Known(Literal::Int(n)), Range(iv, _)) => {
                if iv.contains(n) {
                    Known(Literal::Int(n.clone()))
                } else {
                    Inconsistent
                }
            }
            (Range(iv, _), Known(lit)) | (Known(lit), Range(iv, _)) => {
                panic!("type error: merged an integer range {iv:?} with {lit:?}")
            }
            (Range(r, rd), Range(q, qd)) => match r.meet(q) {
                None => Inconsistent,
                Some(m) => {
                    let acyclic = (qd.0 as usize) <= self.ids.get();
                    let m = if exact || acyclic { m } else { r.narrow(&m) };
                    let depth = if m == *r {
                        *rd
                    } else if m == *q {
                        *qd
                    } else {
                        Depth(rd.0.max(qd.0))
                    };
                    Data::of_interval(m, depth)
                }
            },
        }
    }

    /// Queue the comparisons of `class`, which now `holds` (or fails), to narrow
    /// their operands. An `==` that holds is left to the union it implies.
    fn queue_comparisons(egraph: &EGraph<Symbolic, Self>, class: Id, holds: bool) {
        let mut queue = egraph.analysis.decided.borrow_mut();
        for node in &egraph[class].nodes {
            if let Symbolic::Binary(op, [x, y]) = node
                && (*op == BinOp::LtI || (*op == BinOp::Eq && !holds))
            {
                queue.push((*op, *x, *y, holds));
            }
        }
    }

    /// Narrow the operands of every queued decided comparison. Each narrowing
    /// re-evaluates the operand's parents, which re-queues the comparisons over
    /// it (see [`Analysis::remake`]); [`Depth`] makes that chain finite.
    fn narrow_decided(egraph: &mut EGraph<Symbolic, Self>) {
        if egraph.analysis.draining.replace(true) {
            return;
        }
        loop {
            let next = egraph.analysis.decided.borrow_mut().pop();
            let Some((op, x, y, holds)) = next else {
                break;
            };
            let (x, y) = (egraph.find(x), egraph.find(y));
            let iv = |egraph: &EGraph<Symbolic, Self>, c: Id| {
                egraph[c].data.interval().unwrap_or(Interval::TOP)
            };
            match op {
                BinOp::LtI => {
                    let narrowed = iv(egraph, x).below(&iv(egraph, y), holds);
                    Self::narrow_to(egraph, x, narrowed, y);
                    let (x, y) = (egraph.find(x), egraph.find(y));
                    let narrowed = iv(egraph, y).above(&iv(egraph, x), holds);
                    Self::narrow_to(egraph, y, narrowed, x);
                }
                // `x != k` cuts `k` off an endpoint of `x`.
                BinOp::Eq => {
                    let point = |egraph: &EGraph<Symbolic, Self>, c: Id| {
                        egraph[c].data.interval().and_then(|i| i.as_point())
                    };
                    if let Some(k) = point(egraph, y) {
                        let narrowed = iv(egraph, x).without(k);
                        Self::narrow_to(egraph, x, narrowed, y);
                    }
                    let (x, y) = (egraph.find(x), egraph.find(y));
                    if let Some(k) = point(egraph, x) {
                        let narrowed = iv(egraph, y).without(k);
                        Self::narrow_to(egraph, y, narrowed, x);
                    }
                }
                _ => {}
            }
        }
        egraph.analysis.draining.set(false);
    }

    /// Narrow `class` (an `Int` class) to `to` (`None`: no value is left), a
    /// derived step from the range of `by`.
    fn narrow_to(egraph: &mut EGraph<Symbolic, Self>, class: Id, to: Option<Interval>, by: Id) {
        let depth = Depth::after(&[&egraph[by].data]);
        let cur = &egraph[class].data;
        let to = to.map_or(Data::Inconsistent, |iv| Data::of_interval(iv, depth));
        let next = egraph.analysis.meet(cur, &to, false);
        if next != *cur {
            stats::bump(|s| s.range_narrowings += 1);
            egraph.set_analysis_data(class, next);
        }
    }
}

/// The data of an arithmetic or comparison node whose operands are not both
/// literals, from their ranges. `Unknown` unless one operand has one (an
/// unbounded `Int` operand is [`Interval::TOP`]).
fn binary_from_ranges(op: BinOp, l: &Data, r: &Data) -> Data {
    let (li, ri) = (l.interval(), r.interval());
    if li.is_none() && ri.is_none() {
        return Data::Unknown;
    }
    let (lv, rv) = (li.unwrap_or(Interval::TOP), ri.unwrap_or(Interval::TOP));
    let range = |iv| Data::of_interval(iv, Depth::after(&[l, r]));
    match op {
        BinOp::AddI => range(lv.add(&rv)),
        BinOp::SubI => range(lv.sub(&rv)),
        BinOp::MulI => range(lv.mul(&rv)),
        BinOp::Mod => lv.euclid_mod(&rv).map_or(Data::Unknown, range),
        BinOp::DivI => lv.euclid_div(&rv).map_or(Data::Unknown, range),
        BinOp::LtI => lv
            .lt(&rv)
            .map_or(Data::Unknown, |b| Data::Known(Literal::Bool(b))),
        // Both sides are `Int` only when both have a range; disjoint ranges
        // cannot hold equal values.
        BinOp::Eq if li.is_some() && ri.is_some() && lv.meet(&rv).is_none() => {
            Data::Known(Literal::Bool(false))
        }
        _ => Data::Unknown,
    }
}

impl Analysis<Symbolic> for ConstFold {
    type Data = Data;

    fn make(egraph: &mut EGraph<Symbolic, Self>, enode: &Symbolic, _id: Id) -> Self::Data {
        use Data::{Ctor, Inconsistent, Known, Range, Unknown};
        egraph.analysis.ids.set(egraph.nodes().len() + 1);
        match enode {
            Symbolic::Lit(lit) => Known(lit.clone()),

            Symbolic::FuncApp(f, tys, _) if egraph.analysis.head_of(*f).is_some() => {
                Ctor(*f, tys.clone())
            }

            // A quantifier is opaque to constant folding: its truth is decided by
            // the instantiation rule (guarded merges with `true`), never by its
            // payload or its capture children.
            Symbolic::Fresh(_) | Symbolic::FuncApp(..) | Symbolic::Forall(..) => Unknown,

            Symbolic::RealCast(c) => match &egraph[*c].data {
                Known(Literal::Int(n)) => Known(Literal::Real(BigRational::from(n.clone()))),
                Known(_) | Ctor(..) => unreachable!("RealCast operand must be an integer literal"),
                Inconsistent => Inconsistent,
                Unknown | Range(..) => Unknown,
            },

            Symbolic::Binary(op, [l, r]) => match (&egraph[*l].data, &egraph[*r].data) {
                (Inconsistent, _) | (_, Inconsistent) => Inconsistent,
                // Disequality, the other half of what the SMT `tag` encoding buys:
                // distinct constructors of one ADT are distinct values, so an `==`
                // between them folds to `false` outright. No `tag` term and no
                // `tag_bounds` axiom needed — the constructor identity is right
                // there in the operand's e-class.
                (Ctor(f, ftys), Ctor(g, gtys))
                    if *op == BinOp::Eq
                        && f != g
                        && ftys == gtys
                        && egraph.analysis.head_of(*f) == egraph.analysis.head_of(*g) =>
                {
                    Known(Literal::Bool(false))
                }
                (Known(lv), Known(rv)) => eval_binary(*op, lv, rv).map_or(Unknown, Known),
                (ld, rd) => binary_from_ranges(*op, ld, rd),
            },

            Symbolic::Ite([c, t, e]) => match &egraph[*c].data {
                Known(Literal::Bool(true)) => egraph[*t].data.clone(),
                Known(Literal::Bool(false)) => egraph[*e].data.clone(),
                Known(_) | Ctor(..) | Range(..) => {
                    unreachable!("Condition of ITE must be a boolean literal")
                }
                Inconsistent => Inconsistent,
                // Undecided: the value is one of the arms, so their hull
                // bounds it. A class with a range has the sort `Int`, and so
                // has its sibling arm.
                Unknown => match (egraph[*t].data.interval(), egraph[*e].data.interval()) {
                    (Some(ti), Some(ei)) => Data::of_interval(
                        ti.hull(&ei),
                        Depth::after(&[&egraph[*t].data, &egraph[*e].data]),
                    ),
                    _ => Unknown,
                },
            },
        }
    }

    /// A decided comparison is re-evaluated whenever an operand's data changes;
    /// that is when the other operand may narrow further.
    fn remake(egraph: &mut EGraph<Symbolic, Self>, enode: &Symbolic, id: Id) -> Self::Data {
        let data = Self::make(egraph, enode, id);
        if let Symbolic::Binary(op, [x, y]) = enode
            && let Some(holds) = egraph[id].data.known_bool()
            && (*op == BinOp::LtI || (*op == BinOp::Eq && !holds))
        {
            egraph
                .analysis
                .decided
                .borrow_mut()
                .push((*op, *x, *y, holds));
            Self::narrow_decided(egraph);
        }
        data
    }

    /// Meet the data. In a union both sides are facts about the class, met
    /// exactly; a node re-evaluated inside `rebuild` is a derived step, which
    /// narrows exactly only within the cycle bound of [`Depth`].
    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        let in_union = self.in_union.get();
        let m = self.meet(a, &b, in_union);
        if !in_union && a.known_bool().is_none() && m.known_bool().is_some() {
            self.decided_by_remake.set(true);
        }
        let did = DidMerge(m != *a, m != b);
        *a = m;
        did
    }

    /// Mark the union for `merge`, and queue the comparisons it decides: those
    /// of an undecided class merging into a decided one.
    ///
    /// When `union_log` is on, also record the classes whose nodes will read
    /// differently after this union: the root that survives (it gains nodes),
    /// the parents of the class absorbed (a child's identity changes), and the
    /// parents of the survivor if its data changes. egg keeps the class with more
    /// parents as the root, the first on a tie; were that to change, only which
    /// classes get walked again would, never what a walk may derive.
    fn pre_union(egraph: &EGraph<Symbolic, Self>, id1: Id, id2: Id, _: &Option<Justification>) {
        let (a, b) = (egraph.find(id1), egraph.find(id2));
        if a == b {
            return;
        }
        egraph.analysis.in_union.set(true);
        let (da, db) = (&egraph[a].data, &egraph[b].data);
        if !da.is_inconsistent() && !db.is_inconsistent() {
            match (da.known_bool(), db.known_bool()) {
                (Some(holds), None) => Self::queue_comparisons(egraph, b, holds),
                (None, Some(holds)) => Self::queue_comparisons(egraph, a, holds),
                _ => {}
            }
        }
        let mut log = egraph.analysis.union_log.borrow_mut();
        let Some(log) = log.as_mut() else {
            return;
        };
        let (keep, gone) = if egraph[a].parents().len() < egraph[b].parents().len() {
            (b, a)
        } else {
            (a, b)
        };
        log.push(keep);
        log.extend(egraph[gone].parents());
        let (kept, absorbed) = (&egraph[keep].data, &egraph[gone].data);
        if kept != absorbed && *absorbed != Data::Unknown && *kept != Data::Inconsistent {
            log.extend(egraph[keep].parents());
        }
    }

    fn modify(egraph: &mut EGraph<Symbolic, Self>, id: Id) {
        egraph.analysis.in_union.set(false);
        if egraph.analysis.decided_by_remake.replace(false)
            && let Some(holds) = egraph[id].data.known_bool()
        {
            Self::queue_comparisons(egraph, id, holds);
        }
        match egraph[id].data.clone() {
            Data::Known(lit) => {
                let lit_id = egraph.add(Symbolic::Lit(lit));
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
            Data::Unknown | Data::Range(..) | Data::Ctor(..) => {}
        }
        Self::narrow_decided(egraph);
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
