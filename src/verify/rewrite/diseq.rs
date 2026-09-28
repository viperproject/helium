//! Disequality unit propagation: rules that turn a known-`false` equality into
//! information about its arguments, plus the contra-congruence and
//! distinguishing-observation appliers.

use crate::dhash::HashMap;
use std::sync::Arc;

use egg::{Applier, EGraph, Id, PatternAst, Rewrite, SearchMatches, Searcher, Subst, Symbol, Var};

use crate::verify::analysis::ConstFold;
use crate::verify::lang::{Discriminant, FuncId, Symbolic};
use crate::vmir::{BinOp, Literal, Type};

use super::*;

/// Unit propagation from a **disproven** `==` through `ite` towers — NOT
/// eq-over-ite distribution (see the applier doc). Split by which arm the other
/// operand matches; nested towers unwind one level per saturation iteration (the
/// derived disequality re-enters the `Eq` bucket).
pub(super) fn disequality_unit_prop_rules() -> Vec<Rule> {
    vec![
        Rewrite::new(
            "eq-false-then",
            EqBucketSearcher,
            EqFalseUnitApplier {
                then_side: true,
                memo: Memo::new(),
            },
        )
        .expect("eq-false-then rule"),
        Rewrite::new(
            "eq-false-else",
            EqBucketSearcher,
            EqFalseUnitApplier {
                then_side: false,
                memo: Memo::new(),
            },
        )
        .expect("eq-false-else rule"),
    ]
}

/// Searcher for the guarded eq-over-ite distribution: every e-class holding an
/// `Eq` node, via the `classes_by_op` bucket (no whole-graph scan). One empty
/// subst per class; the applier re-reads the nodes.
pub(super) struct EqBucketSearcher;

impl Searcher<Symbolic, ConstFold> for EqBucketSearcher {
    fn search_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        limit: usize,
    ) -> Vec<SearchMatches<'_, Symbolic>> {
        let Some(classes) = egraph.classes_for_op(&Discriminant::Binary(BinOp::Eq)) else {
            return vec![];
        };
        classes
            .take(limit)
            .map(|eclass| SearchMatches {
                eclass,
                substs: vec![Subst::default()],
                ast: None,
            })
            .collect()
    }

    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _limit: usize,
    ) -> Option<SearchMatches<'_, Symbolic>> {
        egraph[eclass]
            .nodes
            .iter()
            .any(|n| matches!(n, Symbolic::Binary(BinOp::Eq, _)))
            .then(|| SearchMatches {
                eclass,
                substs: vec![Subst::default()],
                ast: None,
            })
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

/// Applier for `eq-false-mirror`: **disequality symmetry**. A disproven
/// `a == b` implies the mirrored `b == a` is false too — land it in the same
/// class so a goal built in the other operand order sees the known boolean.
/// Proven equalities need no mirror — `eq-true-union` merges the args and
/// congruence collapses both orders.
pub(super) struct EqFalseMirrorApplier;

impl Applier<Symbolic, ConstFold> for EqFalseMirrorApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        if known_bool(egraph, eclass) != Some(false) {
            return vec![];
        }
        let mirrors: Vec<[Id; 2]> = egraph[eclass]
            .nodes
            .iter()
            .filter_map(|n| match n {
                Symbolic::Binary(BinOp::Eq, [l, r]) if l != r => Some([*r, *l]),
                _ => None,
            })
            .collect();
        let mut changed = Vec::new();
        for [r, l] in mirrors {
            let mirrored = egraph.add(Symbolic::Binary(BinOp::Eq, [r, l]));
            if egraph.union(eclass, mirrored) {
                changed.push(egraph.find(eclass));
            }
        }
        changed
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

/// Applier for `contra-congruence`: **narrowing contrapositive congruence**.
/// Congruence gives `a⃗ ≡ b⃗ ⟹ f(a⃗) ≡ f(b⃗)`; its contrapositive, from a
/// *disproven* `f(a⃗) == f(b⃗)` with `f` **n-ary**, is the disjunction
/// `a₁≠b₁ ∨ … ∨ aₙ≠bₙ`. An e-graph cannot hold a disjunction of disequalities
/// (it would have to case-split on *which* argument differs — the true/false
/// asymmetry), so we fire **only when the disjunction collapses to a unit**:
/// when every argument pair but one is already proven equal (`aᵢ ≡ bᵢ`), the
/// lone remaining pair must differ — `aⱼ == bⱼ` is false. Sound for **any** `f`,
/// injective or not; unlike [`inj_rule`] the contrapositive of congruence needs
/// no free constructor.
///
/// Connects an unboxed comparison to its boxed source: `value(v) != 0` with the
/// axiom instance `value(cons(0)) ≡ 0` in `0`'s class disproves `v == cons(0)`,
/// which then feeds `eq-false-then/else`'s ite unit propagation and collapses
/// predicate-body disjunction towers via `ite-reduce`.
pub(super) struct ContraCongruenceApplier {
    /// Cost guard, keyed by the function and the canonical argument pair it
    /// disproved (re-deriving is idempotent — the unions no-op).
    pub(super) memo: Memo<(FuncId, [Id; 2])>,
}

impl Applier<Symbolic, ConstFold> for ContraCongruenceApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        if known_bool(egraph, eclass) != Some(false) {
            return vec![];
        }
        let mut contras: Vec<[Id; 2]> = Vec::new();
        for node in &egraph[eclass].nodes {
            let Symbolic::Binary(BinOp::Eq, [l, r]) = node else {
                continue;
            };
            let (l, r) = (egraph.find(*l), egraph.find(*r));
            for lapp in &egraph[l].nodes {
                let Symbolic::FuncApp(lf, ltys, largs) = lapp else {
                    continue;
                };
                for rapp in &egraph[r].nodes {
                    let Symbolic::FuncApp(rf, rtys, rargs) = rapp else {
                        continue;
                    };
                    if lf != rf || ltys != rtys || largs.len() != rargs.len() {
                        continue;
                    }
                    // Narrowing: collect the argument positions not yet proven
                    // equal. Exactly one differing ⟹ the disjunction is a unit and
                    // that pair is disequal. Zero means the apps are congruent
                    // (`ConstFold` handles the conflict); two or more is a
                    // disjunction the e-graph cannot represent, so skip.
                    let mut diff: Option<[Id; 2]> = None;
                    let mut multiple = false;
                    for (la, ra) in largs.iter().zip(rargs.iter()) {
                        let (a, b) = (egraph.find(*la), egraph.find(*ra));
                        if a != b {
                            if diff.is_some() {
                                multiple = true;
                                break;
                            }
                            diff = Some([a, b]);
                        }
                    }
                    if let (false, Some([a, b])) = (multiple, diff)
                        && self.memo.insert((*lf, [a, b]))
                    {
                        contras.push([a, b]);
                    }
                }
            }
        }
        let mut changed = Vec::new();
        for [a, b] in contras {
            let eq = egraph.add(Symbolic::Binary(BinOp::Eq, [a, b]));
            let false_ = egraph.add(Symbolic::Lit(Literal::Bool(false)));
            if egraph.union(eq, false_) {
                changed.push(egraph.find(eq));
            }
        }
        changed
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

/// Applier for `distinguishing-observation`: **congruence, contrapositive at the
/// application**. Congruence gives `a ≡ b ⟹ f(a) ≡ f(b)`. Contrapositively, if
/// some `f` has `f(a)` and `f(b)` sitting in classes whose [`Fingerprint`](crate::verify::analysis::Fingerprint)s
/// differ, then `a ≡ b` is impossible — so an **undecided** `a == b` is `false`.
///
/// Sound for **any** `f`, injective or not: this is the contrapositive of
/// congruence, not of injectivity. Injectivity is merely the usual *reason* an
/// observation exists. Prusti encodes a primitive snapshot (`s_Int_isize`) as a
/// **domain** with a `cons`/`value` retraction rather than an `adt`, so
/// `s_Int_isize_cons(1)` and `s_Int_isize_cons(4)` carry no free-constructor
/// distinctness — but `ax_value` puts `value(cons(1)) ≡ 1` and
/// `value(cons(4)) ≡ 4` in the graph, and those two literals separate them. That
/// is what an enum snapshot tower's guards (`snap == cons(k)`) need pinned.
///
/// The mirror image of [`ContraCongruenceApplier`], which walks the *other*
/// direction: from an already-disproven `f(a⃗) == f(b⃗)` to an argument
/// disequality. Together they close the loop between arguments and applications.
///
/// **Unary observations only.** An n-ary `f` would additionally need every other
/// argument pair proven equal, which is the narrowing search `contra-congruence`
/// already pays for; the retraction pairs this exists for are all unary, so the
/// extra generality would be cost with no measured benefit.
///
/// Success pins the class to `false`, and the `known_bool` gate below then skips
/// it forever. A **failed** scan is the hot path: it finds nothing, and without a
/// memo it repeats in full on every saturation iteration, for every undecided
/// `Eq` class. On a payload enum that dominates everything else — an N=5 grid file
/// spent 11.9s here for 266 unions (45ms per union), and ablating the rule took
/// the file from 18.4s to 0.3s.
///
/// So a failed scan is memoized on `(l, r, |parents(l)|, |parents(r)|)`. Parent
/// lists only grow, so the lengths are a free monotone version stamp: the same
/// pair is re-scanned exactly when either side has gained a parent since the last
/// failure, which is when a new observation can have appeared.
///
/// **This trades completeness, not soundness.** A fingerprint can sharpen without
/// either class gaining a parent (the application's class merges with a literal),
/// and that refinement is not re-scanned. Missing a disproof only fails to prove
/// something — the safe direction — and the corpora pin that nothing regressed.
pub(super) struct DistinguishingObsApplier {
    /// Failed scans, keyed by the pair and its parent-count stamp.
    pub(super) memo: Memo<(Id, Id, usize, usize)>,
}

thread_local! {
    /// Observation maps built during the current saturation run, keyed by the
    /// class and its parent-count stamp. Scoped to one run (see
    /// [`new_scan_generation`]) so a map built on one e-graph can never be read
    /// on another — ground and its clones share ids, and a clone's extra unions
    /// give the same id a different observation set.
    static OBS_CACHE: std::cell::RefCell<HashMap<(Id, usize), Arc<Observations>>> =
        std::cell::RefCell::new(HashMap::default());
}

/// Start a new observation-cache generation. Called once per saturation run: the
/// cached maps describe the graph that run is walking and nothing else.
pub(crate) fn new_scan_generation() {
    OBS_CACHE.with(|c| c.borrow_mut().clear());
}

/// The unary applications of one class: `(f, type args)` to the class of `f(x)`.
pub(super) type Observations = HashMap<(FuncId, Vec<Type>), Id>;

/// Every `(f, tys)` this class is the sole argument of, mapped to the class of
/// that application. The observation map of [`DistinguishingObsApplier`].
///
/// A map rather than a list: for a *unary* `f` and a fixed argument class, egg's
/// congruence memo already collapses every `f(x)` into one class, so a key cannot
/// collide. Building both sides as maps turns the pairing below into a hash join —
/// it used to be a nested loop over both observation lists. Borrows the type slice
/// instead of cloning it; the whole scan holds the graph immutably.
pub(super) fn unary_observations(egraph: &EGraph<Symbolic, ConstFold>, x: Id) -> Arc<Observations> {
    let x = egraph.find(x);
    // Parent lists only grow, so the count is a free monotone version stamp —
    // the same one the failure memo uses. An enum match asks `s == cons(k)` once
    // per variant, and every one of those pairs used to rebuild `s`'s map by
    // walking its whole parent list: 83% of the scans on `enum_v5_p1` were a
    // repeat of a key already built, and the walk is what the rule's time is
    // (937k parents walked on its scratch graphs alone).
    let key = (x, egraph[x].parents().count());
    if let Some(hit) = OBS_CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return hit;
    }
    let mut out = Observations::default();
    for p in egraph[x].parents() {
        let p = egraph.find(p);
        for node in &egraph[p].nodes {
            if let Symbolic::FuncApp(f, tys, args) = node
                && args.len() == 1
                && egraph.find(args[0]) == x
            {
                out.insert((*f, tys.to_vec()), p);
            }
        }
    }
    let out = Arc::new(out);
    OBS_CACHE.with(|c| c.borrow_mut().insert(key, Arc::clone(&out)));
    out
}

impl Applier<Symbolic, ConstFold> for DistinguishingObsApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        // Only undecided equalities: a proven one is congruence's job, and a
        // disproven one is already what `contra-congruence` consumes.
        if known_bool(egraph, eclass).is_some() {
            return vec![];
        }
        // Canonical operand pairs of this class, deduplicated: several `Eq` nodes
        // in one class routinely name the same pair, and each used to pay for its
        // own pair of observation scans.
        let mut pairs: Vec<(Id, Id)> = Vec::new();
        for node in &egraph[eclass].nodes {
            let Symbolic::Binary(BinOp::Eq, [l, r]) = node else {
                continue;
            };
            let (l, r) = (egraph.find(*l), egraph.find(*r));
            if l != r && !pairs.contains(&(l, r)) {
                pairs.push((l, r));
            }
        }

        let mut disprove = false;
        'outer: for (l, r) in pairs {
            // Version stamp: parent lists only grow, so equal lengths mean no new
            // observation can have shown up on either side since the last failure.
            let stamp = (
                l,
                r,
                egraph[l].parents().count(),
                egraph[r].parents().count(),
            );
            if !self.memo.insert(stamp) {
                continue;
            }
            let obs_l = unary_observations(egraph, l);
            if obs_l.is_empty() {
                continue;
            }
            let obs_r = unary_observations(egraph, r);
            // Hash join on the observation key instead of the old nested loop.
            for (key, app_r) in obs_r.iter() {
                let Some(app_l) = obs_l.get(key) else {
                    continue;
                };
                if app_l == app_r {
                    continue;
                }
                let (Some(fp_l), Some(fp_r)) = (
                    egraph[*app_l].data.fingerprint(),
                    egraph[*app_r].data.fingerprint(),
                ) else {
                    continue;
                };
                if fp_l.differs_from(&fp_r) {
                    disprove = true;
                    break 'outer;
                }
            }
        }
        if !disprove {
            return vec![];
        }
        let false_ = egraph.add(Symbolic::Lit(Literal::Bool(false)));
        if egraph.union(eclass, false_) {
            vec![egraph.find(eclass)]
        } else {
            vec![]
        }
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

/// Applier for `eq-false-then`/`eq-false-else`: **unit propagation through an
/// `ite` operand** of a **disproven** equality. From `(ite c x y) == z` proven
/// `false` (an assumed `d != tag`) with one arm already equal to `z`:
///
/// - `then_side` (`x ≡ z`) ⟹ `c = false` (taking the true branch would
///   satisfy the equality) `∧ (y == z) = false` (the value is the false
///   branch's)
/// - else side (`y ≡ z`) ⟹ `c = true ∧ (x == z) = false` (mirrored)
///
/// The second consequence recurses down a nested ite tower (a 3+-variant enum
/// discriminator): the derived disequality lands back in the `Eq` bucket, so the
/// next saturation iteration picks it up, one level per iteration. Two assumed
/// disequalities pinning `c` both ways make the graph inconsistent — which is
/// enum-match exhaustiveness.
///
/// Deliberately **not** syntactic distribution
/// (`(ite c x y) == z ⇒ ite c (x==z) (y==z)`): the pattern form cross-multiplies
/// the ite guard towers `implication()` builds (~75x node blowup). This
/// derivation adds at most one `Eq` node per step and otherwise only unions.
pub(super) struct EqFalseUnitApplier {
    /// Which arm of the ite the other operand must match (see rule doc).
    pub(super) then_side: bool,
    /// Cost guard: one derivation per canonical `[c, x, y, z]` quadruple
    /// (re-deriving is idempotent — the unions no-op).
    pub(super) memo: Memo<[Id; 4]>,
}

impl Applier<Symbolic, ConstFold> for EqFalseUnitApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        if known_bool(egraph, eclass) != Some(false) {
            return vec![];
        }
        // Each derivation: pin `cond` to `cond_val`, and disprove the other
        // arm's comparison `other == z`.
        let mut derivs: Vec<(Id, bool, Id, Id)> = Vec::new();
        for node in &egraph[eclass].nodes {
            let Symbolic::Binary(BinOp::Eq, [l, r]) = node else {
                continue;
            };
            for (ite_side, z) in [(*l, *r), (*r, *l)] {
                let (ite_side, z) = (egraph.find(ite_side), egraph.find(z));
                for inner in &egraph[ite_side].nodes {
                    let Symbolic::Ite([c, x, y]) = inner else {
                        continue;
                    };
                    let (c, x, y) = (egraph.find(*c), egraph.find(*x), egraph.find(*y));
                    let (matched, cond_val, other) = if self.then_side {
                        (x == z, false, y)
                    } else {
                        (y == z, true, x)
                    };
                    if !matched || !self.memo.insert([c, x, y, z]) {
                        continue;
                    }
                    derivs.push((c, cond_val, other, z));
                }
            }
        }
        let mut changed = Vec::new();
        for (cond, cond_val, other, z) in derivs {
            let lit = egraph.add(Symbolic::Lit(Literal::Bool(cond_val)));
            if egraph.union(cond, lit) {
                changed.push(egraph.find(cond));
            }
            let other_eq = egraph.add(Symbolic::Binary(BinOp::Eq, [other, z]));
            let false_ = egraph.add(Symbolic::Lit(Literal::Bool(false)));
            if egraph.union(other_eq, false_) {
                changed.push(egraph.find(other_eq));
            }
        }
        changed
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

/// Applier for `eq-true-union`: when a matched `Eq` e-class is proven `true`,
/// union its two argument e-classes. Sound (proven `a == b` ⇒ same value) and
/// size-non-increasing (only merges existing e-classes, never adds nodes).
pub(super) struct UnionEqArgs {
    pub(super) a: Var,
    pub(super) b: Var,
}

impl Applier<Symbolic, ConstFold> for UnionEqArgs {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        // Only fire once the equality is actually known true. `Assume` seeds
        // this by unioning the `Eq` e-class with `Lit(true)`, which
        // `ConstFold` records as `Data::Known(Bool(true))`.
        if !matches!(egraph[eclass].data.known(), Some(Literal::Bool(true))) {
            return vec![];
        }
        let a = subst[self.a];
        let b = subst[self.b];
        if egraph.union(a, b) {
            vec![egraph.find(a)]
        } else {
            vec![]
        }
    }

    fn vars(&self) -> Vec<Var> {
        vec![self.a, self.b]
    }
}
