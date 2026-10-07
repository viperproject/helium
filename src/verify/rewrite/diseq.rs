//! Disequality unit propagation: rules that turn a known-`false` equality into
//! information about its arguments, plus the contra-congruence and
//! distinguishing-observation appliers.

use crate::dhash::HashMap;
use std::sync::Arc;

use egg::{
    Applier, Changes, EGraph, Id, Pattern, PatternAst, SearchMatches, Searcher, Subst, Symbol, Var,
};

use crate::verify::analysis::ConstFold;
use crate::verify::lang::{Discriminant, FuncId, Symbolic};
use crate::vmir::{BinOp, Literal, Type};

use super::*;

/// Unit propagation from a **disproven** `==` through `ite` towers — NOT
/// eq-over-ite distribution (see the applier doc). Split by which arm the other
/// operand matches; nested towers unwind one level per pass (the derived
/// disequality is a new `Eq` match).
pub(super) fn disequality_unit_prop_rules() -> Vec<Rule> {
    let rule = |name: &str, then_side: bool| {
        let applier = EqFalseUnitApplier {
            then_side,
            memo: Memo::new(),
        };
        let searcher = EqSearcher::new(EqReads::OperandItes, EqGate::Disproven);
        local_fixpoint_rule(name, searcher, applier)
    };
    vec![rule("eq-false-then", true), rule("eq-false-else", false)]
}

/// What a disequality rule's match reads besides its `Eq` node, the node's class
/// data and its operands' identities — so what else must make it search again.
#[derive(Clone, Copy)]
pub(super) enum EqReads {
    Node,
    /// The `ite` nodes in the operands' classes ([`EqFalseUnitApplier`]).
    OperandItes,
    /// The function applications in the operands' classes ([`ContraCongruenceApplier`]).
    OperandApps,
    /// The unary applications over the operands and their classes' data
    /// ([`DistinguishingObsApplier`]).
    Observations,
}

/// Which `Eq` classes a disequality rule derives from, by their known boolean.
#[derive(Clone, Copy)]
pub(super) enum EqGate {
    Disproven,
    Undecided,
}

impl EqGate {
    fn admits(self, egraph: &EGraph<Symbolic, ConstFold>, class: Id) -> bool {
        match self {
            EqGate::Disproven => known_bool(egraph, class) == Some(false),
            EqGate::Undecided => known_bool(egraph, class).is_none(),
        }
    }
}

/// Searcher for the disequality rules: each `Eq` node, as `(== ?l ?r)`, in a class
/// the [`EqGate`] admits. Searching changes, it also finds the `Eq` nodes whose
/// match reads something else that changed (see [`EqReads`]).
#[derive(Clone)]
pub(super) struct EqSearcher {
    reads: EqReads,
    gate: EqGate,
    pattern: Pattern<Symbolic>,
}

impl EqSearcher {
    pub(super) fn new(reads: EqReads, gate: EqGate) -> Self {
        EqSearcher {
            reads,
            gate,
            // the appliers read the data of the `Eq` node's class only
            pattern: "(== ?l ?r)"
                .parse::<Pattern<Symbolic>>()
                .expect("eq pattern")
                .with_data_reads(true, &[]),
        }
    }

    /// The `Eq` nodes with an operand in `class`, as matches.
    fn eq_parents(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        class: Id,
        out: &mut Vec<(Id, Subst)>,
    ) {
        let vars = [var("?l"), var("?r")];
        let is_eq = |n: &Symbolic| matches!(n, Symbolic::Binary(BinOp::Eq, _));
        for (eq, node) in parents_where(egraph, class, is_eq) {
            out.push((eq, flat_subst(&vars, &node)));
        }
    }
}

impl Searcher<Symbolic, ConstFold> for EqSearcher {
    fn search_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        limit: usize,
    ) -> Vec<SearchMatches<'_, Symbolic>> {
        let mut found = self.pattern.search_with_limit(egraph, limit);
        found.retain(|m| self.gate.admits(egraph, m.eclass));
        found
    }

    fn search_changes(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        changes: &Changes<Symbolic>,
        limit: usize,
    ) -> Vec<SearchMatches<'_, Symbolic>> {
        let mut found: Vec<(Id, Subst)> =
            ungroup(self.pattern.search_changes(egraph, changes, limit)).collect();
        // The operand classes whose reads changed; each walked once.
        let mut operands: Vec<Id> = Vec::new();
        match self.reads {
            EqReads::Node => {}
            EqReads::OperandItes => {
                operands.extend(changes.nodes(&Discriminant::Ite).map(|(class, _)| *class))
            }
            EqReads::OperandApps => operands.extend(
                changes
                    .iter()
                    .filter(|(_, n)| matches!(n, Symbolic::FuncApp(..)))
                    .map(|(class, _)| *class),
            ),
            EqReads::Observations => {
                let unary_arg = |n: &Symbolic| match n {
                    Symbolic::FuncApp(_, _, args) if args.len() == 1 => Some(egraph.find(args[0])),
                    _ => None,
                };
                // a new observation `f(x)`, or one whose class's data changed
                operands.extend(changes.iter().filter_map(|(_, n)| unary_arg(n)));
                for &class in changes.data() {
                    operands.extend(egraph[class].nodes.iter().filter_map(unary_arg));
                }
            }
        }
        operands.sort_unstable();
        operands.dedup();
        for class in operands {
            self.eq_parents(egraph, class, &mut found);
        }
        found.retain(|(class, _)| self.gate.admits(egraph, *class));
        group_matches(found)
    }

    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        eclass: Id,
        limit: usize,
    ) -> Option<SearchMatches<'_, Symbolic>> {
        let found = self.pattern.search_eclass_with_limit(egraph, eclass, limit);
        found.filter(|m| self.gate.admits(egraph, m.eclass))
    }

    fn vars(&self) -> Vec<Var> {
        self.pattern.vars()
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
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        if known_bool(egraph, eclass) != Some(false) {
            return vec![];
        }
        let (l, r) = (egraph.find(subst[var("?l")]), egraph.find(subst[var("?r")]));
        if l == r {
            return vec![];
        }
        let mirrored = egraph.add(Symbolic::Binary(BinOp::Eq, [r, l]));
        if egraph.union(eclass, mirrored) {
            vec![egraph.find(eclass)]
        } else {
            vec![]
        }
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
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        if known_bool(egraph, eclass) != Some(false) {
            return vec![];
        }
        let mut contras: Vec<[Id; 2]> = Vec::new();
        {
            let (l, r) = (egraph.find(subst[var("?l")]), egraph.find(subst[var("?r")]));
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
/// A **failed** scan is the common case. Matching is semi-naive, so an undecided
/// `Eq` node is scanned again only when something its scan reads changed: the node
/// itself, a unary application over either operand (a new observation), or the
/// data of such an application's class (a sharper fingerprint); see
/// [`EqReads::Observations`]. (Scanned every iteration instead, the rule once
/// took an N=5 payload-enum file from 0.3s to 18.4s.)
pub(super) struct DistinguishingObsApplier;

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
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        // Only undecided equalities: a proven one is congruence's job, and a
        // disproven one is already what `contra-congruence` consumes.
        if known_bool(egraph, eclass).is_some() {
            return vec![];
        }
        let (l, r) = (egraph.find(subst[var("?l")]), egraph.find(subst[var("?r")]));
        let mut disprove = false;
        let obs_l = unary_observations(egraph, l);
        if l != r && !obs_l.is_empty() {
            let obs_r = unary_observations(egraph, r);
            // Hash join on the observation key.
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
                    break;
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
/// discriminator): the derived disequality is a new `Eq` match, so the rule's
/// next pass picks it up, one level per pass. Two assumed
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
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        if known_bool(egraph, eclass) != Some(false) {
            return vec![];
        }
        // Each derivation: pin `cond` to `cond_val`, and disprove the other
        // arm's comparison `other == z`.
        let (l, r) = (subst[var("?l")], subst[var("?r")]);
        let mut derivs: Vec<(Id, bool, Id, Id)> = Vec::new();
        for (ite_side, z) in [(l, r), (r, l)] {
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
