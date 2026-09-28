//! Per-block reaching-condition algebra for the method-body CFG linearizer.
//!
//! A block is lowered under its *reaching condition* — the disjunction of the
//! path conditions of its incoming edges. [`Reach`] keeps these conditions as a
//! BDD and lowers each to a `(PathConds, Val)` pair: boolean `Val`s are
//! desugared to ternaries (the IR has no `Not`/`And`/`Or`), constant-folded over
//! `TRUE`/`FALSE`, so straight-line code keeps a trivial `<>` guard. Also builds
//! the phi (`ite`) environment merge at joins.

use crate::dhash::{HashMap, HashSet};

use lasso::Spur;

use crate::translate::sink::Sink;
use crate::vmir::{FALSE, PathConds, Polarity, PureInst, TRUE, Type, Val};

/// `!v`, constant-folded over `true`/`false`.
pub(crate) fn not_val(sink: &mut Sink, v: Val) -> Val {
    if v == TRUE {
        FALSE
    } else if v == FALSE {
        TRUE
    } else {
        sink.emit_pure(Type::Bool, PureInst::Ternary(v, FALSE, TRUE))
    }
}

/// A reaching condition: a node of the method's [`Reach`] BDD.
pub(crate) type ReachRef = u32;
const BDD_FALSE: ReachRef = 0;
const BDD_TRUE: ReachRef = 1;

/// The reaching conditions of a method's blocks, as one reduced ordered BDD
/// over the branch conditions, variables ordered by their branch block's
/// topological position.
///
/// A cube set (DNF) cannot be the representation: `k` sequential partial joins
/// (both arms of a branch rejoin, the double miss leaves) reach
/// `∧ᵢ (aᵢ ∨ bᵢ)`, whose minimal DNF has `2^k` cubes. Its BDD is linear, and
/// in general no larger than the CFG: an acyclic CFG testing each condition
/// once per path *is* an ordered branching program over this order. Being
/// canonical, the BDD also finds every collapse the cube set needed
/// minimization laws for (a diamond, or a chain-decoded `n`-way match
/// telescoping back to its head's reach), and contradictions such as a
/// condition re-tested with the opposite outcome.
pub(crate) struct Reach {
    /// `(var, lo, hi)`; slots 0 and 1 are the `false`/`true` terminals.
    nodes: Vec<(u32, ReachRef, ReachRef)>,
    unique: HashMap<(u32, ReachRef, ReachRef), ReachRef>,
    apply_memo: HashMap<(bool, ReachRef, ReachRef), ReachRef>,
    vars: Vec<Val>,
    var_of: HashMap<Val, u32>,
    /// Materialized boolean of each node, shared by every block reaching it.
    vals: HashMap<ReachRef, Val>,
    /// [`Reach::implied`] of each node.
    implied: HashMap<ReachRef, std::rc::Rc<[(u32, bool)]>>,
}

impl Reach {
    pub(crate) const FALSE: ReachRef = BDD_FALSE;
    pub(crate) const TRUE: ReachRef = BDD_TRUE;

    pub(crate) fn new() -> Self {
        Reach {
            nodes: vec![(u32::MAX, BDD_FALSE, BDD_FALSE), (u32::MAX, BDD_TRUE, BDD_TRUE)],
            unique: HashMap::default(),
            apply_memo: HashMap::default(),
            vars: Vec::new(),
            var_of: HashMap::default(),
            vals: HashMap::default(),
            implied: HashMap::default(),
        }
    }

    /// Fix `cond`'s place in the variable order. Called as each branch block is
    /// lowered, i.e. in topological order.
    pub(crate) fn declare(&mut self, cond: &Val) -> u32 {
        if let Some(&v) = self.var_of.get(cond) {
            return v;
        }
        let v = self.vars.len() as u32;
        self.vars.push(cond.clone());
        self.var_of.insert(cond.clone(), v);
        v
    }

    pub(crate) fn lit(&mut self, cond: &Val, pol: Polarity) -> ReachRef {
        let v = self.declare(cond);
        match pol {
            Polarity::Positive => self.mk(v, BDD_FALSE, BDD_TRUE),
            Polarity::Negative => self.mk(v, BDD_TRUE, BDD_FALSE),
        }
    }

    pub(crate) fn and(&mut self, a: ReachRef, b: ReachRef) -> ReachRef {
        self.apply(true, a, b)
    }

    pub(crate) fn or(&mut self, a: ReachRef, b: ReachRef) -> ReachRef {
        self.apply(false, a, b)
    }

    fn mk(&mut self, var: u32, lo: ReachRef, hi: ReachRef) -> ReachRef {
        if lo == hi {
            return lo;
        }
        if let Some(&n) = self.unique.get(&(var, lo, hi)) {
            return n;
        }
        let n = self.nodes.len() as ReachRef;
        self.nodes.push((var, lo, hi));
        self.unique.insert((var, lo, hi), n);
        n
    }

    fn apply(&mut self, is_and: bool, a: ReachRef, b: ReachRef) -> ReachRef {
        let (absorbing, unit) = if is_and {
            (BDD_FALSE, BDD_TRUE)
        } else {
            (BDD_TRUE, BDD_FALSE)
        };
        if a == absorbing || b == absorbing {
            return absorbing;
        }
        if a == unit || a == b {
            return b;
        }
        if b == unit {
            return a;
        }
        let key = (is_and, a.min(b), a.max(b));
        if let Some(&r) = self.apply_memo.get(&key) {
            return r;
        }
        let ((va, alo, ahi), (vb, blo, bhi)) = (self.nodes[a as usize], self.nodes[b as usize]);
        let v = va.min(vb);
        let (alo, ahi) = if va == v { (alo, ahi) } else { (a, a) };
        let (blo, bhi) = if vb == v { (blo, bhi) } else { (b, b) };
        let lo = self.apply(is_and, alo, blo);
        let hi = self.apply(is_and, ahi, bhi);
        let r = self.mk(v, lo, hi);
        self.apply_memo.insert(key, r);
        r
    }

    /// The literals every path to `true` sets the same way: the strongest cube
    /// `f` implies.
    fn implied(&mut self, f: ReachRef) -> std::rc::Rc<[(u32, bool)]> {
        if f == BDD_TRUE {
            return std::rc::Rc::from([]);
        }
        if let Some(c) = self.implied.get(&f) {
            return c.clone();
        }
        let (v, lo, hi) = self.nodes[f as usize];
        let mut arms = Vec::with_capacity(2);
        for (c, pos) in [(lo, false), (hi, true)] {
            if c != BDD_FALSE {
                let mut lits = self.implied(c).to_vec();
                lits.push((v, pos));
                arms.push(lits);
            }
        }
        let out: std::rc::Rc<[(u32, bool)]> = match arms.as_slice() {
            [only] => only.as_slice().into(),
            [a, b] => a.iter().filter(|l| b.contains(l)).copied().collect(),
            _ => unreachable!("a non-false node reaches true"),
        };
        self.implied.insert(f, out.clone());
        out
    }

    /// `f` with the variables of `cube` fixed to their values.
    fn restrict(
        &mut self,
        f: ReachRef,
        cube: &HashMap<u32, bool>,
        memo: &mut HashMap<ReachRef, ReachRef>,
    ) -> ReachRef {
        if f == BDD_FALSE || f == BDD_TRUE {
            return f;
        }
        if let Some(&r) = memo.get(&f) {
            return r;
        }
        let (v, lo, hi) = self.nodes[f as usize];
        let r = match cube.get(&v) {
            Some(true) => self.restrict(hi, cube, memo),
            Some(false) => self.restrict(lo, cube, memo),
            None => {
                let lo = self.restrict(lo, cube, memo);
                let hi = self.restrict(hi, cube, memo);
                self.mk(v, lo, hi)
            }
        };
        memo.insert(f, r);
        r
    }

    /// A function that agrees with `f` wherever `care` holds — `g ∧ care =
    /// f ∧ care` — and is usually much smaller (Coudert–Madre `restrict`):
    /// wherever `care` fixes a variable, the branch it excludes is dropped.
    pub(crate) fn simplify(&mut self, f: ReachRef, care: ReachRef) -> ReachRef {
        self.simplify_memo(f, care, &mut HashMap::default())
    }

    fn simplify_memo(
        &mut self,
        f: ReachRef,
        care: ReachRef,
        memo: &mut HashMap<(ReachRef, ReachRef), ReachRef>,
    ) -> ReachRef {
        if care == BDD_TRUE || care == BDD_FALSE || f == BDD_FALSE || f == BDD_TRUE {
            return f;
        }
        if f == care {
            return BDD_TRUE;
        }
        if let Some(&r) = memo.get(&(f, care)) {
            return r;
        }
        let ((vf, flo, fhi), (vc, clo, chi)) = (self.nodes[f as usize], self.nodes[care as usize]);
        let r = if vc < vf {
            // `f` does not test `care`'s top variable: keep only what `care`
            // says about the rest.
            let rest = self.or(clo, chi);
            self.simplify_memo(f, rest, memo)
        } else if vc > vf {
            let lo = self.simplify_memo(flo, care, memo);
            let hi = self.simplify_memo(fhi, care, memo);
            self.mk(vf, lo, hi)
        } else if clo == BDD_FALSE {
            self.simplify_memo(fhi, chi, memo)
        } else if chi == BDD_FALSE {
            self.simplify_memo(flo, clo, memo)
        } else {
            let lo = self.simplify_memo(flo, clo, memo);
            let hi = self.simplify_memo(fhi, chi, memo);
            self.mk(vf, lo, hi)
        };
        memo.insert((f, care), r);
        r
    }

    /// The materialized boolean of `f`: one `var ? hi : lo` per BDD node,
    /// emitted once per method.
    pub(crate) fn val(&mut self, sink: &mut Sink, f: ReachRef) -> Val {
        match f {
            BDD_FALSE => return FALSE,
            BDD_TRUE => return TRUE,
            _ => {}
        }
        if let Some(v) = self.vals.get(&f) {
            return v.clone();
        }
        let (var, lo, hi) = self.nodes[f as usize];
        let c = self.vars[var as usize].clone();
        let out = match (lo, hi) {
            (BDD_FALSE, BDD_TRUE) => c,
            (BDD_TRUE, BDD_FALSE) => not_val(sink, c),
            _ => {
                let (h, l) = (self.val(sink, hi), self.val(sink, lo));
                sink.emit_pure(Type::Bool, PureInst::Ternary(c, h, l))
            }
        };
        self.vals.insert(f, out.clone());
        out
    }

    /// A block's lowering guard for reach `f`: the cube `f` implies, followed —
    /// unless that cube is all of `f` — by the rest of `f` as one materialized
    /// literal (a conjunctive pc cannot express a genuine disjunction). The flag
    /// says whether that literal was needed.
    pub(crate) fn pc(&mut self, sink: &mut Sink, f: ReachRef) -> (PathConds, bool) {
        if f == BDD_FALSE {
            let conds = vec![(FALSE, Polarity::Positive)];
            return (PathConds { conds }, false);
        }
        let mut cube = self.implied(f).to_vec();
        cube.sort_unstable();
        let mut conds: Vec<(Val, Polarity)> = Vec::with_capacity(cube.len() + 1);
        let fixed: HashMap<u32, bool> = cube.iter().copied().collect();
        for &(v, pos) in &cube {
            let pol = if pos { Polarity::Positive } else { Polarity::Negative };
            conds.push((self.vars[v as usize].clone(), pol));
        }
        let rest = self.restrict(f, &fixed, &mut HashMap::default());
        let disjunctive = rest != BDD_TRUE;
        if disjunctive {
            conds.push((self.val(sink, rest), Polarity::Positive));
        }
        (PathConds { conds }, disjunctive)
    }
}

/// Phi-merge two predecessor environments under a binary join guard: `then_env`
/// is selected when `cond` holds, `els_env` otherwise (the **unguarded**
/// fall-through, so the select reads `cond ? then : els`). A variable that
/// agrees on both sides passes through unchanged; one defined on a single side
/// inherits from that side. Binary because the block IR normalises every join to
/// a chain of these (a diamond is one; an n-way merge nests them).
///
/// Equivalent to a two-edge `ite(cond, then_v, els_v)` phi; the block lowerer
/// folds it right-to-left over an n-ary merge's arms.
pub(crate) fn merge_two_envs(
    sink: &mut Sink,
    cond: Val,
    then_env: &HashMap<Spur, Val>,
    els_env: &HashMap<Spur, Val>,
    var_types: &HashMap<Spur, Type>,
) -> HashMap<Spur, Val> {
    let mut names: HashSet<Spur> = HashSet::default();
    names.extend(then_env.keys().copied());
    names.extend(els_env.keys().copied());
    let mut out: HashMap<Spur, Val> = HashMap::default();
    for name in names {
        let merged = match (then_env.get(&name), els_env.get(&name)) {
            (Some(t), Some(e)) if t == e => t.clone(),
            (Some(t), Some(e)) => {
                let ty = var_types.get(&name).cloned().unwrap_or(Type::Int);
                sink.emit_pure(ty, PureInst::Ternary(cond.clone(), t.clone(), e.clone()))
            }
            (Some(v), None) | (None, Some(v)) => v.clone(),
            (None, None) => continue,
        };
        out.insert(name, merged);
    }
    out
}
