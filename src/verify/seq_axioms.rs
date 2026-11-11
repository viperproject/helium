//! The axioms of the built-in `Seq[T]` type, as quantifier recipes the one
//! `forall` rule (`rewrite::forall_rule`) instantiates, in place of a Viper
//! prelude domain.
//!
//! Adapted from Silicon's `dafny_axioms/sequences.vpr`. As recipes, they need
//! none of the domain's helper functions: `Seq_add`/`Seq_sub` are plain `+`/`-`
//! (their round-trip equalities stay, see [`round_trip`]),
//! `Seq_contains_trigger` is gone, and `Seq_equal` is the [`SeqAux::Eq`] tag in
//! the equality's own e-class. The two skolem functions stay (see
//! [`SeqSkolems`]), as functions the registry mints rather than domain
//! functions: a witness is a term of the instance's binders, so building an
//! instance again finds the same witness instead of adding another one.
//!
//! Each axiom reads `forall vars :: {triggers} conds ==> conclusions` and holds
//! at every element type `E`: its recipe is generic over `E` (`Generic(0)`), so
//! a trigger match fixes `E` from the matched sequence node's element type, as
//! for a generic domain axiom. An axiom only about ranges is ground at
//! `E = Int`. Its body is that implication, an ordinary quantifier body. An
//! axiom whose body was a conjunction of implications is split into one axiom
//! per conjunct.
//!
//! Separately, an axiom may have a guard, the recipe's [`Guard`]: an estimate
//! of when an instance is worth building, not part of what the axiom says. An
//! instance is built only where the guard is already known `true`. Today every
//! axiom with a condition is guarded by that condition, as an instance where it
//! is not yet known says nothing yet but adds terms the next round matches.
//!
//! The Dafny axioms' matching-loop cuts (`s != Seq()`, `s != s ++ t`) are left
//! out: the e-graph could almost never prove them (`s != Seq()` rarely has a
//! term), and its generation bound already cuts matching loops.

use std::sync::LazyLock;

use crate::verify::func_registry::SkolemSig;
use crate::verify::lang::{FuncId, SeqAux};
use crate::verify::quant::{Guard, QuantRecipe};
use crate::verify::rewrite::{AxiomInst, AxiomPure, PreparedTerm, TermHead};
use crate::vmir::{BinOp, Literal, SeqOp, Type, Val};

/// What a bound variable of an axiom ranges over, given the element type `E`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VarSort {
    Seq,
    Elem,
    Int,
}

/// A term of an axiom, over its bound variables and the element type `E`.
/// Boolean connectives use the shapes the program's own lowering builds
/// (`translate::pure_exp`), so a guard finds the program's terms by lookup:
/// `!a` is `ite(a, false, true)`, `a && b` is `ite(a, b, false)`, `a <= b` is
/// `!(b < a)`.
#[derive(Clone, Debug)]
pub(crate) enum STerm {
    Var(usize),
    Int(i64),
    Bool(bool),
    /// `Seq[E]()`.
    Empty,
    /// `Seq(e)`.
    Unit(Box<STerm>),
    /// A sequence operation on `Seq[E]` (a `Range` is on `Seq[Int]`), operands
    /// in `Symbolic::Seq` order. Never `Empty`, which is [`STerm::Empty`].
    Op(SeqOp, Vec<STerm>),
    Aux(SeqAux, Box<[STerm; 2]>),
    /// A witness: a skolem function of [`SeqSkolems`] applied to its operands.
    Skolem(Witness, Box<[STerm; 2]>),
    Bin(BinOp, Box<[STerm; 2]>),
    Ite(Box<[STerm; 3]>),
}

/// The skolem functions of the sequence axioms, each the witness of an
/// existential in a conclusion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Witness {
    /// `Seq_skolem(s, x)`: an index of `x` in `s`, when `x in s`.
    IndexOf,
    /// `Seq_skolem_diff(a, b)`: an index where `a` and `b` differ, when they
    /// have one. Both equality axioms use it, as in the domain: one function
    /// satisfies both.
    Diff,
}

/// The function ids of the [`Witness`]es, generic over the element type
/// (`Generic(0)`), minted by the registry (`FuncRegistry::seq_axioms`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SeqSkolems {
    index_of: FuncId,
    diff: FuncId,
}

impl SeqSkolems {
    /// Mint every witness function through `mint`, with its name and signature.
    pub(crate) fn mint(mut mint: impl FnMut(&str, SkolemSig) -> FuncId) -> Self {
        let seq = || Type::Seq(Box::new(Type::Generic(0)));
        SeqSkolems {
            index_of: mint(
                "Seq_skolem",
                SkolemSig {
                    params: vec![seq(), Type::Generic(0)],
                    ret: Type::Int,
                },
            ),
            diff: mint(
                "Seq_skolem_diff",
                SkolemSig {
                    params: vec![seq(), seq()],
                    ret: Type::Int,
                },
            ),
        }
    }

    fn func(&self, w: Witness) -> FuncId {
        match w {
            Witness::IndexOf => self.index_of,
            Witness::Diff => self.diff,
        }
    }
}

/// One axiom: `forall vars :: {triggers} guards ==> concls`.
pub(crate) struct SeqAxiom {
    pub(crate) name: &'static str,
    pub(crate) vars: Vec<VarSort>,
    /// Alternative trigger groups, each a conjunctive multi-pattern of
    /// applications covering every variable.
    pub(crate) triggers: Vec<Vec<STerm>>,
    /// The left-hand side, as a conjunction.
    pub(crate) conds: Vec<STerm>,
    /// The right-hand side, as a conjunction of boolean terms.
    pub(crate) concls: Vec<STerm>,
    /// The instantiation guard, as a conjunction: an instance is built only
    /// where it is already known `true` (see [`Guard`]). Empty: every match.
    pub(crate) guard: Vec<STerm>,
    /// Only meaningful at `E = Int`: the axiom is about `[a..b)`.
    pub(crate) int_only: bool,
}

pub(crate) fn axioms() -> &'static [SeqAxiom] {
    &AXIOMS
}

static AXIOMS: LazyLock<Vec<SeqAxiom>> = LazyLock::new(|| {
    let all = build();
    for ax in &all {
        ax.check();
    }
    all
});

impl SeqAxiom {
    /// Every trigger group must bind every variable: the e-graph rule reads σ off
    /// a match, and Z3 rejects a pattern that leaves a binder out.
    fn check(&self) {
        assert!(
            !self.triggers.is_empty(),
            "seq axiom {} has no trigger",
            self.name
        );
        for group in &self.triggers {
            let mut seen = vec![false; self.vars.len()];
            for t in group {
                t.mark_vars(&mut seen);
            }
            assert!(
                seen.iter().all(|&s| s),
                "a trigger group of seq axiom {} misses a variable",
                self.name
            );
        }
    }
}

impl STerm {
    fn mark_vars(&self, seen: &mut [bool]) {
        match self {
            STerm::Var(i) => seen[*i] = true,
            STerm::Int(_) | STerm::Bool(_) | STerm::Empty => {}
            STerm::Skolem(_, args) => args.iter().for_each(|a| a.mark_vars(seen)),
            STerm::Unit(e) => e.mark_vars(seen),
            STerm::Op(_, args) => args.iter().for_each(|a| a.mark_vars(seen)),
            STerm::Aux(_, args) => args.iter().for_each(|a| a.mark_vars(seen)),
            STerm::Bin(_, args) => args.iter().for_each(|a| a.mark_vars(seen)),
            STerm::Ite(args) => args.iter().for_each(|a| a.mark_vars(seen)),
        }
    }
}

// Term constructors, named after the Viper syntax they stand for.

fn v(i: usize) -> STerm {
    STerm::Var(i)
}
fn int(n: i64) -> STerm {
    STerm::Int(n)
}
fn empty() -> STerm {
    STerm::Empty
}
fn unit(e: STerm) -> STerm {
    STerm::Unit(Box::new(e))
}
fn op(o: SeqOp, args: Vec<STerm>) -> STerm {
    STerm::Op(o, args)
}
fn len(s: STerm) -> STerm {
    op(SeqOp::Len, vec![s])
}
fn cat(a: STerm, b: STerm) -> STerm {
    op(SeqOp::Concat, vec![a, b])
}
fn idx(s: STerm, i: STerm) -> STerm {
    op(SeqOp::Index, vec![s, i])
}
fn upd(s: STerm, i: STerm, x: STerm) -> STerm {
    op(SeqOp::Update, vec![s, i, x])
}
fn take(s: STerm, n: STerm) -> STerm {
    op(SeqOp::Take, vec![s, n])
}
fn drop(s: STerm, n: STerm) -> STerm {
    op(SeqOp::Drop, vec![s, n])
}
/// `x in s`.
fn contains(s: STerm, x: STerm) -> STerm {
    op(SeqOp::Contains, vec![s, x])
}
fn range(lo: STerm, hi: STerm) -> STerm {
    op(SeqOp::Range, vec![lo, hi])
}
fn aux(a: SeqAux, l: STerm, r: STerm) -> STerm {
    STerm::Aux(a, Box::new([l, r]))
}
fn skolem(w: Witness, l: STerm, r: STerm) -> STerm {
    STerm::Skolem(w, Box::new([l, r]))
}
fn bin(o: BinOp, l: STerm, r: STerm) -> STerm {
    STerm::Bin(o, Box::new([l, r]))
}
fn ite(c: STerm, t: STerm, e: STerm) -> STerm {
    STerm::Ite(Box::new([c, t, e]))
}
fn eq(l: STerm, r: STerm) -> STerm {
    bin(BinOp::Eq, l, r)
}
fn add(l: STerm, r: STerm) -> STerm {
    bin(BinOp::AddI, l, r)
}
fn sub(l: STerm, r: STerm) -> STerm {
    bin(BinOp::SubI, l, r)
}
fn lt(l: STerm, r: STerm) -> STerm {
    bin(BinOp::LtI, l, r)
}
fn not(x: STerm) -> STerm {
    ite(x, STerm::Bool(false), STerm::Bool(true))
}
fn and(a: STerm, b: STerm) -> STerm {
    ite(a, b, STerm::Bool(false))
}
fn ne(l: STerm, r: STerm) -> STerm {
    not(eq(l, r))
}
fn le(l: STerm, r: STerm) -> STerm {
    not(lt(r, l))
}
fn ge(l: STerm, r: STerm) -> STerm {
    not(lt(l, r))
}
fn gt(l: STerm, r: STerm) -> STerm {
    lt(r, l)
}

/// `undo(shifted, by) == back`: the shifted index of an axiom's conclusion,
/// shifted back, is the index it came from (Dafny's
/// `Seq_add(Seq_sub(n, k), k) == n`). This is not decoration: a solver that
/// matches modulo equalities, not arithmetic, otherwise reads
/// `s[(i - n) + n]` as a new term, and `drop-index1` after `drop-index2` (or
/// `append-index2` after `append-index3`) feeds itself new index terms forever.
fn round_trip(shifted: STerm, by: STerm, back: STerm, undo: fn(STerm, STerm) -> STerm) -> STerm {
    eq(undo(shifted, by), back)
}

/// `0 <= i && i < n`, as two guards.
fn in_range(i: STerm, n: STerm) -> [STerm; 2] {
    [le(int(0), i.clone()), lt(i, n)]
}

struct Ax {
    name: &'static str,
    vars: Vec<VarSort>,
    triggers: Vec<Vec<STerm>>,
    conds: Vec<STerm>,
    concls: Vec<STerm>,
    guard: Vec<STerm>,
}

impl Ax {
    fn new(name: &'static str, vars: &[VarSort]) -> Self {
        Ax {
            name,
            vars: vars.to_vec(),
            triggers: vec![],
            conds: vec![],
            concls: vec![],
            guard: vec![],
        }
    }
    fn trig(mut self, group: Vec<STerm>) -> Self {
        self.triggers.push(group);
        self
    }
    /// Conjuncts of the left-hand side.
    fn when(mut self, cs: impl IntoIterator<Item = STerm>) -> Self {
        self.conds.extend(cs);
        self
    }
    /// Conjuncts of the instantiation guard.
    fn guard(mut self, gs: impl IntoIterator<Item = STerm>) -> Self {
        self.guard.extend(gs);
        self
    }
    /// The left-hand side so far as the instantiation guard as well.
    fn guard_when(self) -> Self {
        let conds = self.conds.clone();
        self.guard(conds)
    }
    fn then(mut self, cs: impl IntoIterator<Item = STerm>) -> Self {
        self.concls.extend(cs);
        self
    }
    fn done(self) -> SeqAxiom {
        self.finish(false)
    }
    fn int_only(self) -> SeqAxiom {
        self.finish(true)
    }
    fn finish(self, int_only: bool) -> SeqAxiom {
        SeqAxiom {
            name: self.name,
            vars: self.vars,
            triggers: self.triggers,
            conds: self.conds,
            concls: self.concls,
            guard: self.guard,
            int_only,
        }
    }
}

fn build() -> Vec<SeqAxiom> {
    use VarSort::{Elem, Int, Seq};
    let (s, t, n) = (v(0), v(1), v(2));
    vec![
        // Start basics
        Ax::new("seq-length-nonneg", &[Seq])
            .trig(vec![len(s.clone())])
            .then([le(int(0), len(s.clone()))])
            .done(),
        Ax::new("seq-empty-length", &[])
            .trig(vec![empty()])
            .then([eq(len(empty()), int(0))])
            .done(),
        Ax::new("seq-length-zero", &[Seq])
            .trig(vec![len(s.clone())])
            .when([eq(len(s.clone()), int(0))])
            .guard_when()
            .then([eq(s.clone(), empty())])
            .done(),
        Ax::new("seq-singleton-length", &[Elem])
            .trig(vec![unit(v(0))])
            .then([eq(len(unit(v(0))), int(1))])
            .done(),
        Ax::new("seq-append-length", &[Seq, Seq])
            .trig(vec![len(cat(s.clone(), t.clone()))])
            .then([eq(
                len(cat(s.clone(), t.clone())),
                add(len(s.clone()), len(t.clone())),
            )])
            .done(),
        Ax::new("seq-append-empty-left", &[Seq, Seq])
            .trig(vec![cat(s.clone(), t.clone())])
            .when([eq(s.clone(), empty())])
            .guard_when()
            .then([eq(cat(s.clone(), t.clone()), t.clone())])
            .done(),
        Ax::new("seq-append-empty-right", &[Seq, Seq])
            .trig(vec![cat(s.clone(), t.clone())])
            .when([eq(t.clone(), empty())])
            .guard_when()
            .then([eq(cat(s.clone(), t.clone()), s.clone())])
            .done(),
        Ax::new("seq-singleton-index", &[Elem])
            .trig(vec![unit(v(0))])
            .then([eq(idx(unit(v(0)), int(0)), v(0))])
            .done(),
        // End basics
        // Start index-append-update
        Ax::new("seq-append-index1", &[Seq, Seq, Int])
            .trig(vec![idx(cat(s.clone(), t.clone()), n.clone())])
            .trig(vec![idx(s.clone(), n.clone()), cat(s.clone(), t.clone())])
            .when(in_range(n.clone(), len(s.clone())))
            .guard_when()
            .then([eq(
                idx(cat(s.clone(), t.clone()), n.clone()),
                idx(s.clone(), n.clone()),
            )])
            .done(),
        Ax::new("seq-append-index2", &[Seq, Seq, Int])
            .trig(vec![idx(cat(s.clone(), t.clone()), n.clone())])
            .when([
                le(len(s.clone()), n.clone()),
                lt(n.clone(), len(cat(s.clone(), t.clone()))),
            ])
            .guard_when()
            .then([
                round_trip(
                    sub(n.clone(), len(s.clone())),
                    len(s.clone()),
                    n.clone(),
                    add,
                ),
                eq(
                    idx(cat(s.clone(), t.clone()), n.clone()),
                    idx(t.clone(), sub(n.clone(), len(s.clone()))),
                ),
            ])
            .done(),
        Ax::new("seq-append-index3", &[Seq, Seq, Int])
            .trig(vec![idx(t.clone(), n.clone()), cat(s.clone(), t.clone())])
            .when(in_range(n.clone(), len(t.clone())))
            .guard_when()
            .then([
                round_trip(
                    add(n.clone(), len(s.clone())),
                    len(s.clone()),
                    n.clone(),
                    sub,
                ),
                eq(
                    idx(cat(s.clone(), t.clone()), add(n.clone(), len(s.clone()))),
                    idx(t.clone(), n.clone()),
                ),
            ])
            .done(),
        Ax::new("seq-update-length", &[Seq, Int, Elem])
            .trig(vec![len(upd(s.clone(), v(1), v(2)))])
            .trig(vec![len(s.clone()), upd(s.clone(), v(1), v(2))])
            .when(in_range(v(1), len(s.clone())))
            .guard_when()
            .then([eq(len(upd(s.clone(), v(1), v(2))), len(s.clone()))])
            .done(),
        // `s[i := x][n]`, split on whether `i == n`.
        Ax::new("seq-update-index-same", &[Seq, Int, Elem, Int])
            .trig(vec![idx(upd(s.clone(), v(1), v(2)), v(3))])
            .trig(vec![idx(s.clone(), v(3)), upd(s.clone(), v(1), v(2))])
            .when(in_range(v(3), len(s.clone())))
            .when([eq(v(1), v(3))])
            .guard_when()
            .then([eq(idx(upd(s.clone(), v(1), v(2)), v(3)), v(2))])
            .done(),
        Ax::new("seq-update-index-other", &[Seq, Int, Elem, Int])
            .trig(vec![idx(upd(s.clone(), v(1), v(2)), v(3))])
            .trig(vec![idx(s.clone(), v(3)), upd(s.clone(), v(1), v(2))])
            .when(in_range(v(3), len(s.clone())))
            .when([ne(v(1), v(3))])
            .guard_when()
            .then([eq(
                idx(upd(s.clone(), v(1), v(2)), v(3)),
                idx(s.clone(), v(3)),
            )])
            .done(),
        // End index-append-update
        // Start take/drop. Here `t` is the count `n` (variable 1).
        Ax::new("seq-take-length", &[Seq, Int])
            .trig(vec![len(take(s.clone(), t.clone()))])
            .trig(vec![take(s.clone(), t.clone()), len(s.clone())])
            .when([le(int(0), t.clone()), le(t.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(len(take(s.clone(), t.clone())), t.clone())])
            .done(),
        Ax::new("seq-take-length-over", &[Seq, Int])
            .trig(vec![len(take(s.clone(), t.clone()))])
            .trig(vec![take(s.clone(), t.clone()), len(s.clone())])
            .when([le(int(0), t.clone()), lt(len(s.clone()), t.clone())])
            .guard_when()
            .then([eq(len(take(s.clone(), t.clone())), len(s.clone()))])
            .done(),
        Ax::new("seq-take-length-negative", &[Seq, Int])
            .trig(vec![len(take(s.clone(), t.clone()))])
            .trig(vec![take(s.clone(), t.clone()), len(s.clone())])
            .when([lt(t.clone(), int(0))])
            .guard_when()
            .then([eq(len(take(s.clone(), t.clone())), int(0))])
            .done(),
        Ax::new("seq-take-index", &[Seq, Int, Int])
            .trig(vec![idx(take(s.clone(), v(1)), v(2))])
            .trig(vec![idx(s.clone(), v(2)), take(s.clone(), v(1))])
            .when([le(int(0), v(2)), lt(v(2), v(1)), lt(v(2), len(s.clone()))])
            .guard_when()
            .then([eq(idx(take(s.clone(), v(1)), v(2)), idx(s.clone(), v(2)))])
            .done(),
        Ax::new("seq-drop-length", &[Seq, Int])
            .trig(vec![len(drop(s.clone(), t.clone()))])
            .trig(vec![len(s.clone()), drop(s.clone(), t.clone())])
            .when([le(int(0), t.clone()), le(t.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(
                len(drop(s.clone(), t.clone())),
                sub(len(s.clone()), t.clone()),
            )])
            .done(),
        Ax::new("seq-drop-length-over", &[Seq, Int])
            .trig(vec![len(drop(s.clone(), t.clone()))])
            .trig(vec![len(s.clone()), drop(s.clone(), t.clone())])
            .when([le(int(0), t.clone()), lt(len(s.clone()), t.clone())])
            .guard_when()
            .then([eq(len(drop(s.clone(), t.clone())), int(0))])
            .done(),
        Ax::new("seq-drop-length-negative", &[Seq, Int])
            .trig(vec![len(drop(s.clone(), t.clone()))])
            .trig(vec![len(s.clone()), drop(s.clone(), t.clone())])
            .when([lt(t.clone(), int(0))])
            .guard_when()
            .then([eq(len(drop(s.clone(), t.clone())), len(s.clone()))])
            .done(),
        Ax::new("seq-drop-index1", &[Seq, Int, Int])
            .trig(vec![idx(drop(s.clone(), v(1)), v(2))])
            .when([
                lt(int(0), v(1)),
                le(int(0), v(2)),
                lt(v(2), sub(len(s.clone()), v(1))),
            ])
            .guard_when()
            .then([
                round_trip(add(v(2), v(1)), v(1), v(2), sub),
                eq(
                    idx(drop(s.clone(), v(1)), v(2)),
                    idx(s.clone(), add(v(2), v(1))),
                ),
            ])
            .done(),
        Ax::new("seq-drop-index2", &[Seq, Int, Int])
            .trig(vec![drop(s.clone(), v(1)), idx(s.clone(), v(2))])
            .when([lt(int(0), v(1)), le(v(1), v(2)), lt(v(2), len(s.clone()))])
            .guard_when()
            .then([
                round_trip(sub(v(2), v(1)), v(1), v(2), add),
                eq(
                    idx(drop(s.clone(), v(1)), sub(v(2), v(1))),
                    idx(s.clone(), v(2)),
                ),
            ])
            .done(),
        Ax::new("seq-append-take1", &[Seq, Seq, Int])
            .trig(vec![take(cat(s.clone(), t.clone()), n.clone())])
            .when([lt(int(0), n.clone()), le(n.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(
                take(cat(s.clone(), t.clone()), n.clone()),
                take(s.clone(), n.clone()),
            )])
            .done(),
        Ax::new("seq-append-take2", &[Seq, Seq, Int])
            .trig(vec![take(cat(s.clone(), t.clone()), n.clone())])
            .when([
                gt(n.clone(), int(0)),
                gt(n.clone(), len(s.clone())),
                lt(n.clone(), len(cat(s.clone(), t.clone()))),
            ])
            .guard_when()
            .then([
                round_trip(
                    sub(n.clone(), len(s.clone())),
                    len(s.clone()),
                    n.clone(),
                    add,
                ),
                eq(
                    take(cat(s.clone(), t.clone()), n.clone()),
                    cat(s.clone(), take(t.clone(), sub(n.clone(), len(s.clone())))),
                ),
            ])
            .done(),
        Ax::new("seq-append-drop1", &[Seq, Seq, Int])
            .trig(vec![drop(cat(s.clone(), t.clone()), n.clone())])
            .when([lt(int(0), n.clone()), le(n.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(
                drop(cat(s.clone(), t.clone()), n.clone()),
                cat(drop(s.clone(), n.clone()), t.clone()),
            )])
            .done(),
        Ax::new("seq-append-drop2", &[Seq, Seq, Int])
            .trig(vec![drop(cat(s.clone(), t.clone()), n.clone())])
            .when([
                gt(n.clone(), int(0)),
                gt(n.clone(), len(s.clone())),
                lt(n.clone(), len(cat(s.clone(), t.clone()))),
            ])
            .guard_when()
            .then([
                round_trip(
                    sub(n.clone(), len(s.clone())),
                    len(s.clone()),
                    n.clone(),
                    add,
                ),
                eq(
                    drop(cat(s.clone(), t.clone()), n.clone()),
                    drop(t.clone(), sub(n.clone(), len(s.clone()))),
                ),
            ])
            .done(),
        // These cut matching loops the axioms above can otherwise introduce, as
        // those create new append terms.
        Ax::new("seq-take-zero", &[Seq, Int])
            .trig(vec![take(s.clone(), t.clone())])
            .when([le(t.clone(), int(0))])
            .guard_when()
            .then([eq(take(s.clone(), t.clone()), empty())])
            .done(),
        Ax::new("seq-drop-zero", &[Seq, Int])
            .trig(vec![drop(s.clone(), t.clone())])
            .when([le(t.clone(), int(0))])
            .guard_when()
            .then([eq(drop(s.clone(), t.clone()), s.clone())])
            .done(),
        Ax::new("seq-take-all", &[Seq, Int])
            .trig(vec![take(s.clone(), t.clone())])
            .when([ge(t.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(take(s.clone(), t.clone()), s.clone())])
            .done(),
        Ax::new("seq-drop-all", &[Seq, Int])
            .trig(vec![drop(s.clone(), t.clone())])
            .when([ge(t.clone(), len(s.clone()))])
            .guard_when()
            .then([eq(drop(s.clone(), t.clone()), empty())])
            .done(),
        // End take/drop
        // Start contains. Here `t` is the element `x` (variable 1).
        Ax::new("seq-contains-witness", &[Seq, Elem])
            .trig(vec![contains(s.clone(), t.clone())])
            .when([contains(s.clone(), t.clone())])
            .guard_when()
            .then({
                let k = || skolem(Witness::IndexOf, s.clone(), t.clone());
                let [lo, hi] = in_range(k(), len(s.clone()));
                [lo, hi, eq(idx(s.clone(), k()), t.clone())]
            })
            .done(),
        Ax::new("seq-contains-index", &[Seq, Elem, Int])
            .trig(vec![
                contains(s.clone(), t.clone()),
                idx(s.clone(), n.clone()),
            ])
            .when(in_range(n.clone(), len(s.clone())))
            .when([eq(idx(s.clone(), n.clone()), t.clone())])
            .guard_when()
            .then([contains(s.clone(), t.clone())])
            .done(),
        // End contains
        // Start equals. `Seq_equal(a, b)` is the `SeqAux::Eq` tag, which shares
        // its e-class with `a == b`, so `Seq_equal_ext` holds by construction and
        // `Seq_equal_def` reduces to its third disjunct: unequal sequences of one
        // length differ at the in-range index `k = Seq_skolem_diff(a, b)`.
        Ax::new("seq-equal-diff", &[Seq, Seq])
            .trig(vec![aux(SeqAux::Eq, s.clone(), t.clone())])
            .when([ne(s.clone(), t.clone()), eq(len(s.clone()), len(t.clone()))])
            .guard_when()
            .then({
                let k = || skolem(Witness::Diff, s.clone(), t.clone());
                let [lo, hi] = in_range(k(), len(s.clone()));
                [
                    eq(k(), skolem(Witness::Diff, t.clone(), s.clone())),
                    lo,
                    hi,
                    ne(idx(s.clone(), k()), idx(t.clone(), k())),
                ]
            })
            .done(),
        // The same fact read the other way, which is the one that *proves* an
        // equality: of one length, `a == b` exactly when they agree at `k` (or
        // `k` is out of range, which only happens when they are equal). The
        // implication is spelled as a program's `0 <= i && i < |a| ==> a[i] ==
        // b[i]` lowers, so an instance of such a quantifier at `k` is this term.
        Ax::new("seq-equal-ext", &[Seq, Seq])
            .trig(vec![aux(SeqAux::Eq, s.clone(), t.clone())])
            .when([eq(len(s.clone()), len(t.clone()))])
            .guard_when()
            .then({
                let k = || skolem(Witness::Diff, s.clone(), t.clone());
                let [lo, hi] = in_range(k(), len(s.clone()));
                [eq(
                    eq(s.clone(), t.clone()),
                    ite(
                        and(lo, hi),
                        eq(idx(s.clone(), k()), idx(t.clone(), k())),
                        STerm::Bool(true),
                    ),
                )]
            })
            .done(),
        // End equals
        // Start extras
        Ax::new("seq-singleton-contains", &[Elem, Elem])
            .trig(vec![contains(unit(v(0)), v(1))])
            .then([eq(contains(unit(v(0)), v(1)), eq(v(0), v(1)))])
            .done(),
        Ax::new("seq-range-length", &[Int, Int])
            .trig(vec![len(range(v(0), v(1)))])
            .when([lt(v(0), v(1))])
            .guard_when()
            .then([eq(len(range(v(0), v(1))), sub(v(1), v(0)))])
            .int_only(),
        Ax::new("seq-range-length-empty", &[Int, Int])
            .trig(vec![len(range(v(0), v(1)))])
            .when([le(v(1), v(0))])
            .guard_when()
            .then([eq(len(range(v(0), v(1))), int(0))])
            .int_only(),
        Ax::new("seq-range-index", &[Int, Int, Int])
            .trig(vec![idx(range(v(0), v(1)), v(2))])
            .when(in_range(v(2), sub(v(1), v(0))))
            .guard_when()
            .then([eq(idx(range(v(0), v(1)), v(2)), add(v(0), v(2)))])
            .int_only(),
        Ax::new("seq-range-contains", &[Int, Int, Int])
            .trig(vec![contains(range(v(0), v(1)), v(2))])
            .then([eq(
                contains(range(v(0), v(1)), v(2)),
                and(le(v(0), v(2)), lt(v(2), v(1))),
            )])
            .int_only(),
        // End extras
    ]
}

/// One recipe per axiom, in [`axioms`] order, its witnesses applications of
/// `skolems`. See the module docs.
pub(crate) fn recipes(skolems: &SeqSkolems) -> Vec<QuantRecipe> {
    axioms().iter().map(|ax| ax.recipe(*skolems)).collect()
}

impl SeqAxiom {
    fn recipe(&self, skolems: SeqSkolems) -> QuantRecipe {
        // A range is a `Seq[Int]` whatever `E` is, so its axioms are ground.
        let (elem, n_tys) = if self.int_only {
            (Type::Int, 0)
        } else {
            (Type::Generic(0), 1)
        };
        // The guard and the body are separate step lists, each starting after
        // the binders.
        let steps = || Steps {
            next: self.vars.len(),
            insts: Vec::new(),
            elem: elem.clone(),
            skolems,
        };
        let mut guard = steps();
        let conds: Vec<Val> = self.guard.iter().map(|g| guard.emit(g)).collect();
        let mut body = steps();
        let concl = conj(&self.concls);
        let res = if self.conds.is_empty() {
            body.emit(&concl)
        } else {
            body.emit(&ite(conj(&self.conds), concl, STerm::Bool(true)))
        };
        let groups = self
            .triggers
            .iter()
            .map(|g| g.iter().map(|t| trigger(t, &elem)).collect())
            .collect();
        QuantRecipe {
            n_caps: 0,
            n_bound: self.vars.len(),
            n_tys,
            own_tys: (0..n_tys).collect(),
            groups,
            insts: body.insts,
            res,
            guard: (!conds.is_empty()).then_some(Guard {
                insts: guard.insts,
                conds,
            }),
        }
    }
}

/// The steps of a recipe under construction, in its temp space: the binders
/// first, then one temp per value-producing step.
struct Steps {
    next: usize,
    insts: Vec<AxiomInst>,
    elem: Type,
    skolems: SeqSkolems,
}

impl Steps {
    fn push(&mut self, p: AxiomPure) -> Val {
        self.insts.push(AxiomInst::Val(p));
        self.next += 1;
        Val::Temp(self.next - 1)
    }

    /// The value of `t`, adding the steps that build it.
    fn emit(&mut self, t: &STerm) -> Val {
        let elem = self.elem.clone();
        match t {
            STerm::Var(i) => Val::Temp(*i),
            STerm::Int(n) => Val::Literal(Literal::Int((*n).into())),
            STerm::Bool(b) => Val::Literal(Literal::Bool(*b)),
            STerm::Empty => self.push(AxiomPure::Seq {
                elem_ty: elem,
                op: SeqOp::Empty,
                args: vec![],
            }),
            STerm::Unit(e) => {
                let e = self.emit(e);
                self.push(AxiomPure::SeqLit {
                    elem_ty: elem,
                    elem: e,
                })
            }
            STerm::Op(op, args) => {
                let args = args.iter().map(|a| self.emit(a)).collect();
                let elem_ty = if *op == SeqOp::Range { Type::Int } else { elem };
                self.push(AxiomPure::Seq {
                    elem_ty,
                    op: *op,
                    args,
                })
            }
            STerm::Aux(..) => {
                panic!("a sequence axiom has a `SeqAux` term only in a trigger, not {t:?}")
            }
            // Generic in the element type like the axiom, so an instance's
            // type substitution fixes it.
            STerm::Skolem(w, args) => {
                let args = args.iter().map(|a| self.emit(a)).collect();
                self.push(AxiomPure::App {
                    func: self.skolems.func(*w),
                    type_args: vec![elem],
                    args,
                })
            }
            STerm::Bin(op, args) => {
                let (l, r) = (self.emit(&args[0]), self.emit(&args[1]));
                self.push(AxiomPure::Binary(*op, l, r))
            }
            STerm::Ite(args) => {
                let c = self.emit(&args[0]);
                let t = self.emit(&args[1]);
                let e = self.emit(&args[2]);
                self.push(AxiomPure::Ternary(c, t, e))
            }
        }
    }
}

/// `t1 && .. && tn` (`true` when empty), in the shape the program's own
/// lowering builds, so the e-graph's `ite` rule splits a true one.
fn conj(ts: &[STerm]) -> STerm {
    match ts {
        [] => STerm::Bool(true),
        [t] => t.clone(),
        [t, rest @ ..] => and(t.clone(), conj(rest)),
    }
}

/// A trigger term at element type `elem`: always rooted at a sequence term.
fn trigger(t: &STerm, elem: &Type) -> PreparedTerm {
    let app = |head, args: &[STerm]| PreparedTerm::App {
        head,
        args: args.iter().map(|a| trigger(a, elem)).collect(),
    };
    match t {
        STerm::Var(i) => PreparedTerm::Bound(*i),
        STerm::Int(n) => PreparedTerm::Lit(Literal::Int((*n).into())),
        STerm::Bool(b) => PreparedTerm::Lit(Literal::Bool(*b)),
        STerm::Empty => app(TermHead::Seq(elem.clone(), SeqOp::Empty), &[]),
        STerm::Unit(e) => app(TermHead::SeqLit(elem.clone()), std::slice::from_ref(&**e)),
        STerm::Op(SeqOp::Range, args) => app(TermHead::Seq(Type::Int, SeqOp::Range), args),
        STerm::Op(op, args) => app(TermHead::Seq(elem.clone(), *op), args),
        STerm::Aux(aux, args) => app(TermHead::SeqAux(elem.clone(), *aux), &args[..]),
        STerm::Bin(..) | STerm::Ite(..) | STerm::Skolem(..) => {
            panic!("a sequence-axiom trigger is built from sequence terms, not {t:?}")
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_trigger_group_binds_every_variable() {
        // `axioms()` checks each axiom when the table is first built.
        assert!(!super::axioms().is_empty());
    }
}
