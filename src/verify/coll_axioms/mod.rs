//! The axioms of the built-in collection types, as quantifier recipes the one
//! `forall` rule (`rewrite::forall_rule`) instantiates, in place of a Viper
//! prelude domain. One table per [`CollKind`]: `Seq[T]` ([`seq`]), `Set[T]`
//! ([`set`]), `Map[K, V]` ([`map`]) and `Multiset[T]` ([`multiset`]).
//!
//! Adapted from Silicon's `dafny_axioms/sequences.vpr`, `sets.vpr`,
//! `maps.vpr` and `multisets.vpr`. As recipes, they need none of the domains'
//! helper functions:
//! `Seq_add`/`Seq_sub` are plain `+`/`-` (their round-trip equalities stay,
//! see [`seq`]), `Seq_contains_trigger` is gone, and `Seq_equal`/
//! `Set_equal`/`Map_equal`/`Multiset_equal` are a program's typed collection
//! equality
//! (`CollOp::is_eq`, in the untyped equality's own e-class). The skolem functions stay (see [`Witness`]), as
//! functions the registry mints rather than domain functions: a witness is a
//! term of the instance's binders, so building an instance again finds the
//! same witness instead of adding another one.
//!
//! Each axiom reads `forall vars :: {triggers} conds ==> conclusions` and holds
//! at every element type `E` (for a map, every key and value type `K`, `V`):
//! its recipe is generic over them (`Generic(0)`, and `Generic(1)` for `V`),
//! so a trigger match fixes them from the matched collection node's type
//! arguments, as for a generic domain axiom. An axiom only about ranges is
//! ground at `E = Int`. Its body is that implication, an ordinary quantifier body. An
//! axiom whose body was a conjunction of implications is split into one axiom
//! per conjunct.
//!
//! Separately, an axiom may have a guard, the recipe's [`Guard`]: an estimate
//! of when an instance is worth building, not part of what the axiom says. An
//! instance is built only where the guard is already known `true`. Today every
//! axiom with a condition is guarded by that condition, as an instance where it
//! is not yet known says nothing yet but adds terms the next round matches.
//!
//! Every axiom of one table is about one collection kind: its [`VarSort::Coll`]
//! binders, [`STerm::Empty`] and [`STerm::Unit`] are of that kind, and so is
//! every [`STerm::Op`] it applies (checked when the table is built). Only an
//! [`STerm::OpOn`] is about another kind: the map axioms test and count the
//! sets `domain(m)` and `range(m)`, whose element types are the map's key and
//! value types. That is why a program with a map also gets the set axioms
//! (`declaration::assume_axioms`).

pub(crate) mod map;
pub(crate) mod multiset;
pub(crate) mod seq;
pub(crate) mod set;

use std::sync::LazyLock;

use crate::verify::func_registry::SkolemSig;
use crate::verify::lang::FuncId;
use crate::verify::quant::{Guard, QuantRecipe};
use crate::verify::rewrite::{AxiomInst, AxiomPure, PreparedTerm, TermHead};
use crate::vmir::{BinOp, CollKind, CollOp, Literal, Type, Val};

/// What a bound variable of an axiom ranges over, given the collection's type
/// arguments: the element type `E` of a `Seq` or `Set`, the key and value
/// types `K`, `V` of a `Map`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VarSort {
    /// A collection of the table's kind.
    Coll,
    /// An element `E` of a `Seq`, `Set` or `Multiset`.
    Elem,
    /// A key `K` of a `Map`.
    Key,
    /// A value `V` of a `Map`.
    Value,
    Int,
}

impl VarSort {
    /// The binder type, for a collection of `kind` at type arguments `tys`.
    fn ty(self, kind: CollKind, tys: &[Type]) -> Type {
        let arg = |i: usize, map: bool| {
            assert_eq!(
                kind == CollKind::Map,
                map,
                "a {kind:?} axiom binds a {self:?}"
            );
            tys[i].clone()
        };
        match self {
            VarSort::Coll => Type::coll(kind, tys),
            VarSort::Elem => arg(0, false),
            VarSort::Key => arg(0, true),
            VarSort::Value => arg(1, true),
            VarSort::Int => Type::Int,
        }
    }
}

/// A term of an axiom, over its bound variables and the element type `E`.
/// Boolean connectives use the shapes the program's own lowering builds
/// (`translate::pure_exp`), so a guard finds the program's terms by lookup:
/// `!a` is `ite(a, false, true)`, `a && b` is `ite(a, b, false)`, `a || b` is
/// `ite(a, true, b)`, `a ==> b` is `ite(a, b, true)`, `a <= b` is `!(b < a)`.
#[derive(Clone, Debug)]
pub(crate) enum STerm {
    Var(usize),
    Int(i64),
    Bool(bool),
    /// The empty collection of the table's kind.
    Empty,
    /// The singleton collection of the table's kind, `Seq(e)`, `Set(e)` or
    /// `Multiset(e)` (never a map's: see `CollKind::has_unit`).
    Unit(Box<STerm>),
    /// A collection operation on a collection of `E` (a `Range` is on
    /// `Seq[Int]`), operands in `PureInst::Coll` order. Never an `Empty`,
    /// which is [`STerm::Empty`].
    Op(CollOp, Vec<STerm>),
    /// An operation on a collection of another kind, whose type arguments are
    /// the table's own at these positions: `k in domain(m)` in a map axiom is
    /// a `SetOp::Contains` on a set of `K`, `[0]`, and `v in range(m)` one on
    /// a set of `V`, `[1]`.
    OpOn(CollOp, Box<[usize]>, Vec<STerm>),
    /// A witness: a skolem function applied to its operands.
    Skolem(Witness, Vec<STerm>),
    Bin(BinOp, Box<[STerm; 2]>),
    Ite(Box<[STerm; 3]>),
}

/// The skolem functions of the collection axioms, each the witness of an
/// existential in a conclusion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Witness {
    /// `Seq_skolem(s, x)`: an index of `x` in `s`, when `x in s`.
    SeqIndexOf,
    /// `Seq_skolem_diff(a, b)`: an index where `a` and `b` differ, when they
    /// have one. Both equality axioms use it, as in the domain: one function
    /// satisfies both.
    SeqDiff,
    /// `Set_skolem_diff(a, b)`: an element of exactly one of `a` and `b`, when
    /// they have one.
    SetDiff,
    /// An element of `s`, when `|s| != 0`: the existential of Dafny's
    /// `Set_card_zero`.
    SetMember,
    /// An element of `a` that is not in `b`, when there is one: the
    /// counterexample to `a subset b`, which skolemizes the nested `forall` of
    /// Dafny's `Set_subset_def`.
    SetSubset,
    /// A key where the maps `a` and `b` differ, when they do: in one domain
    /// and not the other, or in both with different values. Skolemizes the
    /// two `forall`s of Dafny's `Map_equal_def`.
    MapDiff,
    /// A key of `m`, when `|m| != 0`: the existential of `Map_card_zero`.
    MapMember,
    /// `Map_range_domain_skolem(m, v)`: a key of `m` with value `v`, when
    /// `v in range(m)`.
    MapRangeKey,
    /// `Multiset_skolem_diff(a, b)`: an element whose count in `a` and `b`
    /// differs, when there is one.
    MultisetDiff,
    /// An element that occurs in `s`, when `|s| != 0`: the existential of
    /// Dafny's `MS_card_zero`.
    MultisetMember,
    /// An element that occurs more often in `a` than in `b`, when there is
    /// one: the counterexample to `a subset b`, which skolemizes the nested
    /// `forall` of Dafny's `MS_subset_def`.
    MultisetSubset,
}

impl Witness {
    /// The witnesses the axioms of `kind` use, in minting order.
    fn of(kind: CollKind) -> &'static [Witness] {
        match kind {
            CollKind::Seq => &[Witness::SeqIndexOf, Witness::SeqDiff],
            CollKind::Set => &[Witness::SetDiff, Witness::SetMember, Witness::SetSubset],
            CollKind::Map => &[Witness::MapDiff, Witness::MapMember, Witness::MapRangeKey],
            CollKind::Multiset => &[
                Witness::MultisetDiff,
                Witness::MultisetMember,
                Witness::MultisetSubset,
            ],
        }
    }

    fn kind(self) -> CollKind {
        match self {
            Witness::SeqIndexOf | Witness::SeqDiff => CollKind::Seq,
            Witness::SetDiff | Witness::SetMember | Witness::SetSubset => CollKind::Set,
            Witness::MapDiff | Witness::MapMember | Witness::MapRangeKey => CollKind::Map,
            Witness::MultisetDiff | Witness::MultisetMember | Witness::MultisetSubset => {
                CollKind::Multiset
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            Witness::SeqIndexOf => "Seq_skolem",
            Witness::SeqDiff => "Seq_skolem_diff",
            Witness::SetDiff => "Set_skolem_diff",
            Witness::SetMember => "Set_skolem_member",
            Witness::SetSubset => "Set_skolem_subset",
            Witness::MapDiff => "Map_skolem_diff",
            Witness::MapMember => "Map_skolem_member",
            Witness::MapRangeKey => "Map_range_domain_skolem",
            Witness::MultisetDiff => "Multiset_skolem_diff",
            Witness::MultisetMember => "Multiset_skolem_member",
            Witness::MultisetSubset => "Multiset_skolem_subset",
        }
    }

    /// The signature, generic over the collection's type arguments: the
    /// element type (`Generic(0)`), or a map's key and value types
    /// (`Generic(0)`, `Generic(1)`).
    fn sig(self) -> SkolemSig {
        let (e, v) = (|| Type::Generic(0), || Type::Generic(1));
        let coll = || match self.kind() {
            CollKind::Map => Type::map(e(), v()),
            kind => Type::coll(kind, [e()]),
        };
        let (params, ret) = match self {
            Witness::SeqIndexOf => (vec![coll(), e()], Type::Int),
            Witness::SeqDiff => (vec![coll(), coll()], Type::Int),
            Witness::SetDiff
            | Witness::SetSubset
            | Witness::MultisetDiff
            | Witness::MultisetSubset => (vec![coll(), coll()], e()),
            Witness::SetMember | Witness::MultisetMember => (vec![coll()], e()),
            Witness::MapDiff => (vec![coll(), coll()], e()),
            Witness::MapMember => (vec![coll()], e()),
            Witness::MapRangeKey => (vec![coll(), v()], e()),
        };
        SkolemSig { params, ret }
    }
}

/// The function ids of one collection kind's [`Witness`]es, generic over the
/// collection's type arguments (see [`Witness::sig`]), minted by the registry
/// (`FuncRegistry::coll_axioms`).
#[derive(Clone, Debug)]
pub(crate) struct Skolems {
    funcs: Vec<(Witness, FuncId)>,
}

impl Skolems {
    /// Mint every witness function of `kind` through `mint`, with its name and
    /// signature.
    pub(crate) fn mint(kind: CollKind, mut mint: impl FnMut(&str, SkolemSig) -> FuncId) -> Self {
        Skolems {
            funcs: Witness::of(kind)
                .iter()
                .map(|w| (*w, mint(w.name(), w.sig())))
                .collect(),
        }
    }

    fn func(&self, w: Witness) -> FuncId {
        self.funcs
            .iter()
            .find(|(x, _)| *x == w)
            .unwrap_or_else(|| panic!("witness {w:?} was not minted with its kind's axioms"))
            .1
    }
}

/// One axiom: `forall vars :: {triggers} guards ==> concls`.
pub(crate) struct CollAxiom {
    pub(crate) name: &'static str,
    /// The collection kind every collection term of the axiom has.
    pub(crate) kind: CollKind,
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

/// The axioms of `kind`, checked once when first built.
pub(crate) fn axioms(kind: CollKind) -> &'static [CollAxiom] {
    static SEQ: LazyLock<Vec<CollAxiom>> = LazyLock::new(|| checked(seq::build()));
    static SET: LazyLock<Vec<CollAxiom>> = LazyLock::new(|| checked(set::build()));
    static MAP: LazyLock<Vec<CollAxiom>> = LazyLock::new(|| checked(map::build()));
    static MULTISET: LazyLock<Vec<CollAxiom>> = LazyLock::new(|| checked(multiset::build()));
    match kind {
        CollKind::Seq => &SEQ,
        CollKind::Set => &SET,
        CollKind::Map => &MAP,
        CollKind::Multiset => &MULTISET,
    }
}

fn checked(all: Vec<CollAxiom>) -> Vec<CollAxiom> {
    for ax in &all {
        ax.check();
    }
    all
}

impl CollAxiom {
    /// Every trigger group must bind every variable: the e-graph rule reads σ off
    /// a match, and Z3 rejects a pattern that leaves a binder out. Every term
    /// must be about the axiom's own kind.
    fn check(&self) {
        assert!(
            !self.triggers.is_empty(),
            "collection axiom {} has no trigger",
            self.name
        );
        for group in &self.triggers {
            let mut seen = vec![false; self.vars.len()];
            for t in group {
                t.mark_vars(&mut seen);
            }
            assert!(
                seen.iter().all(|&s| s),
                "a trigger group of collection axiom {} misses a variable",
                self.name
            );
        }
        let terms = self.triggers.iter().flatten();
        for t in terms
            .chain(&self.conds)
            .chain(&self.concls)
            .chain(&self.guard)
        {
            t.check_kind(self.kind, self.name);
        }
    }
}

impl STerm {
    fn mark_vars(&self, seen: &mut [bool]) {
        match self {
            STerm::Var(i) => seen[*i] = true,
            STerm::Int(_) | STerm::Bool(_) | STerm::Empty => {}
            STerm::Unit(e) => e.mark_vars(seen),
            STerm::Skolem(_, args) | STerm::Op(_, args) | STerm::OpOn(_, _, args) => {
                args.iter().for_each(|a| a.mark_vars(seen))
            }
            STerm::Bin(_, args) => args.iter().for_each(|a| a.mark_vars(seen)),
            STerm::Ite(args) => args.iter().for_each(|a| a.mark_vars(seen)),
        }
    }

    fn check_kind(&self, kind: CollKind, axiom: &str) {
        match self {
            STerm::Var(_) | STerm::Int(_) | STerm::Bool(_) | STerm::Empty => {}
            STerm::Unit(e) => e.check_kind(kind, axiom),
            STerm::Op(op, args) => {
                assert_eq!(op.kind(), kind, "collection axiom {axiom} applies {op:?}");
                args.iter().for_each(|a| a.check_kind(kind, axiom));
            }
            // Another kind's operation, at type arguments the table has.
            STerm::OpOn(op, at, args) => {
                assert_ne!(
                    op.kind(),
                    kind,
                    "collection axiom {axiom} applies its own {op:?} as another kind's"
                );
                assert_eq!(
                    at.len(),
                    op.kind().arity(),
                    "collection axiom {axiom} applies {op:?} at {at:?}"
                );
                assert!(
                    at.iter().all(|&i| i < kind.arity()),
                    "collection axiom {axiom} applies {op:?} at {at:?}"
                );
                args.iter().for_each(|a| a.check_kind(kind, axiom));
            }
            STerm::Skolem(w, args) => {
                assert_eq!(
                    w.kind(),
                    kind,
                    "collection axiom {axiom} uses witness {w:?}"
                );
                args.iter().for_each(|a| a.check_kind(kind, axiom));
            }
            STerm::Bin(_, args) => args.iter().for_each(|a| a.check_kind(kind, axiom)),
            STerm::Ite(args) => args.iter().for_each(|a| a.check_kind(kind, axiom)),
        }
    }
}

// Term constructors shared by the tables, named after the Viper syntax they
// stand for.

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
fn op(o: impl Into<CollOp>, args: Vec<STerm>) -> STerm {
    STerm::Op(o.into(), args)
}
fn op_on(o: impl Into<CollOp>, at: &[usize], args: Vec<STerm>) -> STerm {
    STerm::OpOn(o.into(), at.into(), args)
}
fn skolem(w: Witness, args: Vec<STerm>) -> STerm {
    STerm::Skolem(w, args)
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
fn or(a: STerm, b: STerm) -> STerm {
    ite(a, STerm::Bool(true), b)
}
fn implies(a: STerm, b: STerm) -> STerm {
    ite(a, b, STerm::Bool(true))
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

/// An axiom under construction.
struct Ax {
    name: &'static str,
    kind: CollKind,
    vars: Vec<VarSort>,
    triggers: Vec<Vec<STerm>>,
    conds: Vec<STerm>,
    concls: Vec<STerm>,
    guard: Vec<STerm>,
}

impl Ax {
    fn new(name: &'static str, kind: CollKind, vars: &[VarSort]) -> Self {
        Ax {
            name,
            kind,
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
    fn done(self) -> CollAxiom {
        self.finish(false)
    }
    fn int_only(self) -> CollAxiom {
        self.finish(true)
    }
    fn finish(self, int_only: bool) -> CollAxiom {
        CollAxiom {
            name: self.name,
            kind: self.kind,
            vars: self.vars,
            triggers: self.triggers,
            conds: self.conds,
            concls: self.concls,
            guard: self.guard,
            int_only,
        }
    }
}

/// One recipe per axiom of `kind`, in [`axioms`] order, its witnesses
/// applications of `skolems`. See the module docs.
pub(crate) fn recipes(kind: CollKind, skolems: &Skolems) -> Vec<QuantRecipe> {
    axioms(kind).iter().map(|ax| ax.recipe(skolems)).collect()
}

impl CollAxiom {
    fn recipe(&self, skolems: &Skolems) -> QuantRecipe {
        // A range is a `Seq[Int]` whatever `E` is, so its axioms are ground.
        // Otherwise the collection's type arguments are the recipe's own,
        // `Generic(0..arity)`.
        let (tys, n_tys): (Box<[Type]>, usize) = if self.int_only {
            (Box::new([Type::Int]), 0)
        } else {
            let n = self.kind.arity();
            ((0..n).map(Type::Generic).collect(), n)
        };
        // The guard and the body are separate step lists, each starting after
        // the binders.
        let steps = || Steps {
            next: self.vars.len(),
            insts: Vec::new(),
            kind: self.kind,
            tys: tys.clone(),
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
            .map(|g| g.iter().map(|t| trigger(t, self.kind, &tys)).collect())
            .collect();
        QuantRecipe {
            n_caps: 0,
            bound: self.vars.iter().map(|v| v.ty(self.kind, &tys)).collect(),
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
struct Steps<'a> {
    next: usize,
    insts: Vec<AxiomInst>,
    kind: CollKind,
    /// The collection's type arguments.
    tys: Box<[Type]>,
    skolems: &'a Skolems,
}

impl Steps<'_> {
    fn push(&mut self, p: AxiomPure) -> Val {
        self.insts.push(AxiomInst::Val(p));
        self.next += 1;
        Val::Temp(self.next - 1)
    }

    /// The value of `t`, adding the steps that build it.
    fn emit(&mut self, t: &STerm) -> Val {
        let tys = self.tys.clone();
        match t {
            STerm::Var(i) => Val::Temp(*i),
            STerm::Int(n) => Val::Literal(Literal::Int((*n).into())),
            STerm::Bool(b) => Val::Literal(Literal::Bool(*b)),
            STerm::Empty => self.push(AxiomPure::Coll {
                tys,
                op: CollOp::empty(self.kind),
                args: vec![],
            }),
            STerm::Unit(e) => {
                let e = self.emit(e);
                self.push(AxiomPure::CollLit {
                    kind: self.kind,
                    elem_ty: unit_elem(self.kind, &tys),
                    elem: e,
                })
            }
            STerm::Op(op, args) => {
                assert!(
                    !op.is_eq(),
                    "a collection axiom has the program's typed equality only in a trigger"
                );
                let args = args.iter().map(|a| self.emit(a)).collect();
                self.push(AxiomPure::Coll {
                    tys: op.coll_tys(&tys),
                    op: *op,
                    args,
                })
            }
            STerm::OpOn(op, at, args) => {
                assert!(
                    !op.is_eq(),
                    "a collection axiom has the program's typed equality only in a trigger"
                );
                let args = args.iter().map(|a| self.emit(a)).collect();
                self.push(AxiomPure::Coll {
                    tys: at.iter().map(|&i| tys[i].clone()).collect(),
                    op: *op,
                    args,
                })
            }
            // Generic in the collection's type arguments like the axiom, so an
            // instance's type substitution fixes them.
            STerm::Skolem(w, args) => {
                let args = args.iter().map(|a| self.emit(a)).collect();
                self.push(AxiomPure::App {
                    func: self.skolems.func(*w),
                    type_args: tys.to_vec(),
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

/// The element type of the singleton collection of `kind`, a `Seq` or `Set`
/// of type arguments `tys`.
fn unit_elem(kind: CollKind, tys: &[Type]) -> Type {
    assert!(
        kind.has_unit(),
        "a {kind:?} axiom builds a one-element literal"
    );
    tys[0].clone()
}

/// A trigger term of a `kind` axiom at collection type arguments `tys`:
/// always rooted at a collection term.
fn trigger(t: &STerm, kind: CollKind, tys: &[Type]) -> PreparedTerm {
    let app = |head, args: &[STerm]| PreparedTerm::App {
        head,
        args: args.iter().map(|a| trigger(a, kind, tys)).collect(),
    };
    match t {
        STerm::Var(i) => PreparedTerm::Bound(*i),
        STerm::Int(n) => PreparedTerm::Lit(Literal::Int((*n).into())),
        STerm::Bool(b) => PreparedTerm::Lit(Literal::Bool(*b)),
        STerm::Empty => app(TermHead::Coll(tys.into(), CollOp::empty(kind)), &[]),
        STerm::Unit(e) => app(
            TermHead::CollLit(kind, unit_elem(kind, tys)),
            std::slice::from_ref(&**e),
        ),
        STerm::Op(op, args) => app(TermHead::Coll(op.coll_tys(tys), *op), args),
        STerm::OpOn(op, at, args) => app(
            TermHead::Coll(at.iter().map(|&i| tys[i].clone()).collect(), *op),
            args,
        ),
        STerm::Bin(..) | STerm::Ite(..) | STerm::Skolem(..) => {
            panic!("a collection-axiom trigger is built from collection terms, not {t:?}")
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::vmir::CollKind;

    #[test]
    fn every_trigger_group_binds_every_variable() {
        // `axioms()` checks each axiom when the table is first built.
        assert!(!super::axioms(CollKind::Seq).is_empty());
        assert!(!super::axioms(CollKind::Set).is_empty());
        assert!(!super::axioms(CollKind::Map).is_empty());
        assert!(!super::axioms(CollKind::Multiset).is_empty());
    }

    /// Every multiset axiom keeps to its own kind: a multiset table builds
    /// no set term, so `x in m` is always the count, never a set membership.
    #[test]
    fn multiset_recipes_are_generic_over_the_element() {
        use super::*;
        let skolems = Skolems::mint(CollKind::Multiset, {
            let mut next = 0;
            move |_, _| {
                next += 1;
                FuncId(2000 + next)
            }
        });
        let recipes = recipes(CollKind::Multiset, &skolems);
        let names: Vec<&str> = axioms(CollKind::Multiset).iter().map(|a| a.name).collect();
        for (r, name) in recipes.iter().zip(&names) {
            assert_eq!(r.n_tys, 1, "{name}");
        }
        let at = names
            .iter()
            .position(|n| *n == "multiset-count-nonneg")
            .expect("multiset-count-nonneg");
        let PreparedTerm::App { head, .. } = &recipes[at].groups[0][0] else {
            panic!("multiset-count-nonneg triggers on an application");
        };
        assert!(
            *head
                == TermHead::Coll(
                    Box::new([Type::Generic(0)]),
                    crate::vmir::MultisetOp::Count.into()
                ),
            "`x in m` is the multiset count at the element type"
        );
    }

    /// A map recipe is generic over the key and the value type, and a set
    /// term in it is at the type it tests: `|domain(m)|` at the keys.
    #[test]
    fn map_recipes_are_generic_over_key_and_value() {
        use super::*;
        let skolems = Skolems::mint(CollKind::Map, {
            let mut next = 0;
            move |_, _| {
                next += 1;
                FuncId(1000 + next)
            }
        });
        let recipes = recipes(CollKind::Map, &skolems);
        let names: Vec<&str> = axioms(CollKind::Map).iter().map(|a| a.name).collect();
        for (r, name) in recipes.iter().zip(&names) {
            assert_eq!(r.n_tys, 2, "{name}");
        }
        let at = names
            .iter()
            .position(|n| *n == "map-domain-card")
            .expect("map-domain-card");
        let PreparedTerm::App { head, args } = &recipes[at].groups[0][0] else {
            panic!("map-domain-card triggers on an application");
        };
        assert!(
            *head
                == TermHead::Coll(
                    Box::new([Type::Generic(0)]),
                    crate::vmir::SetOp::Card.into()
                ),
            "`|domain(m)|` is a set cardinality at the key type"
        );
        let [PreparedTerm::App { head, .. }] = &args[..] else {
            panic!("`|domain(m)|` applies to `domain(m)`");
        };
        assert!(
            *head
                == TermHead::Coll(
                    Box::new([Type::Generic(0), Type::Generic(1)]),
                    crate::vmir::MapOp::Domain.into()
                ),
            "`domain(m)` is at the key and value types"
        );
    }

    /// A set term in a map axiom names type arguments the map has.
    #[test]
    #[should_panic(expected = "applies")]
    fn foreign_term_at_a_missing_type_argument_is_rejected() {
        use super::*;
        let bad = STerm::OpOn(
            crate::vmir::SetOp::Card.into(),
            Box::new([2]),
            vec![STerm::Var(0)],
        );
        bad.check_kind(CollKind::Map, "test");
    }
}
