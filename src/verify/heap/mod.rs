// The heap's internals: only `interface` is used from outside.
mod agree;
mod algebra;
mod cond;
mod holder;
mod interface;
mod quantified_permissions;
mod value;
mod view;

pub(crate) use interface::{
    Access, Captures, ConsumeMode, Framing, Loc, QpCheck, QuantSpec, ReadRecipe, Value,
};
pub(crate) use quantified_permissions::{InverseAxioms, ReadResolver, resolve_reads};

use std::borrow::Cow;
use std::cell::Cell;
use std::rc::Rc;

use lasso::Spur;

use crate::verify::context::VerifyContext;
use crate::verify::error::VerifyError;
use crate::verify::lang::Symbolic;
use crate::vmir::{Bound, Literal, Polarity, Type};
use cond::{Cube, Lit};
use holder::{Chunk, QuantChunk, SetBody};

/// S0 merge instrumentation (gated by `SILVER_OXIDE_TRACE_MERGE`). Counts the
/// join-merge structures the pc-hoist plan aims to flatten: branches actually
/// built (a `Select`, or one literal added to a `Gate` — the one-sided branch a
/// zero-armed `Select` was), `0`-leaf constructions (absence encoded as a `?:0`
/// amount), and the deepest branch nesting seen. Reset/dumped per
/// method by [`MergeTrace::take`]. Pure diagnostics — no effect when the flag is
/// off (the `enabled()` check short-circuits every hot-path bump).
#[derive(Default, Clone, Copy)]
pub struct MergeTrace {
    pub selects_built: u64,
    pub zero_leaves: u64,
    pub max_depth: u32,
}

thread_local! {
    static MERGE_TRACE: Cell<MergeTrace> = const { Cell::new(MergeTrace {
        selects_built: 0,
        zero_leaves: 0,
        max_depth: 0,
    }) };
    static MERGE_TRACE_ON: Cell<Option<bool>> = const { Cell::new(None) };
}

impl MergeTrace {
    fn enabled() -> bool {
        MERGE_TRACE_ON.with(|c| match c.get() {
            Some(b) => b,
            None => {
                let b = std::env::var_os("SILVER_OXIDE_TRACE_MERGE").is_some();
                c.set(Some(b));
                b
            }
        })
    }

    fn bump_select(depth: u32) {
        if !Self::enabled() {
            return;
        }
        MERGE_TRACE.with(|c| {
            let mut t = c.get();
            t.selects_built += 1;
            t.max_depth = t.max_depth.max(depth);
            c.set(t);
        });
    }

    fn bump_zero() {
        if !Self::enabled() {
            return;
        }
        MERGE_TRACE.with(|c| {
            let mut t = c.get();
            t.zero_leaves += 1;
            c.set(t);
        });
    }

    /// Read and reset the per-thread counters. `None` when tracing is off.
    pub fn take() -> Option<MergeTrace> {
        if !Self::enabled() {
            return None;
        }
        MERGE_TRACE.with(|c| {
            let t = c.get();
            c.set(MergeTrace::default());
            Some(t)
        })
    }
}

/// A permission amount as an explicit term, held OUTSIDE the union-find so a
/// branch structure collapses without saturation. The conditions *on the amount
/// itself* (a join's divergence, a gated `acc`, an address gate, a chunk set's
/// domain) live in this tree. The condition under which its holder is held at
/// all is the holder's presence, kept apart and applied by
/// [`holder::Holder::gate`]: a bare amount is not a holder's permission.
///
/// - `Zero` is no permission. Structural: it becomes an e-node (`0/1`) only when
///   lowered by [`ChunkPerm::to_id`], so absence costs nothing to build or test.
/// - `Leaf` is an e-class id (a literal `1/1`, a symbolic real, a wildcard),
///   treated as OPAQUE: never structurally decomposed.
/// - `Gate` is `when ? then : 0` over a flat cube: a presence guard, an address
///   gate, a gated `acc`. Built only by [`ChunkPerm::gate`] (or by
///   [`ChunkPerm::select`] when one arm is zero), so `when` is never empty and
///   `then` is never zero. Its literals are a chain read **last-outermost**: it
///   means what `select(lₙ, … select(l₁, then, 0) …, 0)` does.
/// - `Select` is a genuine two-sided branch: a control-flow **join** whose arms
///   hold different amounts, or a gated `acc` with a non-zero else arm.
///
/// Recursive operations over the tree should not match on `Gate` themselves:
/// [`ChunkPerm::shape`] presents it as the branch on its outermost literal.
#[derive(Debug, Clone)]
pub enum ChunkPerm {
    Zero,
    Leaf {
        id: egg::Id,
        /// Whether this amount **came from** a `wildcard` — origin, not sign. A
        /// `1/2` is positive too, and must not trigger [`perm_add_wildcard`]'s
        /// fresh-share rule; a `wildcard` must, however its class const-folds.
        /// Carried structurally because the e-graph cannot answer it: congruence
        /// puts wildcard-bearing terms into a literal's class, and by the time the
        /// merge side sees two amounts their provenance is otherwise gone.
        wild: bool,
    },
    Gate {
        when: Cube,
        then: Box<ChunkPerm>,
    },
    Select {
        cond: egg::Id,
        then: Box<ChunkPerm>,
        els: Box<ChunkPerm>,
    },
}

/// One node of a [`ChunkPerm`] seen as at most a single branch: a `Gate` is
/// peeled to the branch on its outermost literal, with `Zero` on the other arm.
/// What recursive operations match on, so they handle a `Gate` exactly as the
/// `Select` chain it stands for.
pub(super) enum Shape<'a> {
    Zero,
    Leaf {
        id: egg::Id,
        wild: bool,
    },
    Branch {
        cond: egg::Id,
        then: Cow<'a, ChunkPerm>,
        els: Cow<'a, ChunkPerm>,
    },
}

impl ChunkPerm {
    /// A concrete leaf: an amount with no `wildcard` in its provenance.
    pub(crate) fn leaf(id: egg::Id) -> Self {
        ChunkPerm::Leaf { id, wild: false }
    }

    /// A leaf minted from a `wildcard` (or from a term one flowed into). Selects
    /// the fresh-share rule on the produce side; see the `wild` field.
    pub(crate) fn wild_leaf(id: egg::Id) -> Self {
        ChunkPerm::Leaf { id, wild: true }
    }

    /// This node as at most one branch (see [`Shape`]).
    pub(super) fn shape(&self) -> Shape<'_> {
        match self {
            ChunkPerm::Zero => Shape::Zero,
            ChunkPerm::Leaf { id, wild } => Shape::Leaf {
                id: *id,
                wild: *wild,
            },
            ChunkPerm::Select { cond, then, els } => Shape::Branch {
                cond: *cond,
                then: Cow::Borrowed(then),
                els: Cow::Borrowed(els),
            },
            ChunkPerm::Gate { when, then } => {
                let (&(cond, pol), rest) = when.split_last().expect("a gate's cube is never empty");
                let inner = if rest.is_empty() {
                    Cow::Borrowed(&**then)
                } else {
                    Cow::Owned(ChunkPerm::Gate {
                        when: Cube::from(rest),
                        then: then.clone(),
                    })
                };
                let zero = Cow::Owned(ChunkPerm::Zero);
                match pol {
                    Polarity::Positive => Shape::Branch {
                        cond,
                        then: inner,
                        els: zero,
                    },
                    Polarity::Negative => Shape::Branch {
                        cond,
                        then: zero,
                        els: inner,
                    },
                }
            }
        }
    }

    /// The branch nesting depth of this perm tree (a leaf is 0). Diagnostic.
    fn depth(&self) -> u32 {
        match self {
            ChunkPerm::Zero | ChunkPerm::Leaf { .. } => 0,
            ChunkPerm::Gate { when, then } => when.len() as u32 + then.depth(),
            ChunkPerm::Select { then, els, .. } => 1 + then.depth().max(els.depth()),
        }
    }

    /// Whether any leaf of this tree came from a `wildcard`.
    pub(crate) fn has_wild(&self) -> bool {
        match self {
            ChunkPerm::Zero => false,
            ChunkPerm::Leaf { wild, .. } => *wild,
            ChunkPerm::Gate { then, .. } => then.has_wild(),
            ChunkPerm::Select { then, els, .. } => then.has_wild() || els.has_wild(),
        }
    }

    /// Scale every leaf by `pm`, preserving the branch structure.
    ///
    /// Distributing rather than wrapping (`ite(c, pm*t, pm*e)` over
    /// `pm * ite(c, t, e)`) is what keeps the tree intact through an
    /// `unfolding`'s or `unfold`'s multiplier. The two are equal but not
    /// identical; `mul-one-real-l` puts them in one class once `ctx.reduce()`
    /// runs, which is why the common `1/1` scale costs nothing. `pm_wild`: the
    /// scale itself came from a `wildcard`, which makes every scaled leaf
    /// wildcard-derived, whatever its own amount was.
    pub(crate) fn scaled(
        self,
        ctx: &mut VerifyContext<'_>,
        pm: egg::Id,
        pm_wild: bool,
    ) -> ChunkPerm {
        match self {
            ChunkPerm::Zero => ChunkPerm::Zero,
            ChunkPerm::Leaf { id, wild } => ChunkPerm::Leaf {
                id: ctx.add(Symbolic::Binary(crate::vmir::BinOp::MulR, [pm, id])),
                wild: wild || pm_wild,
            },
            ChunkPerm::Gate { when, then } => {
                let t = then.scaled(ctx, pm, pm_wild);
                ChunkPerm::gate(ctx, &when, t)
            }
            ChunkPerm::Select { cond, then, els } => {
                let t = then.scaled(ctx, pm, pm_wild);
                let e = els.scaled(ctx, pm, pm_wild);
                ChunkPerm::select(ctx, cond, t, e)
            }
        }
    }

    /// Whether this is no permission: `Zero`, or a leaf whose class const-folds
    /// to `0`. Const-fold only, no prover.
    pub(super) fn is_zero(&self, ctx: &VerifyContext<'_>) -> bool {
        match self {
            ChunkPerm::Zero => true,
            ChunkPerm::Leaf { id, .. } => matches!(
                ctx.egraph[ctx.egraph.find(*id)].data.known(),
                Some(Literal::Real(r)) if *r == num::BigRational::from(num::BigInt::from(0))
            ),
            ChunkPerm::Gate { .. } | ChunkPerm::Select { .. } => false,
        }
    }

    /// Structural equality with LEAVES compared by e-class `find` (so a `1/1`
    /// from either arm counts as equal), a `Gate` compared as the chain it
    /// stands for, and every zero equal to every other. O(size), no saturation.
    fn same(ctx: &VerifyContext<'_>, a: &ChunkPerm, b: &ChunkPerm) -> bool {
        if a.is_zero(ctx) || b.is_zero(ctx) {
            return a.is_zero(ctx) && b.is_zero(ctx);
        }
        match (a.shape(), b.shape()) {
            (Shape::Leaf { id: x, .. }, Shape::Leaf { id: y, .. }) => {
                ctx.egraph.find(x) == ctx.egraph.find(y)
            }
            (
                Shape::Branch {
                    cond: c1,
                    then: t1,
                    els: e1,
                },
                Shape::Branch {
                    cond: c2,
                    then: t2,
                    els: e2,
                },
            ) => {
                ctx.egraph.find(c1) == ctx.egraph.find(c2)
                    && Self::same(ctx, &t1, &t2)
                    && Self::same(ctx, &e1, &e2)
            }
            _ => false,
        }
    }

    /// Descend into an arm while it re-branches on the SAME condition class,
    /// taking the then-side (`take_then`) or els-side. This is the `ite`
    /// idempotence identity `ite(c, ite(c, a, b), e) = ite(c, a, e)` (and the
    /// dual for the els arm): under `c` an inner `ite(c, …)` is decided, so only
    /// its matching branch survives. CFG lowering makes many joins in one match
    /// arm share the arm's reach `cond`, so without this the perm accretes a
    /// redundant `Select` layer per such join.
    ///
    /// A `Gate` mentioning `cond` anywhere in its cube is decided too: under the
    /// literal's own polarity it drops out, under the other the gate is `Zero`.
    fn collapse_same_cond(
        ctx: &VerifyContext<'_>,
        cond_c: egg::Id,
        arm: ChunkPerm,
        take_then: bool,
    ) -> ChunkPerm {
        match arm {
            ChunkPerm::Select {
                cond: c2,
                then,
                els,
            } if ctx.egraph.find(c2) == cond_c => {
                let inner = if take_then { *then } else { *els };
                Self::collapse_same_cond(ctx, cond_c, inner, take_then)
            }
            ChunkPerm::Gate { when, then } => {
                let Some(k) = when.iter().position(|(i, _)| ctx.egraph.find(*i) == cond_c) else {
                    return ChunkPerm::Gate { when, then };
                };
                if (when[k].1 == Polarity::Positive) != take_then {
                    return ChunkPerm::Zero;
                }
                let rest: Vec<Lit> = when
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != k)
                    .map(|(_, l)| *l)
                    .collect();
                let inner = if rest.is_empty() {
                    *then
                } else {
                    ChunkPerm::Gate {
                        when: Cube::from(rest),
                        then,
                    }
                };
                Self::collapse_same_cond(ctx, cond_c, inner, take_then)
            }
            other => other,
        }
    }

    /// This tree as seen from *inside* one arm of `cond`: every re-branch on the
    /// same condition class is decided, so only the matching side survives. The
    /// public face of [`Self::collapse_same_cond`], used by `perm_add` to restrict
    /// one addend while descending into the other.
    fn restrict(
        ctx: &VerifyContext<'_>,
        cond: egg::Id,
        arm: ChunkPerm,
        take_then: bool,
    ) -> ChunkPerm {
        Self::collapse_same_cond(ctx, ctx.egraph.find(cond), arm, take_then)
    }

    /// `when ? perm : 0`, the gate smart constructor. Literals are applied in
    /// order, each becoming the new outermost one, and each is simplified the way
    /// [`Self::select`] simplifies a branch: a literal already decided inside
    /// `perm` is collapsed, a zero amount stays `Zero`, and a literal that
    /// const-folds drops (true) or zeroes the whole gate (false). With an empty
    /// cube this is `perm` unchanged.
    pub(crate) fn gate(ctx: &VerifyContext<'_>, when: &[Lit], perm: ChunkPerm) -> ChunkPerm {
        when.iter()
            .fold(perm, |acc, &lit| Self::gate_lit(ctx, lit, acc))
    }

    fn gate_lit(ctx: &VerifyContext<'_>, (id, pol): Lit, perm: ChunkPerm) -> ChunkPerm {
        let c = ctx.egraph.find(id);
        let positive = pol == Polarity::Positive;
        let perm = Self::collapse_same_cond(ctx, c, perm, positive);
        if perm.is_zero(ctx) {
            return ChunkPerm::Zero;
        }
        match ctx.egraph[c].data.known() {
            Some(Literal::Bool(b)) if *b == positive => return perm,
            Some(Literal::Bool(_)) => return ChunkPerm::Zero,
            _ => {}
        }
        let out = match perm {
            ChunkPerm::Gate { when, then } => {
                let mut lits = when.to_vec();
                lits.push((id, pol));
                ChunkPerm::Gate {
                    when: Cube::from(lits),
                    then,
                }
            }
            other => ChunkPerm::Gate {
                when: Cube::from(vec![(id, pol)]),
                then: Box::new(other),
            },
        };
        // A gate literal is the one-sided branch a `Select` with a zero arm used
        // to be: count it as one, so the trace means what it did.
        MergeTrace::bump_select(out.depth());
        out
    }

    /// The join-select smart constructor. `cond` is the then-edge reach value.
    /// Applies, in order: (0) `ite`-idempotence flattening of arms that branch on
    /// the same `cond`; (i) same-amount collapse (`then ≡ els ⇒ then`, the
    /// give-back / untouched kill); (ii) dead-arm drop when `cond` const-folds to
    /// a boolean literal; (iii) a `Gate` when one arm is zero; (iv) otherwise a
    /// `Select`. Deterministic, bounded.
    pub fn select(
        ctx: &mut VerifyContext<'_>,
        cond: egg::Id,
        then: ChunkPerm,
        els: ChunkPerm,
    ) -> Self {
        let cond_c = ctx.egraph.find(cond);
        let then = Self::collapse_same_cond(ctx, cond_c, then, true);
        let els = Self::collapse_same_cond(ctx, cond_c, els, false);
        if Self::same(ctx, &then, &els) {
            return then;
        }
        match ctx.egraph[cond_c].data.known() {
            Some(Literal::Bool(true)) => return then,
            Some(Literal::Bool(false)) => return els,
            _ => {}
        }
        if els.is_zero(ctx) {
            return Self::gate_lit(ctx, (cond, Polarity::Positive), then);
        }
        if then.is_zero(ctx) {
            return Self::gate_lit(ctx, (cond, Polarity::Negative), els);
        }
        let out = ChunkPerm::Select {
            cond,
            then: Box::new(then),
            els: Box::new(els),
        };
        MergeTrace::bump_select(out.depth());
        out
    }

    /// Lower to an e-graph id — ONLY where the prover needs an e-class
    /// (sufficiency, bound/non-alias axioms, `perm > 0` framing). For a `Leaf`
    /// this is the identity (no node added). A `Gate` lowers to exactly the
    /// `ite(l, …, 0)` chain it stands for.
    pub fn to_id(&self, ctx: &mut VerifyContext<'_>) -> egg::Id {
        match self {
            ChunkPerm::Zero => expr!(ctx, 0 / 1),
            ChunkPerm::Leaf { id, .. } => *id,
            ChunkPerm::Gate { when, then } => {
                let mut acc = then.to_id(ctx);
                for &(lit, pol) in when.iter() {
                    let zero = expr!(ctx, 0 / 1);
                    let arms = match pol {
                        Polarity::Positive => [lit, acc, zero],
                        Polarity::Negative => [lit, zero, acc],
                    };
                    acc = ctx.add(Symbolic::Ite(arms));
                }
                acc
            }
            ChunkPerm::Select { cond, then, els } => {
                let t = then.to_id(ctx);
                let e = els.to_id(ctx);
                ctx.add(Symbolic::Ite([*cond, t, e]))
            }
        }
    }

    /// The leaf amount id if this is a bare `Leaf` (no branch structure).
    pub fn as_leaf(&self) -> Option<egg::Id> {
        match self {
            ChunkPerm::Leaf { id, .. } => Some(*id),
            _ => None,
        }
    }

    /// Visit every non-zero leaf amount id **with the branch conditions that
    /// reach it** — the conditions on the path from the root, outermost first,
    /// each with the polarity of the arm taken.
    ///
    /// A leaf only describes the permission *under its own arm*, so any fact
    /// stated about it must be gated by that cube. `Zero` arms hold nothing and
    /// are skipped.
    fn for_each_leaf_under(&self, f: &mut impl FnMut(egg::Id, &[Lit])) {
        fn go(p: &ChunkPerm, path: &mut Vec<Lit>, f: &mut impl FnMut(egg::Id, &[Lit])) {
            match p {
                ChunkPerm::Zero => {}
                ChunkPerm::Leaf { id, .. } => f(*id, path),
                ChunkPerm::Gate { when, then } => {
                    let n = path.len();
                    path.extend(when.iter().rev());
                    go(then, path, f);
                    path.truncate(n);
                }
                ChunkPerm::Select { cond, then, els } => {
                    path.push((*cond, Polarity::Positive));
                    go(then, path, f);
                    path.pop();
                    path.push((*cond, Polarity::Negative));
                    go(els, path, f);
                    path.pop();
                }
            }
        }
        go(self, &mut Vec::new(), f);
    }

    /// A representative e-class id for debug display, WITHOUT mutating the graph
    /// (a `Leaf`'s id, or a branch's condition); `None` for `Zero`, which has no
    /// e-class until lowered. Viz and tests only — not a real perm id.
    fn repr_id(&self) -> Option<egg::Id> {
        match self {
            ChunkPerm::Zero => None,
            ChunkPerm::Leaf { id, .. } => Some(*id),
            ChunkPerm::Gate { when, .. } => when.last().map(|&(c, _)| c),
            ChunkPerm::Select { cond, .. } => Some(*cond),
        }
    }
}

/// The **kind** of a heap location: a field/predicate group, the held value
/// type, and the permission bound. This is exactly the content of a location's
/// `Type::Addr` — the chunks of one kind share a group in the heap. Sourced
/// straight from VMIR (the address's `Type::Addr`), never from e-graph inference.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocationKind {
    pub group: Spur,
    pub value: Type,
    pub bound: Bound,
}

impl LocationKind {
    /// Extract the kind from an address type. `None` for a non-`Addr` type.
    pub fn from_addr_type(ty: &Type) -> Option<Self> {
        match ty {
            Type::Addr {
                group,
                value,
                bound,
            } => Some(Self {
                group: *group,
                value: (**value).clone(),
                bound: bound.clone(),
            }),
            _ => None,
        }
    }
}

/// The identity of a chunk set: minted where the set is created (its inhale,
/// or a produced footprint slot) and kept by every later version of it (a
/// take, a write, a join). Two sets with one origin are copies of one inhale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SetId(u32);

/// The identity of one exhale recorded against a chunk set ([`QPTaken`]),
/// minted when it is recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TakeId(u32);

impl SetId {
    pub(crate) fn fresh(ctx: &mut VerifyContext<'_>) -> SetId {
        SetId(ctx.fresh_heap_id())
    }
}

/// The identity of one [`Handover`], minted when it is recorded. Copies of one
/// set share the handovers recorded before they diverged, and only those.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct HandoverId(u32);

impl HandoverId {
    pub(crate) fn fresh(ctx: &mut VerifyContext<'_>) -> HandoverId {
        HandoverId(ctx.fresh_heap_id())
    }
}

/// What a chunk set `kept` learned when an exhale took from another set `from`
/// of its kind: wherever both held a positive share just before that exhale,
/// their values were one. Recorded on the set that keeps its share, so the
/// value survives the exhale through it.
///
/// ```text
/// pc ∧ 0 < perm(from, r) ∧ 0 < perm(kept, r)  ==>  lookup(kept.map, r) == value(from, r)
/// ```
///
/// `from` and `kept` are the two sets **as they were just before the exhale**:
/// their amounts are `qp_perm_at` of those versions, so what either had given
/// up before is already subtracted, and the fact is about their maps then. It
/// stays true after: a later take or write changes the sets, not what held at
/// that moment. `value(from, r)` is read through `from` itself, so its own
/// handovers carry along (`A → B → C`).
///
/// Like [`Restored`] it is assumed per address, where a read meets the set
/// (`View::shares`): the sets are quantified, and no ground term names the
/// addresses they share.
#[derive(Debug, Clone)]
struct Handover {
    pub(crate) id: HandoverId,
    pub(crate) from: QuantChunk,
    pub(crate) kept: QuantChunk,
    /// The exhale's path condition: both sets were held only there.
    pub(crate) pc: Cube,
}

impl TakeId {
    pub(crate) fn fresh(ctx: &mut VerifyContext<'_>) -> TakeId {
        TakeId(ctx.fresh_heap_id())
    }
}

/// A value map minted when a set taken from a chunk set came back (a loop
/// exit hands back what its head took, see `algebra::combine::union_heaps`):
/// inside the taken `region` it reads the returned values `new`, elsewhere the
/// set's own `old` ones.
///
/// ```text
/// lookup(map, r) == (0 < perm(region, r) ? lookup(new, r) : lookup(old, r))
/// 0 < perm(region, r) && 0 < perm(kept, r)  ==>  lookup(new, r) == lookup(old, r)
/// ```
///
/// The second line is value agreement: where the set `kept` (the one given
/// back to, as it was before) held a share too, the values cannot have changed.
///
/// `map` is fresh, so this only defines it. It is assumed per address, where
/// a read meets it (`View::shares`): the region is a chunk set, which no
/// ground term names.
#[derive(Debug, Clone)]
struct Restored {
    pub(crate) map: egg::Id,
    pub(crate) region: QuantChunk,
    pub(crate) kept: QuantChunk,
    pub(crate) new: egg::Id,
    pub(crate) old: egg::Id,
}

/// One exhale recorded against a [`QuantChunk`] (its `taken` list): what was
/// taken, and which exhale took it. Recorded as a [`holder::Debit`], under the
/// presence it counts under: empty for a single location, the exhaled set's
/// own otherwise, and extended per join like any holder's.
#[derive(Debug, Clone)]
struct QPTaken {
    /// Which exhale this is. Copies of one set share the entries recorded
    /// before they diverged, and only those share an id.
    pub(crate) id: TakeId,
    pub(crate) what: Taken,
}

/// What an exhale took from a chunk set (see [`QPTaken`]). Its presence is the
/// debit's ([`holder::Debit`]), so none of these carry one.
#[derive(Debug, Clone)]
enum Taken {
    /// A whole chunk set exhaled from this one. It applies the same address
    /// function; its `pc` is the exhale's path condition. Read back as a set
    /// held under the debit's presence ([`holder::Debit::as_set`]).
    Set(SetBody),
    /// `perm` exhaled at the single location `addr`. Counts only at an address
    /// equal to `addr`.
    Single { addr: egg::Id, perm: ChunkPerm },
    /// This set's share of an exhaled set `demand` that no single holder
    /// covered: the exhale split it greedily over every holder of `before`
    /// (the heap just before it), each giving `min(what it holds, what is still
    /// needed)` in `View::all_shares` order. This set is `before`'s set number
    /// `holder`; at an address its share is recomputed from `before` there.
    ///
    /// `pc` is the exhale's path condition, under which (with the debit's
    /// presence) the split is recomputed.
    Share {
        demand: QuantChunk,
        before: Heap,
        holder: usize,
        pc: Cube,
    },
}

/// Everything a heap holds at one location kind: the single chunks and the
/// quantified chunk sets. Each part is an `Rc<[..]>` so cloning a heap
/// (frequent — one per instruction) shares them; changing one copies just that
/// slice (copy-on-write).
#[derive(Debug, Clone)]
struct Group {
    /// Identity is the chunk's `addr` e-class: at most one chunk per address.
    points: Rc<[Chunk]>,
    sets: Rc<[QuantChunk]>,
}

impl Group {
    fn empty() -> Self {
        Group {
            points: Rc::from([]),
            sets: Rc::from([]),
        }
    }

    fn is_empty(&self) -> bool {
        self.points.is_empty() && self.sets.is_empty()
    }
}

/// Symbolic heap: holders partitioned by [`LocationKind`], one [`Group`] per
/// kind. The outer `im::HashMap` is structurally shared.
///
/// Every holder's reachability is its own presence cube (see
/// [`holder::Holder`]), relative to the heap's [`Heap::region`]: the cube of
/// the block it was built in. Every use goes through [`Heap::at`], which
/// proves the use's path condition establishes that region.
#[derive(Debug, Clone)]
pub struct Heap {
    /// Unseeded hasher: iteration order (and so the order join/consume code mints
    /// terms in) must be a function of the heap alone, not of the process.
    groups: im::HashMap<LocationKind, Group, rustc_hash::FxBuildHasher>,

    /// Where this heap exists: the cube of the block it was built in (empty in
    /// a body without blocks, and for the empty heap, which is the same
    /// everywhere). Every presence is relative to it: a holder is held on
    /// `region ∧ presence`. Derived heaps (a produce, a consume, a write)
    /// inherit it; the walk rehomes a heap into the block that uses it
    /// ([`Heap::at`]) before building on it, so a result's region is that
    /// block's cube.
    region: Cube,
}

impl Heap {
    /// The empty heap. Its region is empty: holding nothing, it is the same
    /// heap wherever it is used.
    pub fn empty() -> Self {
        Self {
            groups: im::HashMap::default(),
            region: Cube::empty(),
        }
    }

    /// The empty heap of the region `region`: the start of a heap built for a
    /// block, such as a join's merge.
    fn empty_in(region: &[Lit]) -> Self {
        Self {
            region: Cube::from(region),
            ..Self::empty()
        }
    }

    /// This heap used under the path condition `pc`, with `pc` as its region.
    ///
    /// A heap exists only where its region was reached, so using it is
    /// well-defined only where `pc` establishes that region (Silicon's "did not
    /// reach labelled state"): a use from a block the heap's block dominates
    /// always does; an `old[L]` read whose label `L` does not dominate it must
    /// prove it. Then `pc` implies the old region, so every presence means the
    /// same relative to it.
    ///
    /// The reach condition belongs to the *use*, not to the heap's contents:
    /// emptying the heap off its region instead would reject only the reads
    /// that need a chunk (a value), and let those an empty heap satisfies
    /// (`perm(x.f) == none`) hold vacuously.
    pub(crate) fn at(&self, ctx: &mut VerifyContext<'_>, pc: &[Lit]) -> Result<Heap, VerifyError> {
        if !self.region.holds_under(ctx, pc) {
            return Err(VerifyError::SideCondition(
                "label may not have been reached",
            ));
        }
        Ok(Heap {
            region: Cube::from(pc),
            ..self.clone()
        })
    }

    /// The path condition a use of this heap under `pc` stands for: the heap's
    /// region, the paths it exists on, and `pc`, region first (the ambient cube
    /// before the delta, as for any instruction's effective pc). An instruction
    /// passes its own pc, which inside a block is only the delta past the
    /// block's cube; everything the heap proves or states is relative to the
    /// effective pc instead, so that no fact about this heap holds on a path
    /// where it does not exist.
    pub(crate) fn effective_pc(&self, ctx: &VerifyContext<'_>, pc: &[Lit]) -> Vec<Lit> {
        // Disjoint: the use is unreachable, and the conjunction says so.
        cond::meet(ctx, &self.region, pc)
            .unwrap_or_else(|| self.region.iter().chain(pc).copied().collect())
    }

    fn group(&self, kind: &LocationKind) -> Option<&Group> {
        self.groups.get(kind)
    }

    /// This heap with `kind`'s group replaced by `f` of it (dropped when it
    /// comes back empty).
    fn with_group(&self, kind: &LocationKind, f: impl FnOnce(Group) -> Group) -> Self {
        let group = f(self.group(kind).cloned().unwrap_or_else(Group::empty));
        let groups = if group.is_empty() {
            self.groups.without(kind)
        } else {
            self.groups.update(kind.clone(), group)
        };
        Self {
            groups,
            region: self.region.clone(),
        }
    }

    /// The single chunks of one location kind (empty slice if none held).
    fn chunks_of(&self, kind: &LocationKind) -> &[Chunk] {
        self.group(kind).map_or(&[], |g| &g.points[..])
    }

    /// The chunk sets of one location kind (empty slice if none held).
    fn qp_chunks_of(&self, kind: &LocationKind) -> &[QuantChunk] {
        self.group(kind).map_or(&[], |g| &g.sets[..])
    }

    /// The chunk held at `addr` within `kind`'s group (exact e-class match).
    fn chunk(&self, kind: &LocationKind, addr: egg::Id) -> Option<&Chunk> {
        self.chunks_of(kind).iter().find(|c| c.addr == addr)
    }

    /// Insert `chunk` into `kind`'s group, replacing any chunk already at the
    /// same `addr` (one chunk per address).
    fn with_chunk(&self, kind: &LocationKind, chunk: Chunk) -> Self {
        self.with_group(kind, |g| {
            let mut v = g.points.to_vec();
            match v.iter_mut().find(|c| c.addr == chunk.addr) {
                Some(slot) => *slot = chunk,
                None => v.push(chunk),
            }
            Group {
                points: v.into(),
                ..g
            }
        })
    }

    /// Drop the chunk at `addr` in `kind`'s group.
    fn without_chunk(&self, kind: &LocationKind, addr: egg::Id) -> Self {
        if self.group(kind).is_none() {
            return self.clone();
        }
        self.with_group(kind, |g| Group {
            points: g
                .points
                .iter()
                .filter(|c| c.addr != addr)
                .cloned()
                .collect(),
            ..g
        })
    }

    /// Replace the `i`-th chunk set of `kind` (as in [`Self::qp_chunks_of`]).
    fn with_qp_chunk_replaced(&self, kind: &LocationKind, i: usize, set: QuantChunk) -> Self {
        self.with_group(kind, |g| {
            let mut v = g.sets.to_vec();
            v[i] = set;
            Group {
                sets: v.into(),
                ..g
            }
        })
    }

    /// Drop the `i`-th chunk set of `kind`.
    fn with_qp_chunk_removed(&self, kind: &LocationKind, i: usize) -> Self {
        self.with_group(kind, |g| {
            let mut v = g.sets.to_vec();
            v.remove(i);
            Group {
                sets: v.into(),
                ..g
            }
        })
    }

    /// Add the chunk set `set` to `kind`'s group.
    fn with_qp_chunk(&self, kind: &LocationKind, set: QuantChunk) -> Self {
        self.with_group(kind, |g| {
            let mut v = g.sets.to_vec();
            v.push(set);
            Group {
                sets: v.into(),
                ..g
            }
        })
    }

    /// Every single chunk in the heap, paired with its location kind.
    pub fn entries(&self) -> impl Iterator<Item = (&LocationKind, &Chunk)> {
        self.groups
            .iter()
            .flat_map(|(k, g)| g.points.iter().map(move |c| (k, c)))
    }

    /// The location kinds the heap holds anything of.
    fn kinds(&self) -> impl Iterator<Item = &LocationKind> {
        self.groups.keys()
    }

    /// The chunk at `addr` within `kind`'s group, matched by **canonical**
    /// e-class (`ctx.egraph.find`) so an address that only aliases the stored
    /// key under the currently-assumed context still resolves. Used by the
    /// join merge (a chunk held on both arms must line up by identity).
    fn chunk_canon(
        &self,
        ctx: &VerifyContext<'_>,
        kind: &LocationKind,
        addr: egg::Id,
    ) -> Option<&Chunk> {
        let canon = ctx.egraph.find(addr);
        self.chunks_of(kind)
            .iter()
            .find(|c| ctx.egraph.find(c.addr) == canon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two heaps built by the same insertions iterate their groups in the same
    /// order. With a per-map random seed they did not, even within one process.
    #[test]
    fn group_iteration_order_is_deterministic() {
        let mut rodeo = lasso::Rodeo::default();
        let kinds: Vec<LocationKind> = (0..32)
            .map(|i| LocationKind {
                group: rodeo.get_or_intern(format!("g{i}")),
                value: Type::Int,
                bound: Bound::Unbounded,
            })
            .collect();
        let build = || {
            kinds.iter().enumerate().fold(Heap::empty(), |h, (i, k)| {
                let id = egg::Id::from(i);
                h.with_chunk(k, Chunk::new(id, id, id))
            })
        };
        let order = |h: &Heap| h.kinds().cloned().collect::<Vec<_>>();
        assert_eq!(order(&build()), order(&build()));
    }

    /// A value read takes its chunk from the demanded kind only. Two kinds at one
    /// address class cannot arise from Viper on main (each group has its own
    /// address head and nothing unions address classes), so the collision is built
    /// directly: a field `f` and a predicate `P` whose addresses are made equal.
    #[test]
    fn chunk_canon_is_kind_scoped() {
        let mut rodeo = lasso::Rodeo::default();
        let mut kind = |name: &str| LocationKind {
            group: rodeo.get_or_intern(name),
            value: Type::Int,
            bound: Bound::Unbounded,
        };
        let (field, pred, other) = (kind("f"), kind("P"), kind("g"));
        let interner = lasso::Rodeo::new();
        let mut ctx = crate::verify::test_support::fresh_ctx(&interner);
        let fresh =
            |ctx: &mut VerifyContext<'_>, i| ctx.add(crate::verify::lang::Symbolic::Fresh(i));
        let (a_f, a_p) = (fresh(&mut ctx, 0), fresh(&mut ctx, 1));
        let (v_f, v_p, one) = (fresh(&mut ctx, 2), fresh(&mut ctx, 3), fresh(&mut ctx, 4));
        let h = Heap::empty()
            .with_chunk(&pred, Chunk::new(a_p, one, v_p))
            .with_chunk(&field, Chunk::new(a_f, one, v_f));
        ctx.union(a_f, a_p);
        ctx.egraph.rebuild();

        let value_of =
            |k: &LocationKind, addr| h.chunk_canon(&ctx, k, addr).map(|c| c.stored_value());
        assert_eq!(value_of(&field, a_p), Some(v_f));
        assert_eq!(value_of(&pred, a_f), Some(v_p));
        assert_eq!(value_of(&other, a_f), None);
    }
}
