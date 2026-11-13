use crate::vmir::display::VmirDisplay;
use crate::vmir::{MemberId, QuantChunks, ResourceCall, Type, Val};
use std::fmt::{self, Display, Formatter};

/// Heap-typed value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeapVal {
    Empty,
    Temp(usize),
}

/// A permission value: the `p` namespace, VMIR's third kind of operand beside
/// values (`e`) and heaps (`h`).
///
/// A `wildcard` is a symbolic positive-but-unspecified share; it is legal
/// **only** here (never as a first-class [`Val`]), matching Viper. Keeping
/// permissions in their own namespace is what makes "a permission never lands in
/// the e-graph" a property of the *types* rather than of a check.
///
/// Leaves are inline. A plain `acc(x.f, write)` carries its amount here and emits
/// no instruction, so the overwhelmingly common concrete permission costs
/// nothing; only *composition* needs a name, and that is [`PermInst`]. A `Temp`
/// names an earlier `PermInst` in the same block — permissions never cross a
/// block boundary, because a permission has no join (heaps own those).
///
/// Generic over the **amount** operand `A`, with two instantiations:
///
/// - `PermVal<Val>` (the default) — the IR, where an amount is a body temp.
/// - `PermVal<BodyRecipe>` — recipe space (`verify::cert::PermRecipe`), where
///   each amount is a standalone sliced recipe.
///
/// The point of the parameter is that [`PermInst::try_map`] is then **one**
/// function serving both spaces.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PermVal<A = Val> {
    /// A concrete permission amount (`1/1`, `1/2`, a symbolic real, …). This is
    /// the only variant a non-wildcard, unbranched program ever produces, and it
    /// lowers to exactly the same e-graph term as a bare amount always did.
    Amount(A),
    /// Viper's `wildcard`: a positive-but-unspecified share, **picked when the
    /// permission is evaluated** rather than named here. Deliberately carries no
    /// operand — that is what keeps it out of value space.
    Wildcard,
    /// An earlier [`PermInst`] in this block.
    Temp(usize),
}

/// A permission-producing instruction — the definition of a `p` temp.
///
/// Only *conditional composition* needs one: `Ite` carries both
/// `Sink::gate_perm`'s branch gating and the `p > 0 ? wildcard : 0` lowering of
/// function-context permissions (so a dead branch reduces to `0`). Nesting is a
/// chain of `p` temps rather than a boxed expression, so a permission reads like
/// every other VMIR operand.
///
/// Deliberately **not** a [`PureInst`](crate::vmir::PureInst): its result is a
/// `PermVal`, not a `Val`, so nothing can route a permission into a pure term.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PermInst<A = Val> {
    /// `cond ? then : else` over permissions. `cond` is a boolean operand of the
    /// *value* space; the arms are permissions.
    Ite(A, PermVal<A>, PermVal<A>),
}

impl<A> PermVal<A> {
    /// Whether this value is *itself* a `wildcard`. A `Temp` cannot answer on its
    /// own — ask the evaluated `ChunkPerm`'s provenance (`ChunkPerm::has_wild`)
    /// or, statically, scan the defining instructions.
    pub fn is_wildcard(&self) -> bool {
        matches!(self, PermVal::Wildcard)
    }

    /// Rebuild with the amount operand mapped through `f`.
    pub fn try_map<B, E>(&self, f: &mut impl FnMut(&A) -> Result<B, E>) -> Result<PermVal<B>, E> {
        Ok(match self {
            PermVal::Amount(a) => PermVal::Amount(f(a)?),
            PermVal::Wildcard => PermVal::Wildcard,
            PermVal::Temp(i) => PermVal::Temp(*i),
        })
    }
}

impl<A> PermInst<A> {
    /// Whether either arm is a bare `wildcard`.
    pub fn has_wildcard_arm(&self) -> bool {
        match self {
            PermInst::Ite(_, t, e) => t.is_wildcard() || e.is_wildcard(),
        }
    }

    /// Rebuild with every operand (the condition included) mapped through `f`,
    /// preserving the shape. The `Val`-operand form a certificate walk
    /// accumulates becomes a `BodyRecipe`-operand one this way, one `slice` per
    /// operand.
    pub fn try_map<B, E>(&self, f: &mut impl FnMut(&A) -> Result<B, E>) -> Result<PermInst<B>, E> {
        Ok(match self {
            PermInst::Ite(c, t, e) => PermInst::Ite(f(c)?, t.try_map(f)?, e.try_map(f)?),
        })
    }
}

impl PermVal<Val> {
    /// The zero permission (`none`).
    pub fn none() -> Self {
        PermVal::Amount(crate::vmir::none())
    }
    /// The full permission (`write`, `1/1`).
    pub fn write() -> Self {
        PermVal::Amount(crate::vmir::write())
    }
}

impl Display for PermVal<Val> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            PermVal::Amount(v) => write!(f, "{v}"),
            PermVal::Wildcard => write!(f, "wildcard"),
            PermVal::Temp(i) => write!(f, "p{i}"),
        }
    }
}

impl Display for VmirDisplay<'_, &PermInst> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.item {
            PermInst::Ite(c, t, e) => write!(f, "{c} ? {t} : {e}"),
        }
    }
}

/// Heap instructions. All heap instructions produce new heaps.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HeapInst {
    /// `h := base + <loc> @ <perm> with <bind>`. Adds the single location
    /// chunk at `loc` with permission `perm` to `base`. Pure heap accounting —
    /// no boolean is assumed or asserted (cf. `Inhale`/`Exhale`).
    ///
    /// `bind` says where the chunk's **value** comes from. It is always written
    /// and never defaulted: producing a value is the operation with soundness
    /// consequences, so the dangerous case (`Fresh`) must not be the one a
    /// reader skips.
    Add {
        base: HeapVal,
        loc: Val,
        perm: PermVal,
        bind: Bind,
    },
    /// `h := base - <loc> @ <perm>`. Subtracts the single location chunk at
    /// `loc` with permission `perm` from `base`. Pure heap accounting — no
    /// boolean is assumed or asserted (cf. `Inhale`/`Exhale`).
    Sub {
        base: HeapVal,
        loc: Val,
        perm: PermVal,
        /// Whether the removed value is **requested**. A `Sub` always discovers
        /// what the location held -- it is the one instruction that learns
        /// something the caller did not know -- but only the desugared `unfold`
        /// wants it, as `Option<T>` (`None` iff nothing was removed). Elsewhere
        /// the binder is `_` and no `Val` is pushed, so temp numbering is
        /// unchanged. Output arity, exactly like `Exhale::frame_only`.
        yields_value: bool,
    },

    AddForall { base: HeapVal, q: Box<QuantChunks>, bind: Bind },
    SubForall { base: HeapVal, q: Box<QuantChunks> },


    /// `h[, s] := base inhale <call> @ <perm>`. Add the resource's delta (scaled by
    /// `perm`) to `base` **and assume** its boolean condition. When the callee is
    /// **self-framed** the inst additionally yields a pure `Val` `s : Snap(callee)`
    /// — the snapshot of the just-inhaled resource (see [`HeapInst::snap_yield`]),
    /// passed on as the trailing snapshot argument of a two-state resource call.
    Inhale {
        base: HeapVal,
        /// Where the produced footprint's values come from. `Fresh` havocs each
        /// slot (a call's post-state); `Bound(s)` recovers them from an in-scope
        /// snapshot as `unwrap(proj_i(s))`, which is what makes a method's entry
        /// heap and the pre-state its `#ensures` reconstructs the *same* terms.
        bind: Bind,
        call: ResourceCall,
        perm: PermVal,
    },
    /// `h[, s] := base exhale <call> @ <perm>`. Subtract the resource's delta (scaled by
    /// `perm`) from `base` **and assert** its boolean condition. Yields a snapshot
    /// `Val` exactly like `Inhale` (values = the consumed caller chunk values).
    Exhale {
        /// When `true` this exhale is a **frame check**: it proves the callee's
        /// footprint is held and asserts its boolean, but produces **no heap** --
        /// the binder is written `_`. Functions frame, they don't consume.
        ///
        /// This is an output-arity property, not a mode flag: the instruction
        /// declares which of its results are wanted, and the evaluator reads that
        /// rather than being told by a side channel. It must never be *derived*
        /// (by liveness, say) and no pass may introduce it -- blanking a merely
        /// unused heap binder would silently turn a consume into a frame check.
        frame_only: bool,
        base: HeapVal,
        call: ResourceCall,
        perm: PermVal,
    },
    /// Assign a value to a heap location in a given heap.
    /// SIDECOND: the location must have at least `write` permission.
    Assign(HeapVal, Assign),
    /// `h := heap_of R(args), snap` — widen a snapshot value back into a heap:
    /// `h := merge <cond> ? <then_h> : <els_h>` — the block-IR heap join: select
    /// between two predecessor exit heaps under a binary join condition. The
    /// *structural* per-chunk merge (`design/block-vmir/30`) that collapses the
    /// exit permission tower lives in its evaluation. **Not emitted yet** — the
    /// block lowering threads a single linear heap for now; this variant is
    /// declared so the type is stable ahead of the structural-join stage.
    Merge {
        cond: Val,
        then_h: HeapVal,
        els_h: HeapVal,
    },
    /// `h := union <a> <b>` — the **sum** of two heaps held simultaneously, as
    /// opposed to [`HeapInst::Merge`], which *selects* between two alternative
    /// predecessor states. Permissions at a shared location add; values at a
    /// shared location are assumed equal (two chunks of one location cannot
    /// disagree).
    ///
    /// Emitted where a loop is left: the body holds the invariant's footprint and
    /// the head set the rest aside as the frame, so the state after the loop is
    /// their sum. Silicon does the same at a `Kind.Out` edge; our heaps are explicit
    /// values, so the frame is simply named.
    Union { a: HeapVal, b: HeapVal },
}

impl HeapInst {
    /// The resource whose snapshot this instruction *additionally* yields as a
    /// pure `Val` (bumping the `Val` counter): an `Inhale`/`Exhale` of a
    /// **self-framed** resource produces `s : Snap(callee)` alongside the new
    /// heap. Two-state callees (and every other heap inst) yield none. Derived
    /// from the callee declaration — not stored on the inst.
    /// Whether this instruction pushes a pure `Val`: a snapshot-yielding
    /// resource op, or a value-yielding `Sub`. Display and temp numbering both
    /// key off this, so they cannot disagree with the evaluator about arity.
    pub fn yields_val(
        &self,
        decls: &typed_index_collections::TiVec<MemberId, crate::vmir::Declaration>,
    ) -> bool {
        matches!(
            self,
            HeapInst::Sub {
                yields_value: true,
                ..
            }
        ) || self.snap_yield(decls).is_some()
    }

    /// Whether this instruction produces a heap. Everything does except a
    /// **frame-only** exhale, whose binder is `_` (see
    /// [`HeapInst::Exhale::frame_only`]). Output arity is part of an
    /// instruction's shape, so this drives both the display and the temp
    /// numbering rather than being stored alongside them.
    pub fn produces_heap(&self) -> bool {
        !matches!(
            self,
            HeapInst::Exhale {
                frame_only: true,
                ..
            }
        )
    }

    /// The extra pure `Val` this instruction yields, and its type.
    ///
    ///  - `Inhale`/`Exhale` of a **self-framed** callee → `Snap(callee)`, plain.
    ///  - `Sub` → `Option<T>`, where `T` is the held value type of `loc`: the
    ///    instruction *discovers* whether the location held anything, and `None`
    ///    reports that it did not. This is the one position where optionality
    ///    belongs, because it is the one place presence is discovered rather than
    ///    supplied — `Add`'s `bind` and both resource ops hand a value over.
    ///  - everything else → none.
    ///
    /// Derived, never stored, like [`HeapInst::snap_yield`] which it generalizes.
    /// `val_ty` resolves an operand's VMIR type; both callers already track it
    /// (the translator in its `Sink`, the verifier in `EvalState::val_types`).
    ///
    /// The `Sub` arm's consumer is the desugared `unfold` (`Sub` + `Inhale`),
    /// which is the only site that sets `yields_value`. Every other `Sub` leaves
    /// the binder `_` and pushes no `Val`, so temp numbering is unaffected.
    pub fn val_yield(
        &self,
        decls: &typed_index_collections::TiVec<MemberId, crate::vmir::Declaration>,
        val_ty: impl Fn(&Val) -> Option<Type>,
    ) -> Option<Type> {
        match self {
            HeapInst::Sub {
                loc,
                yields_value: true,
                ..
            } => {
                let held = val_ty(loc)?.addr_value()?.clone();
                Some(Type::Option(Box::new(held)))
            }
            HeapInst::Sub { .. } => None,
            _ => self.snap_yield(decls).map(Type::Snap),
        }
    }

    pub fn snap_yield(
        &self,
        decls: &typed_index_collections::TiVec<MemberId, crate::vmir::Declaration>,
    ) -> Option<MemberId> {
        match self {
            // `Inhale` yields nothing: its value source arrives through `bind`,
            // so there is no snapshot left for it to hand back. A caller that
            // needs the handle mints one and binds the inhale to it.
            HeapInst::Exhale { call, .. } => match &decls[call.resource] {
                crate::vmir::Declaration::Resource(r) if r.is_self_framed() => Some(call.resource),
                _ => None,
            },
            _ => None,
        }
    }
}

/// Where a produced chunk's value comes from — the IR half of the verifier's
/// `ValueSource`. Deliberately *not* the same type: `ValueSource` names e-class
/// ids and recipe terms, which the IR does not have.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Bind {
    /// Havoc — surface `with fresh`. An unconstrained fresh value per slot.
    /// Legal only at an instruction in a **method** body: a resource body that
    /// minted a fresh observable value would not be deterministic.
    Fresh,
    /// Bound to an in-scope term — surface `with <val>`.
    Bound(Val),
    /// The **next** slot of the enclosing resource's own footprint — surface
    /// `with self`. Deliberately **unnumbered**: writing `self.2` would
    /// presuppose the slot layout, and the layout is *derived* from the body, so
    /// the body would reference its own derived structure. Unnumbered, the
    /// ordinal is a consequence of position and the circularity is gone.
    SelfSlot,
}

impl Display for Bind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Bind::Fresh => write!(f, "fresh"),
            Bind::Bound(v) => write!(f, "{v}"),
            Bind::SelfSlot => write!(f, "self"),
        }
    }
}

/// Assign a value to a heap location.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Assign {
    pub loc: Val,
    pub val: Val,
}

// ======================
// DISPLAY INFRASTRUCTURE
// ======================

impl Display for HeapVal {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            HeapVal::Empty => write!(f, "empty"),
            HeapVal::Temp(i) => write!(f, "h{i}"),
        }
    }
}

impl<'a> Display for VmirDisplay<'a, &'a HeapInst> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Render `name(arg, ...)` for a resource call.
        let call_head = |f: &mut Formatter<'_>, call: &ResourceCall| -> fmt::Result {
            write!(f, "{}(", self.member(call.resource))?;
            for (i, arg) in call.args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{arg}")?;
            }
            write!(f, ")")
        };
        // Render `base <kw> call perm` for a resource inhale/exhale.
        let resource_combine = |f: &mut Formatter<'_>,
                                base: &HeapVal,
                                kw: &str,
                                call: &ResourceCall,
                                perm: &PermVal| {
            write!(f, "{base} {kw} ")?;
            call_head(f, call)?;
            write!(f, " @ {perm}")
        };
        match self.item {
            HeapInst::Add {
                base,
                loc,
                perm,
                bind,
            } => write!(f, "{base} + {loc} @ {perm} with {bind}"),
            HeapInst::AddForall { q, bind, base } => {
                write!(f, "{base} + {} with {bind}", self.with(&**q))
            }
            HeapInst::Sub {
                base, loc, perm, ..
            } => write!(f, "{base} - {loc} @ {perm}"),
            HeapInst::Inhale {
                base,
                bind,
                call,
                perm,
            } => {
                resource_combine(f, base, "inhale", call, perm)?;
                write!(f, " with {bind}")
            }
            HeapInst::Exhale {
                base, call, perm, ..
            } => resource_combine(f, base, "exhale", call, perm),
            HeapInst::Assign(base, Assign { loc, val }) => {
                write!(f, "{base} assign {loc} {val}")
            }
            HeapInst::Merge {
                cond,
                then_h,
                els_h,
            } => write!(f, "merge {cond} ? {then_h} : {els_h}"),
            HeapInst::Union { a, b } => write!(f, "union {a} {b}"),

            HeapInst::SubForall { base, q } => write!(f, "{base} - {}", self.with(&**q)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmir::{Bound, PermVal};
    use lasso::Rodeo;
    use typed_index_collections::TiVec;

    fn addr_ty(value: Type) -> Type {
        let mut groups: Rodeo<lasso::Spur> = Rodeo::new();
        Type::Addr {
            group: groups.get_or_intern("f"),
            value: Box::new(value),
            bound: Bound::Bounded(num::BigRational::from(num::BigInt::from(1))),
        }
    }

    /// `Sub` yields `Option<T>` for the *held value* type of its location — the
    /// one place presence is discovered rather than supplied.
    #[test]
    fn sub_yields_option_of_the_held_value_type() {
        let decls: TiVec<MemberId, crate::vmir::Declaration> = TiVec::new();
        let inst = HeapInst::Sub {
            base: HeapVal::Empty,
            loc: Val::Temp(0),
            perm: PermVal::write(),
            yields_value: true,
        };
        let got = inst.val_yield(&decls, |_| Some(addr_ty(Type::Int)));
        assert_eq!(got, Some(Type::Option(Box::new(Type::Int))));
    }

    /// A predicate location yields `Option<Snap(P)>`, which is what makes the
    /// desugared `unfold` able to hand the snapshot to its paired `inhale`.
    #[test]
    fn sub_on_a_predicate_location_yields_option_of_its_snapshot() {
        let decls: TiVec<MemberId, crate::vmir::Declaration> = TiVec::new();
        let pred = MemberId(7);
        let inst = HeapInst::Sub {
            base: HeapVal::Empty,
            loc: Val::Temp(0),
            perm: PermVal::write(),
            yields_value: true,
        };
        let got = inst.val_yield(&decls, |_| Some(addr_ty(Type::Snap(pred))));
        assert_eq!(got, Some(Type::Option(Box::new(Type::Snap(pred)))));
    }

    /// `Add` supplies its value through `bind`, so it discovers nothing and
    /// yields nothing.
    #[test]
    fn add_yields_nothing() {
        let decls: TiVec<MemberId, crate::vmir::Declaration> = TiVec::new();
        let inst = HeapInst::Add {
            base: HeapVal::Empty,
            loc: Val::Temp(0),
            perm: PermVal::write(),
            bind: Bind::Fresh,
        };
        assert_eq!(inst.val_yield(&decls, |_| Some(addr_ty(Type::Int))), None);
    }

    /// A bind renders as `with <source>`, and the `Fresh` case is never silent.
    #[test]
    fn bind_display_is_explicit() {
        assert_eq!(Bind::Fresh.to_string(), "fresh");
        assert_eq!(Bind::SelfSlot.to_string(), "self");
        assert_eq!(Bind::Bound(Val::Temp(4)).to_string(), "e4");
    }
}
