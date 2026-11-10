use crate::vmir::display::VmirDisplay;
use crate::vmir::{
    Adt, AdtVariant, Bind, Bound, Function, HeapInst, HeapVal, Inst, InstKind, MemberId, Type, Val,
};
use lasso::Spur;
use std::fmt::{self, Display, Formatter};

/// A reusable unit of proof.
///
/// A resource computes a heap delta and a boolean condition, and it **always has
/// a body** — a bodyless Silver predicate is not a resource at all, but the
/// `function` + `domain` pair it implied (see `translate::decl::predicate`). Its
/// address function and snapshot type are not stored — they are mechanically
/// implied by the definition and derived on demand (the `@addr` function via
/// [`Resource::derive_location`]; the snapshot via [`Resource::derive_snapshot`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Resource {
    pub name: Spur,
    pub params: Vec<Type>,
    pub precond: Precond,
    pub body: ResourceBody,
}

/// A resource's precondition mode.
///
/// - `SelfFramed`: one-state — the body reads only its own footprint. Predicates,
///   `#requires`, and function preconditions. Snapshottable / foldable.
/// - `Ctx(req, args)`: two-state — the body additionally reads the pre-state of
///   the precondition resource `req` applied to `args`, received as a trailing
///   snapshot parameter `s : Snap(req)` and widened back into a heap by the
///   body's entry `HeapInst::a bound inhale`. `#ensures`. Opaque-only; never
///   snapshotted or folded. (The payload is metadata — the entry bound `inhale`
///   carries the same information explicitly.)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Precond {
    SelfFramed,
    Ctx(MemberId, Vec<Val>),
}

impl Resource {
    /// Whether the body reads only its own footprint (no ctx/pre-state heap).
    /// Only self-framed resources may be snapshotted / folded / unfolded.
    pub fn is_self_framed(&self) -> bool {
        matches!(self.precond, Precond::SelfFramed)
    }

    /// This resource's `@addr` **function**: `params -> &[group] Snap(id) @ *`.
    /// The address is a generic-`ADDR` application; all location metadata (group
    /// tag, snapshot value, unbounded permission) rides in the return type.
    /// Derived on demand — not emitted as a declaration. `group` is the predicate's
    /// interned group tag (`Program.groups`).
    pub fn derive_location(&self, id: MemberId, group: lasso::Spur) -> Function {
        Function {
            name: self.name,
            ty_params: 0.into(),
            params: self.params.clone().into(),
            ret: Type::addr(group, Type::Snap(id), Bound::Unbounded),
            body: None,
            requires: None,
            ensures: None,
        }
    }

    /// Derive this resource's snapshot type: a single-constructor [`Adt`] over
    /// the footprint slots, each typed `Option[T]` (a slot is present-or-absent
    /// — `fold` packs `present ? Some(v) : None`).
    ///
    /// `None` for a non self-framed resource (two-state; no foldable snapshot)
    /// — that [`Precond::Ctx`] case is now the *only* reason this returns
    /// `None`. Not stored in the IR — the verifier mints the snapshot's
    /// constructor/projection ids from this on demand.
    ///
    /// Slot types come from the body: only `Pure` insts produce a `Val`, and
    /// params occupy `Val::Temp(0..n)`, so a `Val -> Type` map is just the params
    /// followed by each `Pure`'s result type. Every footprint slot is a
    /// `HeapInst::Add` whose `loc` is an address of type `Addr<T>`; the slot
    /// type is `Option[T]`.
    ///
    /// The layout is **defined by the body's `with self` binds**, not inferred
    /// from the shape of its instructions. That distinction is load-bearing: a
    /// body may legitimately contain heap operations that are *not* footprint
    /// slots — a scoped `unfolding` region produces and consumes chunks of
    /// another predicate — and those carry `Bind::Bound`/no bind, so they are
    /// skipped here rather than silently becoming phantom slots.
    pub fn derive_snapshot(&self) -> Option<Adt> {
        if !self.is_self_framed() {
            return None;
        }
        let body = &self.body;
        let mut val_types: Vec<Type> = self.params.clone();
        let mut field_types = Vec::new();
        for inst in &body.insts {
            match &inst.kind {
                InstKind::Pure(ty, _) => val_types.push(ty.clone()),
                // A frame-only exhale yields the callee's snapshot, so it
                // occupies a `Val` slot even though it is a *heap* instruction.
                // The table is positional, so skipping it would shift every
                // later `loc` lookup and miscount the footprint. Its type is
                // never an `Addr`, so a placeholder keeps the alignment without
                // needing the declarations to resolve the real snapshot type.
                InstKind::Heap(HeapInst::Exhale {
                    frame_only: true, ..
                }) => val_types.push(Type::Bool),
                // Only a `with self` add declares a footprint slot.
                InstKind::Heap(HeapInst::Add {
                    loc,
                    bind: Bind::SelfSlot,
                    ..
                }) => {
                    let ty = match loc {
                        Val::Temp(n) => val_types.get(*n),
                        Val::Literal(_) => None,
                    };
                    if let Some(Type::Addr { value, .. }) = ty {
                        field_types.push(Type::Option(value.clone()));
                    }
                }
                _ => {}
            }
        }
        Some(Adt {
            name: self.name,
            ty_params: 0.into(),
            // A snapshot's single constructor is synthetic — no source name.
            variants: vec![AdtVariant {
                name: None,
                field_types,
            }],
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResourceBody {
    pub insts: Vec<Inst>,
    pub res: (HeapVal, Val),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResourceCall {
    pub resource: MemberId,
    /// Call arguments. A two-state resource's pre-state snapshot is an ordinary
    /// trailing argument here (matching its trailing `Snap(req)` param).
    pub args: Vec<Val>,
}

// ======================
// DISPLAY INFRASTRUCTURE
// ======================

impl<'a> Display for VmirDisplay<'a, &'a Resource> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let name = self.interner.resolve(&self.item.name);
        write!(f, "resource {name}(")?;
        for (i, param) in self.item.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            // Params occupy `Val::Temp(0..n)`, so label them `e0`, `e1`, … to
            // match the temporaries the body refers to.
            write!(f, "e{i}: {}", self.with(param))?;
        }
        write!(f, ")")?;

        // Body heaps always count from `h0` (a two-state resource's pre-state is
        // reconstructed by its explicit entry bound `inhale`, which is `h0`
        // itself — no reserved slot).
        let body = &self.item.body;
        writeln!(f, " {{")?;
        write!(
            f,
            "{}",
            self.with((self.item.params.len(), 0usize, 0usize, &body.insts[..]))
        )?;
        writeln!(f, "  result: ({}, {})", body.res.0, body.res.1)?;
        write!(f, "}}")
    }
}

impl<'a> Display for VmirDisplay<'a, &'a ResourceBody> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "{{")?;
        write!(
            f,
            "{}",
            self.with((0usize, 0usize, 0usize, &self.item.insts[..]))
        )?;
        writeln!(f, "  result: ({}, {})", self.item.res.0, self.item.res.1)?;
        write!(f, "}}")
    }
}
