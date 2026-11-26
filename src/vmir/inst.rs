use crate::vmir::display::VmirDisplay;
use crate::vmir::{HeapInst, HeapVal, PermInst, PureInst, Type, Val};

use std::fmt::{self, Display, Formatter};

/// An instruction gated by a path condition.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Inst {
    /// The instruction's path condition, as a **delta over the ambient cube of
    /// the phase it sits in**:
    ///
    /// ```text
    /// effective_pc(inst) = ambient_cube(phase) ++ inst.pc
    /// ```
    ///
    /// | phase | ambient cube |
    /// |---|---|
    /// | [`Block::body`](crate::vmir::Block::body) | [`Block::cube`](crate::vmir::Block::cube) |
    /// | [`Block::join`](crate::vmir::Block::join) | empty |
    /// | resource / function body, `forall` body | empty (no block) |
    ///
    /// A join's ambient is empty because a join *materializes* its own cube
    /// literals — in `bb6 <e10> join e9 [..]`, `e10` is defined by a reach inst
    /// inside the join phase — so there the delta is the whole condition.
    ///
    /// Consequences worth knowing: a block cube is stated **once**, on
    /// `Block::cube`, and so can never be restated or contradicted by an inst;
    /// and it can never reach a permission amount, because gating reads the
    /// delta. Use [`Method::iter_insts_with_pc`](crate::vmir::Method::iter_insts_with_pc)
    /// when you need the effective condition. The dump prints the delta, and the
    /// block header prints the cube, so a reader can reconstruct it.
    pub pc: PathConds,
    /// The heap this instruction's side condition must be **checked in** — the
    /// heap the verifier will consolidate (materialising aliasing-dependent
    /// facts) before discharging the obligation. Delivered like [`PathConds`]:
    /// snapshotted onto the `Inst` by the guarded emitters. `None` when the
    /// instruction has no obligation, or when the relevant heap is already
    /// embedded in `kind` (a `Deref`/`Perm` heap, an `Exhale`/`Fold`/`Assign`
    /// base, a `FunctionCall` ctx heap). Populated only for the heapless
    /// obligations `Assert`, `Refute`, and `Div`/`Mod`.
    pub heap: Option<HeapVal>,
    pub kind: InstKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InstKind {
    /// Produces a temporary of a given type.
    Pure(Type, PureInst),
    /// Produces a heap value.
    Heap(HeapInst),
    /// Produces a **permission** temp (`p`). Separate from `Pure` because its
    /// result is a `PermVal`, not a `Val`: a permission is consumed by heap
    /// operations and by further `Perm` insts, and by nothing else, so it can
    /// never reach the e-graph as a term. Block-local — a permission has no
    /// join, so nothing carries one across a block boundary.
    Perm(PermInst),
    /// Assume a boolean fact. Produces no value.
    Assume(Val),
    /// Assert a boolean obligation. Produces no value.
    Assert(Val),
    /// Refute a boolean: verification succeeds iff it is **not** provable.
    /// Produces no value.
    Refute(Val),
}

/// Conjunction of literals over previously-emitted `Val`s.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct PathConds {
    pub conds: Vec<(Val, Polarity)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Polarity {
    Positive,
    Negative,
}

impl From<bool> for Polarity {
    fn from(b: bool) -> Self {
        if b {
            Polarity::Positive
        } else {
            Polarity::Negative
        }
    }
}

impl Inst {
    /// Construct an instruction with no check-in heap. A non-empty `pc` gates the
    /// instruction's side condition; the translation attaches one only where it is
    /// needed (see the `*_guarded` emitters in `translate`).
    pub fn new(pc: PathConds, kind: InstKind) -> Self {
        Self {
            pc,
            heap: None,
            kind,
        }
    }

    /// Construct an obligation instruction carrying the `heap` its side condition
    /// is checked in (see [`Inst::heap`]).
    pub fn in_heap(pc: PathConds, heap: HeapVal, kind: InstKind) -> Self {
        Self {
            pc,
            heap: Some(heap),
            kind,
        }
    }

    /// Visit every `Val` this instruction reads, path condition included, in
    /// source order. A nested `forall` is descended into: its triggers and body
    /// mention temps of *this* stream, which is what makes implicit capture work
    /// across nesting levels (see [`Forall::free_temps`](crate::vmir::Forall)).
    pub fn for_each_operand(&self, f: &mut impl FnMut(&Val)) {
        for (v, _) in &self.pc.conds {
            f(v);
        }
        match &self.kind {
            InstKind::Pure(_, pi) => pi.for_each_operand(f),
            InstKind::Assume(v) | InstKind::Assert(v) | InstKind::Refute(v) => f(v),
            // The condition of a permission `ite` is a *value* operand and must
            // be walked; the arms are permissions, not `Val`s.
            InstKind::Perm(PermInst::Ite(c, _, _)) => f(c),
            // Operand walking is used only on quantifier bodies, whose one heap
            // step is a heap-dependent call's frame-only exhale.
            InstKind::Heap(hi) => match hi.frame_exhale() {
                Some((_, call)) => call.args.iter().for_each(&mut *f),
                None => panic!(
                    "a quantifier body holds a heap step other than a frame-only exhale: {hi:?}"
                ),
            },
        }
    }
}

// ======================
// DISPLAY INFRASTRUCTURE
// ======================

/// Walk an instruction stream. Wraps `(val_base, heap_base, perm_base, &[Inst])`
/// in a `VmirDisplay` so the iteration lives behind a regular `Display` impl.
/// Callers — `Display for VmirDisplay<&Method>`, `&Resource>`, `&ResourceBody>`
/// — invoke via `self.with((val_base, heap_base, perm_base, &insts[..]))`.
impl<'a> Display for VmirDisplay<'a, (usize, usize, usize, &'a [Inst])> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let (val_base, heap_base, perm_base, insts) = self.item;
        let mut e_idx = val_base;
        let mut h_idx = heap_base;
        let mut p_idx = perm_base;
        let indent = self.indent();
        for inst in insts {
            // The check-in heap of an obligation, rendered `[h3]` where it belongs
            // (after a `Pure` expression as a suffix, after the `assert`/`refute`
            // keyword). Empty for a `None` heap.
            let heap = HeapSuffix(&inst.heap);
            match &inst.kind {
                InstKind::Pure(ty, pi) => {
                    writeln!(
                        f,
                        "{indent}e{e_idx}: {} := {}{}{heap}",
                        self.with(ty),
                        PcPrefix(&inst.pc),
                        self.with(pi)
                    )?;
                    e_idx += 1;
                }
                InstKind::Heap(hi) => {
                    // A snapshot-yielding inhale/exhale also produces a pure
                    // temp: `h1, e5 := h0 inhale R(...) @ 1/1`. A frame-only exhale
                    // produces the temp but no heap: `_, e5 := h0 exhale R(..)`.
                    if hi.yields_val(self.decls) {
                        let heap_binder = if hi.produces_heap() {
                            format!("h{h_idx}")
                        } else {
                            "_".to_string()
                        };
                        writeln!(
                            f,
                            "{indent}{heap_binder}, e{e_idx} := {}{}",
                            PcPrefix(&inst.pc),
                            self.with(hi)
                        )?;
                        e_idx += 1;
                    } else {
                        writeln!(
                            f,
                            "{indent}h{h_idx} := {}{}",
                            PcPrefix(&inst.pc),
                            self.with(hi)
                        )?;
                    }
                    if hi.produces_heap() {
                        h_idx += 1;
                    }
                }
                InstKind::Perm(pi) => {
                    writeln!(
                        f,
                        "{indent}p{p_idx}: perm := {}{}",
                        PcPrefix(&inst.pc),
                        self.with(pi)
                    )?;
                    p_idx += 1;
                }
                InstKind::Assume(v) => writeln!(f, "{indent}{}assume {v}", PcPrefix(&inst.pc))?,
                InstKind::Assert(v) => {
                    writeln!(f, "{indent}{}assert {v}{heap}", PcPrefix(&inst.pc))?
                }
                InstKind::Refute(v) => {
                    writeln!(f, "{indent}{}refute {v}{heap}", PcPrefix(&inst.pc))?
                }
            }
        }
        Ok(())
    }
}

impl Display for PathConds {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.conds.is_empty() {
            return Ok(());
        }
        write!(f, "<")?;
        for (i, (val, pol)) in self.conds.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if matches!(pol, Polarity::Negative) {
                write!(f, "!")?;
            }
            write!(f, "{val}")?;
        }
        write!(f, ">")
    }
}

/// Helper that renders a `PathConds` followed by a trailing space when
/// the guard is non-empty, and emits nothing at all when the guard is
/// the trivial `<>`. Use it at instruction-rendering sites so empty
/// guards don't waste a token.
pub(crate) struct PcPrefix<'a>(pub &'a PathConds);

impl Display for PcPrefix<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.0.conds.is_empty() {
            Ok(())
        } else {
            write!(f, "{} ", self.0)
        }
    }
}

/// Renders an obligation's check-in heap as a trailing ` [h3]` suffix (a leading
/// space so it detaches from the expression), or nothing when there is none.
pub(crate) struct HeapSuffix<'a>(pub &'a Option<HeapVal>);

impl Display for HeapSuffix<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(h) => write!(f, " [{h}]"),
            None => Ok(()),
        }
    }
}
