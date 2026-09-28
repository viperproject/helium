//! The instruction [`Sink`]: a mutable buffer of emitted VMIR `Inst`s plus the
//! running SSA counters and path-condition stack. The instruction set is
//! uniform; how the resulting stream is interpreted is the caller's concern
//! (resource delta+bool, method effects, function result).

use crate::dhash::HashMap;

use crate::vmir::{
    self, BinOp, HeapInst, HeapVal, Inst, InstKind, Literal, PathConds, PermInst, PermVal,
    Polarity, PureInst, ResourceCall, Type, Val, none,
};

/// Why a condition sits on the path-condition stack. Both kinds gate the
/// **side conditions** of the instructions under them (they go on the emitted
/// `pc`), but only a `Branch` gates **permission amounts**:
/// - `Branch` — a case split (`b ==> ..`, `c ? .. : ..`); the dead arm needs 0
///   permission, so the perm is wrapped `b ? p : 0`.
/// - `Fact` — the left operand of a separating conjunction `A && B`; an
///   *assertion* that aborts if false, so `B`'s permissions stay ungated (no
///   spurious `A ? p : 0`).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PcKind {
    Branch,
    Fact,
    /// A CFG **block cube** under the Stage-4 fork model. It rides in every
    /// inst's `pc` (so the verifier proves under it and the join merge folds
    /// dead edges), but — unlike a `Branch` — it does NOT gate permissions:
    /// fork arms run unguarded and the join's structural SELECT carries the
    /// branch. Excluded from [`Sink::branch_conds`].
    Cube,
}

/// A mutable sink for emitted instructions plus the running counters. The
/// instruction set is uniform; how the resulting stream is interpreted is the
/// caller's concern (resource delta+bool, method effects, function result).
pub(crate) struct Sink {
    pub insts: Vec<Inst>,
    pub val_base: usize,
    pub val_count: usize,
    pub heap_count: usize,
    /// Running `p` counter. Starts at 0 in every sink: permissions have no
    /// parameters and no cross-stream continuation (see [`Sink::gate_perm`]).
    pub perm_count: usize,
    /// Running path condition of the lowering point. Sidecond instructions
    /// are emitted gated by this; branch arms push/pop guards via `with_cond`.
    pub pc: Vec<(Val, Polarity, PcKind)>,
    /// Watermark into `pc`: everything below it is the **ambient cube** of the
    /// phase being lowered, already recorded on the enclosing `vmir::Block` and
    /// re-derived by the verifier, so it is elided from every emitted `Inst`.
    /// An `Inst.pc` is therefore a *delta*:
    ///
    /// ```text
    /// effective_pc(inst) = ambient_cube(phase) ++ inst.pc
    /// ```
    ///
    /// Set only for a block's **body** phase (see [`Sink::with_ambient_cube`]);
    /// `0` everywhere else — a block's *join* phase materializes its own cube
    /// literals, so there the pc must stay full.
    ambient: usize,
    /// The current heap an obligation's side condition is checked in. Delivered
    /// like `pc`: set for a lowering region via [`Sink::with_heap`], snapshotted
    /// onto a heapless obligation's `Inst` by the emitters. `None` outside any
    /// heap-bearing region (e.g. before the first heap is threaded).
    pub heap: Option<HeapVal>,
    /// Value-numbering memo for **total** pure insts: an identical
    /// `(ty, inst, pc)` triple reuses the earlier temp instead of re-emitting.
    /// Keeps the VMIR for nested control flow linear: a block's reach-cube
    /// conjunction left-folds, so its prefix is a memo hit and each deeper block
    /// adds O(1) insts, and the per-heap-op perm/value gating chains dedupe to
    /// once per block. `Fresh` (nondeterministic) and the guarded emitters
    /// (pc-dependent obligations) are never memoized.
    ///
    /// The `pc` component exists for [`Sink::emit_call`]: a call inst carries its
    /// lowering pc (the verifier assumes the callee's `f%pre` token under it), so
    /// two occurrences of the same call under *different* pcs must not collide —
    /// see that method for what a collision would cost. It is the **full** pc
    /// ([`Sink::full_guard`]), ambient cube included, not the delta written onto
    /// the `Inst`: the memo spans every block of a method, so two blocks whose
    /// cubes are the only difference must still get two temps. Every other emitter
    /// passes `PathConds::default()`, so their behavior is unchanged.
    memo: HashMap<(Type, PureInst, PathConds), Val>,
    /// Read-only (function/pure) lowering: every `acc`/unfolding permission is
    /// weakened to a `wildcard` (or `0`), since a function only ever needs *some*
    /// positive share to read. Set on the sinks of function bodies and function
    /// precondition resources; `false` for methods and predicate bodies. See
    /// [`Sink::perm_amount`].
    pub(crate) read_only: bool,
    /// Whether this sink is lowering a **resource body** (a predicate, or a
    /// method/function contract) rather than a method body. Decides the `Bind`
    /// on a footprint `acc`: `SelfSlot` inside a resource, `Fresh` in a method.
    ///
    /// Carried here rather than threaded because the two are already distinct
    /// entry points — `lower_spatial_never` / `lower_spatial_ensures` mint their
    /// own `Sink` and return a `ResourceBody`, while a method body's statements
    /// share the method's sink. Same shape as `read_only` above.
    pub(crate) in_resource_body: bool,
    /// Whether this sink is lowering a **method body**. A method body is never
    /// replayed anywhere — only functions, contract functions and resources
    /// become recipes a call site grafts — so nothing would ever act on a
    /// [`FunctionCall::export`](crate::vmir::FunctionCall::export) mark here, and
    /// setting one would make the IR dump claim a propagation that cannot
    /// happen. Same shape as `read_only` above.
    pub(crate) in_method_body: bool,
}

impl Sink {
    pub fn new(val_base: usize, heap_base: usize) -> Self {
        Self {
            insts: Vec::new(),
            val_base,
            val_count: 0,
            heap_count: heap_base,
            perm_count: 0,
            pc: Vec::new(),
            ambient: 0,
            heap: None,
            memo: HashMap::default(),
            read_only: false,
            in_resource_body: false,
            in_method_body: false,
        }
    }

    /// Run `f` with `heap` as the current obligation check-in heap, restoring the
    /// prior heap afterwards (analogous to [`Sink::with_cond`] for `pc`). The
    /// restore runs even when `f` returns `Err`.
    pub(crate) fn with_heap<R>(&mut self, heap: HeapVal, f: impl FnOnce(&mut Self) -> R) -> R {
        let prev = self.heap.replace(heap);
        let r = f(self);
        self.heap = prev;
        r
    }

    /// Build an obligation instruction, attaching the current check-in heap when
    /// one is set (see [`Sink::heap`]).
    fn checked_inst(&self, pc: PathConds, kind: InstKind) -> Inst {
        match self.heap {
            Some(h) => Inst::in_heap(pc, h, kind),
            None => Inst::new(pc, kind),
        }
    }

    pub(crate) fn with_cond<R>(
        &mut self,
        cond: Val,
        pol: Polarity,
        kind: PcKind,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.pc.push((cond, pol, kind));
        let r = f(self);
        self.pc.pop();
        r
    }

    /// Run `f` with every literal of `conds` pushed as a `Branch` guard (the
    /// reaching condition of a CFG block), popping them all afterwards. Used by
    /// the method-body linearizer to lower a basic block under its path
    /// condition. The pops run even when `f` returns `Err`.
    pub(crate) fn with_conds<R>(&mut self, conds: &PathConds, f: impl FnOnce(&mut Self) -> R) -> R {
        self.with_conds_kind(conds, PcKind::Branch, f)
    }

    /// Like [`Sink::with_conds`] but pushes each literal under `kind`. The
    /// Stage-4 body lowering uses [`PcKind::Cube`] so the block cube guards
    /// obligations (it is in `pc`) without gating permissions.
    pub(crate) fn with_conds_kind<R>(
        &mut self,
        conds: &PathConds,
        kind: PcKind,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        for (cond, pol) in &conds.conds {
            self.pc.push((cond.clone(), *pol, kind));
        }
        let r = f(self);
        for _ in &conds.conds {
            self.pc.pop();
        }
        r
    }

    /// Run `f` with `conds` pushed as the phase's **ambient cube**: each literal
    /// enters the pc as a [`PcKind::Cube`] (so it is proof context but never gates
    /// a permission), and the watermark moves past them so they are elided from
    /// every `Inst` emitted inside. The block already records the same cube in
    /// [`vmir::Block::cube`], and the verifier re-derives it via `begin_block`.
    ///
    /// Only a block's **body** phase gets an ambient. A join phase defines its own
    /// cube literals (`bb6 <e10> join e9 [..]` mints `e10` in the join itself), so
    /// its insts must carry the full pc; those sites use [`Sink::with_conds_kind`]
    /// and leave the watermark alone.
    pub(crate) fn with_ambient_cube<R>(
        &mut self,
        conds: &PathConds,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let prev = self.ambient;
        for (cond, pol) in &conds.conds {
            self.pc.push((cond.clone(), *pol, PcKind::Cube));
        }
        self.ambient = self.pc.len();
        let r = f(self);
        self.ambient = prev;
        for _ in &conds.conds {
            self.pc.pop();
        }
        r
    }

    /// The currently-active **branch** path-condition literals (the ones that
    /// gate permissions); `Fact` entries are excluded. Sliced from the ambient
    /// watermark, so that a block cube can never reach [`Sink::gate_perm`] is
    /// visible at the one place it would matter (the ambient is all `Cube`
    /// anyway, so the slice changes no result).
    pub(crate) fn branch_conds(&self) -> Vec<(Val, Polarity)> {
        self.pc[self.ambient..]
            .iter()
            .filter(|(_, _, k)| *k == PcKind::Branch)
            .map(|(c, p, _)| (c.clone(), *p))
            .collect()
    }

    /// Gate a permission by the current branch path condition: each branch literal
    /// wraps `perm` with `none` (0) on the *dead* side. A positive literal `b`
    /// yields `b ? perm : none`; a negative literal `b ? none : perm`. The empty
    /// (top-level) path condition returns `perm` unchanged. Only *branch*
    /// conditions gate; a separating-conjunction `Fact` keeps the bare permission.
    ///
    /// Every permission gates *structurally* as a [`PermInst::Ite`] step — a
    /// concrete amount no less than a wildcard-bearing one. The gate is the same
    /// term either way (`eval_perm`/`build_perm` build the `Symbolic::Ite` a folded
    /// `Ternary` temp used to build), but keeping it as permission structure is what
    /// lets a consume align the demand's arms against the held permission's rather
    /// than proving against an opaque `ite`. Folding it into the value instead is
    /// what left `perm_sub_aligned` dead and made `fold q(c, x)` fail while holding
    /// `acc(x.f)`.
    pub(crate) fn gate_perm(&mut self, perm: PermVal) -> PermVal {
        let mut p = perm;
        for (lit, pol) in self.branch_conds().into_iter().rev() {
            p = match pol {
                Polarity::Positive => self.emit_perm(PermInst::Ite(lit, p, PermVal::none())),
                Polarity::Negative => self.emit_perm(PermInst::Ite(lit, PermVal::none(), p)),
            };
        }
        p
    }

    /// Emit a permission instruction, yielding the `p` temp it defines.
    ///
    /// **Never value-numbered**, unlike [`Sink::emit_pure`]. Two syntactically
    /// identical gated wildcards (`acc(x.f, wildcard)` and `acc(y.f, wildcard)`
    /// under one branch) must stay two temps: a `p` temp denotes *one* share, so
    /// sharing the temp would hand two distinct locations the same wildcard.
    pub(crate) fn emit_perm(&mut self, inst: PermInst) -> PermVal {
        self.insts
            .push(Inst::new(PathConds::default(), InstKind::Perm(inst)));
        self.next_perm_temp()
    }

    /// Map a lowered permission *value* to a [`PermVal`], applying the read-only
    /// The [`Bind`] for a footprint `acc` emitted by this sink: the enclosing
    /// resource's next own slot inside a resource body, an unconstrained fresh
    /// value in a method body. There is no third case — a bind to a named term
    /// comes from a desugaring that builds the instruction itself.
    pub(crate) fn acc_bind(&self) -> crate::vmir::Bind {
        if self.in_resource_body {
            crate::vmir::Bind::SelfSlot
        } else {
            crate::vmir::Bind::Fresh
        }
    }

    /// (function) policy when [`Sink::read_only`] is set: a constant `0` stays
    /// `0`; a constant nonzero amount becomes `wildcard`; any other (symbolic)
    /// amount `p` becomes `p > 0 ? wildcard : 0`. Outside read-only context the
    /// amount is kept exactly ([`PermVal::Amount`]).
    pub(crate) fn perm_amount(&mut self, p: Val) -> PermVal {
        if !self.read_only {
            return PermVal::Amount(p);
        }
        match &p {
            Val::Literal(Literal::Real(r)) => {
                if *r == num::BigRational::from(num::BigInt::from(0)) {
                    PermVal::none()
                } else {
                    PermVal::Wildcard
                }
            }
            _ => {
                // p > 0  ==  0 < p
                let gt = self.emit_pure(Type::Bool, PureInst::Binary(BinOp::LtR, none(), p));
                self.emit_perm(PermInst::Ite(gt, PermVal::Wildcard, PermVal::none()))
            }
        }
    }

    /// Gate a written *value* by the current branch path condition, keeping the
    /// prior value `old` on the dead side: each branch literal wraps the value in a
    /// `lit ? val : old` (positive) or `lit ? old : val` (negative). Used for a
    /// field assignment inside an `if` arm, where the heap is a single timeline (no
    /// heap ternary) so the *value* must carry the branch instead of the chunk. The
    /// empty top-level pc returns `val` unchanged.
    pub(crate) fn gate_value(&mut self, val: Val, old: Val, ty: Type) -> Val {
        let mut v = val;
        for (lit, pol) in self.branch_conds().into_iter().rev() {
            let (then_, else_) = match pol {
                Polarity::Positive => (v, old.clone()),
                Polarity::Negative => (old.clone(), v),
            };
            v = self.emit_pure(ty.clone(), PureInst::Ternary(lit, then_, else_));
        }
        v
    }

    /// Drain and return the instructions emitted since `mark` (a prior
    /// `self.insts.len()`). Since emission is append-only, this yields exactly
    /// one lowering phase's insts as an owned `Vec`, letting the block lowerer
    /// route each phase into a `Block`'s `join`/`body`. The SSA counters and memo
    /// are untouched (global-positional ids survive), so a later block may still
    /// reference a `Val` defined in an earlier drained phase.
    pub fn take_since(&mut self, mark: usize) -> Vec<Inst> {
        self.insts.drain(mark..).collect()
    }

    pub fn next_val_temp(&mut self) -> Val {
        let id = self.val_base + self.val_count;
        self.val_count += 1;
        Val::Temp(id)
    }

    pub fn next_perm_temp(&mut self) -> PermVal {
        let id = self.perm_count;
        self.perm_count += 1;
        PermVal::Temp(id)
    }

    pub fn next_heap_temp(&mut self) -> HeapVal {
        let id = self.heap_count;
        self.heap_count += 1;
        HeapVal::Temp(id)
    }

    /// The pc to attach to an emitted instruction: the running path condition
    /// **minus the ambient cube** (see [`Sink::ambient`]). Empty outside any
    /// branch, and empty for an inst emitted directly under a block cube.
    fn guard(&self) -> PathConds {
        PathConds {
            conds: self.pc[self.ambient..]
                .iter()
                .map(|(c, p, _)| (c.clone(), *p))
                .collect(),
        }
    }

    /// The **whole** path-condition stack, ambient included. The only consumer is
    /// [`Sink::emit_call`]'s memo key: the memo is shared across all blocks of a
    /// method, so two calls distinguished only by their block cubes must not
    /// collide — with deltas both would key the empty pc.
    fn full_guard(&self) -> PathConds {
        PathConds {
            conds: self.pc.iter().map(|(c, p, _)| (c.clone(), *p)).collect(),
        }
    }

    /// Emit a **total** pure instruction (no side condition) — flat, no pc.
    /// Deterministic insts are value-numbered (see [`Sink::memo`]): a repeat
    /// `(ty, inst)` returns the earlier temp without emitting.
    pub fn emit_pure(&mut self, ty: vmir::Type, inst: PureInst) -> Val {
        let pc = PathConds::default();
        self.emit_pure_gated(ty, inst, pc.clone(), pc)
    }

    /// Emit a **function call** carrying the running path condition. Unlike every
    /// other total pure inst, a call's `pc` is load-bearing: the verifier assumes
    /// the callee's `f%pre` token under it, and that token's *truth* is what
    /// releases the callee's definitional equality and exported facts. A call
    /// under a ternary or an implication must therefore not claim an
    /// unconditional occurrence's pc, and vice versa.
    ///
    /// Value-numbered per `(ty, inst, pc)`, so an identical call at an identical
    /// pc still dedupes, while the same call under two different pcs mints two
    /// temps — hence two token assumptions, `pc1 ==> tok` and `pc2 ==> tok`,
    /// i.e. the disjunction. That is what a call occurring on both paths means.
    /// Keying on `(ty, inst)` alone would let the two collide, and whichever was
    /// emitted first would decide the damage: conditional-then-unconditional
    /// scopes the token to the branch and the unconditional occurrence loses its
    /// unfold; unconditional-then-conditional makes the token unconditionally
    /// true and lets the callee's facts leak to sibling paths.
    pub fn emit_call(&mut self, ty: vmir::Type, inst: PureInst) -> Val {
        // Emit the delta, but key the memo on the **full** pc. The memo is shared
        // across every block of a method, so `f(x)` in `bb1 <e1>` and `f(x)` in
        // `bb2 <!e1>` have equal deltas (both empty) and would collide on the
        // delta — reintroducing exactly the `f%pre` damage described above.
        self.emit_pure_gated(ty, inst, self.guard(), self.full_guard())
    }

    /// Shared body of [`Sink::emit_pure`] and [`Sink::emit_call`]: emit `inst`
    /// gated by `pc`, value-numbering on `(ty, inst, key_pc)`. The two pcs differ
    /// only for [`Sink::emit_call`], where `pc` is the delta written onto the
    /// `Inst` and `key_pc` is the full path condition the memo must distinguish on.
    fn emit_pure_gated(
        &mut self,
        ty: vmir::Type,
        inst: PureInst,
        pc: PathConds,
        key_pc: PathConds,
    ) -> Val {
        if inst == PureInst::Fresh {
            let v = self.next_val_temp();
            self.insts.push(Inst::new(pc, InstKind::Pure(ty, inst)));
            return v;
        }
        if let Some(v) = self.memo.get(&(ty.clone(), inst.clone(), key_pc.clone())) {
            return v.clone();
        }
        let v = self.next_val_temp();
        self.memo
            .insert((ty.clone(), inst.clone(), key_pc), v.clone());
        self.insts.push(Inst::new(pc, InstKind::Pure(ty, inst)));
        v
    }

    /// Emit a pure instruction into a temp allocated **before** the instruction is
    /// built. A `forall` needs this: its body shares this temp space and is
    /// numbered from its own step's temp, so the temp has to exist first (see
    /// [`Forall`](crate::vmir::Forall)).
    ///
    /// Carries the running path condition. A `forall`'s **well-definedness** is
    /// checked once per syntactic occurrence against fresh binders, and that check
    /// reads this inst's `pc` — so a quantifier under a branch or an implication
    /// (`b ==> (forall i :: .. 10 / n ..)`, with `n != 0` known only under `b`) must
    /// carry the guard, or its side conditions are discharged unconditionally and
    /// fail spuriously.
    ///
    /// Deliberately un-memoized. A memo hit would hand back some earlier temp,
    /// while the body just lowered was numbered against `v` — the same reason
    /// `Fresh` is carved out of [`Sink::emit_pure`].
    pub fn emit_pure_at(&mut self, v: &Val, ty: vmir::Type, inst: PureInst) {
        debug_assert!(
            matches!(v, Val::Temp(i) if *i + 1 == self.val_base + self.val_count),
            "emit_pure_at must fill the most recently allocated temp"
        );
        let pc = self.guard();
        self.insts.push(Inst::new(pc, InstKind::Pure(ty, inst)));
    }

    /// Emit a pure instruction whose side condition (e.g. `Deref` permission,
    /// `Div`/`Mod` divisor) must hold under the running path condition. A
    /// `Div`/`Mod` has no embedded heap, so it snapshots the current check-in heap
    /// (see [`Sink::heap`]); a `Deref`/`Perm` embeds its own heap, so it does not.
    pub fn emit_pure_guarded(&mut self, ty: vmir::Type, inst: PureInst) -> Val {
        let v = self.next_val_temp();
        let pc = self.guard();
        let heapless_obligation = matches!(inst, PureInst::Binary(op, _, _) if op.is_div_or_mod());
        let kind = InstKind::Pure(ty, inst);
        let node = if heapless_obligation {
            self.checked_inst(pc, kind)
        } else {
            Inst::new(pc, kind)
        };
        self.insts.push(node);
        v
    }

    /// Emit a **total** heap instruction (no side condition) — e.g. `Add`.
    pub fn emit_heap(&mut self, inst: HeapInst) -> HeapVal {
        let h = self.next_heap_temp();
        self.insts
            .push(Inst::new(PathConds::default(), InstKind::Heap(inst)));
        h
    }

    /// Emit a heap instruction whose side condition (`Acc` perm ≥ 0, `Sub`
    /// sufficient perm, `Assign` write perm) must hold under the running pc.
    pub fn emit_heap_guarded(&mut self, inst: HeapInst) -> HeapVal {
        let h = self.next_heap_temp();
        let pc = self.guard();
        self.insts.push(Inst::new(pc, InstKind::Heap(inst)));
        h
    }

    pub fn emit_assume(&mut self, v: Val) {
        let pc = self.guard();
        self.insts.push(Inst::new(pc, InstKind::Assume(v)));
    }

    pub fn emit_assert(&mut self, v: Val) {
        let pc = self.guard();
        let node = self.checked_inst(pc, InstKind::Assert(v));
        self.insts.push(node);
    }

    pub fn emit_refute(&mut self, v: Val) {
        let pc = self.guard();
        let node = self.checked_inst(pc, InstKind::Refute(v));
        self.insts.push(node);
    }

    /// Emit a resource inhale (`base inhale call perm`, assumes the bool) or
    /// exhale (`base exhale call perm`, asserts the bool). Inhale is total;
    /// exhale carries the running pc as its side-condition guard.
    ///
    /// When `yields_snap` (the callee is self-framed) the inst additionally
    /// produces a pure `Val` — the snapshot of the in/ex-haled resource — so the
    /// `Val` counter bumps alongside the heap counter.
    /// Fork model: the branch no longer rides in the perm scale, so the inhale
    /// carries the block cube as its pc — the verifier guards the inhaled bool by
    /// it (else a conditional inhale leaks its fact past the branch). An empty pc
    /// (unconditional inhale) guards by nothing.
    /// An inhale takes its values in through `bind`, so it yields **nothing** —
    /// unlike [`Sink::emit_resource_exhale`], there is no snapshot left to hand
    /// back. A caller needing the handle mints one and binds it in.
    pub fn emit_resource_inhale(
        &mut self,
        base: HeapVal,
        call: ResourceCall,
        perm: PermVal,
        bind: crate::vmir::Bind,
    ) -> HeapVal {
        self.emit_heap_guarded(HeapInst::Inhale {
            base,
            bind,
            call,
            perm,
        })
    }

    /// A value-yielding slot consume: `h1, e1 := h0 - <loc> @ <perm>`. Removes
    /// the chunk and hands back what was there as `Option<T>`. Only the desugared
    /// `unfold` wants that, so every other `Sub` leaves the binder `_`.
    pub fn emit_sub_yielding(&mut self, base: HeapVal, loc: Val, perm: PermVal) -> (HeapVal, Val) {
        let h = self.emit_heap_guarded(HeapInst::Sub {
            base,
            loc,
            perm,
            yields_value: true,
        });
        (h, self.next_val_temp())
    }

    /// A **frame-only** exhale: prove the callee's footprint is held and assert
    /// its boolean, producing the snapshot but **no heap** (`_, e := ..`). This is
    /// the implicit precondition check at a heap-dependent function call --
    /// functions frame, they don't consume -- so it bumps the `Val` counter only.
    pub fn emit_resource_frame_exhale(
        &mut self,
        base: HeapVal,
        call: ResourceCall,
        perm: PermVal,
    ) -> Val {
        // NOT `emit_heap_guarded`: that mints a heap temp, and this instruction
        // produces no heap. Only the `Val` counter advances.
        let pc = self.guard();
        self.insts.push(Inst::new(
            pc,
            InstKind::Heap(HeapInst::Exhale {
                frame_only: true,
                base,
                call,
                perm,
            }),
        ));
        self.next_val_temp()
    }

    /// The consume counterpart of [`Sink::emit_resource_inhale`].
    pub fn emit_resource_exhale(
        &mut self,
        base: HeapVal,
        call: ResourceCall,
        perm: PermVal,
        yields_snap: bool,
    ) -> (HeapVal, Option<Val>) {
        let h = self.emit_heap_guarded(HeapInst::Exhale {
            frame_only: false,
            base,
            call,
            perm,
        });
        let snap = yields_snap.then(|| self.next_val_temp());
        (h, snap)
    }
}
