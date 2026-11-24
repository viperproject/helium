use crate::vmir::display::VmirDisplay;
use crate::vmir::{FunctionBody, HeapVal, InstKind, PermVal, PureInst, TyParams, Type, Val};
use std::fmt::{self, Display, Formatter};

/// A pure `forall`, **inline** in the enclosing instruction stream (it is an
/// ordinary [`PureInst`](crate::vmir::PureInst) producing a `Bool`, not a
/// declaration). Capture is **implicit**: the body shares the enclosing temp
/// space, so a free occurrence of an outer value is simply that value's temp —
/// there is no capture list to build or keep in sync.
///
/// Let `p` be the temp of the `forall` step itself. It is allocated *before* the
/// body is lowered, which makes it the scope boundary:
///
/// | temp | meaning |
/// |---|---|
/// | `< p` | free — an enclosing value (this is the capture) |
/// | `binder_base .. binder_base + bound.len()` | a binder |
/// | `>= binder_base + bound.len()` | a body-local step |
///
/// with `binder_base == p`: the numbering starts *at* the `forall` step's own
/// temp, so the first binder **shadows** that boolean and the quantifier has no
/// way to refer to itself. The enclosing stream resumes numbering at `p + 1`, so
/// the body's temps are likewise shadowed by whatever follows the quantifier —
/// ordinary lexical scoping, and harmless because the two scopes never overlap in
/// time: the body can only mention temps that already exist when the quantifier is
/// reached.
///
/// Nesting needs no extra mechanism: an inner `forall` is a `PureInst::Forall`
/// inside the outer's body, and since its own `binder_base` is larger, the outer's
/// binders and captures read as *free* to it under the same rule. The verifier
/// encodes a `forall` as a single e-node (payload = the compiled body, children =
/// the e-classes of [`Forall::free_temps`]), so an outer instantiation materializes
/// the inner quantifier with the outer σ baked into the children.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Forall {
    /// The first temp this quantifier owns — the `forall` step's own temp, which
    /// the first binder therefore shadows. Redundant with that step's position in
    /// the enclosing stream, but kept here so a `Forall` is self-contained: it is
    /// passed around alone (recipe interning, well-definedness checking) with no
    /// access to its host.
    pub binder_base: usize,
    /// The **type** binders: this quantifier binds `Generic(0..ty_binders)`, and
    /// an instantiation fixes them by the type arguments of the trigger terms it
    /// matches. Non-zero only on the root quantifier of a generic axiom (Viper
    /// has no nested type quantifiers), so `Generic(i)` always names that root's
    /// binder `i`; a nested `forall` mentioning one leaves it open for the
    /// enclosing instantiation to fix. Printed `?i: Type` before the binders.
    pub ty_binders: TyParams,
    /// The binder types; binder `i` is `Val::Temp(binder_base + i)`. Empty for
    /// a quantifier that binds only types.
    pub bound: Box<[Type]>,
    /// The trigger groups, in source order — **alternatives**: a match of *any*
    /// one of them instantiates the quantifier. Never empty. Written by the user,
    /// except on a generic axiom's root quantifier that binds only types, whose
    /// triggers translation infers (`translate::decl::domain`).
    pub triggers: Box<[QuantTrigger]>,
    pub body: FunctionBody,
}

impl Forall {
    /// One past the last temp this quantifier owns for binders.
    pub fn step_base(&self) -> usize {
        self.binder_base + self.bound.len()
    }

    /// The enclosing temps this quantifier's triggers and body mention, in
    /// first-appearance order — **triggers first**, so a trigger's variables get
    /// the leading slots. This is the capture list, derived rather than stored:
    /// the verifier resolves each entry against the enclosing evaluation state to
    /// get the e-node's children.
    ///
    /// A nested `forall` contributes whatever *its* subtree mentions below **our**
    /// `binder_base` — the two levels are filtered by the same rule, so an inner
    /// reference to an outer binder stops here rather than escaping further out.
    pub fn free_temps(&self) -> Vec<usize> {
        free_temps_of(self.binder_base, &self.triggers, &self.body.insts, &[])
    }

    /// The heap reads of this quantifier's own body, in instruction order (see
    /// [`HeapRead`]). Derived, like [`Forall::free_temps`].
    pub fn heap_reads(&self) -> Vec<HeapRead> {
        heap_reads_of(self.step_base(), &self.body.insts)
    }

    /// The heap-dependent calls of this quantifier's own body, in instruction
    /// order (see [`HeapCall`]). Derived, like [`Forall::heap_reads`].
    pub fn heap_calls(&self) -> Vec<HeapCall> {
        heap_calls_of(self.binder_base, self.step_base(), &self.body.insts)
    }

    /// Visit every type this quantifier mentions: its binder types, its
    /// triggers' type arguments, and every type of its body, nested quantifiers
    /// included.
    pub fn for_each_type(&self, f: &mut impl FnMut(&Type)) {
        self.bound.iter().for_each(&mut *f);
        for term in self.triggers.iter().flat_map(|g| g.terms.iter()) {
            term.for_each_type(f);
        }
        self.body.for_each_type(f);
    }

    fn for_each_type_mut(&mut self, f: &mut impl FnMut(&mut Type)) {
        self.bound.iter_mut().for_each(&mut *f);
        for term in self.triggers.iter_mut().flat_map(|g| g.terms.iter_mut()) {
            term.for_each_type_mut(f);
        }
        self.body.for_each_type_mut(f);
    }
}

/// The enclosing temps (`< binder_base`) a quantified scope mentions, in
/// first-appearance order: triggers first, then the body's instructions, then
/// `extra` (operands the scope's result carries outside its instruction
/// stream). Shared by [`Forall`] and [`QuantChunks`], so the two capture rules
/// cannot drift apart.
fn free_temps_of(
    binder_base: usize,
    triggers: &[QuantTrigger],
    insts: &[crate::vmir::Inst],
    extra: &[Val],
) -> Vec<usize> {
    let mut out = Vec::new();
    let mut seen = crate::dhash::HashSet::default();
    let mut visit = |v: &Val| {
        if let Val::Temp(k) = v
            && *k < binder_base
            && seen.insert(*k)
        {
            out.push(*k);
        }
    };
    for group in triggers {
        for term in group.terms.iter() {
            term.for_each_var(&mut visit);
        }
    }
    for inst in insts {
        inst.for_each_operand(&mut visit);
    }
    extra.iter().for_each(&mut visit);
    out
}

/// A heap read in a quantifier body: a `Deref` of an **enclosing** heap. A body
/// defines no heap, so every heap it mentions is captured, exactly as a temp
/// below `binder_base` is. The verifier resolves each read to a value map of
/// that heap before it uses the body, so the body it compiles is heap-free.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeapRead {
    /// The enclosing heap it reads (`Empty` in a resource body's first slots).
    pub heap: HeapVal,
    /// The address it reads, a body operand.
    pub addr: Val,
    /// The address's `Type::Addr`: which kind of location it reads.
    pub addr_ty: Type,
}

/// The heap reads of a quantified scope's own instructions (its steps from
/// `step_base` on), in order. A nested `forall`'s reads belong to that
/// quantifier and are not listed: each level resolves its own. Panics on a
/// read at an address the body does not compute: translation builds every
/// address inside the body (`resource::field_addr` in the body's sink).
fn heap_reads_of(step_base: usize, insts: &[crate::vmir::Inst]) -> Vec<HeapRead> {
    let step_types = step_types(insts);
    insts
        .iter()
        .filter_map(|inst| match &inst.kind {
            InstKind::Pure(_, PureInst::Deref(heap, addr)) => {
                let addr_ty = match addr {
                    Val::Temp(t) if *t >= step_base => step_types[t - step_base].clone(),
                    _ => panic!("a quantifier body reads at {addr:?}, which it does not compute"),
                };
                Some(HeapRead {
                    heap: *heap,
                    addr: addr.clone(),
                    addr_ty,
                })
            }
            _ => None,
        })
        .collect()
}

/// The type of the `Val` each step of a quantifier body yields, in order: a
/// body's temps from `step_base` on name these. A pure step yields its own
/// type, a heap-dependent call's frame-only exhale its snapshot.
fn step_types(insts: &[crate::vmir::Inst]) -> Vec<Type> {
    insts
        .iter()
        .filter_map(|i| match &i.kind {
            InstKind::Pure(ty, _) => Some(ty.clone()),
            InstKind::Heap(hi) => Some(Type::Snap(frame_call(hi).resource)),
            _ => None,
        })
        .collect()
}

/// The call of a quantifier body's heap step, which is always a frame-only
/// exhale (see [`HeapCall`]).
fn frame_call(hi: &crate::vmir::HeapInst) -> &crate::vmir::ResourceCall {
    hi.frame_exhale().map(|(_, call)| call).unwrap_or_else(|| {
        panic!("a quantifier body holds a heap step other than a frame-only exhale: {hi:?}")
    })
}

/// A heap-dependent function call in a quantifier body: the frame-only
/// `exhale` of the callee's `#requires` on an **enclosing** heap, which yields
/// the snapshot the call takes. Like a [`HeapRead`], the verifier resolves it
/// before it uses the body: framed per binding, then replaced by stand-ins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeapCall {
    /// The enclosing heap it frames against.
    pub heap: HeapVal,
    pub call: crate::vmir::ResourceCall,
    /// The snapshot temp the exhale yields.
    pub snap: Val,
    /// No argument depends on a binder, so the snapshot is the same for every
    /// binding.
    pub binder_free: bool,
}

/// The heap-dependent calls of a quantified scope's own instructions, in
/// order. A nested `forall`'s calls belong to that quantifier.
fn heap_calls_of(
    binder_base: usize,
    step_base: usize,
    insts: &[crate::vmir::Inst],
) -> Vec<HeapCall> {
    // Per body step: whether its value depends on a binder.
    let mut dep: Vec<bool> = Vec::new();
    let depends = |dep: &[bool], v: &Val| match v {
        Val::Temp(k) if *k >= step_base => dep[k - step_base],
        Val::Temp(k) => *k >= binder_base,
        _ => false,
    };
    let mut out = Vec::new();
    for inst in insts {
        match &inst.kind {
            InstKind::Pure(_, pi) => {
                let mut d = false;
                pi.for_each_operand(&mut |v| d |= depends(&dep, v));
                dep.push(d);
            }
            InstKind::Heap(hi) => {
                let call = frame_call(hi);
                let (heap, _) = hi.frame_exhale().expect("checked by frame_call");
                let d = call.args.iter().any(|v| depends(&dep, v));
                out.push(HeapCall {
                    heap,
                    call: call.clone(),
                    snap: Val::Temp(step_base + dep.len()),
                    binder_free: !d,
                });
                dep.push(d);
            }
            _ => {}
        }
    }
    out
}

/// A quantified set of heap chunks, `{ loc(xs) @ perm(xs) | cond(xs) }` — the
/// payload of a quantified permission. Scoped exactly like [`Forall`]: the body
/// shares the enclosing temp space, binder `i` is `Val::Temp(binder_base + i)`,
/// and a temp below `binder_base` is a free (captured) enclosing value.
///
/// Unlike a `Forall`, the hosting heap instruction defines no `Val` of its own,
/// so `binder_base` is the enclosing stream's *next* temp and the stream resumes
/// at that same number afterwards (the body is shadowed, never overlapped).
///
/// `cond` is kept apart from `perm` on purpose: it is the chunk set's domain,
/// which lookup, inverse functions and injectivity all need. Only the
/// *enclosing* branch conditions are folded into `perm`, as for a single `acc`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantChunks {
    /// The first temp this scope owns; binder `i` is `Val::Temp(binder_base + i)`.
    pub binder_base: usize,
    /// The binder types.
    pub bound: Box<[Type]>,
    /// Trigger groups, as alternatives. Without user triggers, translation
    /// uses the location's address call `f(e(xs))`; empty only when that
    /// cannot serve as one either (e.g. a constant receiver).
    pub triggers: Box<[QuantTrigger]>,
    pub loc_ty: Type,
    pub body: QuantBody,
}

/// The body of a [`QuantChunks`]: `Pure` and `Perm` instructions, plus the
/// `Assert`/`Assume` a call to a contract-bearing function stitches in, then the
/// per-binding condition, location and permission.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantBody {
    pub insts: Vec<crate::vmir::Inst>,
    /// `Bool`: whether the binding contributes a chunk.
    pub cond: Val,
    /// `Addr<T>`: every instance has the same location kind.
    pub loc: Val,
    /// The amount; a `Temp` names a `Perm` instruction of *this body* (body
    /// permission temps count from 0).
    pub perm: PermVal,
}

impl QuantBody {
    /// Returns true iff in any instruction a wildcard appears or the permission is a wildcard.
    pub fn contains_wc_permission(&self) -> bool {
        self.perm.is_wildcard()
            || self
                .insts
                .iter()
                .any(|i| matches!(&i.kind, InstKind::Perm(p) if p.has_wildcard_arm()))
    }
}

impl QuantChunks {
    /// One past the last temp this scope owns for binders.
    pub fn step_base(&self) -> usize {
        self.binder_base + self.bound.len()
    }

    /// The heap reads of this scope's body, in instruction order (see
    /// [`HeapRead`]): a heap-dependent condition, receiver or amount.
    pub fn heap_reads(&self) -> Vec<HeapRead> {
        heap_reads_of(self.step_base(), &self.body.insts)
    }

    /// The heap-dependent calls of this scope's body, in instruction order
    /// (see [`HeapCall`]).
    pub fn heap_calls(&self) -> Vec<HeapCall> {
        heap_calls_of(self.binder_base, self.step_base(), &self.body.insts)
    }

    /// The enclosing temps this scope mentions (its capture list), triggers
    /// first, then the body, then `cond`, `loc` and the permission amount.
    pub fn free_temps(&self) -> Vec<usize> {
        let mut extra = vec![self.body.cond.clone(), self.body.loc.clone()];
        if let PermVal::Amount(a) = &self.body.perm {
            extra.push(a.clone());
        }
        free_temps_of(self.binder_base, &self.triggers, &self.body.insts, &extra)
    }
}

impl FunctionBody {
    /// Visit every type of this **pure** body: each step's result type and the
    /// type arguments it carries, nested quantifiers included. Panics on a heap
    /// or permission step: only quantifier and axiom bodies are walked, and those
    /// are pure.
    pub fn for_each_type(&self, f: &mut impl FnMut(&Type)) {
        for inst in &self.insts {
            match &inst.kind {
                InstKind::Pure(ty, pi) => {
                    f(ty);
                    match pi {
                        PureInst::FunctionCall(fc) => fc.type_args.iter().for_each(&mut *f),
                        PureInst::AdtCons { type_args, .. }
                        | PureInst::AdtProj { type_args, .. }
                        | PureInst::AdtIs { type_args, .. } => type_args.iter().for_each(&mut *f),
                        PureInst::Coll { tys, .. } => tys.iter().for_each(&mut *f),
                        PureInst::CollLit { elem_ty, .. } => f(elem_ty),
                        PureInst::Forall(q) => q.for_each_type(f),
                        _ => {}
                    }
                }
                InstKind::Assume(_) | InstKind::Assert(_) | InstKind::Refute(_) => {}
                // A heap-dependent call's snapshot: `Snap(..)` has no type
                // arguments.
                InstKind::Heap(hi) if hi.frame_exhale().is_some() => {}
                InstKind::Heap(_) | InstKind::Perm(_) => {
                    panic!(
                        "a quantifier or axiom body is pure, but has {:?}",
                        inst.kind
                    )
                }
            }
        }
    }

    fn for_each_type_mut(&mut self, f: &mut impl FnMut(&mut Type)) {
        for inst in &mut self.insts {
            match &mut inst.kind {
                InstKind::Pure(ty, pi) => {
                    f(ty);
                    match pi {
                        PureInst::FunctionCall(fc) => fc.type_args.iter_mut().for_each(&mut *f),
                        PureInst::AdtCons { type_args, .. }
                        | PureInst::AdtProj { type_args, .. }
                        | PureInst::AdtIs { type_args, .. } => {
                            type_args.iter_mut().for_each(&mut *f)
                        }
                        PureInst::Coll { tys, .. } => tys.iter_mut().for_each(&mut *f),
                        PureInst::CollLit { elem_ty, .. } => f(elem_ty),
                        PureInst::Forall(q) => q.for_each_type_mut(f),
                        _ => {}
                    }
                }
                InstKind::Assume(_) | InstKind::Assert(_) | InstKind::Refute(_) => {}
                // A heap-dependent call's snapshot: `Snap(..)` has no type
                // arguments.
                InstKind::Heap(hi) if hi.frame_exhale().is_some() => {}
                InstKind::Heap(_) | InstKind::Perm(_) => {
                    panic!(
                        "a quantifier or axiom body is pure, but has {:?}",
                        inst.kind
                    )
                }
            }
        }
    }

    /// Record every `Generic(i)` this pure body mentions (see
    /// [`FunctionBody::for_each_type`]).
    pub fn collect_generics(&self, out: &mut crate::dhash::HashSet<usize>) {
        self.for_each_type(&mut |t| t.collect_generics(out));
    }

    /// This pure body with every `Generic(i)` replaced by `tys[i]`, nested
    /// quantifiers included. Panics on a quantifier that binds types: its
    /// `Generic`s are its own, not the ones being substituted.
    pub fn subst_generics(&self, tys: &[Type]) -> FunctionBody {
        self.assert_no_type_binders();
        let mut out = self.clone();
        out.for_each_type_mut(&mut |t| *t = t.subst_generics(tys));
        out
    }

    fn assert_no_type_binders(&self) {
        for inst in &self.insts {
            if let InstKind::Pure(_, PureInst::Forall(q)) = &inst.kind {
                assert_eq!(
                    q.ty_binders.count(),
                    0,
                    "only a generic axiom's root quantifier binds types"
                );
                q.body.assert_no_type_binders();
            }
        }
    }
}

impl TrigTerm {
    /// Visit every variable position of the pattern, outermost-first, as the
    /// `Val::Temp` it names.
    pub fn for_each_var(&self, f: &mut impl FnMut(&Val)) {
        match self {
            TrigTerm::Var(k) => f(&Val::Temp(*k)),
            TrigTerm::Lit(_) | TrigTerm::Wild => {}
            TrigTerm::App { args, .. } => args.iter().for_each(|a| a.for_each_var(f)),
        }
    }

    /// Visit the type arguments of every application in the pattern.
    pub fn for_each_type(&self, f: &mut impl FnMut(&Type)) {
        if let TrigTerm::App {
            type_args, args, ..
        } = self
        {
            type_args.iter().for_each(&mut *f);
            args.iter().for_each(|a| a.for_each_type(f));
        }
    }

    fn for_each_type_mut(&mut self, f: &mut impl FnMut(&mut Type)) {
        if let TrigTerm::App {
            type_args, args, ..
        } = self
        {
            type_args.iter_mut().for_each(&mut *f);
            args.iter_mut().for_each(|a| a.for_each_type_mut(f));
        }
    }

    /// Every `Generic(i)` the type arguments of the pattern's applications
    /// mention: the type parameters a match of it fixes.
    pub fn collect_generics(&self, out: &mut crate::dhash::HashSet<usize>) {
        self.for_each_type(&mut |t| t.collect_generics(out));
    }
}

/// One trigger group — a conjunctive multi-pattern (`{f(x), g(x)}`): the
/// quantifier instantiates at a σ only when **every** term matches. Each term's
/// root is an application ([`TrigTerm::App`]), and the group's `Bound` positions
/// jointly cover all binders (typecheck-enforced; repeats allowed).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantTrigger {
    pub terms: Box<[TrigTerm]>,
}

/// A trigger pattern term. A `Var` is a temp of the enclosing space, classified
/// exactly as the body's operands are (see [`Forall`]): at or above the
/// quantifier's `binder_base` it is a binder — matching it determines σ — and
/// below it is a free occurrence, which the match must reproduce. A `Lit`
/// requires the literal, and an `App` requires an application of that head whose
/// arguments recursively match, so a trigger may nest arbitrarily
/// (`{f(g(x), c)}`). A `Wild` matches anything and binds nothing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TrigTerm {
    Var(usize),
    Lit(crate::vmir::Literal),
    /// Any value. Only in an inferred trigger of a quantifier that binds only
    /// types: it has no value binders to bind, and its trigger needs only the
    /// head and type arguments of an application to fix its types. Printed `_`.
    Wild,
    App {
        head: TrigHead,
        type_args: Vec<Type>,
        args: Box<[TrigTerm]>,
    },
}

/// The head of a trigger application — the same heads a body's `PureInst` can
/// produce, so a trigger matches exactly what the program can build.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TrigHead {
    Func(crate::vmir::MemberId),
    AdtCons {
        adt: crate::vmir::MemberId,
        variant: usize,
    },
    AdtProj {
        adt: crate::vmir::MemberId,
        variant: usize,
        field: usize,
    },
    /// The tester of variant `variant`: a trigger `{x.isC}` fires only on
    /// `x.isC`, not on a test of another variant of the same value.
    AdtIs {
        adt: crate::vmir::MemberId,
        variant: usize,
    },
    /// A collection operation. The `App`'s `args` are its operands in
    /// `PureInst::Coll` order (so `x in s` is `Contains(s, x)`), and `type_args`
    /// is the type arguments of the collection it works on (`[E]`, or `[K, V]`
    /// for a map).
    Coll(crate::vmir::CollOp),
    /// The singleton collection of the `App`'s one argument,
    /// `PureInst::CollLit`. `type_args` is `[elem_ty]`.
    CollLit(crate::vmir::CollKind),
}

/// A trigger term, rendered in the enclosing temp syntax: every variable is an
/// `e{k}`, whether it names a binder or a free enclosing value — the same
/// notation the body uses, since they share one space.
impl<'a> Display for VmirDisplay<'a, &'a TrigTerm> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let term = self.item;
        match term {
            TrigTerm::Var(k) => write!(f, "e{k}"),
            TrigTerm::Lit(lit) => write!(f, "{lit}"),
            TrigTerm::Wild => write!(f, "_"),
            TrigTerm::App {
                head,
                type_args,
                args,
            } => {
                match head {
                    TrigHead::Func(id) => write!(f, "{}", self.member(*id))?,
                    TrigHead::AdtCons { adt, variant } => {
                        write!(f, "{}", self.adt_variant(*adt, *variant))?
                    }
                    TrigHead::AdtProj {
                        adt,
                        variant,
                        field,
                    } => write!(f, "{}.{field}", self.adt_variant(*adt, *variant))?,
                    TrigHead::AdtIs { adt, variant } => {
                        write!(f, "{}?", self.adt_variant(*adt, *variant))?
                    }
                    TrigHead::Coll(op) => write!(f, "{}", op.token())?,
                    TrigHead::CollLit(kind) => write!(f, "{}", kind.name())?,
                }
                if !type_args.is_empty() {
                    write!(f, "<")?;
                    for (i, t) in type_args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", self.with(t))?;
                    }
                    write!(f, ">")?;
                }
                write!(f, "(")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", self.with(a))?;
                }
                write!(f, ")")
            }
        }
    }
}

/// An inline `forall`, rendered as a nested block. Free occurrences are just the
/// enclosing temps — there is no capture list to print:
///
/// ```text
///   e1: Bool := forall e1: Int :: {f(e0, e1)} {
///     e2: Bool := f(e0, e1)
///     result: e2
///   }
/// ```
///
/// The binders and the body start the numbering *at* the `forall` step's own temp
/// (`e1` here, shadowed by the first binder), and the enclosing stream resumes at
/// `e2` — so the nesting, not the numbering, is what tells the two scopes apart.
impl<'a> Display for VmirDisplay<'a, &'a Forall> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let q = self.item;
        write!(f, "forall ")?;
        let tys = (0..q.ty_binders.count()).map(|i| format!("?{i}: Type"));
        let vals = q
            .bound
            .iter()
            .enumerate()
            .map(|(k, ty)| format!("e{}: {}", q.binder_base + k, self.with(ty)));
        write!(f, "{}", tys.chain(vals).collect::<Vec<_>>().join(", "))?;
        // Trigger groups, alternatives side by side: `{f(e1), g(e1)}{h(e1)}`.
        write!(f, " :: ")?;
        for group in q.triggers.iter() {
            write!(f, "{{")?;
            for (k, term) in group.terms.iter().enumerate() {
                if k > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.with(term))?;
            }
            write!(f, "}}")?;
        }
        writeln!(f, " {{")?;
        // The body's own steps continue after the binders, rendered one nesting
        // level deeper so an inner `forall` indents further than its parent.
        let body_indent = self.with_nested(()).indent();
        write!(
            f,
            "{}",
            self.with_nested((q.step_base(), 0usize, 0usize, &q.body.insts[..]))
        )?;
        writeln!(f, "{body_indent}result: {}", q.body.res)?;
        write!(f, "{}}}", self.indent())
    }
}

/// A quantified chunk set, rendered as a nested block whose result is the
/// guarded chunk `cond ? loc @ perm`:
///
/// ```text
/// forall e2: Ref :: {
///   e3: Bool := contains(e0, e2)
///   e4: &[f] Int @ 1/1 := f(e2)
///   result: e3 ? e4 @ 1/1
/// }
/// ```
///
/// Triggers are printed only when present (` :: {f(e2)}`).
impl<'a> Display for VmirDisplay<'a, &'a QuantChunks> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let q = self.item;
        write!(f, "forall ")?;
        for (k, ty) in q.bound.iter().enumerate() {
            if k > 0 {
                write!(f, ", ")?;
            }
            write!(f, "e{}: {}", q.binder_base + k, self.with(ty))?;
        }
        write!(f, " ::")?;
        for group in q.triggers.iter() {
            write!(f, " {{")?;
            for (k, term) in group.terms.iter().enumerate() {
                if k > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.with(term))?;
            }
            write!(f, "}}")?;
        }
        writeln!(f, " {{")?;
        let body_indent = self.with_nested(()).indent();
        // Body permission temps are local to the body, so they count from `p0`.
        write!(
            f,
            "{}",
            self.with_nested((q.step_base(), 0usize, 0usize, &q.body.insts[..]))
        )?;
        writeln!(
            f,
            "{body_indent}result: {} ? {} @ {}",
            q.body.cond, q.body.loc, q.body.perm
        )?;
        write!(f, "{}}}", self.indent())
    }
}
