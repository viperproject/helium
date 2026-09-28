use crate::vmir::display::VmirDisplay;
use crate::vmir::{FunctionBody, Type, Val};
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
    /// The binder types; binder `i` is `Val::Temp(binder_base + i)`.
    pub bound: Box<[Type]>,
    /// The trigger groups, in source order — **alternatives**: a match of *any*
    /// one of them instantiates the quantifier. Never empty and never inferred:
    /// a `forall` without a usable trigger is a type error.
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
        let mut out = Vec::new();
        let mut seen = crate::dhash::HashSet::default();
        let mut visit = |v: &Val| {
            if let Val::Temp(k) = v
                && *k < self.binder_base
                && seen.insert(*k)
            {
                out.push(*k);
            }
        };
        for group in self.triggers.iter() {
            for term in group.terms.iter() {
                term.for_each_var(&mut visit);
            }
        }
        for inst in &self.body.insts {
            inst.for_each_operand(&mut visit);
        }
        out
    }
}

impl TrigTerm {
    /// Visit every variable position of the pattern, outermost-first, as the
    /// `Val::Temp` it names.
    pub fn for_each_var(&self, f: &mut impl FnMut(&Val)) {
        match self {
            TrigTerm::Var(k) => f(&Val::Temp(*k)),
            TrigTerm::Lit(_) => {}
            TrigTerm::App { args, .. } => args.iter().for_each(|a| a.for_each_var(f)),
        }
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
/// (`{f(g(x), c)}`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TrigTerm {
    Var(usize),
    Lit(crate::vmir::Literal),
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
    AdtTag {
        adt: crate::vmir::MemberId,
    },
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
                    TrigHead::AdtTag { adt } => write!(f, "{}@tag", self.member(*adt))?,
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
        for (k, ty) in q.bound.iter().enumerate() {
            if k > 0 {
                write!(f, ", ")?;
            }
            write!(f, "e{}: {}", q.binder_base + k, self.with(ty))?;
        }
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
