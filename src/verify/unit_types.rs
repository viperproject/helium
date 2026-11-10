//! The **types of a verification unit**: the ground types whose axioms a unit's
//! e-graph may need. Collected once, before the unit is walked, by
//! [`method_types`] / [`function_types`] / [`resource_types`].
//!
//! A type belongs to a unit `U` (a method, a function, or a predicate — a
//! `Resource`) iff at least one of these holds:
//!
//! 1. it is the type of a value of `U`: a parameter, a return value, a local, any
//!    intermediate value of the flattened body, or a variable bound by a quantifier
//!    in `U`;
//! 2. it is the type of a value in the contract of `U`;
//! 3. it is the type of a value in the contract of a method or function `U` calls
//!    directly;
//! 4. it is the type of a value in the body of a function `U` calls directly, or of
//!    a predicate `U` folds or unfolds (their bodies are unfolded while `U` is
//!    verified, so their values become terms of `U`'s e-graph);
//! 5. it is a type argument of a generic application in any of the places above.
//!
//! All of them are ground, and only the types themselves count, never the types
//! nested inside them: `Box<Int>` is not a type of a unit with a parameter
//! `Option<Box<Int>>` until the unit unwraps it. Types that occur only in an axiom
//! body, or in a function body two calls away, do not count.
//!
//! In VMIR a contract is not attached to its owner: a method's `#requires` and
//! `#ensures` are resources that its body `inhale`s / `exhale`s, a callee's are
//! `exhale`d / `inhale`d at the call, and a `fold` / `unfold` is an `inhale` /
//! `exhale` of the predicate. So rules 2–4 are one thing on a method or predicate:
//! every resource its instructions name contributes the values of its definition.
//! A function carries explicit `requires` / `ensures` links instead.

use crate::dhash::HashSet;
use crate::vmir::{
    Declaration, Function, HeapInst, Inst, InstKind, MemberId, Method, Program, PureInst, Resource,
    Type,
};

/// The ground types of one verification unit.
pub(crate) type UnitTypes = HashSet<Type>;

/// The types of method `method`.
pub(crate) fn method_types(program: &Program, method: &Method) -> UnitTypes {
    let mut c = Collector::new(program);
    for block in &method.blocks {
        c.insts(&block.join, true);
        c.insts(&block.body, true);
    }
    c.out
}

/// The types of function `function`. A generic domain function has no values (it
/// has no body, and its signature is not ground), so its types are empty.
pub(crate) fn function_types(program: &Program, function: &Function) -> UnitTypes {
    let mut c = Collector::new(program);
    if function.ty_params.count() > 0 {
        return c.out;
    }
    c.add_all(function.params.iter());
    c.add(&function.ret);
    if let Some(body) = &function.body {
        c.insts(&body.insts, true);
    }
    // Its own contract: the definitions of its `requires` / `ensures` links.
    if let Some(requires) = &function.requires {
        c.definition(requires.member());
    }
    if let Some(ensures) = &function.ensures {
        c.definition(ensures.member);
    }
    c.out
}

/// The types of predicate (resource) `resource`.
pub(crate) fn resource_types(program: &Program, resource: &Resource) -> UnitTypes {
    let mut c = Collector::new(program);
    c.add_all(resource.params.iter());
    c.insts(&resource.body.insts, true);
    c.out
}

struct Collector<'p> {
    program: &'p Program,
    out: UnitTypes,
    /// Definitions already visited as a callee, so each contributes once.
    visited: HashSet<MemberId>,
}

impl<'p> Collector<'p> {
    fn new(program: &'p Program) -> Self {
        Self {
            program,
            out: UnitTypes::default(),
            visited: HashSet::default(),
        }
    }

    /// Record one type. A unit's types are ground — an axiom body and a generic
    /// domain function's signature are the only places a `Generic` occurs, and
    /// neither is walked — so a `Generic` here is a bug upstream.
    fn add(&mut self, ty: &Type) {
        let mut generics = HashSet::default();
        ty.collect_generics(&mut generics);
        assert!(
            generics.is_empty(),
            "the types of a verification unit are ground, found `{ty}`"
        );
        self.out.insert(ty.clone());
    }

    fn add_all<'t>(&mut self, tys: impl IntoIterator<Item = &'t Type>) {
        for ty in tys {
            self.add(ty);
        }
    }

    /// The values of an instruction stream: each result type, each type argument
    /// of an application, each quantifier binder. With `follow`, also the
    /// definitions it calls *directly* (never theirs: those are walked with
    /// `follow` off).
    fn insts(&mut self, insts: &[Inst], follow: bool) {
        for inst in insts {
            match &inst.kind {
                InstKind::Pure(ty, pi) => {
                    self.add(ty);
                    self.pure(pi, follow);
                }
                // `inhale` / `exhale` of a resource: a contract (own or a
                // callee's), or a predicate being folded or unfolded.
                InstKind::Heap(HeapInst::Inhale { call, .. } | HeapInst::Exhale { call, .. }) => {
                    if follow {
                        self.definition(call.resource);
                    }
                }
                InstKind::Heap(_)
                | InstKind::Perm(_)
                | InstKind::Assume(_)
                | InstKind::Assert(_)
                | InstKind::Refute(_) => {}
            }
        }
    }

    fn pure(&mut self, pi: &PureInst, follow: bool) {
        match pi {
            PureInst::FunctionCall(call) => {
                self.add_all(&call.type_args);
                if follow {
                    self.definition(call.function);
                }
            }
            PureInst::AdtCons { type_args, .. }
            | PureInst::AdtProj { type_args, .. }
            | PureInst::AdtIs { type_args, .. } => self.add_all(type_args),
            // A quantified body is part of the unit, calls included.
            PureInst::Forall(q) => {
                self.add_all(q.bound.iter());
                self.insts(&q.body.insts, follow);
            }
            PureInst::Fresh
            | PureInst::Binary(..)
            | PureInst::Ternary(..)
            | PureInst::RealCast(_)
            | PureInst::Deref(..)
            | PureInst::Perm(..)
            | PureInst::OptionUnwrap(_)
            | PureInst::SeqLit { .. }
            | PureInst::Seq { .. } => {}
        }
    }

    /// The values of declaration `id`'s definition, without following what it
    /// calls: a function's parameters, result, body and contract links; a
    /// resource's parameters and body.
    fn definition(&mut self, id: MemberId) {
        if !self.visited.insert(id) {
            return;
        }
        let program = self.program;
        match &program.decls[id] {
            Declaration::Function(f) => {
                // A generic domain function has no definition to unfold, and its
                // signature is not ground.
                if f.ty_params.count() > 0 {
                    return;
                }
                self.add_all(f.params.iter());
                self.add(&f.ret);
                if let Some(body) = &f.body {
                    self.insts(&body.insts, false);
                }
                // The callee's contract: the values of the definitions it links.
                if let Some(requires) = &f.requires {
                    self.definition(requires.member());
                }
                if let Some(ensures) = &f.ensures {
                    self.definition(ensures.member);
                }
            }
            Declaration::Resource(r) => {
                self.add_all(r.params.iter());
                self.insts(&r.body.insts, false);
            }
            _ => unreachable!(
                "a call, a contract link or an inhale / exhale names a function or a resource"
            ),
        }
    }
}
