//! Interner-aware `Display` for the typed `typed`.
//!
//! `typed::Ident` only holds a `Spur`, so rendering readable names requires
//! the `Interner`. Mirrors `vmir::display`: a `Show<'a, T>` wrapper carries the
//! interner and is rethreaded into children via `.with(..)`.
//!
//! Every `TypedPureExp` is printed as `(expr : Type)` so inferred types are
//! visible inline.

use std::fmt::{self, Display, Formatter};

use crate::viper::interner::Interner;
use crate::viper::typed::{
    AssignLhs, AssignRhs, AxiomExt, BinOp, BuiltinCollection, Call, Declaration, Domain,
    DomainFunction, Field, Forall, FuncEnsuresExt, Function, HeapExt, HeapNode, Ident, Literal,
    Method, MethodBodyExt, MethodEnsuresExt, Predicate, PredicateWithPerm, Program, PureExpKind,
    ResourceExp, ResourceExpKind, SpatialExp, SpatialExpKind, StarOrFields, Statement, StmtBlock,
    Type, TypedIdent, TypedPureExp, UnOp,
};

/// Interner-aware formatting wrapper.
pub struct Show<'a, T> {
    item: T,
    interner: &'a Interner,
}

impl<'a, T> Show<'a, T> {
    pub fn new(item: T, interner: &'a Interner) -> Self {
        Self { item, interner }
    }

    fn with<U>(&self, item: U) -> Show<'a, U> {
        Show {
            item,
            interner: self.interner,
        }
    }

    fn name(&self, id: Ident) -> &'a str {
        self.interner.resolve(&id.0)
    }
}

/// Convenience entry point: `print!("{}", show(&program, &interner))`.
pub fn show<'a>(program: &'a Program, interner: &'a Interner) -> Show<'a, &'a Program> {
    Show::new(program, interner)
}

// ---- Extension nodes (the `Ext` generic of pure expressions) ----

/// Lets `PureExpKind<Ext>` render its context-specific `Ext` node.
pub trait ShowExt {
    fn fmt_ext(&self, f: &mut Formatter<'_>, interner: &Interner) -> fmt::Result;
}

impl ShowExt for ! {
    fn fmt_ext(&self, _f: &mut Formatter<'_>, _interner: &Interner) -> fmt::Result {
        *self
    }
}

/// Render a heap node (`e.f` / `function(..)` / `unfolding .. in ..`).
fn fmt_heap_node<Ext: ShowExt>(
    node: &HeapNode<Ext>,
    f: &mut Formatter<'_>,
    interner: &Interner,
) -> fmt::Result {
    match node {
        HeapNode::Field(e, field) => {
            write!(
                f,
                "{}.{}",
                Show::new(e, interner),
                interner.resolve(&field.0)
            )
        }
        HeapNode::FunctionCall(call) => write!(f, "{}", Show::new(call, interner)),
        HeapNode::Unfolding(p, e) => {
            write!(
                f,
                "unfolding {} in {}",
                Show::new(p, interner),
                Show::new(e, interner)
            )
        }
    }
}

/// Render a `forall` quantifier (binders, trigger groups, body).
fn fmt_forall(q: &Forall, f: &mut Formatter<'_>, interner: &Interner) -> fmt::Result {
    write!(f, "forall ")?;
    for (i, bv) in q.bound.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}", Show::new(bv, interner))?;
    }
    write!(f, " :: ")?;
    for group in &q.triggers {
        write!(f, "{{")?;
        for (i, t) in group.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", Show::new(t, interner))?;
        }
        write!(f, "}}")?;
    }
    write!(f, " {}", Show::new(&q.body, interner))
}

impl ShowExt for HeapExt {
    fn fmt_ext(&self, f: &mut Formatter<'_>, interner: &Interner) -> fmt::Result {
        match self {
            HeapExt::Heap(node) => fmt_heap_node(node, f, interner),
            HeapExt::Forall(q) => fmt_forall(q, f, interner),
        }
    }
}

impl ShowExt for AxiomExt {
    fn fmt_ext(&self, f: &mut Formatter<'_>, interner: &Interner) -> fmt::Result {
        match self {
            AxiomExt::FunctionCall(call) => write!(f, "{}", Show::new(call, interner)),
            AxiomExt::Forall(q) => fmt_forall(q, f, interner),
        }
    }
}

impl ShowExt for FuncEnsuresExt {
    fn fmt_ext(&self, f: &mut Formatter<'_>, interner: &Interner) -> fmt::Result {
        match self {
            FuncEnsuresExt::Heap(node) => fmt_heap_node(node, f, interner),
            FuncEnsuresExt::Result => write!(f, "result"),
            FuncEnsuresExt::Old(e) => write!(f, "old({})", Show::new(e, interner)),
            FuncEnsuresExt::Forall(q) => fmt_forall(q, f, interner),
        }
    }
}

impl ShowExt for MethodEnsuresExt {
    fn fmt_ext(&self, f: &mut Formatter<'_>, interner: &Interner) -> fmt::Result {
        match self {
            MethodEnsuresExt::Heap(node) => fmt_heap_node(node, f, interner),
            MethodEnsuresExt::Old(e) => write!(f, "old({})", Show::new(e, interner)),
            MethodEnsuresExt::Forall(q) => fmt_forall(q, f, interner),
        }
    }
}

impl ShowExt for MethodBodyExt {
    fn fmt_ext(&self, f: &mut Formatter<'_>, interner: &Interner) -> fmt::Result {
        match self {
            MethodBodyExt::Heap(node) => fmt_heap_node(node, f, interner),
            MethodBodyExt::Old(None, e) => write!(f, "old({})", Show::new(e, interner)),
            MethodBodyExt::Old(Some(label), e) => {
                write!(
                    f,
                    "old[{}]({})",
                    interner.resolve(label),
                    Show::new(e, interner)
                )
            }
            MethodBodyExt::Perm(res) => write!(f, "perm({})", Show::new(res, interner)),
            MethodBodyExt::Forall(q) => fmt_forall(q, f, interner),
        }
    }
}

// ---- Top level ----

impl<'a> Display for Show<'a, &'a Program> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (i, decl) in self.item.decls.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            writeln!(f, "{}", self.with(decl))?;
        }
        Ok(())
    }
}

impl<'a> Display for Show<'a, &'a Declaration> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.item {
            Declaration::Function(d) => write!(f, "{}", self.with(d)),
            Declaration::Predicate(d) => write!(f, "{}", self.with(d)),
            Declaration::Method(d) => write!(f, "{}", self.with(d)),
            Declaration::Field(d) => write!(f, "{}", self.with(d)),
            // ADT decls are not yet consumed by translation; a minimal header
            // keeps the typed dump total.
            Declaration::Adt(d) => write!(f, "adt {}", self.name(d.name)),
            Declaration::Domain(d) => write!(f, "{}", self.with(d)),
        }
    }
}

impl<'a> Display for Show<'a, &'a Field> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "field {}", self.with(&self.item.0))
    }
}

impl<'a> Display for Show<'a, &'a TypedIdent> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {}",
            self.name(self.item.name),
            self.with(&self.item.ty)
        )
    }
}

impl<'a> Display for Show<'a, &'a Type> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.item {
            Type::Bool => write!(f, "Bool"),
            Type::Int => write!(f, "Int"),
            Type::Real => write!(f, "Real"),
            Type::Ref => write!(f, "Ref"),
            Type::Generic(id) => write!(f, "{}", self.name(*id)),
            Type::Collection(BuiltinCollection::Seq(t)) => write!(f, "Seq[{}]", self.with(&**t)),
            Type::Collection(BuiltinCollection::Set(t)) => write!(f, "Set[{}]", self.with(&**t)),
            Type::Collection(BuiltinCollection::Map(k, v)) => {
                write!(f, "Map[{}, {}]", self.with(&**k), self.with(&**v))
            }
            Type::Collection(BuiltinCollection::MultiSet(t)) => {
                write!(f, "Multiset[{}]", self.with(&**t))
            }
            Type::Domain(id, args) => {
                write!(f, "{}", self.name(*id))?;
                if !args.is_empty() {
                    write!(f, "[")?;
                    for (i, a) in args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", self.with(a))?;
                    }
                    write!(f, "]")?;
                }
                Ok(())
            }
        }
    }
}

fn fmt_params(
    f: &mut Formatter<'_>,
    show: &Show<'_, impl Sized>,
    params: &[TypedIdent],
) -> fmt::Result {
    write!(f, "(")?;
    for (i, p) in params.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}", show.with(p))?;
    }
    write!(f, ")")
}

impl<'a> Display for Show<'a, &'a Function> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "function {}", self.name(self.item.name))?;
        fmt_params(f, self, &self.item.params)?;
        write!(f, ": {}", self.with(&self.item.ret))?;
        if let Some(req) = &self.item.requires {
            write!(f, "\n  requires {}", self.with(req))?;
        }
        if let Some(ens) = &self.item.ensures {
            write!(f, "\n  ensures {}", self.with(ens))?;
        }
        if let Some(body) = &self.item.body {
            write!(f, "\n{{ {} }}", self.with(body))?;
        }
        Ok(())
    }
}

impl<'a> Display for Show<'a, &'a Domain> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "domain {}", self.name(self.item.name))?;
        if !self.item.type_params.is_empty() {
            write!(f, "[")?;
            for (i, p) in self.item.type_params.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.name(*p))?;
            }
            write!(f, "]")?;
        }
        writeln!(f, " {{")?;
        for func in &self.item.functions {
            writeln!(f, "  {}", self.with(func))?;
        }
        for ax in &self.item.axioms {
            write!(f, "  axiom")?;
            if let Some(n) = &ax.name {
                write!(f, " {}", self.name(*n))?;
            }
            writeln!(f, " {{ {} }}", self.with(&ax.exp))?;
        }
        write!(f, "}}")
    }
}

impl<'a> Display for Show<'a, &'a DomainFunction> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "function {}", self.name(self.item.name))?;
        fmt_params(f, self, &self.item.params)?;
        write!(f, ": {}", self.with(&self.item.ret))
    }
}

impl<'a> Display for Show<'a, &'a Predicate> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "predicate {}", self.name(self.item.name))?;
        fmt_params(f, self, &self.item.params)?;
        if let Some(body) = &self.item.body {
            write!(f, "\n{{ {} }}", self.with(body))?;
        }
        Ok(())
    }
}

impl<'a> Display for Show<'a, &'a Method> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "method {}", self.name(self.item.name))?;
        fmt_params(f, self, &self.item.params)?;
        if !self.item.rets.is_empty() {
            write!(f, " returns ")?;
            fmt_params(f, self, &self.item.rets)?;
        }
        if let Some(req) = &self.item.requires {
            write!(f, "\n  requires {}", self.with(req))?;
        }
        if let Some(ens) = &self.item.ensures {
            write!(f, "\n  ensures {}", self.with(ens))?;
        }
        if let Some(body) = &self.item.body {
            write!(f, "\n{}", self.with(body))?;
        }
        Ok(())
    }
}

impl<'a> Display for Show<'a, &'a StmtBlock> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "{{")?;
        for stmt in &self.item.0 {
            writeln!(f, "  {}", self.with(stmt))?;
        }
        write!(f, "}}")
    }
}

impl<'a> Display for Show<'a, &'a Statement> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.item {
            Statement::Assume(e) => write!(f, "assume {}", self.with(e)),
            Statement::Assert(e) => write!(f, "assert {}", self.with(e)),
            Statement::Refute(e) => write!(f, "refute {}", self.with(e)),
            Statement::Inhale(e) => write!(f, "inhale {}", self.with(e)),
            Statement::Exhale(e) => write!(f, "exhale {}", self.with(e)),
            Statement::If(cond, then, else_) => {
                write!(f, "if ({}) {}", self.with(cond), self.with(then))?;
                if let Some(e) = else_ {
                    write!(f, " else {}", self.with(e))?;
                }
                Ok(())
            }
            Statement::Var(decls, init) => {
                write!(f, "var ")?;
                for (i, d) in decls.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", self.with(d))?;
                }
                if let Some(rhs) = init {
                    write!(f, " := {}", self.with(rhs))?;
                }
                Ok(())
            }
            Statement::Assign(lhs, rhs) => {
                for (i, l) in lhs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", self.with(l))?;
                }
                if lhs.is_empty() {
                    write!(f, "{}", self.with(rhs))
                } else {
                    write!(f, " := {}", self.with(rhs))
                }
            }
            Statement::Block(b) => write!(f, "{}", self.with(b)),
            Statement::Fold(p) => write!(f, "fold {}", self.with(p)),
            Statement::Unfold(p) => write!(f, "unfold {}", self.with(p)),
            Statement::Label(l, invs) => {
                write!(f, "label {}", self.interner.resolve(l))?;
                for inv in invs {
                    write!(f, " invariant {}", self.with(inv))?;
                }
                Ok(())
            }
            Statement::Goto(l) => write!(f, "goto {}", self.interner.resolve(l)),
            Statement::While(cond, invs, body) => {
                write!(f, "while ({})", self.with(cond))?;
                for inv in invs {
                    write!(f, " invariant {}", self.with(inv))?;
                }
                write!(f, " {{ {} }}", self.with(body))
            }
        }
    }
}

impl<'a> Display for Show<'a, &'a AssignLhs> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.item {
            AssignLhs::Var(id) => write!(f, "{}", self.name(*id)),
            AssignLhs::Field(base, field) => {
                write!(f, "{}.{}", self.with(base), self.name(*field))
            }
        }
    }
}

impl<'a> Display for Show<'a, &'a AssignRhs> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.item {
            AssignRhs::New(s) => match s {
                StarOrFields::Star => write!(f, "new(*)"),
                StarOrFields::Fields(fields) => {
                    write!(f, "new(")?;
                    for (i, fld) in fields.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", self.name(*fld))?;
                    }
                    write!(f, ")")
                }
            },
            AssignRhs::MethodCall(call) => write!(f, "{}", self.with(call)),
            AssignRhs::Exp(e) => write!(f, "{}", self.with(e)),
        }
    }
}

// ---- Spatial expressions ----
//
// `SpatialExp<Ext>` recurses into itself, so a fully generic blanket impl would
// need to bound `Show<&SpatialExp<Ext>>: Display` on itself. Generate one
// concrete impl per real `Ext` instead; each child `self.with(..)` then resolves
// to the same concrete impl.

macro_rules! impl_spatial_display {
    ($ext:ty) => {
        impl<'a> Display for Show<'a, &'a SpatialExp<$ext>> {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                match self.item.0.as_ref() {
                    SpatialExpKind::Implies(c, rhs) => {
                        write!(f, "({} ==> {})", self.with(c), self.with(rhs))
                    }
                    SpatialExpKind::Conj(l, r) => {
                        write!(f, "({} && {})", self.with(l), self.with(r))
                    }
                    SpatialExpKind::Ternary { if_, then, else_ } => write!(
                        f,
                        "({} ? {} : {})",
                        self.with(if_),
                        self.with(then),
                        self.with(else_)
                    ),
                    SpatialExpKind::Acc(res, perm) => {
                        write!(f, "acc({}, {})", self.with(res), self.with(perm))
                    }
                    SpatialExpKind::Pure(e) => write!(f, "{}", self.with(e)),
                }
            }
        }
    };
}

impl_spatial_display!(!);
impl_spatial_display!(HeapExt);
impl_spatial_display!(FuncEnsuresExt);
impl_spatial_display!(MethodEnsuresExt);
impl_spatial_display!(MethodBodyExt);

// ---- Resource expressions ----

impl<'a, Ext: ShowExt> Display for Show<'a, &'a ResourceExp<Ext>>
where
    Show<'a, &'a TypedPureExp<Ext>>: Display,
    Show<'a, &'a Call<Ext>>: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.item.0.as_ref() {
            ResourceExpKind::Field(base, field) => {
                write!(f, "{}.{}", self.with(base), self.name(*field))
            }
            ResourceExpKind::PredicateCall(call) => write!(f, "{}", self.with(call)),
        }
    }
}

impl<'a, Ext: ShowExt> Display for Show<'a, &'a PredicateWithPerm<Ext>>
where
    Show<'a, &'a TypedPureExp<Ext>>: Display,
    Show<'a, &'a Call<Ext>>: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "acc({}, {})",
            self.with(&self.item.pred_call),
            self.with(&self.item.perm)
        )
    }
}

// ---- Calls ----

impl<'a, Ext: ShowExt> Display for Show<'a, &'a Call<Ext>>
where
    Show<'a, &'a TypedPureExp<Ext>>: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.name(self.item.name))?;
        for (i, arg) in self.item.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", self.with(arg))?;
        }
        write!(f, ")")
    }
}

// ---- Pure expressions ----

impl<'a, Ext: ShowExt> Display for Show<'a, &'a TypedPureExp<Ext>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        fmt_pure_kind(f, self, self.item.exp.as_ref())?;
        write!(f, " : {})", self.with(&self.item.ty))
    }
}

fn fmt_pure_kind<'a, Ext: ShowExt>(
    f: &mut Formatter<'_>,
    show: &Show<'a, &'a TypedPureExp<Ext>>,
    kind: &PureExpKind<Ext>,
) -> fmt::Result {
    match kind {
        PureExpKind::Ident(id) => write!(f, "{}", show.name(*id)),
        PureExpKind::Const(lit) => write!(f, "{}", show.with(lit)),
        PureExpKind::Unary(op, e) => write!(f, "{}{}", show.with(op), show.with(e)),
        PureExpKind::Binary(op, l, r) => {
            write!(f, "{} {} {}", show.with(l), show.with(op), show.with(r))
        }
        PureExpKind::Ternary { if_, then, else_ } => {
            write!(
                f,
                "{} ? {} : {}",
                show.with(if_),
                show.with(then),
                show.with(else_)
            )
        }
        PureExpKind::DomainFunctionCall(call) | PureExpKind::AdtConstructor(call) => {
            write!(f, "{}", show.with(call))
        }
        PureExpKind::LetIn { binder, value, exp } => write!(
            f,
            "let {} := {} in {}",
            show.name(*binder),
            show.with(value),
            show.with(exp)
        ),
        PureExpKind::AdtDestructor(e, field) => {
            write!(f, "{}.{}", show.with(e), show.name(*field))
        }
        PureExpKind::AdtDiscriminator(e, variant) => {
            write!(f, "{}.is{}", show.with(e), show.name(*variant))
        }
        PureExpKind::SeqLit(elems)
        | PureExpKind::SetLit(elems)
        | PureExpKind::MultisetLit(elems) => {
            let name = match kind {
                PureExpKind::SeqLit(_) => "Seq",
                PureExpKind::SetLit(_) => "Set",
                _ => "Multiset",
            };
            write!(f, "{name}(")?;
            for (i, e) in elems.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", show.with(e))?;
            }
            write!(f, ")")
        }
        PureExpKind::SeqIndex(base, op) => {
            let s = show.with(base);
            match op {
                crate::viper::typed::SeqIndexOp::Index(i) => write!(f, "{s}[{}]", show.with(i)),
                crate::viper::typed::SeqIndexOp::Take(n) => write!(f, "{s}[..{}]", show.with(n)),
                crate::viper::typed::SeqIndexOp::Drop(n) => write!(f, "{s}[{}..]", show.with(n)),
                crate::viper::typed::SeqIndexOp::Slice(a, b) => {
                    write!(f, "{s}[{}..{}]", show.with(a), show.with(b))
                }
                crate::viper::typed::SeqIndexOp::Update(i, v) => {
                    write!(f, "{s}[{} := {}]", show.with(i), show.with(v))
                }
            }
        }
        PureExpKind::MapLit(entries) => {
            write!(f, "Map(")?;
            for (i, (k, v)) in entries.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{} := {}", show.with(k), show.with(v))?;
            }
            write!(f, ")")
        }
        PureExpKind::MapIndex(base, op) => {
            let m = show.with(base);
            match op {
                crate::viper::typed::MapIndexOp::Lookup(k) => write!(f, "{m}[{}]", show.with(k)),
                crate::viper::typed::MapIndexOp::Update(k, v) => {
                    write!(f, "{m}[{} := {}]", show.with(k), show.with(v))
                }
            }
        }
        PureExpKind::MapDomain(m) => write!(f, "domain({})", show.with(m)),
        PureExpKind::MapRange(m) => write!(f, "range({})", show.with(m)),
        PureExpKind::Ext(ext) => ext.fmt_ext(f, show.interner),
    }
}

impl<'a> Display for Show<'a, &'a Literal> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.item {
            Literal::Bool(b) => write!(f, "{b}"),
            Literal::Int(i) => write!(f, "{i}"),
            Literal::Real(r) => write!(f, "{r}"),
            Literal::Null => write!(f, "null"),
            Literal::Wildcard => write!(f, "wildcard"),
        }
    }
}

impl<'a> Display for Show<'a, &'a BinOp> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let op = match self.item {
            BinOp::Or => "||",
            BinOp::And => "&&",
            BinOp::Implies => "==>",
            BinOp::Iff => "<==>",
            BinOp::Eq => "==",
            BinOp::Neq => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Plus => "+",
            BinOp::Minus => "-",
            BinOp::Mult => "*",
            BinOp::Div => "/",
            BinOp::IntDiv => "\\",
            BinOp::Mod => "%",
            BinOp::In => "in",
            BinOp::Union => "union",
            BinOp::SetMinus => "setminus",
            BinOp::Intersection => "intersection",
            BinOp::Subset => "subset",
            BinOp::Concat => "++",
            BinOp::Range => "..",
        };
        write!(f, "{op}")
    }
}

impl<'a> Display for Show<'a, &'a UnOp> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let op = match self.item {
            UnOp::Not => "!",
            UnOp::Neg => "-",
            UnOp::Cardinality => "|.|",
        };
        write!(f, "{op}")
    }
}
