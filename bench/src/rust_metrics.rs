//! Per-function shape metrics of a Rust source file, from a `syn` parse.
//!
//! These are the "program shape" side of Part B of the plan: what a function
//! looks like (size, control flow, pattern matching, signature, calls,
//! mutation, the checks each operation adds, types), to be related to what
//! verifying its Prusti encoding costs.
//!
//! Everything is syntactic and file-local. Types are resolved only against the
//! structs and enums declared in the same file, which is what the benchmark
//! corpora use; a type the file does not declare counts as one opaque leaf.
//! Macro bodies are token streams to `syn` and are not looked into.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use proc_macro2::Span;
use serde::Serialize;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{BinOp, Expr, Fields, Item, Pat, Stmt, Type, UnOp};

/// Metrics of one function. Field names are the keys in the run JSON; a
/// `max_` prefix marks a maximum (aggregated by `max` over a file), anything
/// else a count (aggregated by sum).
#[derive(Debug, Default, Clone, Serialize, PartialEq)]
pub struct FnMetrics {
    // ── Size ──
    /// Lines in the function that are neither blank nor `//` comments.
    pub loc: u64,
    pub stmts: u64,
    pub exprs: u64,

    // ── Control flow ──
    pub loops: u64,
    pub max_loop_depth: u64,
    pub ifs: u64,
    pub max_if_depth: u64,
    /// Most `if`/`match` statements side by side in one block. These multiply
    /// paths (2ⁿ for n two-way branches), where nesting only adds them.
    pub max_seq_branches: u64,
    /// Deepest nesting of `if` and `match` together.
    pub max_branch_depth: u64,
    /// McCabe: 1 + decision points (`if`, loops, extra match arms, `&&`,
    /// `||`, `?`).
    pub cyclomatic: u64,
    /// Estimated number of acyclic paths through the body (loops taken at
    /// most once). A float: it is exponential in sequential branches.
    pub paths: f64,
    /// Explicit `return`s and `?` operators.
    pub early_returns: u64,

    // ── Pattern matching ──
    pub matches: u64,
    pub match_arms: u64,
    pub max_match_arms: u64,
    pub max_match_depth: u64,
    /// Variants of the largest file-local enum matched on.
    pub max_enum_variants: u64,
    /// Payload nesting depth of the file-local enums matched on.
    pub max_payload_depth: u64,

    // ── Signature ──
    pub args: u64,
    pub args_by_value: u64,
    pub args_by_ref: u64,
    pub args_by_mut_ref: u64,
    /// Leaf fields of the argument types once nested structs are flattened.
    pub arg_fields: u64,
    /// Leaf fields of the return type (0 for `()`).
    pub ret_fields: u64,

    // ── Calls ──
    /// Function and method calls (macros excluded).
    pub calls: u64,
    pub distinct_callees: u64,
    /// Longest chain of calls into functions of the same file.
    pub max_call_depth: u64,
    pub calls_in_loops: u64,
    pub calls_in_branches: u64,

    // ── Mutation ──
    pub assigns: u64,
    pub compound_assigns: u64,
    /// Assignments whose place goes through a `&mut` binding or a deref.
    pub mut_ref_writes: u64,
    /// `&mut` borrows of a place already behind a `&mut`, explicit
    /// (`&mut *r`, `&mut r.f`) or implicit (passing `r` on to a call).
    pub reborrows: u64,
    /// Longest place written to, in components (`a.b.c.d = …` is 4).
    pub max_write_path: u64,
    /// References stored into a struct literal's field.
    pub struct_borrows: u64,

    // ── Checks each operation adds ──
    /// `+ - *` (and their compound forms, and unary `-`): overflow checks.
    pub arith_ops: u64,
    /// `/ %`: divide-by-zero (and overflow) checks.
    pub divisions: u64,
    /// `a[i]`: bounds checks.
    pub indexing: u64,
    pub casts: u64,
    /// `.unwrap()` and `.expect(..)`.
    pub unwraps: u64,
    /// `panic!`, `unreachable!`, `assert*!`, `todo!`, `unimplemented!`.
    pub panics: u64,

    // ── Types ──
    /// Deepest struct/enum nesting among the signature's types.
    pub max_struct_depth: u64,
    /// File-local enums mentioned anywhere in the function.
    pub enums_touched: u64,
    /// Generic arguments written out (`Vec<T>`, `f::<T>()`).
    pub generic_insts: u64,
}

/// Metrics of one file: its functions by qualified name (`f`, `Type::f`,
/// `module::f`), plus file-level sizes and the per-function metrics
/// aggregated (sum, or max for `max_` keys).
#[derive(Debug, Clone, Serialize)]
pub struct FileMetrics {
    pub loc: u64,
    pub fns: u64,
    pub structs: u64,
    pub enums: u64,
    pub totals: serde_json::Map<String, serde_json::Value>,
    pub functions: BTreeMap<String, FnMetrics>,
}

pub fn file_metrics(source: &str) -> Result<FileMetrics, String> {
    let file = syn::parse_file(source).map_err(|e| {
        let at = e.span().start();
        format!("{}:{}: {e}", at.line, at.column)
    })?;
    let lines: Vec<&str> = source.lines().collect();

    let mut types = TypeTable::default();
    let mut fns: Vec<(String, &syn::Signature, &syn::Block, Span)> = Vec::new();
    collect_items(&file.items, "", &mut types, &mut fns);

    let local_fns: HashSet<String> = fns
        .iter()
        .map(|(q, ..)| last_segment(q).to_string())
        .collect();
    let mut functions = BTreeMap::new();
    let mut callees_of: HashMap<String, BTreeSet<String>> = HashMap::new();
    for (qualified, sig, block, span) in &fns {
        let (mut m, callees) = fn_metrics(sig, block, &types);
        m.loc = code_lines(&lines, span.start().line, span.end().line);
        callees_of.insert(
            qualified.clone(),
            callees
                .into_iter()
                .filter(|c| local_fns.contains(c))
                .collect(),
        );
        functions.insert(qualified.clone(), m);
    }

    // Call-chain depth over the file-local call graph, by plain name (a call
    // `x.f()` or `T::f()` cannot be resolved further syntactically).
    let by_plain: HashMap<&str, Vec<&String>> =
        fns.iter().fold(HashMap::new(), |mut acc, (q, ..)| {
            acc.entry(last_segment(q)).or_default().push(q);
            acc
        });
    let mut memo = HashMap::new();
    for q in callees_of.keys() {
        let depth = call_depth(q, &callees_of, &by_plain, &mut memo, &mut HashSet::new());
        functions.get_mut(q).unwrap().max_call_depth = depth;
    }

    let mut totals = serde_json::Map::new();
    for m in functions.values() {
        let serde_json::Value::Object(obj) = serde_json::to_value(m).unwrap() else {
            unreachable!()
        };
        for (k, v) in obj {
            let v = v.as_f64().unwrap_or(0.0);
            let slot = totals.entry(k.clone()).or_insert(serde_json::json!(0));
            let cur = slot.as_f64().unwrap_or(0.0);
            let new = if k.starts_with("max_") {
                cur.max(v)
            } else {
                cur + v
            };
            *slot = if new.fract() == 0.0 && new < 9.0e15 {
                serde_json::json!(new as u64)
            } else {
                serde_json::json!(new)
            };
        }
    }

    Ok(FileMetrics {
        loc: code_lines(&lines, 1, lines.len()),
        fns: functions.len() as u64,
        structs: types.structs.len() as u64,
        enums: types.enums.len() as u64,
        totals,
        functions,
    })
}

fn last_segment(q: &str) -> &str {
    q.rsplit("::").next().unwrap_or(q)
}

fn call_depth(
    f: &String,
    callees_of: &HashMap<String, BTreeSet<String>>,
    by_plain: &HashMap<&str, Vec<&String>>,
    memo: &mut HashMap<String, u64>,
    on_stack: &mut HashSet<String>,
) -> u64 {
    if let Some(&d) = memo.get(f) {
        return d;
    }
    if !on_stack.insert(f.clone()) {
        return 0; // recursion: count the cycle once
    }
    let mut depth = 0;
    for callee in callees_of.get(f).into_iter().flatten() {
        for target in by_plain.get(callee.as_str()).into_iter().flatten() {
            depth = depth.max(1 + call_depth(target, callees_of, by_plain, memo, on_stack));
        }
    }
    on_stack.remove(f);
    memo.insert(f.clone(), depth);
    depth
}

/// Lines `from..=to` (1-based) that are neither blank nor `//` comments.
fn code_lines(lines: &[&str], from: usize, to: usize) -> u64 {
    lines
        .iter()
        .skip(from.saturating_sub(1))
        .take(to + 1 - from.max(1))
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with("//"))
        .count() as u64
}

fn collect_items<'a>(
    items: &'a [Item],
    prefix: &str,
    types: &mut TypeTable,
    fns: &mut Vec<(String, &'a syn::Signature, &'a syn::Block, Span)>,
) {
    for item in items {
        match item {
            Item::Fn(f) => fns.push((
                format!("{prefix}{}", f.sig.ident),
                &f.sig,
                &f.block,
                f.span(),
            )),
            Item::Impl(imp) => {
                let self_name = match &*imp.self_ty {
                    Type::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
                    _ => None,
                }
                .unwrap_or_else(|| "impl".into());
                for it in &imp.items {
                    if let syn::ImplItem::Fn(f) = it {
                        fns.push((
                            format!("{prefix}{self_name}::{}", f.sig.ident),
                            &f.sig,
                            &f.block,
                            f.span(),
                        ));
                    }
                }
            }
            Item::Trait(t) => {
                for it in &t.items {
                    if let syn::TraitItem::Fn(f) = it {
                        if let Some(block) = &f.default {
                            fns.push((
                                format!("{prefix}{}::{}", t.ident, f.sig.ident),
                                &f.sig,
                                block,
                                f.span(),
                            ));
                        }
                    }
                }
            }
            Item::Mod(m) => {
                if let Some((_, inner)) = &m.content {
                    collect_items(inner, &format!("{prefix}{}::", m.ident), types, fns);
                }
            }
            Item::Struct(s) => {
                types
                    .structs
                    .insert(s.ident.to_string(), field_types(&s.fields));
            }
            Item::Enum(e) => {
                types.enums.insert(
                    e.ident.to_string(),
                    e.variants.iter().map(|v| field_types(&v.fields)).collect(),
                );
            }
            _ => {}
        }
    }
}

fn field_types(fields: &Fields) -> Vec<Type> {
    fields.iter().map(|f| f.ty.clone()).collect()
}

/// The structs and enums a file declares: a struct's field types, and each
/// enum variant's payload types.
#[derive(Default)]
struct TypeTable {
    structs: HashMap<String, Vec<Type>>,
    enums: HashMap<String, Vec<Vec<Type>>>,
}

impl TypeTable {
    fn local_name(ty: &Type) -> Option<String> {
        match ty {
            Type::Path(p) if p.qself.is_none() => {
                p.path.segments.last().map(|s| s.ident.to_string())
            }
            _ => None,
        }
    }

    /// Leaf fields of `ty` once nested structs are flattened. An enum counts
    /// its discriminant plus its largest variant.
    fn flat_fields(&self, ty: &Type, seen: &mut Vec<String>) -> u64 {
        match ty {
            Type::Reference(r) => self.flat_fields(&r.elem, seen),
            Type::Paren(p) => self.flat_fields(&p.elem, seen),
            Type::Group(g) => self.flat_fields(&g.elem, seen),
            Type::Tuple(t) => t.elems.iter().map(|e| self.flat_fields(e, seen)).sum(),
            Type::Array(a) => {
                let n = match &a.len {
                    Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Int(i),
                        ..
                    }) => i.base10_parse().unwrap_or(1),
                    _ => 1,
                };
                n * self.flat_fields(&a.elem, seen)
            }
            _ => {
                let Some(name) = Self::local_name(ty) else {
                    return 1;
                };
                if seen.contains(&name) {
                    return 1;
                }
                seen.push(name.clone());
                let n = if let Some(fields) = self.structs.get(&name) {
                    fields.iter().map(|f| self.flat_fields(f, seen)).sum()
                } else if let Some(variants) = self.enums.get(&name) {
                    1 + variants
                        .iter()
                        .map(|fs| fs.iter().map(|f| self.flat_fields(f, seen)).sum::<u64>())
                        .max()
                        .unwrap_or(0)
                } else {
                    1
                };
                seen.pop();
                n
            }
        }
    }

    /// Nesting depth of `ty`: 0 for anything not declared in the file, one
    /// more than the deepest field for a struct or enum.
    fn depth(&self, ty: &Type, seen: &mut Vec<String>) -> u64 {
        match ty {
            Type::Reference(r) => self.depth(&r.elem, seen),
            Type::Paren(p) => self.depth(&p.elem, seen),
            Type::Group(g) => self.depth(&g.elem, seen),
            Type::Tuple(t) => t
                .elems
                .iter()
                .map(|e| self.depth(e, seen))
                .max()
                .unwrap_or(0),
            Type::Array(a) => self.depth(&a.elem, seen),
            _ => {
                let Some(name) = Self::local_name(ty) else {
                    return 0;
                };
                if seen.contains(&name) {
                    return 0;
                }
                let fields: Vec<&Type> = if let Some(fields) = self.structs.get(&name) {
                    fields.iter().collect()
                } else if let Some(variants) = self.enums.get(&name) {
                    variants.iter().flatten().collect()
                } else {
                    return 0;
                };
                seen.push(name);
                let d = 1 + fields
                    .iter()
                    .map(|f| self.depth(f, seen))
                    .max()
                    .unwrap_or(0);
                seen.pop();
                d
            }
        }
    }

    /// Payload depth of enum `name`: how deeply its variants' payloads nest.
    fn payload_depth(&self, name: &str) -> u64 {
        let Some(variants) = self.enums.get(name) else {
            return 0;
        };
        let mut seen = vec![name.to_string()];
        variants
            .iter()
            .flatten()
            .map(|t| self.depth(t, &mut seen))
            .max()
            .unwrap_or(0)
    }
}

fn is_mut_ref(ty: &Type) -> bool {
    matches!(ty, Type::Reference(r) if r.mutability.is_some())
}

fn fn_metrics(
    sig: &syn::Signature,
    block: &syn::Block,
    types: &TypeTable,
) -> (FnMetrics, BTreeSet<String>) {
    let mut m = FnMetrics::default();
    let mut mut_bindings = HashSet::new();

    for input in &sig.inputs {
        m.args += 1;
        let ty: &Type = match input {
            syn::FnArg::Receiver(r) => &r.ty,
            syn::FnArg::Typed(t) => {
                if is_mut_ref(&t.ty) {
                    if let Pat::Ident(id) = &*t.pat {
                        mut_bindings.insert(id.ident.to_string());
                    }
                }
                &t.ty
            }
        };
        match ty {
            Type::Reference(r) if r.mutability.is_some() => m.args_by_mut_ref += 1,
            Type::Reference(_) => m.args_by_ref += 1,
            _ => m.args_by_value += 1,
        }
        if let syn::FnArg::Receiver(r) = input {
            if r.reference.is_some() && r.mutability.is_some() {
                mut_bindings.insert("self".into());
            }
        }
        m.arg_fields += types.flat_fields(ty, &mut Vec::new());
        m.max_struct_depth = m.max_struct_depth.max(types.depth(ty, &mut Vec::new()));
    }
    if let syn::ReturnType::Type(_, ty) = &sig.output {
        m.ret_fields = types.flat_fields(ty, &mut Vec::new());
        m.max_struct_depth = m.max_struct_depth.max(types.depth(ty, &mut Vec::new()));
    }

    let mut v = BodyVisitor {
        m,
        types,
        mut_bindings,
        callees: BTreeSet::new(),
        enums: BTreeSet::new(),
        loop_depth: 0,
        if_depth: 0,
        match_depth: 0,
        branch_depth: 0,
    };
    v.visit_signature(sig);
    v.visit_block(block);
    v.m.paths = paths_block(block);
    v.m.cyclomatic += 1;
    v.m.distinct_callees = v.callees.len() as u64;
    v.m.enums_touched = v.enums.len() as u64;
    (v.m, v.callees)
}

struct BodyVisitor<'t> {
    m: FnMetrics,
    types: &'t TypeTable,
    /// Bindings of `&mut` type: parameters, and `let`s initialised with a
    /// `&mut` borrow or annotated with a `&mut` type.
    mut_bindings: HashSet<String>,
    callees: BTreeSet<String>,
    enums: BTreeSet<String>,
    loop_depth: u64,
    if_depth: u64,
    match_depth: u64,
    branch_depth: u64,
}

/// The components of a place expression (`a.b[i].c` → root `a`, 4
/// components), and whether it passes through a deref.
fn place(expr: &Expr) -> Option<(String, u64, bool)> {
    match expr {
        Expr::Path(p) => p.path.get_ident().map(|i| (i.to_string(), 1, false)),
        Expr::Field(f) => place(&f.base).map(|(r, n, d)| (r, n + 1, d)),
        Expr::Index(i) => place(&i.expr).map(|(r, n, d)| (r, n + 1, d)),
        Expr::Paren(p) => place(&p.expr),
        Expr::Unary(u) if matches!(u.op, UnOp::Deref(_)) => {
            place(&u.expr).map(|(r, n, _)| (r, n, true))
        }
        _ => None,
    }
}

impl BodyVisitor<'_> {
    fn is_mut_place(&self, expr: &Expr) -> bool {
        place(expr).is_some_and(|(root, _, deref)| deref || self.mut_bindings.contains(&root))
    }

    fn write(&mut self, lhs: &Expr) {
        if let Some((_, len, _)) = place(lhs) {
            self.m.max_write_path = self.m.max_write_path.max(len);
        }
        if self.is_mut_place(lhs) {
            self.m.mut_ref_writes += 1;
        }
    }

    fn call(&mut self, name: String, args: &syn::punctuated::Punctuated<Expr, syn::Token![,]>) {
        self.m.calls += 1;
        self.callees.insert(name);
        if self.loop_depth > 0 {
            self.m.calls_in_loops += 1;
        }
        if self.branch_depth > 0 {
            self.m.calls_in_branches += 1;
        }
        // Passing a `&mut` binding on is an implicit reborrow.
        for arg in args {
            if let Expr::Path(p) = arg {
                if p.path
                    .get_ident()
                    .is_some_and(|i| self.mut_bindings.contains(&i.to_string()))
                {
                    self.m.reborrows += 1;
                }
            }
        }
    }

    fn note_enum_path(&mut self, path: &syn::Path) {
        for seg in &path.segments {
            let name = seg.ident.to_string();
            if self.types.enums.contains_key(&name) {
                self.enums.insert(name);
            }
        }
    }

    fn block_seq_branches(&mut self, block: &syn::Block) {
        let n = block
            .stmts
            .iter()
            .filter(|s| {
                let e = match s {
                    Stmt::Expr(e, _) => Some(e),
                    Stmt::Local(l) => l.init.as_ref().map(|i| &*i.expr),
                    _ => None,
                };
                matches!(e, Some(Expr::If(_) | Expr::Match(_)))
            })
            .count() as u64;
        self.m.max_seq_branches = self.m.max_seq_branches.max(n);
    }
}

/// The enum a match arm's pattern names, if it is written `Enum::Variant..`.
fn pattern_enum(pat: &Pat) -> Option<String> {
    let path = match pat {
        Pat::TupleStruct(p) => &p.path,
        Pat::Struct(p) => &p.path,
        Pat::Path(p) => &p.path,
        Pat::Ident(p) => return p.subpat.as_ref().and_then(|(_, s)| pattern_enum(s)),
        Pat::Or(o) => return o.cases.iter().find_map(pattern_enum),
        Pat::Reference(r) => return pattern_enum(&r.pat),
        Pat::Paren(p) => return pattern_enum(&p.pat),
        _ => return None,
    };
    let n = path.segments.len();
    (n >= 2).then(|| path.segments[n - 2].ident.to_string())
}

impl<'ast> Visit<'ast> for BodyVisitor<'_> {
    fn visit_stmt(&mut self, s: &'ast Stmt) {
        self.m.stmts += 1;
        if let Stmt::Local(l) = s {
            let binds_mut = match (&l.pat, &l.init) {
                (Pat::Type(t), _) if is_mut_ref(&t.ty) => true,
                (_, Some(init)) => {
                    matches!(&*init.expr, Expr::Reference(r) if r.mutability.is_some())
                }
                _ => false,
            };
            if binds_mut {
                let ident = match &l.pat {
                    Pat::Ident(i) => Some(&i.ident),
                    Pat::Type(t) => match &*t.pat {
                        Pat::Ident(i) => Some(&i.ident),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some(i) = ident {
                    self.mut_bindings.insert(i.to_string());
                }
            }
        }
        visit::visit_stmt(self, s);
    }

    fn visit_block(&mut self, b: &'ast syn::Block) {
        self.block_seq_branches(b);
        visit::visit_block(self, b);
    }

    fn visit_expr(&mut self, e: &'ast Expr) {
        self.m.exprs += 1;
        match e {
            Expr::If(i) => {
                self.m.ifs += 1;
                self.m.cyclomatic += 1;
                self.visit_expr(&i.cond);
                self.if_depth += 1;
                self.branch_depth += 1;
                self.m.max_if_depth = self.m.max_if_depth.max(self.if_depth);
                self.m.max_branch_depth = self.m.max_branch_depth.max(self.branch_depth);
                self.visit_block(&i.then_branch);
                if let Some((_, els)) = &i.else_branch {
                    // `else if` is a chain, not a level of nesting.
                    if matches!(&**els, Expr::If(_)) {
                        self.if_depth -= 1;
                        self.branch_depth -= 1;
                        self.visit_expr(els);
                        self.if_depth += 1;
                        self.branch_depth += 1;
                    } else {
                        self.visit_expr(els);
                    }
                }
                self.if_depth -= 1;
                self.branch_depth -= 1;
                return;
            }
            Expr::Match(mt) => {
                self.m.matches += 1;
                let arms = mt.arms.len() as u64;
                self.m.match_arms += arms;
                self.m.max_match_arms = self.m.max_match_arms.max(arms);
                self.m.cyclomatic += arms.saturating_sub(1);
                if let Some(en) = mt.arms.iter().find_map(|a| pattern_enum(&a.pat)) {
                    if let Some(variants) = self.types.enums.get(&en) {
                        self.m.max_enum_variants =
                            self.m.max_enum_variants.max(variants.len() as u64);
                        self.m.max_payload_depth =
                            self.m.max_payload_depth.max(self.types.payload_depth(&en));
                    }
                }
                self.visit_expr(&mt.expr);
                self.match_depth += 1;
                self.branch_depth += 1;
                self.m.max_match_depth = self.m.max_match_depth.max(self.match_depth);
                self.m.max_branch_depth = self.m.max_branch_depth.max(self.branch_depth);
                for arm in &mt.arms {
                    self.visit_arm(arm);
                }
                self.match_depth -= 1;
                self.branch_depth -= 1;
                return;
            }
            Expr::While(_) | Expr::ForLoop(_) | Expr::Loop(_) => {
                self.m.loops += 1;
                self.m.cyclomatic += 1;
                self.loop_depth += 1;
                self.m.max_loop_depth = self.m.max_loop_depth.max(self.loop_depth);
                visit::visit_expr(self, e);
                self.loop_depth -= 1;
                return;
            }
            Expr::Binary(b) => match b.op {
                BinOp::Add(_) | BinOp::Sub(_) | BinOp::Mul(_) => self.m.arith_ops += 1,
                BinOp::Div(_) | BinOp::Rem(_) => self.m.divisions += 1,
                BinOp::And(_) | BinOp::Or(_) => self.m.cyclomatic += 1,
                BinOp::AddAssign(_) | BinOp::SubAssign(_) | BinOp::MulAssign(_) => {
                    self.m.arith_ops += 1;
                    self.m.compound_assigns += 1;
                    self.write(&b.left);
                }
                BinOp::DivAssign(_) | BinOp::RemAssign(_) => {
                    self.m.divisions += 1;
                    self.m.compound_assigns += 1;
                    self.write(&b.left);
                }
                BinOp::BitAndAssign(_)
                | BinOp::BitOrAssign(_)
                | BinOp::BitXorAssign(_)
                | BinOp::ShlAssign(_)
                | BinOp::ShrAssign(_) => {
                    self.m.compound_assigns += 1;
                    self.write(&b.left);
                }
                _ => {}
            },
            Expr::Unary(u) if matches!(u.op, UnOp::Neg(_)) => self.m.arith_ops += 1,
            Expr::Assign(a) => {
                self.m.assigns += 1;
                self.write(&a.left);
            }
            Expr::Index(_) => self.m.indexing += 1,
            Expr::Cast(_) => self.m.casts += 1,
            Expr::Return(_) => self.m.early_returns += 1,
            Expr::Try(_) => {
                self.m.early_returns += 1;
                self.m.cyclomatic += 1;
            }
            Expr::Reference(r) if r.mutability.is_some() => {
                if self.is_mut_place(&r.expr) {
                    self.m.reborrows += 1;
                }
            }
            Expr::Struct(s) => {
                self.m.struct_borrows += s
                    .fields
                    .iter()
                    .filter(|f| matches!(f.expr, Expr::Reference(_)))
                    .count() as u64;
            }
            Expr::Call(c) => {
                let name = match &*c.func {
                    Expr::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
                    _ => None,
                }
                .unwrap_or_else(|| "<expr>".into());
                self.call(name, &c.args);
            }
            Expr::MethodCall(mc) => {
                let name = mc.method.to_string();
                if name == "unwrap" || name == "expect" {
                    self.m.unwraps += 1;
                }
                self.call(name, &mc.args);
            }
            _ => {}
        }
        visit::visit_expr(self, e);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        let name = mac
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        if matches!(
            name.as_str(),
            "panic"
                | "unreachable"
                | "todo"
                | "unimplemented"
                | "assert"
                | "assert_eq"
                | "assert_ne"
        ) {
            self.m.panics += 1;
        }
        visit::visit_macro(self, mac);
    }

    fn visit_path(&mut self, p: &'ast syn::Path) {
        self.note_enum_path(p);
        for seg in &p.segments {
            if matches!(seg.arguments, syn::PathArguments::AngleBracketed(_)) {
                self.m.generic_insts += 1;
            }
        }
        visit::visit_path(self, p);
    }

    fn visit_expr_method_call(&mut self, mc: &'ast syn::ExprMethodCall) {
        if mc.turbofish.is_some() {
            self.m.generic_insts += 1;
        }
        visit::visit_expr_method_call(self, mc);
    }
}

// ── Path counting ──
//
// The number of acyclic paths, loops entered at most once: a sequence
// multiplies, a branch adds. Subexpressions evaluated in sequence (operands,
// arguments) multiply as well, so a `match` in an argument position counts.

fn paths_block(b: &syn::Block) -> f64 {
    b.stmts.iter().map(paths_stmt).product()
}

fn paths_stmt(s: &Stmt) -> f64 {
    match s {
        Stmt::Local(l) => l.init.as_ref().map_or(1.0, |i| {
            paths_expr(&i.expr) * i.diverge.as_ref().map_or(1.0, |(_, d)| 1.0 + paths_expr(d))
        }),
        Stmt::Expr(e, _) => paths_expr(e),
        Stmt::Item(_) | Stmt::Macro(_) => 1.0,
    }
}

fn paths_all<'a>(es: impl IntoIterator<Item = &'a Expr>) -> f64 {
    es.into_iter().map(paths_expr).product()
}

fn paths_expr(e: &Expr) -> f64 {
    match e {
        Expr::If(i) => {
            let els = i.else_branch.as_ref().map_or(1.0, |(_, e)| paths_expr(e));
            paths_expr(&i.cond) * (paths_block(&i.then_branch) + els)
        }
        Expr::Match(m) => {
            paths_expr(&m.expr)
                * m.arms
                    .iter()
                    .map(|a| paths_expr(&a.body))
                    .sum::<f64>()
                    .max(1.0)
        }
        Expr::While(w) => paths_expr(&w.cond) * (1.0 + paths_block(&w.body)),
        Expr::ForLoop(f) => paths_expr(&f.expr) * (1.0 + paths_block(&f.body)),
        Expr::Loop(l) => paths_block(&l.body),
        Expr::Block(b) => paths_block(&b.block),
        Expr::Unsafe(b) => paths_block(&b.block),
        // `a && b` is not counted as a branch of its own: as an `if` condition
        // it only splits the paths the `if` already has (cyclomatic counts it).
        Expr::Binary(b) => paths_expr(&b.left) * paths_expr(&b.right),
        Expr::Unary(u) => paths_expr(&u.expr),
        Expr::Paren(p) => paths_expr(&p.expr),
        Expr::Group(g) => paths_expr(&g.expr),
        Expr::Reference(r) => paths_expr(&r.expr),
        Expr::Cast(c) => paths_expr(&c.expr),
        Expr::Field(f) => paths_expr(&f.base),
        Expr::Index(i) => paths_expr(&i.expr) * paths_expr(&i.index),
        Expr::Assign(a) => paths_expr(&a.left) * paths_expr(&a.right),
        Expr::Call(c) => paths_expr(&c.func) * paths_all(&c.args),
        Expr::MethodCall(m) => paths_expr(&m.receiver) * paths_all(&m.args),
        Expr::Tuple(t) => paths_all(&t.elems),
        Expr::Array(a) => paths_all(&a.elems),
        Expr::Struct(s) => paths_all(s.fields.iter().map(|f| &f.expr)),
        Expr::Return(r) => r.expr.as_deref().map_or(1.0, paths_expr),
        Expr::Let(l) => paths_expr(&l.expr),
        // `x?` returns early or continues: one more path.
        Expr::Try(t) => paths_expr(&t.expr) + 1.0,
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = r#"
pub struct Cell { pub v: i32, pub w: i32 }
pub struct Grid { pub a: Cell, pub b: Cell }
pub enum Shape { Dot, Box(Cell), Pair(Grid, i32) }

pub fn cell_sum(c: &Cell) -> i32 {
    // a comment
    c.v + c.w
}

pub fn step(g: &mut Grid, k: i32, s: &Shape) -> i32 {
    let mut acc = cell_sum(&g.a);
    if k > 0 {
        acc = acc * 2;
        if k > 10 { acc -= 1; }
    } else {
        g.b.v = k / 3;
    }
    if acc < 0 && k != 0 { acc = -acc; }
    match s {
        Shape::Dot => {}
        Shape::Box(c) => acc += c.v,
        Shape::Pair(p, n) => acc += cell_sum(&p.a) + n,
    }
    grow(g);
    acc
}

pub fn grow(g: &mut Grid) {
    let r = &mut g.a;
    r.v += 1;
    for i in 0..3 { r.w = r.w + i; }
}

impl Grid {
    pub fn total(&self) -> i32 { cell_sum(&self.a) + cell_sum(&self.b) }
}
"#;

    #[test]
    fn metrics_of_a_small_file() {
        let fm = file_metrics(SRC).unwrap();
        assert_eq!(fm.fns, 4);
        assert_eq!((fm.structs, fm.enums), (2, 1));
        assert!(fm.functions.contains_key("Grid::total"));

        let cs = &fm.functions["cell_sum"];
        assert_eq!(cs.loc, 3);
        assert_eq!(
            (cs.args, cs.args_by_ref, cs.arg_fields, cs.ret_fields),
            (1, 1, 2, 1)
        );
        assert_eq!(cs.arith_ops, 1);
        assert_eq!(cs.max_call_depth, 0);

        let s = &fm.functions["step"];
        assert_eq!(
            (s.args, s.args_by_mut_ref, s.args_by_value, s.args_by_ref),
            (3, 1, 1, 1)
        );
        // Grid = 2 Cells of 2 fields; Shape = discriminant + Pair(Grid, i32).
        assert_eq!(s.arg_fields, 4 + 1 + (1 + 5));
        assert_eq!((s.ifs, s.max_if_depth), (3, 2));
        assert_eq!(s.max_seq_branches, 3);
        assert_eq!(
            (s.matches, s.match_arms, s.max_enum_variants),
            (1, 3, 1 + 2)
        );
        assert_eq!(s.max_payload_depth, 2);
        assert_eq!(s.max_branch_depth, 2);
        // if/else with a nested if: 2 + 1 = 3; the second if: 2; match: 3.
        assert_eq!(s.paths, 3.0 * 2.0 * 3.0);
        // 1 + 3 ifs + `&&` + 2 extra arms.
        assert_eq!(s.cyclomatic, 1 + 3 + 1 + 2);
        assert_eq!(s.divisions, 1);
        assert_eq!(s.mut_ref_writes, 1); // g.b.v = ...
        assert_eq!(s.max_write_path, 3);
        assert_eq!(s.reborrows, 1); // grow(g)
        assert_eq!(s.enums_touched, 1);
        assert_eq!(s.max_call_depth, 1); // step -> grow / cell_sum
        assert_eq!(s.calls_in_branches, 1);

        let g = &fm.functions["grow"];
        assert_eq!((g.loops, g.max_loop_depth), (1, 1));
        assert_eq!(g.reborrows, 1); // &mut g.a
        assert_eq!(g.mut_ref_writes, 2); // r.v += 1; r.w = ...
        assert_eq!(g.compound_assigns, 1);
        assert_eq!(g.paths, 2.0);

        assert_eq!(fm.totals["loops"], 1);
        assert_eq!(fm.totals["max_if_depth"], 2);
        assert_eq!(fm.totals["calls"], 3 + 2);
    }
}
