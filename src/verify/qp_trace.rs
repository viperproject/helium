//! Text trace of how quantifiers that touch the heap are handled: quantified
//! permissions (inhale, exhale, inverse axioms) and pure `forall`s whose body
//! reads the heap or calls a heap-dependent function.
//!
//! Enabled by `SILVER_OXIDE_TRACE_QP`. Its value selects the verification
//! units: empty, `1` or `all` traces every unit; otherwise it is a
//! comma-separated list of substrings, and a unit is traced when its name
//! contains one of them (`SILVER_OXIDE_TRACE_QP=foo,bar`). Disabled = one
//! cached env read per unit and nothing else.
//!
//! The trace goes to stderr, one block per event, every line prefixed with
//! `[qp]` so it can be grepped out of the other output. A block is written in
//! one `eprint!`, so other output interleaves whole blocks, never lines. Each instruction that produces an event gets a header
//! naming the unit and the instruction.
//!
//! The events, in the order the verifier meets them:
//!
//! - **resolve**: a body's heap reads and the footprint slots of its
//!   heap-dependent calls, each resolved to the one holder that frames it for
//!   every binding: a chunk set's value map (`Map`, read as `lookup(map, addr)`)
//!   or a single chunk's value (`Value`, captured as is).
//! - **recipe**: the compiled body in its canonical space: captures `c*`
//!   (the enclosing temps, then the read and call stand-ins), binders `x*`,
//!   steps `t*`. A recipe already printed in this unit is only named.
//! - **qp**: a quantified permission produced or consumed, its well-formedness
//!   check, its inverse axiom, and which held sets a consume took from.
//! - **instances**: the instances `rewrite::forall_rule` built of those
//!   recipes (heap ones only: a pure `forall` with heap reads or calls, and
//!   inverse axioms) while an instruction was evaluated, with their binders.
//!   The rule records them as small term trees ([`record_instance`]); the walk
//!   renders them after the instruction ([`flush_instances`]), where the
//!   context can name functions. An instance in a scratch or probe graph is
//!   marked `scratch`: it dies with that graph, and the next scratch graph may
//!   build it again.

use std::cell::RefCell;
use std::fmt::Write as _;
use std::sync::OnceLock;

use crate::dhash::HashSet;
use crate::verify::context::VerifyContext;
use crate::verify::heap::LocationKind;
use crate::verify::lang::{FuncId, RecipeId};
use crate::verify::quant::{Resolved, Stand};
use crate::verify::rewrite::{AxiomInst, AxiomPure, PreparedTerm, TermHead};
use crate::vmir::{self, Polarity, Val, display::VmirDisplay};

/// The unit filter from the env var: `None` = tracing off, `Some([])` = every
/// unit.
fn filter() -> Option<&'static [String]> {
    static FILTER: OnceLock<Option<Vec<String>>> = OnceLock::new();
    FILTER
        .get_or_init(|| {
            let v = std::env::var("SILVER_OXIDE_TRACE_QP").ok()?;
            let v = v.trim();
            if v.is_empty() || v == "1" || v == "all" {
                return Some(Vec::new());
            }
            Some(v.split(',').map(|s| s.trim().to_string()).collect())
        })
        .as_deref()
}

#[derive(Default)]
struct UnitState {
    /// Whether the current unit is traced.
    active: bool,
    unit: String,
    /// The current instruction's header, printed before its first event.
    pending_header: Option<String>,
    /// Recipes already printed in full in this unit.
    seen: HashSet<RecipeId>,
}

thread_local! {
    static STATE: RefCell<UnitState> = RefCell::default();
    /// The program's recipes that touch the heap, whose instances are traced.
    /// Program-wide (a recipe stated in one unit is instantiated in others),
    /// filled whether or not the stating unit is traced.
    static HEAP_RECIPES: RefCell<HashSet<RecipeId>> = RefCell::default();
    /// Instances built since the last [`flush_instances`].
    static INSTANCES: RefCell<Vec<Instance>> = const { RefCell::new(Vec::new()) };
}

/// Start verifying a program: recipe ids are per program.
pub(crate) fn begin_program() {
    HEAP_RECIPES.with(|r| r.borrow_mut().clear());
}

/// Start a verification unit (`kind` is `method`, `function` or `resource`).
pub(crate) fn begin_unit(kind: &str, name: &str) {
    let active = filter().is_some_and(|f| f.is_empty() || f.iter().any(|s| name.contains(s)));
    INSTANCES.with(|i| i.borrow_mut().clear());
    STATE.with(|s| {
        *s.borrow_mut() = UnitState {
            active,
            unit: format!("{kind} {name}"),
            ..UnitState::default()
        }
    });
}

/// Whether the current unit is traced. Callers check it before building
/// anything an event needs.
pub(crate) fn active() -> bool {
    STATE.with(|s| s.borrow().active)
}

/// The walk reached instruction `idx` of `n`; `text` renders it. The header
/// is printed only if the instruction produces an event.
pub(crate) fn at_inst(idx: usize, n: usize, text: impl FnOnce() -> String) {
    if !active() {
        return;
    }
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        // The first line only: a quantifier's body follows in the event.
        let text = text();
        let mut lines = text.lines();
        let first = lines.next().unwrap_or("");
        let more = if lines.next().is_some() { " …" } else { "" };
        let header = format!("── {} · inst {idx}/{n}: {first}{more}", s.unit);
        s.pending_header = Some(header);
    });
}

/// Write one event block: the pending instruction header (once), then `body`,
/// every line prefixed.
fn emit(body: &str) {
    let header = STATE.with(|s| s.borrow_mut().pending_header.take());
    let mut out = String::new();
    if let Some(h) = header {
        out.push_str("[qp]\n");
        let _ = writeln!(out, "[qp] {h}");
    }
    for line in body.lines() {
        let _ = writeln!(out, "[qp]   {line}");
    }
    eprint!("{out}");
}

/// Print `body` as an event if tracing is active. For one-line outcomes.
pub(crate) fn note(body: impl FnOnce() -> String) {
    if active() {
        emit(&body());
    }
}

fn vmir<'a, T>(ctx: &VerifyContext<'a>, item: T) -> VmirDisplay<'a, T> {
    VmirDisplay::new(item, ctx.decls, ctx.interner, ctx.groups)
}

fn kind_name(ctx: &VerifyContext<'_>, kind: &LocationKind) -> String {
    format!(
        "{}:{}",
        ctx.groups.resolve(&kind.group),
        ctx.type_name(&kind.value)
    )
}

/// An e-class as `@id = <one of its nodes>`, children as `@id`.
pub(crate) fn eclass(ctx: &VerifyContext<'_>, id: egg::Id) -> String {
    let c = ctx.egraph.find(id);
    match ctx.egraph[c].nodes.first() {
        Some(n) => format!("@{} = {}", usize::from(c), super::viz::render_node(ctx, n)),
        None => format!("@{}", usize::from(c)),
    }
}

fn stand(ctx: &VerifyContext<'_>, s: Stand) -> String {
    match s {
        Stand::Map(id) => format!("Map({})", eclass(ctx, id)),
        Stand::Value(id) => format!("Value({})", eclass(ctx, id)),
    }
}

/// The resolution of a body's reads and calls; `calls` are its VMIR calls, in
/// the order of `r`'s.
fn render_resolved(
    ctx: &VerifyContext<'_>,
    out: &mut String,
    calls: &[vmir::HeapCall],
    r: &Resolved,
) {
    let mut slot = 0;
    for (j, read) in r.reads.iter().enumerate() {
        let _ = writeln!(
            out,
            "read {j}: {}[{}] -> {}   ({})",
            read.heap,
            read.addr,
            stand(ctx, read.stand),
            kind_name(ctx, &read.kind)
        );
        slot += 1;
    }
    for (k, call) in r.calls.iter().enumerate() {
        let name = calls
            .get(k)
            .map_or_else(|| "?".to_string(), |c| ctx.member_name(c.call.resource));
        let args = calls.get(k).map_or(String::new(), |c| {
            let a: Vec<String> = c.call.args.iter().map(|v| v.to_string()).collect();
            a.join(", ")
        });
        let _ = writeln!(out, "call {k}: {name}({args}) -> e{}", call.temp);
        for (i, s) in call.slots.iter().enumerate() {
            let _ = writeln!(
                out,
                "    slot {i}: {} -> {}",
                kind_name(ctx, &s.kind),
                stand(ctx, s.stand)
            );
            slot += 1;
        }
    }
    if slot == 0 {
        out.push_str("(no heap reads or calls)\n");
    }
}

/// A pure `forall` reading the heap, resolved at this program point
/// (`check_forall_wd`).
pub(crate) fn forall_resolved(ctx: &VerifyContext<'_>, q: &vmir::Forall, r: &Resolved) {
    if !active() || r.is_empty() {
        return;
    }
    let mut out = String::from("pure forall:\n");
    indent_into(&mut out, &vmir(ctx, q).to_string());
    out.push_str("resolve:\n");
    let mut res = String::new();
    render_resolved(ctx, &mut res, &q.heap_calls(), r);
    indent_into(&mut out, &res);
    emit(&out);
}

/// A quantified permission about to be produced or consumed (`what`), with
/// its captures resolved.
pub(crate) fn qp_set(
    ctx: &VerifyContext<'_>,
    what: &str,
    q: &vmir::QuantChunks,
    caps: &crate::verify::heap::Captures,
    check: crate::verify::heap::QpCheck,
) {
    if !active() {
        return;
    }
    let mut out = format!("qp {what} (check: {check:?}):\n");
    indent_into(&mut out, &vmir(ctx, q).to_string());
    if !caps.temps.is_empty() {
        out.push_str("captures:\n");
        for &(k, id) in caps.temps.iter() {
            let _ = writeln!(out, "  e{k} -> {}", eclass(ctx, id));
        }
    }
    if !caps.reads.is_empty() {
        out.push_str("resolve:\n");
        let mut res = String::new();
        render_resolved(ctx, &mut res, &q.heap_calls(), &caps.reads);
        indent_into(&mut out, &res);
    }
    if let Some(w) = caps.wildcard {
        let _ = writeln!(out, "wildcard: {}", ctx.func_name(w));
    }
    emit(&out);
}

/// A recipe stated at this point as a `Symbolic::Forall` node with children
/// `caps` (`what` says what it is for).
pub(crate) fn recipe(ctx: &VerifyContext<'_>, what: &str, rid: RecipeId, caps: &[egg::Id]) {
    if filter().is_none() {
        return;
    }
    HEAP_RECIPES.with(|r| r.borrow_mut().insert(rid));
    if !active() {
        return;
    }
    let first = STATE.with(|s| s.borrow_mut().seen.insert(rid));
    let mut out = format!(
        "{what}: R{}{}\n",
        rid.0,
        if first { "" } else { " (printed above)" }
    );
    for (k, &id) in caps.iter().enumerate() {
        let _ = writeln!(out, "  c{k} := {}", eclass(ctx, id));
    }
    if first {
        let table = ctx.alloc.quant_table().read().expect("recipe table lock");
        indent_into(&mut out, &render_recipe(ctx, table.get(rid)));
    }
    emit(&out);
}

/// A recipe in its canonical space.
fn render_recipe(ctx: &VerifyContext<'_>, r: &crate::verify::quant::QuantRecipe) -> String {
    let n_caps = r.n_caps;
    let n_bound = r.bound.len();
    let name = |v: &Val| match v {
        Val::Literal(l) => l.to_string(),
        Val::Temp(k) if *k < n_caps => format!("c{k}"),
        Val::Temp(k) if *k < n_caps + n_bound => format!("x{}", k - n_caps),
        Val::Temp(k) => format!("t{}", k - n_caps - n_bound),
    };
    let names = |vs: &[Val]| vs.iter().map(&name).collect::<Vec<_>>().join(", ");
    let tys = |ts: &[vmir::Type]| {
        if ts.is_empty() {
            String::new()
        } else {
            let t: Vec<String> = ts.iter().map(|t| ctx.type_name(t)).collect();
            format!("<{}>", t.join(", "))
        }
    };
    let func = |f: FuncId, ts: &[vmir::Type]| format!("{}{}", ctx.func_name(f), tys(ts));

    let mut out = String::new();
    let binders: Vec<String> = r
        .bound
        .iter()
        .enumerate()
        .map(|(j, t)| format!("x{j}: {}", ctx.type_name(t)))
        .collect();
    let _ = writeln!(out, "forall {} ::", binders.join(", "));
    for group in &r.groups {
        let terms: Vec<String> = group.iter().map(|t| term(ctx, t)).collect();
        let _ = writeln!(out, "  trigger {{{}}}", terms.join(", "));
    }
    if r.guard.is_some() {
        out.push_str("  (guarded)\n");
    }
    let mut next = 0;
    for inst in &r.insts {
        match inst {
            AxiomInst::Val(p) => {
                let rhs = match p {
                    AxiomPure::Binary(op, a, b) => format!("{} {op} {}", name(a), name(b)),
                    AxiomPure::Ternary(c, a, b) => {
                        format!("{} ? {} : {}", name(c), name(a), name(b))
                    }
                    AxiomPure::RealCast(a) => format!("real({})", name(a)),
                    AxiomPure::App {
                        func: f,
                        type_args,
                        args,
                    } => format!("{}({})", func(*f, type_args), names(args)),
                    AxiomPure::Coll { tys: t, op, args } => {
                        format!("{op:?}{}({})", tys(t), names(args))
                    }
                    AxiomPure::CollLit {
                        kind,
                        elem_ty,
                        elem,
                    } => format!("{kind:?}<{}>{{{}}}", ctx.type_name(elem_ty), name(elem)),
                };
                let _ = writeln!(out, "  t{next} = {rhs}");
                next += 1;
            }
            AxiomInst::Forall {
                recipe,
                type_args,
                caps,
            } => {
                let _ = writeln!(
                    out,
                    "  t{next} = forall R{}{}({})",
                    recipe.0,
                    tys(type_args),
                    names(caps)
                );
                next += 1;
            }
            AxiomInst::Assume(v) => {
                let _ = writeln!(out, "  assume {}", name(v));
            }
            AxiomInst::Token {
                func: f,
                type_args,
                args,
                guards,
            } => {
                let g: Vec<String> = guards
                    .iter()
                    .map(|(v, p)| match p {
                        Polarity::Positive => name(v),
                        Polarity::Negative => format!("!{}", name(v)),
                    })
                    .collect();
                let when = if g.is_empty() {
                    String::new()
                } else {
                    format!("   if {}", g.join(" && "))
                };
                let _ = writeln!(
                    out,
                    "  token {}({}){when}",
                    func(*f, type_args),
                    names(args)
                );
            }
        }
    }
    let _ = writeln!(out, "  result: {}", name(&r.res));
    out
}

fn term(ctx: &VerifyContext<'_>, t: &PreparedTerm) -> String {
    match t {
        PreparedTerm::Bound(j) => format!("x{j}"),
        PreparedTerm::Capture(k) => format!("c{k}"),
        PreparedTerm::Lit(l) => l.to_string(),
        PreparedTerm::Wild => "_".to_string(),
        PreparedTerm::App { head, args } => {
            let args: Vec<String> = args.iter().map(|a| term(ctx, a)).collect();
            let head = match head {
                TermHead::Func { func, type_args } => {
                    if type_args.is_empty() {
                        ctx.func_name(*func)
                    } else {
                        let t: Vec<String> = type_args.iter().map(|t| ctx.type_name(t)).collect();
                        format!("{}<{}>", ctx.func_name(*func), t.join(", "))
                    }
                }
                TermHead::Coll(_, op) => format!("{op:?}"),
                TermHead::CollLit(kind, _) => format!("{kind:?}"),
            };
            format!("{head}({})", args.join(", "))
        }
    }
}

/// Append `text` to `out`, every line indented two spaces.
fn indent_into(out: &mut String, text: &str) {
    for line in text.lines() {
        let _ = writeln!(out, "  {line}");
    }
}

/// One instance built by `rewrite::forall_rule`.
struct Instance {
    rid: RecipeId,
    binders: Vec<(egg::Id, Term)>,
    generation: u32,
    scratch: bool,
}

/// An e-class captured as one of its nodes, with its children to a bounded
/// depth (`None` past it), so it can be rendered after the graph is gone.
struct Term {
    node: crate::verify::lang::Symbolic,
    kids: Option<Vec<Term>>,
}

/// How deep [`record_instance`] captures a binder's term.
const TERM_DEPTH: usize = 3;

/// Whether instances of `rid` are traced: the unit is traced and `rid` is a
/// heap recipe. Checked once per quantifier per rule application.
pub(crate) fn wants_instances(rid: RecipeId) -> bool {
    active() && HEAP_RECIPES.with(|r| r.borrow().contains(&rid))
}

/// Record an instance of `rid` built at `binders`, at `generation`.
pub(crate) fn record_instance(
    egraph: &egg::EGraph<crate::verify::lang::Symbolic, crate::verify::analysis::ConstFold>,
    rid: RecipeId,
    binders: &[egg::Id],
    generation: u32,
) {
    let binders = binders
        .iter()
        .map(|&b| (egraph.find(b), capture(egraph, b, TERM_DEPTH)))
        .collect();
    let inst = Instance {
        rid,
        binders,
        generation,
        scratch: crate::verify::rewrite::memo::in_scratch_scope(),
    };
    INSTANCES.with(|i| i.borrow_mut().push(inst));
}

fn capture(
    egraph: &egg::EGraph<crate::verify::lang::Symbolic, crate::verify::analysis::ConstFold>,
    id: egg::Id,
    depth: usize,
) -> Term {
    use egg::Language;
    let class = &egraph[egraph.find(id)];
    // A class the analysis folded to a literal reads as that literal.
    if let Some(lit) = class.data.known() {
        return Term {
            node: crate::verify::lang::Symbolic::Lit(lit.clone()),
            kids: Some(Vec::new()),
        };
    }
    let node = class.nodes[0].clone();
    let kids = (depth > 0).then(|| {
        node.children()
            .iter()
            .map(|&c| capture(egraph, c, depth - 1))
            .collect()
    });
    Term { node, kids }
}

fn render_term(ctx: &VerifyContext<'_>, t: &Term) -> String {
    use crate::verify::lang::Symbolic;
    use egg::Language;
    let kids = match &t.kids {
        Some(k) => k.iter().map(|k| render_term(ctx, k)).collect::<Vec<_>>(),
        None if t.node.children().is_empty() => Vec::new(),
        None => vec!["…".to_string()],
    };
    match &t.node {
        Symbolic::Fresh(_) | Symbolic::Lit(_) => t.node.to_string(),
        Symbolic::FuncApp(m, tys, _) => {
            let name = ctx.func_name(*m);
            let head = if tys.is_empty() {
                name
            } else {
                let ts: Vec<String> = tys.iter().map(|t| ctx.type_name(t)).collect();
                format!("{name}<{}>", ts.join(", "))
            };
            format!("{head}({})", kids.join(", "))
        }
        n => format!("{n}({})", kids.join(", ")),
    }
}

/// Print the instances recorded since the last flush, under the current
/// instruction's header. Identical instances (one per scratch graph that
/// rebuilt it) print once, with a count.
pub(crate) fn flush_instances(ctx: &VerifyContext<'_>) {
    let insts = INSTANCES.with(|i| std::mem::take(&mut *i.borrow_mut()));
    if insts.is_empty() || !active() {
        return;
    }
    let mut lines: Vec<(String, usize)> = Vec::new();
    for i in &insts {
        let binders: Vec<String> = i
            .binders
            .iter()
            .enumerate()
            .map(|(j, (id, t))| format!("x{j} := @{} {}", usize::from(*id), render_term(ctx, t)))
            .collect();
        let line = format!(
            "R{}: {}   (gen {}{})",
            i.rid.0,
            binders.join(", "),
            i.generation,
            if i.scratch { ", scratch" } else { "" }
        );
        match lines.iter_mut().find(|(l, _)| *l == line) {
            Some((_, n)) => *n += 1,
            None => lines.push((line, 1)),
        }
    }
    let mut out = format!(
        "instances ({}):
",
        insts.len()
    );
    for (line, n) in lines {
        let times = if n > 1 {
            format!("  ×{n}")
        } else {
            String::new()
        };
        let _ = writeln!(out, "  {line}{times}");
    }
    emit(&out);
}
