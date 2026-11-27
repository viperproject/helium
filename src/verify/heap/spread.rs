//! The value of a quantified permission that no single held set covers, only
//! all holders of its kind together ([`covered_by_sum`]): a **throwaway** value
//! map. It is the value a frame or take of that demand returns and is never
//! stored on a holder, since no single holder holds what it describes.
//!
//! Its values are a definition over the holders it reads,
//!
//! ```text
//! lookup(F(cs), at) == ite(P_1(c_1, at), lookup(m_1, at),
//!                      ite(P_2(c_2, at), lookup(m_2, at), … lookup(m_n, at)))
//! ```
//!
//! where `cs` lists every holder's captures and map, and `P_j(c_j, at)` says
//! whether set `j` holds a positive amount at `at` ([`positivity`]): its body
//! at the binding its inverses give `at`, gated by the rebuilt location being
//! `at`, as [`qp_perm_at`] gates it. Both are definitions, consistent for any
//! arguments and any inverse, so a certificate states them too, over its own
//! recipes ([`crate::verify::cert::RecipeBuilder::record_definition`]), and names
//! the map `F(cs)` there.
//!
//! A single chunk at `a` enters as `ite(at == a ∧ 0 < p, v, …)`. It carries no
//! address recipe, so with one among the holders the map has no recipe.
//!
//! [`covered_by_sum`]: super::quantified_permissions::covered_by_sum
//! [`qp_perm_at`]: super::quantified_permissions::qp_perm_at

use std::rc::Rc;

use crate::dhash::HashMap;
use crate::verify::context::VerifyContext;
use crate::verify::error::VerifyError;
use crate::verify::heap::cond::Lit;
use crate::verify::heap::interface::Value;
use crate::verify::heap::quantified_permissions::{and, positive};
use crate::verify::heap::{Heap, LocationKind, QuantChunk, Scale};
use crate::verify::lang::{FuncId, RecipeId, Symbolic};
use crate::verify::quant::{Canonical, QuantRecipe};
use crate::verify::rewrite::{AxiomInst, AxiomPure, PreparedTerm, TermHead};
use crate::vmir::{self, BinOp, InstKind, PermInst, Type, Val};

/// The functions minted for spread maps, each with its definitional
/// quantifier: one positivity function per set (keyed by its inverses, which
/// fix its body), one map function per holder layout.
#[derive(Default)]
pub(crate) struct SpreadFns {
    positivity: HashMap<Rc<[FuncId]>, Option<(FuncId, RecipeId)>>,
    maps: HashMap<(LocationKind, Vec<Part>), (FuncId, RecipeId)>,
}

/// How one holder enters a spread map's definition.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Part {
    /// A chunk set, by its positivity function: its captures, then its map.
    Set { pos: FuncId, n_caps: usize },
    /// A single chunk: its address, `0 < p`, its value.
    Chunk,
}

impl Part {
    fn arity(&self) -> usize {
        match self {
            Part::Set { n_caps, .. } => n_caps + 1,
            Part::Chunk => 3,
        }
    }
}

/// One holder of a spread map with its arguments: the e-classes, and in a
/// certificate walk their recipes when every one has one.
struct Holder {
    part: Part,
    /// A set's positivity definition, stated over its captures.
    def: Option<RecipeId>,
    args: Vec<egg::Id>,
    recipes: Option<Vec<Val>>,
}

/// The value of a demand of `loc_ty` that all holders of `kind` in `heap` cover
/// together under `pc`. `None` when a holder cannot be described: a set that
/// something was taken from or given back to, held under a presence `pc` does
/// not decide, or scaled by an amount not known positive.
pub(super) fn spread_value(
    ctx: &mut VerifyContext<'_>,
    heap: &Heap,
    kind: &LocationKind,
    loc_ty: &Type,
    pc: &[Lit],
) -> Result<Option<Value>, VerifyError> {
    let mut holders = Vec::new();
    for set in heap.qp_chunks_of(kind) {
        if set.presence_within(ctx, pc).is_none() {
            continue;
        }
        match set_holder(ctx, set, pc)? {
            Some(h) => holders.push(h),
            None => return Ok(None),
        }
    }
    for c in heap.chunks_of(kind) {
        let perm = c.gated_perm(ctx).to_id(ctx);
        let positive = expr!(ctx, (0 / 1) < r { perm });
        holders.push(Holder {
            part: Part::Chunk,
            def: None,
            args: vec![c.addr, positive, c.value().id],
            recipes: None,
        });
    }
    if holders.is_empty() {
        return Ok(None);
    }
    let parts: Vec<Part> = holders.iter().map(|h| h.part.clone()).collect();
    let args: Vec<egg::Id> = holders
        .iter()
        .flat_map(|h| h.args.iter().copied())
        .collect();
    let Some((map_fn, map_def)) = map_function(ctx, kind, loc_ty, parts, &args) else {
        return Ok(None);
    };

    // The definitions hold for any arguments: assumed unguarded.
    for h in &holders {
        if let Some(def) = h.def {
            let caps = &h.args[..h.args.len() - 1];
            assume_definition(ctx, def, caps);
        }
    }
    assume_definition(ctx, map_def, &args);
    let id = ctx.add(Symbolic::FuncApp(map_fn, Box::new([]), args.into()));

    let recipe = match ctx.recipe.as_mut() {
        Some(rb) => holders
            .iter()
            .map(|h| h.recipes.clone())
            .collect::<Option<Vec<_>>>()
            .map(|per_holder| {
                for (h, r) in holders.iter().zip(&per_holder) {
                    if let Some(def) = h.def {
                        let caps = r[..r.len() - 1].to_vec();
                        let fact = rb.emit_forall(def, Vec::new(), caps);
                        rb.record_definition(fact);
                    }
                }
                let args: Vec<Val> = per_holder.into_iter().flatten().collect();
                let fact = rb.emit_forall(map_def, Vec::new(), args.clone());
                rb.record_definition(fact);
                rb.emit(AxiomPure::App {
                    func: map_fn,
                    type_args: Vec::new(),
                    args,
                })
            }),
        None => None,
    };
    Ok(Some(Value { id, recipe }))
}

fn assume_definition(ctx: &mut VerifyContext<'_>, def: RecipeId, caps: &[egg::Id]) {
    crate::verify::qp_trace::recipe(ctx, "spread definition", def, caps);
    let node = ctx.add(Symbolic::Forall(def, Box::new([]), caps.into()));
    ctx.assume_guarded(node, std::iter::empty());
}

/// The set `set` as a holder of a spread map under `pc` (see [`spread_value`]).
fn set_holder(
    ctx: &mut VerifyContext<'_>,
    set: &QuantChunk,
    pc: &[Lit],
) -> Result<Option<Holder>, VerifyError> {
    let pristine = set.taken.is_empty() && set.restored.is_empty() && set.handovers.is_empty();
    if !pristine || !set.presence().holds_under(ctx, pc) {
        return Ok(None);
    }
    if let Some(Scale::Amount(s)) = set.caps.scale {
        let positive = expr!(ctx, (0 / 1) < r { s });
        if !ctx.prove_under_pc(positive, pc) {
            return Ok(None);
        }
    }
    let Some((pos, def)) = positivity(ctx, set)? else {
        return Ok(None);
    };
    let mut args = set.caps.cap_ids();
    let n_caps = args.len();
    args.push(set.value_map.id);
    let recipes = set.caps.cap_recipes().and_then(|mut r| {
        r.push(set.value_map.recipe.clone()?);
        Some(r)
    });
    Ok(Some(Holder {
        part: Part::Set { pos, n_caps },
        def: Some(def),
        args,
        recipes,
    }))
}

/// The positivity function of `set` and its definition,
///
/// ```text
/// forall at :: {P(cs, at)}  P(cs, at) == (c(xs) ∧ e(xs) == at ∧ 0 < p(xs))
///     where xs = inv(at)
/// ```
///
/// over its captures `cs`: whether it holds a positive amount at `at`. Its
/// scale and wildcard leave that unchanged ([`set_holder`] checks the scale).
/// `None` when a capture's type cannot be inferred.
fn positivity(
    ctx: &mut VerifyContext<'_>,
    set: &QuantChunk,
) -> Result<Option<(FuncId, RecipeId)>, VerifyError> {
    if let Some(found) = ctx.spread_fns.positivity.get(&set.inv) {
        return Ok(*found);
    }
    let found = mint_positivity(ctx, set)?;
    ctx.spread_fns.positivity.insert(set.inv.clone(), found);
    Ok(found)
}

fn mint_positivity(
    ctx: &mut VerifyContext<'_>,
    set: &QuantChunk,
) -> Result<Option<(FuncId, RecipeId)>, VerifyError> {
    let (q, caps) = (&set.q, &set.caps);
    let Some(mut params) = types_of(ctx, &caps.cap_ids()) else {
        return Ok(None);
    };
    params.push(q.loc_ty.clone());
    let pos = ctx.alloc.spread_positivity(params);
    ctx.func_ret_types.insert(pos, Type::Bool);

    let free: Vec<usize> = caps.temps.iter().map(|&(k, _)| k).collect();
    let shape = caps.reads.shape();
    let (interner, decls) = (ctx.interner, ctx.decls);
    let names = move |m| crate::verify::context::member_name_in(interner, decls, m);
    let table = std::sync::Arc::clone(ctx.alloc.quant_table());
    let mut c = Canonical::at_binding(
        ctx.alloc,
        &table.read().expect("recipe table lock"),
        &names,
        ctx.certs,
        q.binder_base,
        q.bound.len(),
        &free,
        &q.heap_reads(),
        &shape,
        &q.body.insts,
        1,
    )?;
    let n_caps = c.n_caps();
    let at = Val::Temp(n_caps);
    c.bind(
        set.inv
            .iter()
            .map(|&func| AxiomPure::App {
                func,
                type_args: Vec::new(),
                args: vec![at.clone()],
            })
            .collect(),
    );
    let perms: Vec<&PermInst> = q
        .body
        .insts
        .iter()
        .filter_map(|i| match &i.kind {
            InstKind::Perm(p) => Some(p),
            _ => None,
        })
        .collect();
    let amount = positive(&mut c, &q.body.perm, &perms);
    let cond = c.rename(&q.body.cond);
    let loc = c.rename(&q.body.loc);
    let is_at = c.push(AxiomPure::Binary(BinOp::Eq, loc, at.clone()));
    let held = and(&mut c, is_at, amount);
    let held = and(&mut c, cond, held);
    let mut args: Vec<Val> = (0..n_caps).map(Val::Temp).collect();
    args.push(at);
    let app = c.push(AxiomPure::App {
        func: pos,
        type_args: Vec::new(),
        args,
    });
    let res = c.push(AxiomPure::Binary(BinOp::Eq, app, held));
    let args = (0..n_caps).map(PreparedTerm::Capture);
    let trigger = application(pos, args.chain([PreparedTerm::Bound(0)]));
    let def = push_recipe(ctx, n_caps, &q.loc_ty, vec![vec![trigger]], c.insts, res);
    Ok(Some((pos, def)))
}

/// The map function of a holder layout and its definition (see the module
/// docs), minted on first use. `args` are one use's arguments, whose types
/// type the function. `None` when one cannot be inferred.
fn map_function(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    loc_ty: &Type,
    parts: Vec<Part>,
    args: &[egg::Id],
) -> Option<(FuncId, RecipeId)> {
    let key = (kind.clone(), parts);
    if let Some(&found) = ctx.spread_fns.maps.get(&key) {
        return Some(found);
    }
    let params = types_of(ctx, args)?;
    let map_ty = Type::ValueMap(Box::new(loc_ty.clone()));
    let map_fn = ctx.alloc.spread_map(params, map_ty.clone());
    ctx.func_ret_types.insert(map_fn, map_ty);
    let lookup = ctx.alloc.value_map(kind).lookup;

    let n_caps = args.len();
    let at = Val::Temp(n_caps);
    let mut steps = Steps::new(n_caps + 1);
    let lookup_at = |steps: &mut Steps, map: Val| {
        steps.push(AxiomPure::App {
            func: lookup,
            type_args: Vec::new(),
            args: vec![map, at.clone()],
        })
    };
    let all: Vec<Val> = (0..n_caps).map(Val::Temp).collect();
    let map = steps.push(AxiomPure::App {
        func: map_fn,
        type_args: Vec::new(),
        args: all,
    });
    let value = lookup_at(&mut steps, map);
    // Each holder's slice of the arguments, and what it gives at `at`.
    let mut entries = Vec::new();
    let mut base = 0;
    for part in &key.1 {
        let arg = |i: usize| Val::Temp(base + i);
        let entry = match part {
            Part::Set { pos, n_caps } => {
                let mut pargs: Vec<Val> = (0..*n_caps).map(arg).collect();
                pargs.push(at.clone());
                let held = steps.push(AxiomPure::App {
                    func: *pos,
                    type_args: Vec::new(),
                    args: pargs,
                });
                (held, lookup_at(&mut steps, arg(*n_caps)))
            }
            Part::Chunk => {
                let is_at = steps.push(AxiomPure::Binary(BinOp::Eq, arg(0), at.clone()));
                let held = steps.push(AxiomPure::Ternary(is_at, arg(1), vmir::FALSE));
                (held, arg(2))
            }
        };
        entries.push(entry);
        base += part.arity();
    }
    // Where no holder holds `at` the value is never read: the last one's.
    let mut chosen = entries.last().expect("a spread has a holder").1.clone();
    for (held, v) in entries.into_iter().rev() {
        chosen = steps.push(AxiomPure::Ternary(held, v, chosen));
    }
    let res = steps.push(AxiomPure::Binary(BinOp::Eq, value, chosen));
    let trigger = PreparedTerm::App {
        head: TermHead::Func {
            func: lookup,
            type_args: Box::new([]),
        },
        args: vec![
            application(map_fn, (0..n_caps).map(PreparedTerm::Capture)),
            PreparedTerm::Bound(0),
        ],
    };
    let def = push_recipe(ctx, n_caps, loc_ty, vec![vec![trigger]], steps.insts, res);
    ctx.spread_fns.maps.insert(key, (map_fn, def));
    Some((map_fn, def))
}

/// The trigger term `f(args)`.
fn application(f: FuncId, args: impl Iterator<Item = PreparedTerm>) -> PreparedTerm {
    PreparedTerm::App {
        head: TermHead::Func {
            func: f,
            type_args: Box::new([]),
        },
        args: args.collect(),
    }
}

/// A ground recipe binding one location `at` of `loc_ty` after `n_caps`
/// captures.
fn push_recipe(
    ctx: &mut VerifyContext<'_>,
    n_caps: usize,
    loc_ty: &Type,
    groups: Vec<Vec<PreparedTerm>>,
    insts: Vec<AxiomInst>,
    res: Val,
) -> RecipeId {
    ctx.alloc
        .quant_table()
        .write()
        .expect("recipe table lock")
        .push_fresh(QuantRecipe {
            n_caps,
            bound: Box::new([loc_ty.clone()]),
            n_tys: 0,
            own_tys: vec![],
            groups,
            insts,
            res,
            guard: None,
        })
}

/// Recipe steps whose temps count from `first`.
struct Steps {
    first: usize,
    insts: Vec<AxiomInst>,
}

impl Steps {
    fn new(first: usize) -> Self {
        Steps {
            first,
            insts: Vec::new(),
        }
    }

    fn push(&mut self, p: AxiomPure) -> Val {
        self.insts.push(AxiomInst::Val(p));
        Val::Temp(self.first + self.insts.len() - 1)
    }
}

/// The types of `ids`, inferred; `None` if one cannot be.
fn types_of(ctx: &VerifyContext<'_>, ids: &[egg::Id]) -> Option<Vec<Type>> {
    let mut memo = HashMap::default();
    ids.iter()
        .map(|&id| {
            crate::verify::types::infer_type(
                &ctx.egraph,
                &ctx.fresh_types,
                &ctx.func_ret_types,
                id,
                &mut memo,
            )
        })
        .collect()
}
