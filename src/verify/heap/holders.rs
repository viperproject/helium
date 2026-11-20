//! Who holds a location: every single chunk and quantified chunk set that may
//! hold an address, with its share and value, and what follows from that (the
//! total permission, the one value they agree on, reading and consuming).

use crate::verify::context::VerifyContext;
use crate::verify::error::VerifyError;
use crate::verify::heap::algebra::{
    Demand, chunk_share_at, heap_subtract, perm_add, prove_perm_positive, prove_sufficient_aligned,
};
use crate::verify::heap::interface::{ReadRecipe, Value};
use crate::verify::heap::quantified_permissions::qp_perm_at;
use crate::verify::heap::{Chunk, ChunkPerm, Heap, LocationKind, Restored, Taken};
use crate::verify::lang::Symbolic;
use crate::verify::rewrite::AxiomPure;
use crate::vmir::{BinOp, Literal, Polarity, Type, Val};

/// One holder of a permission at an address (see [`holders_at`]): its share
/// there and the value it holds there.
pub(super) struct Holder {
    pub(super) perm: ChunkPerm,
    pub(super) value: egg::Id,
    source: HolderSource,
    pub(super) at: HolderAt,
}

/// Where a [`Holder`] sits in its heap: an index into `chunks_of(kind)` or
/// `qp_chunks_of(kind)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HolderAt {
    Chunk(usize),
    Set(usize),
}

/// Where a [`Holder`]'s value comes from, with its recipe provenance
/// (certificate walks).
enum HolderSource {
    /// A single chunk; the recipe is the value's own.
    Chunk(Option<Val>),
    /// A quantified chunk set; the recipe is its map's (the value is
    /// `lookup(map, addr)`).
    Set(Option<Val>),
}

/// Everything in `heap` that may hold `addr`: each single chunk that may sit
/// there ([`chunk_share_at`], its aliasing gate folded in) and each quantified
/// chunk set that can name it ([`qp_perm_at`], its guard folded in, what was
/// exhaled from it subtracted). In heap order, single chunks first.
pub(super) fn holders_at(
    ctx: &mut VerifyContext<'_>,
    heap: &Heap,
    kind: &LocationKind,
    addr: egg::Id,
    pc_lits: &[(egg::Id, Polarity)],
) -> Result<Vec<Holder>, VerifyError> {
    let mut holders = Vec::new();
    for (i, c) in heap.chunks_of(kind).iter().enumerate() {
        if let Some((perm, _)) = chunk_share_at(ctx, c, addr, pc_lits) {
            holders.push(Holder {
                perm,
                value: c.value,
                source: HolderSource::Chunk(c.recipe.clone()),
                at: HolderAt::Chunk(i),
            });
        }
    }
    for (i, set) in heap.qp_chunks_of(kind).iter().enumerate() {
        let lookup = ctx.alloc.value_map(kind).lookup;
        let value = ctx.add(Symbolic::FuncApp(
            lookup,
            Box::new([]),
            Box::new([set.value_map, addr]),
        ));
        for r in set.restored.iter() {
            define_restored(ctx, kind, r, addr)?;
        }
        holders.push(Holder {
            perm: qp_perm_at(ctx, set, addr)?,
            value,
            source: HolderSource::Set(set.recipe.clone()),
            at: HolderAt::Set(i),
        });
    }
    Ok(holders)
}

/// Assume what the restored map `rest.map` reads at `addr` (see [`Restored`]).
fn define_restored(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    rest: &Restored,
    addr: egg::Id,
) -> Result<(), VerifyError> {
    let positive = |ctx: &mut VerifyContext<'_>, set: &crate::verify::heap::QuantChunk| {
        let perm = qp_perm_at(ctx, set, addr)?.to_id(ctx);
        Ok::<_, VerifyError>(expr!(ctx, (0 / 1) < r { perm }))
    };
    let inside = positive(ctx, &rest.region)?;
    let kept = positive(ctx, &rest.kept)?;
    let lookup = ctx.alloc.value_map(kind).lookup;
    let [read, new, old] = [rest.map, rest.new, rest.old]
        .map(|m| ctx.add(Symbolic::FuncApp(lookup, Box::new([]), Box::new([m, addr]))));
    let value = ctx.add(Symbolic::Ite([inside, new, old]));
    let fact = expr!(ctx, { read } == { value });
    ctx.assume_guarded(fact, std::iter::empty());
    let eq = expr!(ctx, { new } == { old });
    let both = [(kept, Polarity::Positive), (inside, Polarity::Positive)];
    let agree = ctx.implication(eq, both.into_iter());
    ctx.assume_guarded(agree, std::iter::empty());
    Ok(())
}

/// The sum of the `holders`' shares: everything they hold at the address.
pub(super) fn sum_perms(ctx: &mut VerifyContext<'_>, holders: &[Holder]) -> ChunkPerm {
    let mut total = ChunkPerm::Zero;
    for h in holders {
        total = perm_add(ctx, &total, &h.perm);
    }
    total
}

/// Split `demand` greedily over `holders`, in order: each gives
/// `min(what it holds, what is still needed)`, so the takes sum to the demand
/// wherever the holders hold enough. One take per holder.
pub(super) fn greedy_takes(
    ctx: &mut VerifyContext<'_>,
    holders: &[Holder],
    demand: egg::Id,
) -> Vec<egg::Id> {
    let mut remaining = demand;
    holders
        .iter()
        .map(|h| {
            let hold = h.perm.to_id(ctx);
            let take = expr!(ctx, if ({hold} <r {remaining}) then {hold} else {remaining});
            remaining = ctx.add(Symbolic::Binary(
                crate::vmir::BinOp::SubR,
                [remaining, take],
            ));
            take
        })
        .collect()
}

/// The value at a location from its `holders`. A location has one value, so
/// any two holders with a positive share there agree; that is assumed here,
/// under the pc. The value is the first holder's whose share is known positive,
/// behind an `ite` on each holder before it that may be positive; fresh when no
/// holder is known positive.
fn value_at(
    ctx: &mut VerifyContext<'_>,
    holders: &[Holder],
    ty: &Type,
    pc_lits: &[(egg::Id, Polarity)],
) -> egg::Id {
    let positive: Vec<egg::Id> = holders
        .iter()
        .map(|h| {
            let p = h.perm.to_id(ctx);
            expr!(ctx, (0 / 1) < r { p })
        })
        .collect();
    for i in 0..holders.len() {
        for j in i + 1..holders.len() {
            let (vi, vj) = (holders[i].value, holders[j].value);
            if ctx.egraph.find(vi) == ctx.egraph.find(vj) {
                continue;
            }
            // Inside the fact, not as guards: a scratch graph assumes the fact
            // without its guards (see `assume_guarded`).
            let eq = expr!(ctx, { vi } == { vj });
            let both = [
                (positive[j], Polarity::Positive),
                (positive[i], Polarity::Positive),
            ];
            let agree = ctx.implication(eq, both.into_iter());
            ctx.assume_guarded(agree, pc_lits.iter().rev().copied());
        }
    }
    let known = holders
        .iter()
        .position(|h| prove_perm_positive(ctx, &h.perm, pc_lits));
    let (mut value, upto) = match known {
        Some(k) => (holders[k].value, k),
        None => (ctx.fresh_symbolic_value(ty.clone()), holders.len()),
    };
    for i in (0..upto).rev() {
        value = ctx.add(Symbolic::Ite([positive[i], holders[i].value, value]));
    }
    value
}

/// The value at `addr` (of `kind`) from its `holders`, `None` when there are
/// none. Outside a certificate walk it is [`value_at`]. A certificate names one
/// holder instead: the first known to hold a positive share, with its recipe
/// (see [`holder_recipe`]). `None` too when no holder is known positive there.
///
/// Naming one holder drops what the others hold there, and the walk runs in a
/// scratch graph, so their agreement would not reach a graft site. It is
/// recorded in the certificate instead (see [`record_agreements`]).
pub(super) fn pick_value(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    holders: &[Holder],
    pc_lits: &[(egg::Id, Polarity)],
    read_recipe: Option<&ReadRecipe>,
) -> Option<Value> {
    if holders.is_empty() {
        return None;
    }
    if ctx.recipe.is_none() {
        return Some(Value::plain(value_at(ctx, holders, &kind.value, pc_lits)));
    }
    let k = holders
        .iter()
        .position(|h| prove_perm_positive(ctx, &h.perm, pc_lits))?;
    let addr_r = read_recipe.and_then(|r| r.addr.as_ref());
    let recipe = holder_recipe(ctx, kind, &holders[k], addr_r);
    record_agreements(ctx, kind, holders, k, recipe.as_ref(), read_recipe, pc_lits);
    Some(Value {
        id: holders[k].value,
        recipe,
    })
}

/// The recipe naming what `h` holds at the address in a certificate walk: a
/// single chunk's own recipe, `lookup(map, addr)` for a chunk set (from its
/// map's recipe and the address recipe `addr_r`). `None` when a part is
/// missing.
fn holder_recipe(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    h: &Holder,
    addr_r: Option<&Val>,
) -> Option<Val> {
    match (&h.source, addr_r) {
        (HolderSource::Chunk(r), _) => r.clone(),
        (HolderSource::Set(Some(map_r)), Some(addr_r)) => {
            let lookup = ctx.alloc.value_map(kind).lookup;
            let rb = ctx
                .recipe
                .as_mut()
                .expect("holder recipe outside a certificate walk");
            Some(rb.emit(AxiomPure::App {
                func: lookup,
                type_args: vec![],
                args: vec![map_r.clone(), addr_r.clone()],
            }))
        }
        (HolderSource::Set(_), _) => None,
    }
}

/// In a certificate walk, the agreement between the named holder `holders[k]`
/// and every later holder also known to hold a positive share: a location has
/// one value. Assumed here, under the pc, and recorded in the certificate as
/// `pc_r ==> v_k == v_j` (the read's recipe-space pc folded in), which each
/// graft site assumes (see `walk_footprint`).
///
/// Guarding by the pc alone is sound because positivity was proven here from
/// what the certificate itself carries: the params, the slot permissions and
/// the body's conjuncts, all reproduced (assumed or proven) at a graft site
/// before the agreement is assumed.
///
/// A pair without a recipe on either side, or a read without a recipe-space
/// pc, records nothing: the graft site then knows less, never too much.
fn record_agreements(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    holders: &[Holder],
    k: usize,
    named_r: Option<&Val>,
    read_recipe: Option<&ReadRecipe>,
    pc_lits: &[(egg::Id, Polarity)],
) {
    let addr_r = read_recipe.and_then(|r| r.addr.as_ref());
    let pc_r = read_recipe.and_then(|r| r.pc.as_ref());
    let vk = holders[k].value;
    for h in &holders[k + 1..] {
        if ctx.egraph.find(vk) == ctx.egraph.find(h.value) {
            continue;
        }
        if !prove_perm_positive(ctx, &h.perm, pc_lits) {
            continue;
        }
        let eq = expr!(ctx, { vk } == { h.value });
        ctx.assume_guarded(eq, pc_lits.iter().rev().copied());
        let (Some(named_r), Some(pc_r)) = (named_r, pc_r) else {
            continue;
        };
        let Some(other_r) = holder_recipe(ctx, kind, h, addr_r) else {
            continue;
        };
        let rb = ctx
            .recipe
            .as_mut()
            .expect("agreement outside a certificate walk");
        let mut fact = rb.emit(AxiomPure::Binary(BinOp::Eq, named_r.clone(), other_r));
        let true_ = Val::Literal(Literal::Bool(true));
        // Outermost-first, folded innermost-first (as `implication` does).
        for (g, pol) in pc_r.iter().rev() {
            fact = rb.emit(match pol {
                Polarity::Positive => AxiomPure::Ternary(g.clone(), fact, true_.clone()),
                Polarity::Negative => AxiomPure::Ternary(g.clone(), true_.clone(), fact),
            });
        }
        rb.record_side_fact(fact);
    }
}

/// Consume the single-location `chunk` from `heap`: from the single chunks
/// (`heap_subtract`) when they hold it, otherwise as a single-location take
/// from the first quantified chunk set that holds enough at its address.
///
/// The demand is not split across holders (part from a single chunk, part from
/// a set, or from two sets) yet, and a wildcard demand on a set is
/// unimplemented: it needs a fresh amount below what the set holds.
pub(super) fn subtract_location(
    ctx: &mut VerifyContext<'_>,
    heap: &Heap,
    kind: &LocationKind,
    chunk: Chunk,
    pc_lits: &[(egg::Id, Polarity)],
    demand: Demand,
) -> Result<Heap, VerifyError> {
    if heap.qp_chunks_of(kind).is_empty() {
        return heap_subtract(ctx, heap, kind, chunk, pc_lits, demand);
    }
    match heap_subtract(ctx, heap, kind, chunk.clone(), pc_lits, demand) {
        Err(VerifyError::InsufficientPermission) => {}
        done => return done,
    }
    if demand == Demand::Wildcard {
        return Err(VerifyError::Unimplemented(
            "exhaling a wildcard location from a quantified permission",
        ));
    }
    let needed = chunk.ungated_perm().clone();
    for (i, set) in heap.qp_chunks_of(kind).iter().enumerate() {
        let held = qp_perm_at(ctx, set, chunk.addr)?;
        if prove_sufficient_aligned(ctx, &held, &needed, pc_lits) {
            let set = set.with_taken(
                ctx,
                Taken::Single {
                    addr: chunk.addr,
                    perm: needed,
                    guard: crate::verify::heap::Cube::empty(),
                },
            );
            return Ok(heap.with_qp_chunk_replaced(kind, i, set));
        }
    }
    Err(VerifyError::InsufficientPermission)
}
