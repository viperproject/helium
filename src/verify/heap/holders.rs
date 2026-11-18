//! Who holds a location: every single chunk and quantified chunk set that may
//! hold an address, with its share and value, and what follows from that (the
//! total permission, the one value they agree on, reading and consuming).

use crate::verify::context::VerifyContext;
use crate::verify::error::VerifyError;
use crate::verify::heap::algebra::{
    Demand, chunk_share_at, heap_subtract, perm_add, prove_perm_positive,
    prove_sufficient_aligned,
};
use crate::verify::heap::quantified_permissions::qp_perm_at;
use crate::verify::heap::{
    Chunk, ChunkPerm, Heap, LocationKind, QPTaken, gate_perm_by_guard,
};
use crate::verify::lang::Symbolic;
use crate::verify::heap::interface::Value;
use crate::vmir::{Polarity, Type, Val};

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
#[derive(Clone, Copy, Debug)]
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
        if let Some(p) = qp_perm_at(ctx, set, addr)? {
            let lookup = ctx.alloc.value_map(kind).lookup;
            let value = ctx.add(Symbolic::FuncApp(
                lookup,
                Box::new([]),
                Box::new([set.value_map, addr]),
            ));
            holders.push(Holder {
                perm: gate_perm_by_guard(ctx, &p, &set.guard),
                value,
                source: HolderSource::Set(set.recipe.clone()),
                at: HolderAt::Set(i),
            });
        }
    }
    Ok(holders)
}

/// The sum of the `holders`' shares: everything they hold at the address.
pub(super) fn sum_perms(ctx: &mut VerifyContext<'_>, holders: &[Holder]) -> ChunkPerm {
    let mut total = ChunkPerm::leaf(expr!(ctx, 0 / 1));
    for h in holders {
        total = perm_add(ctx, &total, &h.perm);
    }
    total
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
/// (a chunk set's is `lookup(map, addr)`, built from the address recipe
/// `addr_r`; without one the recipe is `None`). `None` too when no holder is
/// known positive there.
pub(super) fn pick_value(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    holders: &[Holder],
    pc_lits: &[(egg::Id, Polarity)],
    addr_r: Option<&Val>,
) -> Option<Value> {
    if holders.is_empty() {
        return None;
    }
    if ctx.recipe.is_none() {
        return Some(Value::plain(value_at(ctx, holders, &kind.value, pc_lits)));
    }
    let h = holders
        .iter()
        .find(|h| prove_perm_positive(ctx, &h.perm, pc_lits))?;
    let recipe = match (&h.source, addr_r) {
        (HolderSource::Chunk(r), _) => r.clone(),
        (HolderSource::Set(Some(map_r)), Some(addr_r)) => {
            let lookup = ctx.alloc.value_map(kind).lookup;
            let rb = ctx.recipe.as_mut().unwrap();
            Some(rb.emit(crate::verify::rewrite::AxiomPure::App {
                func: lookup,
                type_args: vec![],
                args: vec![map_r.clone(), addr_r.clone()],
            }))
        }
        (HolderSource::Set(_), _) => None,
    };
    Some(Value {
        id: h.value,
        recipe,
    })
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
        let Some(held) = qp_perm_at(ctx, set, chunk.addr)? else {
            continue;
        };
        let held = gate_perm_by_guard(ctx, &held, &set.guard);
        if prove_sufficient_aligned(ctx, &held, &needed, pc_lits) {
            let set = set.with_taken(QPTaken::Single {
                addr: chunk.addr,
                perm: needed,
                guard: pc_lits.into(),
            });
            return Ok(heap.with_qp_chunk_replaced(kind, i, set));
        }
    }
    Err(VerifyError::InsufficientPermission)
}
