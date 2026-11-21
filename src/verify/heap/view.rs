//! A heap as seen from one state: [`View`] binds a heap to the path condition
//! of the instruction that reads it, and answers who holds a location there
//! ([`Shares`]) and what follows from that: the total permission, the one value
//! the holders agree on.
//!
//! Every read of a location goes through [`View::shares`], single chunks and
//! chunk sets alike. Each [`Share`] is already gated by everything that
//! conditions it (the holder's presence, the address gate, a set's domain and
//! what was taken from it), so a share's amount can be summed and proven about
//! as it stands.

use crate::verify::context::VerifyContext;
use crate::verify::error::VerifyError;
use crate::verify::heap::algebra::{chunk_share_at, chunk_under_pc, perm_add, prove_perm_positive};
use crate::verify::heap::cond::Lit;
use crate::verify::heap::interface::{ReadRecipe, Value};
use crate::verify::heap::quantified_permissions::qp_perm_at;
use crate::verify::heap::{Chunk, ChunkPerm, Heap, LocationKind, QuantChunk, Restored};
use crate::verify::lang::Symbolic;
use crate::verify::rewrite::AxiomPure;
use crate::vmir::{BinOp, Literal, Polarity, Type, Val};

/// A heap bound to the path condition of the state it is read in. Built by
/// [`Heap::view`]; every question about who holds a location is asked of one,
/// so the pc an answer is valid under travels with the heap it is about.
pub(super) struct View<'a> {
    heap: &'a Heap,
    pc: &'a [Lit],
}

impl Heap {
    /// This heap as read under `pc`.
    pub(super) fn view<'a>(&'a self, pc: &'a [Lit]) -> View<'a> {
        View { heap: self, pc }
    }
}

/// Where a [`Share`]'s holder sits in its heap: an index into `chunks_of(kind)`
/// or `qp_chunks_of(kind)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HolderAt {
    Chunk(usize),
    Set(usize),
}

/// What one holder holds at an address (see [`View::shares`]): its amount there,
/// gated by everything that conditions it, and the value it holds there.
pub(super) struct Share {
    pub(super) perm: ChunkPerm,
    pub(super) value: egg::Id,
    source: ShareSource,
    pub(super) at: HolderAt,
}

/// Where a [`Share`]'s value comes from, with its recipe provenance
/// (certificate walks).
enum ShareSource {
    /// A single chunk; the recipe is the value's own.
    Chunk(Option<Val>),
    /// A quantified chunk set; the recipe is its map's (the value is
    /// `lookup(map, addr)`).
    Set(Option<Val>),
}

/// Everything a heap holds at one address, in heap order (single chunks first).
pub(super) struct Shares {
    kind: LocationKind,
    shares: Vec<Share>,
    /// The shares are exactly one single chunk that *is* the location under the
    /// pc, and nothing else of its kind is held: its value is the location's,
    /// with no agreement to state and no positivity to prove.
    sole: bool,
}

impl<'a> View<'a> {
    /// Who may hold `addr` (of `kind`) here.
    ///
    /// When the kind has no chunk sets and one single chunk sits at the address
    /// under the pc ([`chunk_under_pc`]: on ground, else in a probe that assumes
    /// the pc), that chunk alone: it is the location. Otherwise every single
    /// chunk that may sit there ([`chunk_share_at`], its aliasing gate folded
    /// in) and every chunk set that can name it ([`qp_perm_at`], its presence
    /// folded in, what was exhaled from it subtracted).
    pub(super) fn shares(
        &self,
        ctx: &mut VerifyContext<'_>,
        kind: &LocationKind,
        addr: egg::Id,
    ) -> Result<Shares, VerifyError> {
        let heap = self.heap;
        let chunks = heap.chunks_of(kind);
        if heap.qp_chunks_of(kind).is_empty()
            && let Some(c) = chunk_under_pc(ctx, chunks, addr, self.pc)
        {
            let i = chunks
                .iter()
                .position(|x| std::ptr::eq(x, c))
                .expect("the matched chunk is one of the group's");
            return Ok(Shares {
                kind: kind.clone(),
                shares: vec![chunk_share(c, i, c.gated_perm(ctx))],
                sole: true,
            });
        }
        self.all_shares(ctx, kind, addr)
    }

    /// Every holder that may hold `addr` (of `kind`), without the sole-chunk
    /// resolution [`View::shares`] tries first.
    pub(super) fn all_shares(
        &self,
        ctx: &mut VerifyContext<'_>,
        kind: &LocationKind,
        addr: egg::Id,
    ) -> Result<Shares, VerifyError> {
        let mut shares = Vec::new();
        for (i, c) in self.heap.chunks_of(kind).iter().enumerate() {
            if let Some((perm, _)) = chunk_share_at(ctx, c, addr, self.pc) {
                shares.push(chunk_share(c, i, perm));
            }
        }
        for (i, set) in self.heap.qp_chunks_of(kind).iter().enumerate() {
            shares.push(set_share(ctx, kind, set, i, addr)?);
        }
        Ok(Shares {
            kind: kind.clone(),
            shares,
            sole: false,
        })
    }
}

fn chunk_share(c: &Chunk, i: usize, perm: ChunkPerm) -> Share {
    Share {
        perm,
        value: c.value,
        source: ShareSource::Chunk(c.recipe.clone()),
        at: HolderAt::Chunk(i),
    }
}

/// What the chunk set `set` (number `i`) holds at `addr`: its amount there and
/// `lookup(map, addr)`. Assumes the definitions of its restored maps there.
fn set_share(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    set: &QuantChunk,
    i: usize,
    addr: egg::Id,
) -> Result<Share, VerifyError> {
    let lookup = ctx.alloc.value_map(kind).lookup;
    let value = ctx.add(Symbolic::FuncApp(
        lookup,
        Box::new([]),
        Box::new([set.value_map, addr]),
    ));
    for r in set.restored.iter() {
        define_restored(ctx, kind, r, addr)?;
    }
    Ok(Share {
        perm: qp_perm_at(ctx, set, addr)?,
        value,
        source: ShareSource::Set(set.recipe.clone()),
        at: HolderAt::Set(i),
    })
}

/// Assume what the restored map `rest.map` reads at `addr` (see [`Restored`]).
fn define_restored(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    rest: &Restored,
    addr: egg::Id,
) -> Result<(), VerifyError> {
    let positive = |ctx: &mut VerifyContext<'_>, set: &QuantChunk| {
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

impl Shares {
    pub(super) fn is_empty(&self) -> bool {
        self.shares.is_empty()
    }

    pub(super) fn iter(&self) -> std::slice::Iter<'_, Share> {
        self.shares.iter()
    }

    /// The sum of the shares: everything held at the address.
    pub(super) fn total(&self, ctx: &mut VerifyContext<'_>) -> ChunkPerm {
        let mut total = ChunkPerm::Zero;
        for s in &self.shares {
            total = perm_add(ctx, &total, &s.perm);
        }
        total
    }

    /// Split `demand` greedily over the shares, in order: each gives
    /// `min(what it holds, what is still needed)`, so the takes sum to the
    /// demand wherever the holders hold enough. One take per share.
    pub(super) fn greedy_takes(
        &self,
        ctx: &mut VerifyContext<'_>,
        demand: egg::Id,
    ) -> Vec<egg::Id> {
        let mut remaining = demand;
        self.shares
            .iter()
            .map(|s| {
                let hold = s.perm.to_id(ctx);
                let take = expr!(ctx, if ({hold} <r {remaining}) then {hold} else {remaining});
                remaining = ctx.add(Symbolic::Binary(BinOp::SubR, [remaining, take]));
                take
            })
            .collect()
    }

    /// The value at the address, `None` when nothing holds it. A sole chunk's
    /// value is the location's. Otherwise, outside a certificate walk, it is
    /// [`value_at`]; a certificate names one holder instead: the first known to
    /// hold a positive share, with its recipe (see [`holder_recipe`]), and
    /// `None` too when no holder is known positive there.
    ///
    /// Naming one holder drops what the others hold there, and the walk runs in a
    /// scratch graph, so their agreement would not reach a graft site. It is
    /// recorded in the certificate instead (see [`record_agreements`]).
    pub(super) fn value(
        &self,
        ctx: &mut VerifyContext<'_>,
        pc_lits: &[Lit],
        read_recipe: Option<&ReadRecipe>,
    ) -> Option<Value> {
        let holders = &self.shares[..];
        if self.sole {
            let s = &holders[0];
            let ShareSource::Chunk(recipe) = &s.source else {
                unreachable!("a sole share is a single chunk")
            };
            return Some(Value {
                id: s.value,
                recipe: recipe.clone(),
            });
        }
        if holders.is_empty() {
            return None;
        }
        if ctx.recipe.is_none() {
            return Some(Value::plain(value_at(
                ctx,
                holders,
                &self.kind.value,
                pc_lits,
            )));
        }
        let k = holders
            .iter()
            .position(|h| prove_perm_positive(ctx, &h.perm, pc_lits))?;
        let addr_r = read_recipe.and_then(|r| r.addr.as_ref());
        let recipe = holder_recipe(ctx, &self.kind, &holders[k], addr_r);
        record_agreements(
            ctx,
            &self.kind,
            holders,
            k,
            recipe.as_ref(),
            read_recipe,
            pc_lits,
        );
        Some(Value {
            id: holders[k].value,
            recipe,
        })
    }
}

/// The value at a location from its `holders`. A location has one value, so
/// any two holders with a positive share there agree; that is assumed here,
/// under the pc. The value is the first holder's whose share is known positive,
/// behind an `ite` on each holder before it that may be positive; fresh when no
/// holder is known positive.
fn value_at(ctx: &mut VerifyContext<'_>, holders: &[Share], ty: &Type, pc_lits: &[Lit]) -> egg::Id {
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

/// The recipe naming what `h` holds at the address in a certificate walk: a
/// single chunk's own recipe, `lookup(map, addr)` for a chunk set (from its
/// map's recipe and the address recipe `addr_r`). `None` when a part is
/// missing.
fn holder_recipe(
    ctx: &mut VerifyContext<'_>,
    kind: &LocationKind,
    h: &Share,
    addr_r: Option<&Val>,
) -> Option<Val> {
    match (&h.source, addr_r) {
        (ShareSource::Chunk(r), _) => r.clone(),
        (ShareSource::Set(Some(map_r)), Some(addr_r)) => {
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
        (ShareSource::Set(_), _) => None,
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
    holders: &[Share],
    k: usize,
    named_r: Option<&Val>,
    read_recipe: Option<&ReadRecipe>,
    pc_lits: &[Lit],
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
