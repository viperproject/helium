//! Consuming a single location from a heap that also holds chunk sets.

use crate::verify::context::VerifyContext;
use crate::verify::error::VerifyError;
use crate::verify::heap::algebra::{Demand, heap_subtract, prove_sufficient_aligned};
use crate::verify::heap::quantified_permissions::qp_perm_at;
use crate::verify::heap::{Chunk, Heap, LocationKind, Taken};
use crate::vmir::Polarity;

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
    // The demand is unconditional, so its gated amount is its bare one.
    let needed = chunk.gated_perm(ctx);
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
