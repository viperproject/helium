//! Heap algebra: the operations over [`Heap`]/[`Chunk`]/[`ChunkPerm`] that add,
//! subtract, join and summarize permission at a location, plus the permission
//! proofs and location axioms they rest on.
//!
//! Single chunks only: chunk sets are `quantified_permissions`, and the two are
//! looked at together in `holders`. Nothing outside the heap module calls this
//! directly; [`Heap`]'s methods in `interface` do.
//!
//! The entry points are [`heap_union`] (produce), [`heap_subtract`] (consume)
//! and [`merge_heaps`]/[`union_heaps`] (control-flow join and loop frame
//! restore). Everything else supports those:
//! - `perm`: permission arithmetic and proofs,
//! - `produce`, `consume`: adding and taking a chunk,
//! - `lookup`: which chunks sit at an address,
//! - `combine`: joining and summing whole heaps.

mod combine;
mod consume;
mod lookup;
mod perm;
mod produce;

pub(super) use combine::*;
pub(super) use consume::*;
pub(super) use lookup::*;
pub(super) use perm::*;
pub(super) use produce::*;

#[cfg(test)]
use crate::verify::{
    context::VerifyContext,
    error::VerifyError,
    heap::{Chunk, Heap, LocationKind},
    lang::Symbolic,
};
#[cfg(test)]
use crate::vmir::{BinOp, Bound, Literal, Polarity};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::test_support::{fresh_ctx, real, test_kind};
    use crate::vmir::Type;

    // A multi-arg bounded location (not Viper-reachable; only via direct VMIR):
    // two chunks of the same location whose perms sum > bound, with all args
    // forced equal, must contradict (the conjunction of arg-equalities collapses
    // to false against the assumed-true equalities).
    #[test]
    fn multiarg_location_nonaliasing_all_args_equal_is_inconsistent() {
        let interner = lasso::Rodeo::new();
        let decls =
            typed_index_collections::TiVec::<crate::vmir::MemberId, crate::vmir::Declaration>::new(
            );
        let mut groups = lasso::Rodeo::<lasso::Spur>::new();
        let g = groups.get_or_intern("g");
        let mut alloc = crate::verify::func_registry::FuncRegistry::empty();
        let mut ctx = VerifyContext::new(&interner, &decls, &groups, &mut alloc);

        // A 2-arg bounded address group `g` of held type `Int`, cap `1/1`. The
        // address is an ordinary `FuncApp` to the group's address function
        // (`FuncId(0)` here); its `Addr{..}` return type is recorded in
        // `func_ret_types` (recovered by `location_chunks`).
        let addr_ty = Type::addr(
            g,
            Type::Int,
            Bound::Bounded(num::BigRational::from(num::BigInt::from(1))),
        );
        let addr_fn = crate::verify::lang::FuncId(0);
        let (x0, y0) = (ctx.add(Symbolic::Fresh(0)), ctx.add(Symbolic::Fresh(1)));
        let (x1, y1) = (ctx.add(Symbolic::Fresh(2)), ctx.add(Symbolic::Fresh(3)));
        let a0 = ctx.add_func_app_id(addr_fn, Box::new([]), addr_ty.clone(), Box::new([x0, y0]));
        let a1 = ctx.add_func_app_id(addr_fn, Box::new([]), addr_ty.clone(), Box::new([x1, y1]));
        let k = LocationKind::from_addr_type(&addr_ty).unwrap();
        let (v0, v1) = (ctx.add(Symbolic::Fresh(4)), ctx.add(Symbolic::Fresh(5)));
        let (p0, p1) = (real(&mut ctx, 3, 4), real(&mut ctx, 1, 2)); // sum 5/4 > 1
        let heap = Heap::empty()
            .with_chunk(&k, Chunk::new(a0, p0, v0))
            .with_chunk(&k, Chunk::new(a1, p1, v1));

        // Assume both argument pairs are equal: x0==x1, y0==y1.
        let true_ = ctx.add(Symbolic::Lit(Literal::Bool(true)));
        for (a, b) in [(x0, x1), (y0, y1)] {
            let eq = ctx.add(Symbolic::Binary(BinOp::Eq, [a, b]));
            ctx.union(eq, true_);
        }
        ctx.egraph.rebuild();

        assume_location_axioms(&mut ctx, &heap);
        ctx.saturate();
        assert!(
            ctx.is_inconsistent(),
            "holding 5/4 across a 2-arg location with all args equal must contradict"
        );
    }

    #[test]
    fn heap_union_merges_egg_equivalent_addresses() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let b = ctx.add(Symbolic::Fresh(1));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let p2 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(2).into())));
        let v1 = ctx.add(Symbolic::Fresh(2));
        let v2 = ctx.add(Symbolic::Fresh(3));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p1, v1));
        ctx.union(a, b);
        ctx.egraph.rebuild();

        let merged = heap_union(&mut ctx, &h1, &test_kind(), Chunk::new(b, p2, v2), &[]);

        let canon = ctx.egraph.find(a);
        let chunk = merged
            .chunk(&test_kind(), canon)
            .expect("merged chunk missing");

        let expected_perm = ctx.add(Symbolic::Binary(BinOp::AddR, [p1, p2]));
        ctx.saturate();
        assert_eq!(
            ctx.egraph
                .find(chunk.perm_repr_id().expect("chunk holds no permission")),
            ctx.egraph.find(expected_perm)
        );
        // Both fractions positive (1, 2) → agreement axiom fuses the values.
        assert_eq!(ctx.egraph.find(v1), ctx.egraph.find(v2));
        assert_eq!(ctx.egraph.find(chunk.value), ctx.egraph.find(v1));
        assert_eq!(merged.entries().count(), 1);
    }

    // Two chunks inserted at *distinct* addresses whose classes collapse
    // later (an aliasing union learned after insertion): the consolidating
    // lookup must re-merge them so a subtract sees the full held amount —
    // first-match lookup only ever saw one 1/2 fragment and failed.
    #[test]
    fn subtract_consolidates_post_hoc_aliased_chunks() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let b = ctx.add(Symbolic::Fresh(1));
        let half = real(&mut ctx, 1, 2);
        let (v0, v1) = (ctx.add(Symbolic::Fresh(2)), ctx.add(Symbolic::Fresh(3)));
        let h = Heap::empty()
            .with_chunk(&test_kind(), Chunk::new(a, half, v0))
            .with_chunk(&test_kind(), Chunk::new(b, half, v1));
        assert_eq!(h.entries().count(), 2);

        // The aliasing fact arrives after both chunks are in the heap.
        ctx.union(a, b);
        ctx.egraph.rebuild();

        let one = real(&mut ctx, 1, 1);
        let out = heap_subtract(
            &mut ctx,
            &h,
            &test_kind(),
            Chunk::new(a, one, v0),
            &[],
            Demand::Concrete,
        )
        .expect("full permission is held across the two aliased fragments");
        // 1/2 + 1/2 − 1/1 = 0 const-folds → the emptied chunk is dropped.
        assert_eq!(out.entries().count(), 0);
    }

    // The borrow / give-back cycle: a chunk whose permission was reduced to
    // `1 − p` gets `p` unioned back in. The merged permission `(1 − p) + p`
    // must land back in `1/1`'s class (the cancellation rewrite), so the next
    // full-permission check is O(1) — the "regain full perm after a match arm"
    // shape, with `p = ite(c, 1, 0)` a branch-scaled borrow.
    #[test]
    fn give_back_restores_full_permission() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let c = ctx.add(Symbolic::Fresh(1));
        let one = real(&mut ctx, 1, 1);
        let zero = real(&mut ctx, 0, 1);
        let p = ctx.add(Symbolic::Ite([c, one, zero]));
        let rest = ctx.add(Symbolic::Binary(BinOp::SubR, [one, p]));
        let (v0, v1) = (ctx.add(Symbolic::Fresh(2)), ctx.add(Symbolic::Fresh(3)));

        let h = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, rest, v0));
        let out = heap_union(&mut ctx, &h, &test_kind(), Chunk::new(a, p, v1), &[]);
        let chunk = out
            .chunk(&test_kind(), ctx.egraph.find(a))
            .expect("merged chunk missing");
        ctx.saturate();
        assert_eq!(
            ctx.egraph
                .find(chunk.perm_repr_id().expect("chunk holds no permission")),
            ctx.egraph.find(one),
            "(1 - p) + p must collapse back to the full permission"
        );
    }

    #[test]
    fn merge_zero_fraction_picks_active_value() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let p0 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(0).into())));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let v0 = ctx.add(Symbolic::Fresh(1));
        let v1 = ctx.add(Symbolic::Fresh(2));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p0, v0));
        let merged = heap_union(&mut ctx, &h1, &test_kind(), Chunk::new(a, p1, v1), &[]);
        let chunk = merged
            .chunk(&test_kind(), ctx.egraph.find(a))
            .expect("merged chunk missing");
        ctx.saturate();

        // p0 = 0 → asymmetric ternary picks the active half v1; no fusion.
        assert_eq!(ctx.egraph.find(chunk.value), ctx.egraph.find(v1));
        assert_ne!(ctx.egraph.find(v0), ctx.egraph.find(v1));
    }

    #[test]
    fn merge_both_active_fuses_values() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let p0 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let v0 = ctx.add(Symbolic::Fresh(1));
        let v1 = ctx.add(Symbolic::Fresh(2));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p0, v0));
        let merged = heap_union(&mut ctx, &h1, &test_kind(), Chunk::new(a, p1, v1), &[]);
        let chunk = merged
            .chunk(&test_kind(), ctx.egraph.find(a))
            .expect("merged chunk missing");
        ctx.saturate();

        // Both fractions positive → agreement axiom fuses the symbolic values.
        assert_eq!(ctx.egraph.find(v0), ctx.egraph.find(v1));
        assert_eq!(ctx.egraph.find(chunk.value), ctx.egraph.find(v0));
    }

    #[test]
    fn merge_under_false_pc_blocks_fusion() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let p0 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let v0 = ctx.add(Symbolic::Fresh(1));
        let v1 = ctx.add(Symbolic::Fresh(2));
        let false_lit = ctx.add(Symbolic::Lit(Literal::Bool(false)));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p0, v0));
        let merged = heap_union(
            &mut ctx,
            &h1,
            &test_kind(),
            Chunk::new(a, p1, v1),
            &[(false_lit, Polarity::Positive)],
        );
        let chunk = merged
            .chunk(&test_kind(), ctx.egraph.find(a))
            .expect("merged chunk missing");
        ctx.saturate();

        // PC literal is `false` → implication collapses to its `true` fallback;
        // values must NOT fuse even though both fractions are positive.
        assert_ne!(ctx.egraph.find(v0), ctx.egraph.find(v1));
        // Value pick is independent of the PC gate.
        assert_eq!(ctx.egraph.find(chunk.value), ctx.egraph.find(v0));
    }

    #[test]
    fn merge_under_true_pc_allows_fusion() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let p0 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let v0 = ctx.add(Symbolic::Fresh(1));
        let v1 = ctx.add(Symbolic::Fresh(2));
        let true_lit = ctx.add(Symbolic::Lit(Literal::Bool(true)));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p0, v0));
        let merged = heap_union(
            &mut ctx,
            &h1,
            &test_kind(),
            Chunk::new(a, p1, v1),
            &[(true_lit, Polarity::Positive)],
        );
        let _chunk = merged
            .chunk(&test_kind(), ctx.egraph.find(a))
            .expect("merged chunk missing");
        ctx.saturate();

        // PC literal is `true` + both fractions positive → agreement fires.
        assert_eq!(ctx.egraph.find(v0), ctx.egraph.find(v1));
    }

    #[test]
    fn subtract_symbolic_perm_fails_without_proof() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let p_have = ctx.add(Symbolic::Fresh(1));
        let p_take = ctx.add(Symbolic::Fresh(2));
        let v1 = ctx.add(Symbolic::Fresh(3));
        let v2 = ctx.add(Symbolic::Fresh(4));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p_have, v1));
        // Symbolic perms → `have >= take` not provable by equality saturation.
        let err = heap_subtract(
            &mut ctx,
            &h1,
            &test_kind(),
            Chunk::new(a, p_take, v2),
            &[],
            Demand::Concrete,
        )
        .expect_err("symbolic-perm exhale must fail without a proof");
        assert!(matches!(
            err.root_cause(),
            VerifyError::InsufficientPermission
        ));
    }

    #[test]
    fn heap_subtract_canonical_match() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let b = ctx.add(Symbolic::Fresh(1));
        let p2 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(2).into())));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let v1 = ctx.add(Symbolic::Fresh(2));
        let v2 = ctx.add(Symbolic::Fresh(3));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p2, v1));
        ctx.union(a, b);
        ctx.egraph.rebuild();

        let result = heap_subtract(
            &mut ctx,
            &h1,
            &test_kind(),
            Chunk::new(b, p1, v2),
            &[],
            Demand::Concrete,
        )
        .expect("subtract should succeed");

        let canon = ctx.egraph.find(a);
        let chunk = result
            .chunk(&test_kind(), canon)
            .expect("result chunk missing");
        let expected_perm = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        ctx.egraph.rebuild();
        assert_eq!(
            ctx.egraph
                .find(chunk.perm_repr_id().expect("chunk holds no permission")),
            ctx.egraph.find(expected_perm)
        );
        assert_eq!(ctx.egraph.find(v1), ctx.egraph.find(v2));
    }

    #[test]
    fn heap_subtract_exact_match_drops_chunk() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let v1 = ctx.add(Symbolic::Fresh(2));
        let v2 = ctx.add(Symbolic::Fresh(3));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p1, v1));
        let result = heap_subtract(
            &mut ctx,
            &h1,
            &test_kind(),
            Chunk::new(a, p1, v2),
            &[],
            Demand::Concrete,
        )
        .expect("subtract should succeed");

        let canon = ctx.egraph.find(a);
        assert!(
            result.chunk(&test_kind(), canon).is_none(),
            "zero-perm chunk must be dropped"
        );
    }

    #[test]
    fn heap_subtract_over_consume_fails() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let p2 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(2).into())));
        let v1 = ctx.add(Symbolic::Fresh(2));
        let v2 = ctx.add(Symbolic::Fresh(3));

        let h1 = Heap::empty().with_chunk(&test_kind(), Chunk::new(a, p1, v1));
        let err = heap_subtract(
            &mut ctx,
            &h1,
            &test_kind(),
            Chunk::new(a, p2, v2),
            &[],
            Demand::Concrete,
        )
        .expect_err("over-consumption must fail");
        assert!(matches!(
            err.root_cause(),
            VerifyError::InsufficientPermission
        ));
    }

    #[test]
    fn heap_subtract_missing_addr_fails() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let a = ctx.add(Symbolic::Fresh(0));
        let _b = ctx.add(Symbolic::Fresh(1));
        let p1 = ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(1).into())));
        let v1 = ctx.add(Symbolic::Fresh(2));

        let h1 = Heap::empty();
        let err = heap_subtract(
            &mut ctx,
            &h1,
            &test_kind(),
            Chunk::new(a, p1, v1),
            &[],
            Demand::Concrete,
        )
        .expect_err("subtract from empty must fail");
        assert!(matches!(
            err.root_cause(),
            VerifyError::InsufficientPermission
        ));
    }
}
