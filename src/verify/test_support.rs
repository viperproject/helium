//! Shared fixtures for the verifier's unit tests.
//!
//! Lives here rather than in one test module because the heap-algebra tests
//! (`heap::algebra`) and the rewrite/const-fold tests (`declaration`) need the same
//! isolated context and the same literal builders.

use crate::verify::context::VerifyContext;
use crate::verify::heap::LocationKind;
use crate::verify::lang::Symbolic;
use crate::vmir::{self, Bound, Literal, Type};

/// A verification context over an empty program.
///
/// Leaks a `'static` empty allocator, names table and group interner so the
/// returned context can borrow them — fine for a test process, and it keeps the
/// call sites to a single line.
pub(crate) fn fresh_ctx(interner: &lasso::Rodeo) -> VerifyContext<'_> {
    let alloc: &'static mut _ =
        Box::leak(Box::new(crate::verify::func_registry::FuncRegistry::empty()));
    let decls: &'static _ = Box::leak(Box::new(typed_index_collections::TiVec::<
        vmir::MemberId,
        vmir::Declaration,
    >::new()));
    let groups: &'static _ = Box::leak(Box::new(lasso::Rodeo::<lasso::Spur>::new()));
    VerifyContext::new(interner, decls, groups, alloc)
}

/// The rational literal `n/d`.
pub(crate) fn real(ctx: &mut VerifyContext<'_>, n: i64, d: i64) -> egg::Id {
    ctx.add(Symbolic::Lit(Literal::Real(num::BigRational::new(
        num::BigInt::from(n),
        num::BigInt::from(d),
    ))))
}

/// An **unbounded** test location kind — like a predicate, it triggers no
/// bound/non-aliasing axioms, so the merge/subtract unit tests exercise the chunk
/// accounting in isolation.
pub(crate) fn test_kind() -> LocationKind {
    LocationKind {
        group: <lasso::Spur as lasso::Key>::try_from_usize(0).unwrap(),
        value: Type::Int,
        bound: Bound::Unbounded,
    }
}
