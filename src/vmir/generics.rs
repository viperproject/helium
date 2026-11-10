//! Generic-parameter bookkeeping for domain function signatures: renumbering the
//! parameters a signature actually mentions to `0..k`.

use crate::vmir::Type;

/// Substitution arguments mapping each used generic `order[k]` to `Generic(k)`.
/// Unused indices never occur, so any filler is fine.
pub(crate) fn compaction(order: &[usize], declared: usize) -> Vec<Type> {
    let mut args = vec![Type::Generic(0); declared];
    for (new, &old) in order.iter().enumerate() {
        args[old] = Type::Generic(new);
    }
    args
}
