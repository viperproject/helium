#![feature(never_type)]
pub mod json;
pub mod peak_memory;
pub mod pipeline;
/// Deterministic hash maps and sets for the whole crate. `std`'s default hasher
/// is seeded per process, so iterating a map (candidate chunks, rule matches,
/// disjuncts) visited entries in a different order on every run, and a few
/// verdicts depended on that order. FxHash is unseeded: the order is a function
/// of the insertions alone.
pub mod dhash {
    pub type HashMap<K, V> = std::collections::HashMap<K, V, rustc_hash::FxBuildHasher>;
    pub type HashSet<K> = std::collections::HashSet<K, rustc_hash::FxBuildHasher>;
}
pub mod translate;
mod util;
pub mod verify;
pub mod viper;
pub mod vmir;
pub use util::*;
pub use viper::viper_parser;
