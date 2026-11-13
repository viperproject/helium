//! The symbolic [`Heap`] and the [`Region`] it exists on, in a module of their
//! own so that the rest of `heap` (the algebra above all) sees only their
//! functions, never their fields.

use std::rc::Rc;

use super::{Chunk, HeapPc, LocationKind, cube_eq};
use crate::verify::context::VerifyContext;
use crate::vmir::Polarity;

/// Symbolic heap: chunks partitioned by [`LocationKind`]. Each group is an
/// `Rc<[Chunk]>` so cloning a heap (frequent — one per instruction) shares the
/// group slices; mutating one group copies just that slice (copy-on-write). The
/// outer `im::HashMap` is itself structurally shared.
///
/// Within a group, identity is the chunk's `addr` e-class — today at most one
/// chunk per address (the same-address merge happens in `declaration.rs`); the vec
/// shape is a prerequisite for the lazy Σ-ite permission model.
///
/// A chunk is held on `region ∧ guard`: the heap's [`Region`], and relative to it
/// the chunk's own [`Chunk::guard`] (the join merge appends the branch literal).
#[derive(Debug, Clone)]
pub struct Heap {
    /// Unseeded hasher: iteration order (and so the order join/consume code mints
    /// terms in) must be a function of the heap alone, not of the process.
    groups: im::HashMap<LocationKind, Rc<[Chunk]>, rustc_hash::FxBuildHasher>,
    region: Region,
}

/// The path a [`Heap`] exists on: the absolute block cube of the phase it is used
/// in (`fetch_heap` sets it), never a delta pc nor the walker's ambient cube, which
/// is empty in a join phase. The algebra reads only this, through [`Heap::region`].
#[derive(Debug, Clone)]
pub struct Region(HeapPc);

impl Region {
    pub(crate) fn of_cube(cube: Vec<(egg::Id, Polarity)>) -> Self {
        Self(cube.into())
    }

    pub(crate) fn lits(&self) -> &[(egg::Id, Polarity)] {
        &self.0
    }

    pub(crate) fn same(&self, ctx: &VerifyContext<'_>, other: &Region) -> bool {
        cube_eq(ctx, &self.0, &other.0)
    }
}

impl Heap {
    pub fn empty(region: Region) -> Self {
        Self {
            groups: im::HashMap::default(),
            region,
        }
    }

    pub(crate) fn region(&self) -> &Region {
        &self.region
    }

    /// The same contents, used on `region`.
    pub(crate) fn at(&self, region: Region) -> Self {
        Self {
            groups: self.groups.clone(),
            region,
        }
    }

    #[cfg(test)]
    pub(crate) fn top() -> Self {
        Self::empty(Region::of_cube(Vec::new()))
    }

    /// The chunks of one location kind (empty slice if none held).
    pub fn chunks_of(&self, kind: &LocationKind) -> &[Chunk] {
        self.groups.get(kind).map(|r| &r[..]).unwrap_or(&[])
    }

    /// The chunk held at `addr` within `kind`'s group (exact e-class match).
    pub fn chunk(&self, kind: &LocationKind, addr: egg::Id) -> Option<&Chunk> {
        self.chunks_of(kind).iter().find(|c| c.addr == addr)
    }

    /// Insert `chunk` into `kind`'s group, replacing any chunk already at the
    /// same `addr` (preserving today's one-chunk-per-address semantics).
    pub fn with_chunk(&self, kind: &LocationKind, chunk: Chunk) -> Self {
        let mut v: Vec<Chunk> = self
            .groups
            .get(kind)
            .map(|r| r.to_vec())
            .unwrap_or_default();
        match v.iter_mut().find(|c| c.addr == chunk.addr) {
            Some(slot) => *slot = chunk,
            None => v.push(chunk),
        }
        Self {
            groups: self.groups.update(kind.clone(), v.into()),
            region: self.region.clone(),
        }
    }

    /// Drop the chunk at `addr` in `kind`'s group (removing the group if empty).
    pub fn without_chunk(&self, kind: &LocationKind, addr: egg::Id) -> Self {
        let Some(group) = self.groups.get(kind) else {
            return self.clone();
        };
        let v: Vec<Chunk> = group.iter().filter(|c| c.addr != addr).cloned().collect();
        let groups = if v.is_empty() {
            self.groups.without(kind)
        } else {
            self.groups.update(kind.clone(), v.into())
        };
        Self {
            groups,
            region: self.region.clone(),
        }
    }

    /// Every chunk in the heap, paired with its location kind.
    pub fn entries(&self) -> impl Iterator<Item = (&LocationKind, &Chunk)> {
        self.groups
            .iter()
            .flat_map(|(k, cs)| cs.iter().map(move |c| (k, c)))
    }

    /// The location kinds (group keys) present in the heap.
    pub fn kinds(&self) -> impl Iterator<Item = &LocationKind> {
        self.groups.keys()
    }

    /// The chunk at `addr` within `kind`'s group, matched by **canonical**
    /// e-class (`ctx.egraph.find`) so an address that only aliases the stored
    /// key under the currently-assumed context still resolves. Used by the
    /// Stage-4 join merge (a chunk held on both arms must line up by identity).
    pub fn chunk_canon(
        &self,
        ctx: &VerifyContext<'_>,
        kind: &LocationKind,
        addr: egg::Id,
    ) -> Option<&Chunk> {
        let canon = ctx.egraph.find(addr);
        self.chunks_of(kind)
            .iter()
            .find(|c| ctx.egraph.find(c.addr) == canon)
    }
}
