//! The per-block scratch e-graph: ground plus the block cube, plus what the rules
//! derive from the two. Nothing else may enter it, so its graph is private and
//! the only writes are [`BlockScratch::build`] (clone ground, assume the cube),
//! the mirrors of a ground `add`/`union`, [`BlockScratch::import`] of a ground
//! term, and rule runs. There is deliberately no way to assume a fact in the
//! scratch alone: it would hold on every path of the block
//! (`tests/cases/failing/scratch_drops_guards/`). Reads go through
//! [`BlockScratch::graph`].
//!
//! The scratch records the cube it assumed ([`BlockScratch::assumed`]): anything
//! read off it holds under that cube only. Its ids are [`ScratchId`]s, so a scratch
//! id cannot be handed to ground (or a ground id to the scratch) by accident.

use crate::dhash::HashMap;
use crate::verify::{analysis::ConstFold, lang::Symbolic, rewrite};
use crate::vmir::Polarity;
use egg::Language as _;

type Graph = egg::EGraph<Symbolic, ConstFold>;

/// An e-class id in a [`BlockScratch`]'s id space, never ground's.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct ScratchId(egg::Id);

impl ScratchId {
    /// The plain id, for egg calls on the scratch's [`BlockScratch::graph`] or on
    /// a clone of it (a clone keeps its source's ids).
    pub(super) fn raw(self) -> egg::Id {
        self.0
    }
}

/// A parallel "scratch" e-graph for one method block: a clone of ground taken at
/// build time with the block cube assumed, kept in sync with ground through the
/// `VerifyContext::add`/`VerifyContext::union` hooks. It saturates under the
/// full rule set (independently of ground), so it can discharge every obligation
/// in the block without re-cloning ground per obligation.
///
/// Id-space handling: ids present at build time are identical in both graphs
/// (clone preserves them), so a ground id below `watermark` translates to itself.
/// Ids minted after the build are mirrored through the `add`/`union` hooks into
/// `map`. A ground id at-or-above `watermark` that is *not* in the map (a
/// rule-derived operand, or an unmirrored recipe `build`) is imported on demand
/// from `id_to_node` — the node minted at that exact uncanonical id, never a
/// canonical-class representative.
pub(super) struct BlockScratch {
    egraph: Graph,
    /// The cube assumed at build time, as ground literals: everything the scratch
    /// derives holds only under it.
    assumed: Vec<(egg::Id, Polarity)>,
    /// ground id → scratch id, for mints recorded since the clone.
    map: HashMap<egg::Id, ScratchId>,
    /// Number of ids ground had minted at build time — the id-space boundary.
    watermark: usize,
    /// The `true` literal's ground id at build time — identity-valid in the
    /// scratch. Canonicalize with `find` before comparing.
    true_id: egg::Id,
    /// Set on every mirrored op and import; cleared by a saturation.
    dirty: bool,
    /// As `dirty`, for the cheap reduce-only run used by framing. A saturation
    /// clears both; a reduce clears only this one.
    dirty_reduce: bool,
    /// Applier-memo scope for this graph, resumed by each of its runs (the graph
    /// outlives a run, so its memo must too — see `rewrite`).
    scope: u64,
}

impl BlockScratch {
    /// Clone `ground` and assume `cube`. `true_id`/`false_id` are ground's boolean
    /// literals. A contradictory cube folds to `Inconsistent`, not a panic.
    pub(super) fn build(
        ground: &Graph,
        cube: &[(egg::Id, Polarity)],
        true_id: egg::Id,
        false_id: egg::Id,
    ) -> Self {
        let mut egraph = ground.clone();
        // Id-space boundary = the size of the *id-indexed* node vector, i.e. the
        // number of ids ever minted. Not `total_size()` (= `memo.len()`, which
        // dedups and *shrinks* on a reduce): an under-count there pushes genuine
        // pre-clone ids onto the import path, and importing rebuilds an
        // **unmerged** copy of a class whose ground merges the clone already had.
        let watermark = egraph.nodes().len();
        for (id, pol) in cube {
            let lit = match pol {
                Polarity::Positive => true_id,
                Polarity::Negative => false_id,
            };
            egraph.union(*id, lit);
        }
        egraph.rebuild();
        Self {
            egraph,
            assumed: cube.to_vec(),
            map: HashMap::default(),
            watermark,
            true_id,
            dirty: true,
            dirty_reduce: true,
            scope: rewrite::new_scope_id(),
        }
    }

    /// Read-only view. Writing goes through the methods of this type.
    pub(super) fn graph(&self) -> &Graph {
        &self.egraph
    }

    /// The cube this scratch assumed: what it derives holds under it only.
    pub(super) fn assumed(&self) -> &[(egg::Id, Polarity)] {
        &self.assumed
    }

    /// The `true` literal's class.
    pub(super) fn true_id(&self) -> ScratchId {
        ScratchId(self.true_id)
    }

    pub(super) fn find(&self, id: ScratchId) -> ScratchId {
        ScratchId(self.egraph.find(id.0))
    }

    pub(super) fn watermark(&self) -> usize {
        self.watermark
    }

    pub(super) fn needs_saturate(&self) -> bool {
        self.dirty
    }

    pub(super) fn needs_reduce(&self) -> bool {
        self.dirty_reduce
    }

    /// Translate a **ground** id into the scratch's id space, importing the ground
    /// term if the scratch lacks it. Adds a term, never a fact.
    pub(super) fn import(&mut self, ground: &Graph, g: egg::Id) -> ScratchId {
        // NB: **no** ground `find` here. Ground canonicalization is not
        // meaning-preserving for the scratch: the first union that merges a term
        // into the `true` class makes ground `find` return the `true` leader for it,
        // so canonicalizing first would translate *every* such term to the scratch's
        // `true` — silently turning each mirrored union/assume into `true == true`.
        // The raw id a caller holds is a stable name: egg never reuses an id, and
        // `id_to_node` returns the node minted at it. Ground does not grow only
        // through mirrored ops — `prove_under_pc` saturates it in-block, and recipe
        // builds write to it directly — so ids past the watermark can be
        // unmirrored; the import below handles those.
        if let Some(&s) = self.map.get(&g) {
            return s;
        }
        if usize::from(g) < self.watermark {
            return ScratchId(g);
        }
        // `id_to_node(g)`, the node *minted at that id*: `ground[g].nodes[0]` picks
        // an arbitrary member of `g`'s class, which is `Lit(true)` once any union
        // merged `g` into the `true` class.
        let s = self.add_ground_node(ground, ground.id_to_node(g).clone());
        self.map.insert(g, s);
        s
    }

    /// Add a ground node (its children are ground ids), translating the children.
    fn add_ground_node(&mut self, ground: &Graph, mut node: Symbolic) -> ScratchId {
        for c in node.children_mut() {
            *c = self.import(ground, *c).0;
        }
        self.dirty = true;
        self.dirty_reduce = true;
        ScratchId(self.egraph.add(node))
    }

    /// Repeat a ground `add` of `node`, which produced ground id `ground_id`.
    pub(super) fn mirror_add(&mut self, ground: &Graph, ground_id: egg::Id, node: Symbolic) {
        // Key on the id `add` returned, raw — see the `find` note in `import`.
        if self.map.contains_key(&ground_id) {
            self.dirty = true;
            self.dirty_reduce = true;
            return;
        }
        let s = self.add_ground_node(ground, node);
        self.map.insert(ground_id, s);
    }

    /// Repeat a ground `union(a, b)`. A conflict under the cube folds to
    /// `Inconsistent`, which makes the block's goals vacuously provable.
    pub(super) fn mirror_union(&mut self, ground: &Graph, a: egg::Id, b: egg::Id) {
        let (ta, tb) = (self.import(ground, a), self.import(ground, b));
        self.egraph.union(ta.0, tb.0);
        self.dirty = true;
        self.dirty_reduce = true;
    }

    /// Run the full rule set (`run` returns the graph saturated), in this graph's
    /// memo scope.
    pub(super) fn saturate(&mut self, run: impl FnOnce(Graph) -> Graph) {
        let _scope = rewrite::ScratchScope::resume(self.scope);
        self.egraph = run(std::mem::take(&mut self.egraph));
        self.dirty = false;
        self.dirty_reduce = false;
    }

    /// Run the terminating reductions. Not a saturation: `dirty` stays set.
    pub(super) fn reduce(&mut self, run: impl FnOnce(Graph) -> Graph) {
        let _scope = rewrite::ScratchScope::resume(self.scope);
        self.egraph = run(std::mem::take(&mut self.egraph));
        self.dirty_reduce = false;
    }
}
