//! The prover: obligation discharge and the method's working e-graph.
//!
//! Split out of `context::mod` — same `VerifyContext`, second `impl` block. This is
//! the piece with real invariants: the working graph is a copy of ground taken once
//! per method, scoped along the method's dominator tree, kept in sync with ground
//! through the `add`/`union` hooks, and its id space is related to ground's by
//! `watermark`/`map`/[`VerifyContext::tr`].
//!
//! `add` and `union` deliberately stay in `mod.rs`: each touches the e-graph and
//! the fixpoint cache (engine state) *and* the working graph (prover state), so the
//! outer type keeps that coordination.

use crate::dhash::HashMap;

use egg::Language as _;

use crate::{
    verify::{
        analysis::ConstFold,
        context::{VerifyContext, graph_inconsistent, run_rules, run_rules_until},
        lang::Symbolic,
        rewrite::{self, ScratchScope},
        stats,
    },
    vmir::{Literal, Polarity},
};

type EGraph = egg::EGraph<Symbolic, ConstFold>;

/// The method's **working graph**: where block obligations are proven.
///
/// It is a copy of ground taken once per method, on the first obligation that
/// needs it, and then kept for the rest of the method. Its contents are organised
/// by the method's dominator tree:
///
/// - the **base** (no scope) holds ground as it was when the copy was taken, plus
///   everything mirrored into it since outside any scope: facts that hold on every
///   path;
/// - a **scope** is opened for a block whose facts must not outlive it (a branch
///   arm: some block walked later is not dominated by it, see
///   [`BlockOrder::needs_scope`]). The block's cube is assumed and its facts are
///   assumed **unguarded** inside the scope; blocks it dominates are walked inside
///   it and see all of that without re-deriving it. Leaving the block's dominator
///   subtree rolls the scope back (an egg checkpoint) and re-asserts, in the
///   enclosing scope, the **guarded** form of every fact recorded on ground while
///   the scope was open — exactly what ground holds about those blocks — so the
///   blocks after a join see the closed arm's facts the way ground shows them.
///
/// Invariant: every fact in the base or an open scope holds on every path to the
/// block being walked. A block without a scope of its own puts its facts into the
/// innermost open scope, which is sound because every block walked while that
/// scope stays open is dominated by it.
///
/// Id-space handling: ids present when the copy was taken are identical in both
/// graphs, so a ground id below `watermark` translates to itself. Ids minted after
/// are mirrored through the `add`/`union` hooks into `map`, or imported on demand by
/// [`VerifyContext::tr`] from `id_to_node` — the node minted at that exact
/// uncanonical id, never a canonical-class representative. A map entry made inside
/// a scope is removed with the scope (its target id is rolled back).
pub(super) struct WorkGraph {
    pub(super) egraph: EGraph,
    /// ground id → working id, for mints since the copy.
    map: HashMap<egg::Id, egg::Id>,
    /// The keys of `map` in insertion order, so a scope can remove its own.
    map_log: Vec<egg::Id>,
    /// Ground id-space size when the copy was taken.
    watermark: usize,
    /// The `true` / `false` literals (same ids in both graphs when the copy was
    /// taken). Canonicalize with `find`.
    pub(super) true_id: egg::Id,
    false_id: egg::Id,
    /// The open scopes, outermost first.
    scopes: Vec<Scope>,
    /// Applier-memo scope of the base; each scope adds its own.
    memo: u64,
    /// Changed since the last saturation / reduction.
    pub(super) dirty: bool,
    pub(super) dirty_reduce: bool,
}

struct Scope {
    checkpoint: egg::Checkpoint,
    /// The block the scope belongs to.
    block: usize,
    map_len: usize,
    /// `VerifyContext::facts` length when the scope was opened.
    facts_len: usize,
    memo: u64,
    dirty: bool,
    dirty_reduce: bool,
}

/// Dominance over a method's blocks, in walk (stored, topological) order, and
/// which blocks the walk has entered so far.
pub(crate) struct BlockOrder {
    /// Preorder number and the last preorder number of the subtree, per block, in
    /// the dominator tree.
    interval: Vec<(usize, usize)>,
    /// Entered blocks, counted by preorder number (a Fenwick tree), so the
    /// entered part of any dominator subtree is a range count.
    entered: Vec<usize>,
    /// Blocks entered so far.
    n_entered: usize,
}

impl BlockOrder {
    /// `preds[b]` are the predecessors of block `b`; every predecessor precedes its
    /// successor.
    pub(crate) fn new(preds: &[Vec<usize>]) -> Self {
        let n = preds.len();
        // Immediate dominators: in a topological order the dominators of a block
        // form a chain ending in the nearest common dominator of its predecessors.
        let mut idom: Vec<Option<usize>> = vec![None; n];
        let mut depth = vec![0usize; n];
        for b in 0..n {
            let mut ps = preds[b].iter().copied();
            let Some(first) = ps.next() else { continue };
            let mut d = first;
            for p in ps {
                let mut p = p;
                while d != p {
                    if depth[d] >= depth[p] && d > p {
                        d = idom[d].expect("non-entry block has a dominator");
                    } else {
                        p = idom[p].expect("non-entry block has a dominator");
                    }
                }
            }
            idom[b] = Some(d);
            depth[b] = depth[d] + 1;
        }
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut roots = Vec::new();
        for (b, d) in idom.iter().enumerate() {
            match *d {
                Some(d) => children[d].push(b),
                None => roots.push(b),
            }
        }
        let mut interval = vec![(0, 0); n];
        let mut next = 0;
        let mut stack: Vec<(usize, bool)> = roots.into_iter().rev().map(|r| (r, false)).collect();
        while let Some((b, done)) = stack.pop() {
            if done {
                interval[b].1 = next - 1;
                continue;
            }
            interval[b].0 = next;
            next += 1;
            stack.push((b, true));
            stack.extend(children[b].iter().rev().map(|&c| (c, false)));
        }
        BlockOrder {
            interval,
            entered: vec![0; n + 1],
            n_entered: 0,
        }
    }

    /// Record that the walk entered `block`. Blocks are entered in walk order,
    /// each once.
    pub(crate) fn enter(&mut self, block: usize) {
        let mut i = self.interval[block].0 + 1;
        while i < self.entered.len() {
            self.entered[i] += 1;
            i += i.isolate_lowest_one();
        }
        self.n_entered += 1;
    }

    /// Entered blocks with preorder number below `pre`.
    fn entered_below(&self, pre: usize) -> usize {
        let (mut i, mut n) = (pre, 0);
        while i > 0 {
            n += self.entered[i];
            i -= i.isolate_lowest_one();
        }
        n
    }

    /// Blocks of `root`'s dominator subtree (`None`: all blocks) not yet entered.
    fn remaining(&self, root: Option<usize>) -> usize {
        match root {
            None => self.interval.len() - self.n_entered,
            Some(r) => {
                let (lo, hi) = self.interval[r];
                (hi + 1 - lo) - (self.entered_below(hi + 1) - self.entered_below(lo))
            }
        }
    }

    /// Whether `a` dominates `b` (reflexive).
    pub(crate) fn dominates(&self, a: usize, b: usize) -> bool {
        let (lo, hi) = self.interval[a];
        (lo..=hi).contains(&self.interval[b].0)
    }

    /// Whether `block`, just entered with the innermost open scope belonging to
    /// `outer` (`None`: the base), needs a scope of its own: whether a block walked
    /// after it would still be walked inside `outer`'s scope without being
    /// dominated by `block`. The blocks `block` dominates all follow it in a
    /// topological order, so the question is whether `outer`'s subtree has more
    /// blocks left to walk than `block`'s own: two range counts.
    fn needs_scope(&self, block: usize, outer: Option<usize>) -> bool {
        let (lo, hi) = self.interval[block];
        self.remaining(outer) > hi - lo
    }
}

impl WorkGraph {
    /// Fast translate: `Some(working id)` for a mapped or pre-copy id; `None` when
    /// a post-copy ground id is unmapped and so must be imported by
    /// [`VerifyContext::tr`].
    fn fast_translate(&self, g: egg::Id) -> Option<egg::Id> {
        if let Some(&s) = self.map.get(&g) {
            Some(s)
        } else if usize::from(g) < self.watermark {
            Some(g)
        } else {
            None
        }
    }

    fn map_insert(&mut self, g: egg::Id, s: egg::Id) {
        self.map.insert(g, s);
        self.map_log.push(g);
    }

    /// Memo scopes for a run on this graph: the base's and every open scope's, so
    /// instances built in an enclosing scope are remembered and those of a closed
    /// one are forgotten with it.
    fn memo_scopes(&self) -> Vec<ScratchScope> {
        std::iter::once(self.memo)
            .chain(self.scopes.iter().map(|s| s.memo))
            .map(ScratchScope::resume)
            .collect()
    }
}

/// How hard a decomposition's working graph has been run since its last
/// assumption — the escalation ladder [`VerifyContext::prove_by_ite_decomposition`]
/// climbs when its chain stalls, cheapest rung first.
///
/// A link itself only ever needs `Rebuilt`; the rungs above it are bought on the
/// evidence of a stall, so a chain that telescopes cleanly runs no rules at all.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Rung {
    /// Congruence and const-fold only (`EGraph::rebuild`).
    Rebuilt,
    /// Plus the terminating structural reductions (`run_reduce`).
    Reduced,
    /// Plus the full rule set (`run_probe`). Nothing above this.
    Saturated,
}

impl<'a> VerifyContext<'a> {
    /// Translate a **ground** id into the working graph's id space. Identity
    /// without a working graph. Fast path (a mapped or pre-copy id) is O(1);
    /// otherwise the ground term at `g` is imported (its children translated,
    /// then re-added).
    pub(super) fn tr(&mut self, g: egg::Id) -> egg::Id {
        // NB: **no** ground `find` here. Ground canonicalization is not
        // meaning-preserving for the working graph: the first union that merges a
        // term into the `true` class makes ground `find` return the `true` leader
        // for it, so canonicalizing first would translate *every* such term to the
        // working graph's `true` — silently turning each mirrored union/assume into
        // `true == true`.
        let Some(w) = self.work.as_ref() else {
            return g;
        };
        // The one exception: the literals. `expr!(ctx, true)` is the canonical id
        // of ground's `true` class, i.e. whichever node the class's root was
        // minted as, and the root moves when a union makes another class the
        // larger one. Translated as that node, it would be a class W need not know
        // is `true`. So the current root of ground's `true` (`false`) class is
        // W's `true` (`false`). Everything in that class holds unconditionally, so
        // this is sound; a term that became the root is learned only as `true`.
        if g == self.egraph.find(g) {
            if g == self.egraph.find(w.true_id) {
                return w.true_id;
            }
            if g == self.egraph.find(w.false_id) {
                return w.false_id;
            }
        }
        if let Some(s) = w.fast_translate(g) {
            return s;
        }
        // Miss: an unmirrored ground mint (recipe `build`s and rule runs add
        // straight to `ctx.egraph`). Import it **faithfully** via `id_to_node(g)`,
        // the node *minted at that id*: `self.egraph[g].nodes[0]` picks an arbitrary
        // member of `g`'s canonical class, which is `Lit(true)` once any union
        // merged `g` into the `true` class.
        let node = self.egraph.id_to_node(g).clone();
        let kids: Vec<egg::Id> = node.children().to_vec();
        let tkids: Vec<egg::Id> = kids.iter().map(|c| self.tr(*c)).collect();
        let mut wnode = node;
        for (slot, tk) in wnode.children_mut().iter_mut().zip(tkids) {
            *slot = tk;
        }
        let w = self.work.as_mut().unwrap();
        let s = w.egraph.add(wnode);
        w.map_insert(g, s);
        w.dirty = true;
        w.dirty_reduce = true;
        s
    }

    /// Whether we are inside a method block's **body** walk (the join phase runs
    /// outside one — its cube's reach boolean is materialized *by* the join).
    pub(crate) fn in_block(&self) -> bool {
        self.in_block
    }

    /// The current block's control cube, as live ids. Empty outside a block body.
    pub(crate) fn current_cube(&self) -> &[(egg::Id, Polarity)] {
        &self.current_cube
    }

    /// Assume `fact` in the working graph, guarded only by the part of `guards` (in
    /// innermost-first fold order) beyond the current block's cube. The working
    /// graph has the cube assumed, so a fact whose guard is the cube holds there
    /// outright (invariant 4 of `design/block-vmir/82-two-egraph-block-model.md`:
    /// ground guarded, working graph unguarded); an intra-block condition stays a
    /// guard, as the fact outlives the instruction in blocks the current one
    /// dominates.
    pub(super) fn work_assume(&mut self, fact: egg::Id, guards: &[(egg::Id, Polarity)]) {
        if self.work.is_none() {
            return;
        }
        let cube = self.current_cube.len();
        let delta = match guards.len().checked_sub(cube) {
            Some(n)
                if guards[n..]
                    .iter()
                    .zip(self.current_cube.iter().rev())
                    .all(|(g, c)| g == c) =>
            {
                &guards[..n]
            }
            _ => guards,
        };
        let delta = delta.to_vec();
        let mut imp = self.tr(fact);
        let true_w = self.work.as_ref().unwrap().true_id;
        for (id, pol) in delta {
            let c = self.tr(id);
            let w = self.work.as_mut().unwrap();
            imp = match pol {
                Polarity::Positive => w.egraph.add(Symbolic::Ite([c, imp, true_w])),
                Polarity::Negative => w.egraph.add(Symbolic::Ite([c, true_w, imp])),
            };
        }
        let w = self.work.as_mut().unwrap();
        w.egraph.union(imp, true_w);
        w.dirty = true;
        w.dirty_reduce = true;
    }

    /// Mirror a ground `add` that minted `node` at the fresh ground id
    /// `ground_id` into the working graph, translating its children through
    /// [`Self::tr`]. Only a fresh id names `node` itself: an `add` that finds the
    /// node already present returns its class's canonical id, which may have been
    /// minted as a different node (see [`Self::tr`]), so it is not mirrored.
    pub(super) fn mirror_add(&mut self, ground_id: egg::Id, node: Symbolic) {
        let kids: Vec<egg::Id> = node.children().to_vec();
        let tkids: Vec<egg::Id> = kids.iter().map(|c| self.tr(*c)).collect();
        let mut wnode = node;
        for (slot, tk) in wnode.children_mut().iter_mut().zip(tkids) {
            *slot = tk;
        }
        let w = self.work.as_mut().unwrap();
        let s = w.egraph.add(wnode);
        w.map_insert(ground_id, s);
        w.dirty = true;
        w.dirty_reduce = true;
    }

    /// Mirror a ground union into the working graph. A same-typed conflict there
    /// folds to `Inconsistent`, not a panic — that just makes the block's goals
    /// vacuously provable.
    pub(super) fn mirror_union(&mut self, a: egg::Id, b: egg::Id) {
        let (ta, tb) = (self.tr(a), self.tr(b));
        let w = self.work.as_mut().unwrap();
        w.egraph.union(ta, tb);
        w.dirty = true;
        w.dirty_reduce = true;
    }

    /// Record that a certificate build unioned each of `released` with `true` in
    /// ground (a recipe build adds to `ctx.egraph` directly, past the `union`
    /// hook): a fact like any other, so the working graph gets it too.
    pub(crate) fn note_released(&mut self, released: &[egg::Id]) {
        if self.work.is_none() {
            return;
        }
        let true_ = expr!(self, true);
        for &id in released {
            self.facts.push((id, true_));
            self.mirror_union(id, true_);
        }
    }

    /// Start walking a method whose blocks are ordered by `order`.
    pub(crate) fn begin_method(&mut self, order: BlockOrder) {
        self.end_method();
        self.blocks = Some(order);
    }

    /// Finish the method: drop the working graph and the block state.
    pub(crate) fn end_method(&mut self) {
        if let Some(w) = self.work.take() {
            rewrite::retire_scope(w.memo);
            w.scopes.iter().for_each(|s| rewrite::retire_scope(s.memo));
        }
        self.blocks = None;
        self.facts.clear();
        self.current_block = None;
        self.current_cube.clear();
        self.in_block = false;
        self.block_dead = false;
    }

    /// Enter block `block`, before its join phase: close the scopes of blocks that
    /// do not dominate it and open its own if it needs one. Obligations of the join
    /// phase are proven on ground (its cube's reach boolean is materialized *by*
    /// the join), but what the join records is mirrored into the working graph.
    pub(crate) fn enter_block(&mut self, block: usize) {
        self.current_block = Some(block);
        self.current_cube.clear();
        self.in_block = false;
        self.block_dead = false;
        self.blocks.as_mut().unwrap().enter(block);
        if self.work.is_none() {
            return;
        }
        // Scopes nest along the dominator tree, so the ones to close are a suffix:
        // everything above the innermost whose block dominates this one. Scanned
        // from the top, so the scan costs what it closes.
        let blocks = self.blocks.as_ref().unwrap();
        let scopes = &self.work.as_ref().unwrap().scopes;
        let mut first = scopes.len();
        while first > 0 && !blocks.dominates(scopes[first - 1].block, block) {
            first -= 1;
        }
        if first < scopes.len() {
            self.close_scopes(first);
        }
        self.open_scope_if_needed(block);
    }

    /// Enter the body of the current block, under its control cube.
    pub(crate) fn begin_block(&mut self, cube: Vec<(egg::Id, Polarity)>) {
        self.block_dead = false;
        self.current_cube = cube;
        self.in_block = true;
        if self.work.is_some() {
            self.assume_cube();
        }
    }

    fn open_scope_if_needed(&mut self, block: usize) {
        let w = self.work.as_ref().unwrap();
        let outer = w.scopes.last().map(|s| s.block);
        if !self.blocks.as_ref().unwrap().needs_scope(block, outer) {
            return;
        }
        let facts_len = self.facts.len();
        let w = self.work.as_mut().unwrap();
        let scope = Scope {
            checkpoint: w.egraph.checkpoint(),
            block,
            map_len: w.map_log.len(),
            facts_len,
            memo: rewrite::new_scope_id(),
            dirty: w.dirty,
            dirty_reduce: w.dirty_reduce,
        };
        w.scopes.push(scope);
        stats::bump(|s| s.work_scopes += 1);
    }

    /// Roll back the scopes from the `first`-th on (it and everything nested in
    /// it) and hand the enclosing one every union recorded on ground while they
    /// were open, in the form ground holds it: a fact of a closed block is guarded
    /// by the block's path condition there, so it is learned as that implication.
    /// Closing nested scopes together hands each fact over once, not once per
    /// level.
    fn close_scopes(&mut self, first: usize) {
        let w = self.work.as_mut().unwrap();
        let mut closed = w.scopes.split_off(first);
        closed.iter().for_each(|s| rewrite::retire_scope(s.memo));
        let scope = closed.swap_remove(0);
        drop(closed);
        w.egraph.rollback(scope.checkpoint);
        for g in w.map_log.drain(scope.map_len..) {
            w.map.remove(&g);
        }
        w.dirty = scope.dirty;
        w.dirty_reduce = scope.dirty_reduce;
        let facts = self.facts[scope.facts_len..].to_vec();
        stats::bump(|s| s.work_replayed_facts += facts.len() as u64);
        for (a, b) in facts {
            self.mirror_union(a, b);
        }
        self.work.as_mut().unwrap().egraph.rebuild();
    }

    /// Assume the current cube in the working graph; a contradiction makes the
    /// block (and everything it dominates) unreachable.
    fn assume_cube(&mut self) {
        let true_g = expr!(self, true);
        let false_g = expr!(self, false);
        let cube = std::mem::take(&mut self.current_cube);
        for (id, pol) in &cube {
            let lit = if matches!(pol, Polarity::Positive) {
                true_g
            } else {
                false_g
            };
            self.mirror_union(*id, lit);
        }
        self.current_cube = cube;
        let w = self.work.as_mut().unwrap();
        w.egraph.rebuild();
        // `pc ⇒ false`, in its cheap *assume* form: a contradiction among the cube
        // literals has surfaced as `true == false`. Recording it here decides the
        // block's reachability once; every later obligation short-circuits at the
        // top of `prove_under_pc`.
        self.block_dead = graph_inconsistent(&w.egraph);
    }

    /// Build the working graph if absent: copy ground, open the current block's
    /// scope if it needs one, assume the block cube.
    pub(super) fn ensure_work(&mut self) {
        if self.work.is_some() {
            return;
        }
        let t_clone = std::time::Instant::now();
        // The literals, minted before the copy so both graphs have them.
        let true_id = expr!(self, true);
        let false_id = expr!(self, false);
        let egraph = self.egraph.clone();
        // Id-space boundary = the size of the *id-indexed* node vector, i.e. the
        // number of ids ever minted. Not `total_size()` (= `memo.len()`, which
        // dedups and *shrinks* on a reduce): an under-count there pushes genuine
        // pre-copy ids onto `tr`'s import path, and importing rebuilds an
        // **unmerged** copy of a class whose ground merges the copy already had.
        let watermark = egraph.nodes().len();
        self.work = Some(WorkGraph {
            egraph,
            map: HashMap::default(),
            map_log: Vec::new(),
            watermark,
            true_id,
            false_id,
            scopes: Vec::new(),
            memo: rewrite::new_scope_id(),
            dirty: true,
            dirty_reduce: true,
        });
        self.facts.clear();
        if let Some(block) = self.current_block {
            self.open_scope_if_needed(block);
        }
        self.assume_cube();
        stats::bump(|s| s.graph_timing.0.scratch_clone += t_clone.elapsed().as_secs_f64());
        stats::bump(|s| s.block_scratch_clones += 1);
    }

    /// Saturate the working graph under the full rule set if it changed since the
    /// last saturation.
    pub(super) fn saturate_work(&mut self) {
        if !self.work.as_ref().is_some_and(|w| w.dirty) {
            return;
        }
        let mut w = self.work.take().expect("working graph");
        let _scopes = w.memo_scopes();
        let t = std::time::Instant::now();
        let before = stats::with_stats(|s| s.sat_iterations);
        w.egraph = self.saturate_flat(std::mem::take(&mut w.egraph));
        stats::bump(|s| s.block_scratch_saturations += 1);
        stats::bump(|s| s.block_scratch_iterations += s.sat_iterations - before);
        w.dirty = false;
        w.dirty_reduce = false;
        stats::bump(|s| s.graph_timing.0.scratch += t.elapsed().as_secs_f64());
        drop(_scopes);
        self.work = Some(w);
    }

    /// Run only the terminating structural reductions on the working graph — what
    /// **framing** needs: address matching wants snapshot towers collapsed, not the
    /// full rule set.
    pub(super) fn reduce_work(&mut self) {
        if !self.work.as_ref().is_some_and(|w| w.dirty_reduce) {
            return;
        }
        let mut w = self.work.take().expect("working graph");
        let _scopes = w.memo_scopes();
        let t = std::time::Instant::now();
        let (egraph, iterations) = run_rules(
            std::mem::take(&mut w.egraph),
            self.static_reduce.iter().chain(self.alloc.rules()),
            None,
        );
        w.egraph = egraph;
        stats::bump(|s| s.record_run(&iterations));
        // A reduce is not a saturation: leave `dirty` set so the next obligation
        // still runs the full rule set.
        w.dirty_reduce = false;
        stats::bump(|s| s.graph_timing.0.scratch += t.elapsed().as_secs_f64());
        drop(_scopes);
        self.work = Some(w);
    }

    /// Prove `pc ⇒ goal` against the live e-graph, escalating through named
    /// tiers (cheapest first) and **memoizing** the result. In execution order,
    /// each named by the stat it bumps:
    ///
    /// 1. `inconsistent` — the graph holds `true == false`, so everything is
    ///    vacuously provable. Two `find`s, hence first: it is both the cheapest
    ///    verdict and the one that makes all further work pointless.
    /// 2. `dead_block` — the current block's control cube is unsatisfiable
    ///    (`pc ⇒ false`), decided when the cube was assumed in the working graph.
    /// 3. `goal_true` — the goal is *unconditionally* true, so the implication
    ///    holds whatever the pc is. Checked before the implication chain is
    ///    built, because building it allocates nodes and const-folding
    ///    obligations (`0 < 1/1`) are the bulk of the stream.
    /// 4. `memo` — the implication itself is already `true` (a prior identical
    ///    obligation merged it, or it is trivial), or sits in `proven_imps`.
    /// 5. `saturate` — saturate the live graph (only **unconditional** facts
    ///    live there) and re-check.
    /// 6. `probe` — in a method block, the working graph; outside one (functions
    ///    and resources, which have no CFG, and join phases), ground under a
    ///    checkpoint with the path condition assumed and saturated.
    /// 7. `ite_decompose` — non-forking `ite`-goal decomposition on that probe,
    ///    see [`Self::prove_by_ite_decomposition`]. The last resort: there is
    ///    **no case split**, so a goal needing genuine reasoning-by-cases over
    ///    an opaque condition is reported unproven.
    ///
    /// In a block these last two are **one** step rather than two: the pc is not
    /// assumed by a loop of its own, it is handed to the decomposition *as* the
    /// implication and telescoped there — see [`Self::prove_in_work`].
    ///
    /// On success the implication is merged with `true` in the live graph so the
    /// next identical obligation hits `memo`. (Tiers 1–5 already have it merged.)
    ///
    /// A PC literal that already folds to the opposite boolean means the path is
    /// unsatisfiable, so the goal holds vacuously and we short-circuit — which also
    /// keeps the union below from making the whole graph contradictory.
    pub(crate) fn prove_under_pc(
        &mut self,
        goal: egg::Id,
        pc_lits: &[(egg::Id, Polarity)],
    ) -> bool {
        stats::bump(|s| s.prove_calls += 1);
        // Inconsistent: the held facts are contradictory (e.g. a field location
        // holds > 1/1 permission), so every goal is vacuously provable. Two
        // `find`s — cheaper than building anything, hence first.
        if self.is_inconsistent() {
            stats::bump(|s| s.prove_inconsistent += 1);
            return true;
        }
        // Dead block: the block's control cube is unsatisfiable, so nothing in it
        // is reachable and all its obligations hold vacuously. Decided once, when
        // the cube was assumed, instead of rediscovered per obligation.
        if self.block_dead {
            stats::bump(|s| s.prove_dead_block += 1);
            return true;
        }
        let true_ = expr!(self, true);
        // `goal_true`: unconditionally true, checked before the implication chain
        // is built (see the tier list above).
        if self.egraph.find(goal) == self.egraph.find(true_) {
            stats::bump(|s| s.prove_goal_true += 1);
            return true;
        }
        let imp = self.implication(goal, pc_lits.iter().rev().copied());

        // `memo`: already true (memoized / trivial). Under `oob_memo` a proven
        // *conditional* obligation lives in `proven_imps` rather than the `true`
        // class, so consult it too.
        if self.egraph.find(imp) == self.egraph.find(true_)
            || (self.oob_memo && self.proven_imps.contains(&self.egraph.find(imp)))
        {
            stats::bump(|s| s.prove_memo += 1);
            return true;
        }

        // `saturate`: saturate the live graph and re-check (no clone). Saturation
        // can also expose a contradiction, so re-check inconsistency too.
        self.saturate();
        if self.is_inconsistent() || self.egraph.find(imp) == self.egraph.find(true_) {
            stats::bump(|s| s.prove_saturate += 1);
            return true;
        }

        // Inside a method block, discharge the `probe` tier against the working
        // graph: it has the block cube assumed and every dominating block's facts
        // unguarded, it is kept in sync with ground, and its `tr` imports any ground
        // operand it is missing.
        if self.in_block {
            let proven = self.prove_in_work(goal, pc_lits);
            if proven {
                // Export under the **full** path condition. `pc_lits` is only the
                // instruction's delta; the block's cube is ambient — assumed by the
                // working graph, absent from `imp`. Recording `imp` would publish a
                // verdict that only holds under the cube as if it held under the
                // delta alone, and for an empty delta that is the bare goal: a dead
                // arm's `assert false` merged `false` with `true` in ground and
                // poisoned every later block. Guarded by the cube it is exactly the
                // fact the arm licenses (`¬b ⇒ false`), true even where the cube is
                // unsat.
                let guards: Vec<_> = self.current_cube.iter().chain(pc_lits).copied().collect();
                let full_imp = if self.current_cube.is_empty() {
                    imp
                } else {
                    self.implication(goal, guards.iter().rev().copied())
                };
                self.record_proven(full_imp, guards.is_empty());
            }
            return proven;
        }

        // Outside a block: probe ground itself under a checkpoint, with the pc
        // assumed. The probe never touches live state (its every change is rolled
        // back), so it needs no copy of the graph.
        stats::bump(|s| s.prove_probe += 1);
        let pc_lits = pc_lits.to_vec();
        let proven = self.probe_ground(|ctx, mut probe| {
            // Tier 3 shortcut: if every PC literal already carries its required
            // polarity in the just-saturated live graph, assuming the PC adds
            // nothing — skip straight to the `ite_decompose` tier.
            let all_known = pc_lits.iter().all(|(id, pol)| {
                matches!(
                    probe[*id].data.known(),
                    Some(Literal::Bool(b)) if *b == matches!(pol, Polarity::Positive)
                )
            });
            if all_known {
                return ctx.prove_by_ite_decomposition(probe, goal, Rung::Saturated);
            }
            let true_p = probe.add(Symbolic::Lit(Literal::Bool(true)));
            let false_p = probe.add(Symbolic::Lit(Literal::Bool(false)));
            for (id, pol) in &pc_lits {
                let want_true = matches!(pol, Polarity::Positive);
                match probe[*id].data.known() {
                    Some(Literal::Bool(b)) if *b != want_true => {
                        // PC literal contradicts its required polarity → off-path,
                        // so `pc ⇒ goal` is vacuously true. (Guard also avoids a
                        // `true == false` ConstFold conflict from the union below.)
                        return (true, probe);
                    }
                    _ => {
                        probe.union(*id, if want_true { true_p } else { false_p });
                    }
                }
            }
            let probe = ctx.run_probe_until(probe, goal);
            // Last resort is the non-forking ite-goal decomposition; there is no
            // case split beyond it.
            if probe.find(goal) == probe.find(true_p) {
                return (true, probe);
            }
            ctx.prove_by_ite_decomposition(probe, goal, Rung::Saturated)
        });
        if !proven {
            stats::bump(|s| s.prove_probe -= 1);
        }

        // Persist the result so future identical obligations hit the `memo` tier.
        if proven {
            self.record_proven(imp, pc_lits.is_empty());
        }
        proven
    }

    /// Run `f` on ground under a checkpoint and roll everything it did back:
    /// the graph, and the fixpoint cache. `f` receives ground by value (it may run
    /// rules on it) and returns it. The working graph is detached for the extent,
    /// so nothing `f` does is mirrored into it.
    pub(crate) fn probe_ground<T>(
        &mut self,
        f: impl FnOnce(&mut Self, EGraph) -> (T, EGraph),
    ) -> T {
        let clean = self.clean;
        let work = self.work.take();
        let mut ground = std::mem::take(&mut self.egraph);
        let checkpoint = ground.checkpoint();
        let (out, mut ground) = f(self, ground);
        ground.rollback(checkpoint);
        self.egraph = ground;
        self.work = work;
        self.clean = clean;
        out
    }

    /// Canonical ids of `ids` in ground with `pc_lits` assumed and the full rule set
    /// run, or `None` when a literal of `pc_lits` is already decided against its
    /// polarity (the path is unsatisfiable). Ground itself is left untouched.
    pub(crate) fn classes_under_pc(
        &mut self,
        pc_lits: &[(egg::Id, Polarity)],
        ids: &[egg::Id],
    ) -> Option<Vec<egg::Id>> {
        self.probe_ground(|ctx, mut probe| {
            for (id, pol) in pc_lits {
                let want_true = matches!(pol, Polarity::Positive);
                if matches!(probe[*id].data.known(), Some(Literal::Bool(b)) if *b != want_true) {
                    return (None, probe);
                }
                let lit = probe.add(Symbolic::Lit(Literal::Bool(want_true)));
                probe.union(*id, lit);
            }
            probe.rebuild();
            let probe = ctx.run_probe(probe);
            let out = ids.iter().map(|&i| probe.find(i)).collect();
            (Some(out), probe)
        })
    }

    /// The `probe` tier against the working graph. It already has the block cube
    /// assumed and is saturated on demand, so an obligation whose pc is fully
    /// implied by the cube is discharged with no per-obligation probe (a "free
    /// hit"). Obligations carrying extra pc literals (a perm-`Select` branch
    /// condition) hand `pc ⇒ goal` — the extra literals only — to
    /// [`Self::prove_by_ite_decomposition`] under a checkpoint of the working graph,
    /// which assumes them by telescoping the chain.
    pub(super) fn prove_in_work(&mut self, goal: egg::Id, pc_lits: &[(egg::Id, Polarity)]) -> bool {
        self.ensure_work();
        // Assuming the cube is what decides `pc ⇒ false`; if the cube is
        // unsatisfiable there is nothing to translate or run.
        if self.block_dead {
            return true;
        }
        // Translate goal + pc into working space first: `tr` may *import* ground
        // operands, which dirties the graph — so import before the runs below.
        let tg = self.tr(goal);
        let tpc: Vec<(egg::Id, Polarity)> =
            pc_lits.iter().map(|(id, p)| (self.tr(*id), *p)).collect();
        // Try the cheap reductions first and saturate only if the goal is still
        // open: with the cube assumed *unguarded* the `ite` guards no longer
        // throttle function-unfold / axiom-trigger / forall cascades, so a full run
        // can grow the graph a lot for a goal the reductions already close.
        self.reduce_work();
        if self.work_decided(tg) {
            return true;
        }
        self.saturate_work();
        if self.work_decided(tg) {
            return true;
        }
        let w = self.work.as_ref().expect("working graph");
        let all_sat = tpc.iter().all(|(tid, pol)| {
            matches!(
                w.egraph[w.egraph.find(*tid)].data.known(),
                Some(Literal::Bool(b)) if *b == matches!(pol, Polarity::Positive)
            )
        });
        // Off-path: a literal that already folds against its required polarity
        // makes `pc ⇒ goal` vacuously true. O(k) folds, no allocation.
        let off_path = tpc.iter().any(|(tid, pol)| {
            matches!(
                w.egraph[w.egraph.find(*tid)].data.known(),
                Some(Literal::Bool(b)) if *b != matches!(pol, Polarity::Positive)
            )
        });
        if all_sat {
            stats::bump(|s| s.block_scratch_freehits += 1);
        } else if off_path {
            return true;
        }
        self.probe_work(|ctx, mut probe| {
            if all_sat {
                return ctx.prove_by_ite_decomposition(probe, tg, Rung::Saturated);
            }
            // Extra literals: rather than assume them and saturate, hand the
            // *implication* to the decomposition and let it telescope them — one
            // reduce per literal instead of one saturation for the batch. The chain
            // is built exactly as `VerifyContext::implication` builds the ground
            // one, but in working space, so the nodes die with the probe;
            // `prove_under_pc` still memoizes the ground `imp`.
            let true_p = probe.add(Symbolic::Lit(Literal::Bool(true)));
            let mut imp = tg;
            for (id, pol) in tpc.iter().rev() {
                imp = match pol {
                    Polarity::Positive => probe.add(Symbolic::Ite([*id, imp, true_p])),
                    Polarity::Negative => probe.add(Symbolic::Ite([*id, true_p, imp])),
                };
            }
            probe.rebuild();
            // The working graph arrives saturated and the chain assumes nothing
            // until its first pick, so no saturation is owed up front.
            ctx.prove_by_ite_decomposition(probe, imp, Rung::Saturated)
        })
    }

    /// Whether the (reduced or saturated) working graph already settles the block:
    /// a contradiction kills the block, a goal in `true`'s class is a free hit.
    fn work_decided(&mut self, tg: egg::Id) -> bool {
        let w = self.work.as_ref().expect("working graph");
        // A cube that only contradicts once run still kills the block, and
        // latching it here spares every later obligation the same discovery.
        if graph_inconsistent(&w.egraph) {
            self.block_dead = true;
            return true;
        }
        if w.egraph.find(tg) == w.egraph.find(w.true_id) {
            stats::bump(|s| s.block_scratch_freehits += 1);
            return true;
        }
        false
    }

    /// Run `f` on the working graph under a checkpoint and roll back what it did.
    fn probe_work<T>(&mut self, f: impl FnOnce(&mut Self, EGraph) -> (T, EGraph)) -> T {
        let mut w = self.work.take().expect("working graph");
        let scopes = w.memo_scopes();
        let mut probe = std::mem::take(&mut w.egraph);
        let checkpoint = probe.checkpoint();
        let (out, mut probe) = f(self, probe);
        probe.rollback(checkpoint);
        w.egraph = probe;
        drop(scopes);
        self.work = Some(w);
        out
    }

    /// Persist a proven obligation so future identical ones hit the `memo` tier.
    ///
    /// Default (and always for an **empty-pc** goal, where `imp == goal`): union
    /// `imp` with `true`. That path is *productive* — a proven `Eq`/discriminator
    /// goal must collapse its argument classes via `eq-true-union` /
    /// `contra-congruence`.
    ///
    /// Under `oob_memo`, a **conditional** obligation (`imp` is an
    /// `ite(pc.., goal, true)` chain) is instead recorded out of band: unioning it
    /// drags the whole chain permanently into the `true` class (the measured #1
    /// growth driver) when the verdict alone is what the memo needs. The cost is
    /// losing auto-propagation of `goal` once its pc lands unconditionally.
    fn record_proven(&mut self, imp: egg::Id, pc_empty: bool) {
        if self.oob_memo && !pc_empty {
            let canon = self.egraph.find(imp);
            self.proven_imps.insert(canon);
        } else {
            let true_ = expr!(self, true);
            self.union(imp, true_);
            self.egraph.rebuild();
        }
    }

    /// Tier 3.5 — **non-forking `ite`-goal decomposition** on `work`, a graph the
    /// caller rolls back afterwards (it is handed back with the verdict). When the
    /// goal's class holds an `ite` with a *`true` constant arm*, that arm's world is
    /// already discharged, so the goal reduces to proving the **other** arm under
    /// the corresponding condition polarity — **one** probe, never a fork:
    ///
    /// - `ite(c, true, e)  ⟸  e` proven under `¬c`
    /// - `ite(c, e, true)  ⟸  e` proven under `c`   (this is `c ⟹ e`, i.e. a
    ///   guarded fact / implication under a branch)
    ///
    /// This *assumes* the one condition and re-saturates the single surviving arm —
    /// half of a case split, with the condition read off the goal rather than
    /// searched — and unlike a split it never forks, so one arm must already be
    /// discharged. It then loops on the surviving arm, so a nested guard chain
    /// `c₁ ⟹ c₂ ⟹ … ⟹ φ` telescopes one assumption per iteration.
    ///
    /// The `false`-constant shapes fall out of the same loop: `ite(c, false, true)`
    /// leaves `false` as the surviving arm under `c`, which is discharged exactly
    /// when assuming `c` **refutes itself** — checked here as the graph turning
    /// inconsistent. That is weaker than asking saturation to derive `¬c` outright,
    /// and it is how a negated goal (`!(x == 0)` under `0 < x`, the spelling Prusti
    /// emits for MIR asserts) reaches its path-condition contradiction.
    ///
    /// Terminates without a depth cap: each iteration assumes one
    /// *previously-unknown* condition and the e-graph has finitely many, which the
    /// `assumed` set makes explicit. `SILVER_OXIDE_NO_ITE_DECOMPOSE=1` disables it.
    ///
    /// **Cost model.** A link only needs the *terminating* reductions: assuming
    /// `cᵢ` const-folds the `ite` and the next link is the surviving arm's own
    /// class, so the loop runs [`Self::run_reduce`] per pick and escalates to a
    /// full [`Self::run_probe`] only when the chain **stalls** — no candidate in
    /// the goal class. That is where rules are genuinely needed: a guarded fact
    /// that reaches the goal class through an axiom or a quantifier instantiation,
    /// or an arm whose `true` only shows after saturation. Saturations are then
    /// bounded by *stalls*, not by picks (one, for a chain that telescopes
    /// cleanly), where the previous shape paid one per pick.
    ///
    /// `entry` says how hard `work` has already been run; every assumption drops
    /// it back to [`Rung::Rebuilt`], since a union invalidates the fixpoint.
    fn prove_by_ite_decomposition(
        &mut self,
        mut work: EGraph,
        goal: egg::Id,
        entry: Rung,
    ) -> (bool, EGraph) {
        if std::env::var_os("SILVER_OXIDE_NO_ITE_DECOMPOSE").is_some() {
            return (false, work);
        }
        let mut goal = goal;
        let mut rung = entry;
        let mut assumed: crate::dhash::HashSet<egg::Id> = crate::dhash::HashSet::default();
        loop {
            // The assumptions accumulated along this chain are contradictory, so
            // the surviving arm is unreachable and the goal holds vacuously. This
            // is what closes the `ite(c, false, true)` shape — `¬c` need not be
            // *derivable*, it is enough that assuming `c` refutes itself, which is
            // strictly weaker than asking saturation to produce `¬c` outright.
            if graph_inconsistent(&work) {
                stats::bump(|s| s.prove_ite_decompose += 1);
                return (true, work);
            }
            let g = work.find(goal);
            if Self::known_bool_class(&work, g, true) {
                stats::bump(|s| s.prove_ite_decompose += 1);
                return (true, work);
            }
            // Pick a `true`-constant-arm ite: the surviving arm is the *other*
            // branch, to be proven under the condition that reaches it.
            //
            // A class routinely holds several such nodes, and picking the wrong
            // one dead-ends the chain — the class of `!(0 < d)` under a negated
            // branch guard holds both `ite(c, true, self)`, whose surviving arm is
            // the goal class itself, and the useful `ite(c', goal', true)`. So
            // candidates that cannot make progress are skipped rather than taken
            // and abandoned: a surviving arm equal to the current goal is a no-op,
            // and a condition already fixed on this chain would re-saturate an
            // identical graph. Within what is left, the positive implication
            // `c ⟹ e` (the as-written direction) is preferred over its negated
            // dual.
            let usable = |work: &EGraph,
                          assumed: &crate::dhash::HashSet<egg::Id>,
                          cond: egg::Id,
                          branch: egg::Id| {
                work.find(branch) != g && !assumed.contains(&work.find(cond))
            };
            let mut plan: Option<(egg::Id, bool, egg::Id)> = None;
            for node in &work[g].nodes {
                let Symbolic::Ite([c, x, y]) = node else {
                    continue;
                };
                let (c, x, y) = (work.find(*c), work.find(*x), work.find(*y));
                // ite(c, e, true) ⟸ e under c
                if Self::known_bool_class(&work, y, true) && usable(&work, &assumed, c, x) {
                    plan = Some((c, true, x));
                    break;
                }
                // ite(c, true, e) ⟸ e under ¬c — the negated dual, taken only if
                // no positive candidate in this class works out.
                if Self::known_bool_class(&work, x, true)
                    && usable(&work, &assumed, c, y)
                    && plan.is_none()
                {
                    plan = Some((c, false, y));
                }
            }
            let Some((cond, want, branch)) = plan else {
                // Stalled: no usable candidate in the goal class. Escalate one
                // rung and retry — the missing link may be one the reductions
                // collapse, or one only the full rule set produces (a guarded fact
                // carried in by an axiom or a quantifier instantiation). Rules are
                // spent here, on evidence that the cheap rung was not enough, and
                // never on a link that did not need them.
                match rung {
                    Rung::Rebuilt => {
                        work = self.run_reduce(work);
                        rung = Rung::Reduced;
                    }
                    Rung::Reduced => {
                        work = self.run_probe_until(work, goal);
                        rung = Rung::Saturated;
                    }
                    Rung::Saturated => return (false, work),
                }
                continue;
            };
            // If the condition already can't take `want`, the constant-`true`
            // arm is the only reachable one — the goal holds outright.
            if Self::known_bool_class(&work, cond, !want) {
                stats::bump(|s| s.prove_ite_decompose += 1);
                return (true, work);
            }
            // Progress guard: assuming a condition already assumed on this chain
            // would re-saturate an identical graph — give up instead of looping.
            if !assumed.insert(work.find(cond)) {
                return (false, work);
            }
            let lit = work.add(Symbolic::Lit(Literal::Bool(want)));
            work.union(cond, lit);
            // A rebuild is the whole cost of a link. `ConstFold` folds an `Ite`
            // whose condition is a known bool by propagating the taken arm's data
            // (see `analysis`), rebuild propagates that to parents, and the next
            // pick is a scan of the surviving arm's own class — no rule derives
            // either. Anything more is bought at the stall above, not here.
            work.rebuild();
            rung = Rung::Rebuilt;
            goal = branch;
        }
    }

    /// Whether e-class `id` folds to the boolean literal `b` in `probe`.
    fn known_bool_class(probe: &EGraph, id: egg::Id, b: bool) -> bool {
        matches!(probe[probe.find(id)].data.known(), Some(Literal::Bool(v)) if *v == b)
    }

    /// Which held chunk addresses coincide with `addr` in the **working graph** —
    /// the graph that already has this block's cube assumed. `None` outside a block
    /// (functions and resources have no CFG); the caller then falls back to a
    /// ground probe.
    ///
    /// This is the pc-alias question [`pc_alias_partners`](crate::verify::heap::algebra::pc_alias_partners)
    /// asks, answered by `find` instead of by a fresh saturation.
    pub(crate) fn scratch_alias_partners(
        &mut self,
        addr: egg::Id,
        chunk_addrs: &[egg::Id],
    ) -> Option<Vec<egg::Id>> {
        if !self.in_block {
            return None;
        }
        self.ensure_work();
        if self.block_dead {
            return None;
        }
        // Translate everything first: `tr` can import nodes and dirty the graph.
        let ta = self.tr(addr);
        let tcs: Vec<egg::Id> = chunk_addrs.iter().map(|c| self.tr(*c)).collect();
        self.saturate_work();
        let ground = self.egraph.find(addr);
        let w = self.work.as_ref().unwrap();
        let canon = w.egraph.find(ta);
        let mut out = Vec::new();
        for (c, tc) in chunk_addrs.iter().zip(tcs) {
            if w.egraph.find(tc) == canon && self.egraph.find(*c) != ground {
                out.push(*c);
            }
        }
        Some(out)
    }

    /// Whether `pc_lits` *is* the current block's control cube — the precondition
    /// for reading an answer off the working graph, which has assumed exactly that
    /// cube.
    pub(crate) fn cube_matches(&self, pc_lits: &[(egg::Id, Polarity)]) -> bool {
        self.current_cube.len() == pc_lits.len()
            && self
                .current_cube
                .iter()
                .zip(pc_lits)
                .all(|(a, b)| self.egraph.find(a.0) == self.egraph.find(b.0) && a.1 == b.1)
    }

    /// Saturate a detached probe e-graph with the full rule set, inside a
    /// scratch memo scope (its instantiations die with the probe).
    pub(crate) fn run_probe(&mut self, probe: EGraph) -> EGraph {
        let _scope = ScratchScope::enter();
        let t = std::time::Instant::now();
        let iters_before = stats::with_stats(|s| s.sat_iterations);
        let out = self.saturate_flat(probe);
        stats::bump(|s| s.graph_timing.0.probe += t.elapsed().as_secs_f64());
        stats::bump(|s| s.probe_saturations += 1);
        stats::bump(|s| s.probe_iterations += s.sat_iterations - iters_before);
        out
    }

    /// As [`Self::run_probe`], but stopping as soon as `goal` is settled — for a
    /// probe whose whole purpose is that one question. Sound only because the graph
    /// is rolled back afterwards; see `run_rules_until`.
    pub(crate) fn run_probe_until(&mut self, probe: EGraph, goal: egg::Id) -> EGraph {
        let _scope = ScratchScope::enter();
        let t = std::time::Instant::now();
        let iters_before = stats::with_stats(|s| s.sat_iterations);
        let (out, iterations) = run_rules_until(
            probe,
            self.static_rules
                .iter()
                .chain(self.alloc.rules())
                .chain(self.axiom_rules.iter()),
            None,
            Some(goal),
        );
        stats::bump(|s| s.record_run(&iterations));
        stats::bump(|s| s.graph_timing.0.probe += t.elapsed().as_secs_f64());
        stats::bump(|s| s.probe_saturations += 1);
        stats::bump(|s| s.probe_iterations += s.sat_iterations - iters_before);
        out
    }

    /// Run only the terminating structural reductions over a probe — the `reduce`
    /// counterpart of [`Self::run_probe`], driving the same rule set as
    /// [`VerifyContext::reduce`] and [`Self::reduce_work`].
    ///
    /// This is what an `ite_decompose` link costs between picks: assuming a
    /// condition makes the surviving arm's `ite` collapse by const-fold plus these
    /// reductions, which is all the *telescope* needs — the full rule set is only
    /// required when the chain stalls (see [`Self::prove_by_ite_decomposition`]).
    fn run_reduce(&mut self, probe: EGraph) -> EGraph {
        let _scope = ScratchScope::enter();
        let t = std::time::Instant::now();
        let (out, iterations) = run_rules(
            probe,
            self.static_reduce.iter().chain(self.alloc.rules()),
            None,
        );
        stats::bump(|s| s.graph_timing.0.probe += t.elapsed().as_secs_f64());
        stats::bump(|s| s.probe_reduces += 1);
        stats::bump(|s| s.record_run(&iterations));
        out
    }

    /// Run `f` against a throwaway state of the live graph: everything `f` does to
    /// ground (and its fixpoint cache, and `proven_imps`) is rolled back afterwards.
    /// The whole extent is a scratch memo scope.
    ///
    /// The working graph is **detached** for the extent: `f`'s ground is a
    /// throwaway whose ids are rolled back afterwards, so mirroring would leave
    /// `map` entries keyed by ids that cease to mean anything, and a stale entry
    /// makes `tr` hand back an unrelated class. Detached, `f`'s obligations take
    /// the non-block path (ground probes).
    pub(crate) fn with_scratch_graph<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let _scope = ScratchScope::enter();
        let clean = self.clean;
        let proven_imps = self.proven_imps.clone();
        let work = self.work.take();
        let in_block = std::mem::replace(&mut self.in_block, false);
        let checkpoint = self.egraph.checkpoint();
        let out = f(self);
        self.egraph.rollback(checkpoint);
        self.in_block = in_block;
        self.work = work;
        self.clean = clean;
        self.proven_imps = proven_imps;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::BlockOrder;

    /// `0 → {1, 2} → 3 → {4, 5}` (two diamonds in sequence), plus a block 6
    /// reached from 4 and 5 only through 3's dominator subtree.
    fn two_diamonds() -> BlockOrder {
        let preds = vec![
            vec![],
            vec![0],
            vec![0],
            vec![1, 2],
            vec![3],
            vec![3],
            vec![4, 5],
        ];
        BlockOrder::new(&preds)
    }

    #[test]
    fn dominance_follows_the_diamonds() {
        let o = two_diamonds();
        assert!(o.dominates(0, 6) && o.dominates(3, 6) && o.dominates(3, 3));
        assert!(!o.dominates(1, 3) && !o.dominates(2, 3));
        assert!(!o.dominates(4, 6) && !o.dominates(1, 2));
    }

    #[test]
    fn only_arms_need_a_scope() {
        let mut o = two_diamonds();
        // Walked in order with the base as the enclosing scope: the entry and the
        // joins dominate everything walked after them, so their facts hold on
        // every later path; an arm does not dominate its sibling or its join (the
        // last arm still does not dominate the join after it).
        let expect = [false, true, true, false, true, true, false];
        for (b, &want) in expect.iter().enumerate() {
            o.enter(b);
            assert_eq!(o.needs_scope(b, None), want, "block {b}");
        }
    }

    /// Inside an arm's scope only what that arm's subtree still has to walk
    /// counts: a nested diamond's arms need scopes, the arm's own join does not.
    #[test]
    fn nested_arms_count_the_enclosing_subtree() {
        // 0 -> {1, 5}; 1 -> {2, 3} -> 4; {4, 5} -> 6
        let preds = vec![
            vec![],
            vec![0],
            vec![1],
            vec![1],
            vec![2, 3],
            vec![0],
            vec![4, 5],
        ];
        let mut o = BlockOrder::new(&preds);
        o.enter(0);
        o.enter(1);
        assert!(o.needs_scope(1, None));
        o.enter(2);
        assert!(o.needs_scope(2, Some(1)));
        o.enter(3);
        assert!(o.needs_scope(3, Some(1)));
        o.enter(4);
        assert!(!o.needs_scope(4, Some(1)));
        o.enter(5);
        assert!(o.needs_scope(5, None));
        o.enter(6);
        assert!(!o.needs_scope(6, None));
    }
}

#[cfg(test)]
mod working_graph_tests {
    use super::BlockOrder;
    use crate::verify::lang::Symbolic;
    use crate::vmir::{Literal, Polarity};

    /// Ground's `true` class translates to W's `true` even once the class's root
    /// is a node W does not know is true: here ground learns `x` outside the
    /// hooks (as a rule run would), its root moves, and a re-`add` of `x` returns
    /// that root after the scope the working graph was built in has closed.
    #[test]
    fn true_stays_true_in_the_working_graph() {
        let interner = lasso::Rodeo::new();
        let mut ctx = crate::verify::test_support::fresh_ctx(&interner);
        let c = ctx.add(Symbolic::Fresh(0));
        let x = ctx.add(Symbolic::Fresh(1));
        let t = ctx.add(Symbolic::Lit(Literal::Bool(true)));
        // 0 -> {1, 2} -> 3
        ctx.begin_method(BlockOrder::new(&[vec![], vec![0], vec![0], vec![1, 2]]));
        ctx.enter_block(0);
        ctx.begin_block(vec![]);
        ctx.enter_block(1);
        ctx.begin_block(vec![(c, Polarity::Positive)]);
        ctx.ensure_work();
        ctx.egraph.union(x, t);
        ctx.egraph.rebuild();
        ctx.enter_block(2);
        ctx.begin_block(vec![(c, Polarity::Negative)]);
        let again = ctx.add(Symbolic::Fresh(1));
        assert_eq!(ctx.egraph.find(again), ctx.egraph.find(t));
        for id in [again, ctx.egraph.find(t)] {
            let tw = ctx.tr(id);
            let w = ctx.work.as_ref().unwrap();
            assert_eq!(w.egraph.find(tw), w.egraph.find(w.true_id));
        }
        // The cube of block 2 (`!c`) reached W's `false`, not some other class.
        let tc = ctx.tr(c);
        let w = ctx.work.as_ref().unwrap();
        assert_eq!(w.egraph.find(tc), w.egraph.find(w.false_id));
    }
}
