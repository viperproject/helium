//! The prover: obligation discharge and the per-block scratch e-graph.
//!
//! Split out of `context::mod` — same `VerifyContext`, second `impl` block, no
//! type or signature changes. This is the piece with real invariants: the
//! scratch is a clone of ground kept in sync through the `add`/`union` hooks, and
//! its id space is related to ground's by `watermark`/`map`/[`VerifyContext::tr`].
//!
//! `add` and `union` deliberately stay in `mod.rs`: each touches the e-graph and
//! the fixpoint cache (engine state) *and* the scratch (prover state), so the
//! outer type keeps that coordination.

use crate::dhash::HashMap;

use egg::Language as _;

use crate::{
    verify::{
        analysis::ConstFold,
        context::{VerifyContext, graph_inconsistent, run_rules, run_rules_until},
        lang::Symbolic,
        rewrite::{self},
        stats,
    },
    vmir::{Literal, Polarity},
};

/// A parallel "scratch" e-graph for one method block: a clone of ground taken at
/// build time with the block cube assumed, kept in sync with ground through the
/// [`VerifyContext::add`]/[`VerifyContext::union`] hooks. It saturates under the
/// full rule set (independently of ground), so it can discharge every obligation
/// in the block without re-cloning ground per obligation.
///
/// Id-space handling: ids present at build time are identical in both graphs
/// (clone preserves them), so a ground id below `watermark` translates to itself.
/// Ids minted after the build are mirrored through the `add`/`union` hooks into
/// `map`. A ground id at-or-above `watermark` that is *not* in the map (a
/// rule-derived operand, or an unmirrored recipe `build`) is imported on demand by
/// [`VerifyContext::tr`] from `id_to_node` — the node minted at that exact
/// uncanonical id, never a canonical-class representative.
pub(super) struct BlockScratch {
    pub(super) egraph: egg::EGraph<Symbolic, ConstFold>,
    /// ground id → scratch id, for mints recorded since the clone.
    pub(super) map: HashMap<egg::Id, egg::Id>,
    /// Ground e-node count (`total_size`) at build time — the id-space boundary:
    /// a ground id `< watermark` existed in the clone (identity-valid in the
    /// scratch). `total_size` only ever under-counts after a rebuild, which is
    /// safe here (a pre-build id then just gets re-imported, a no-op via
    /// hash-consing).
    pub(super) watermark: usize,
    /// The `true` literal's ground id at build time — identity-valid in the
    /// scratch. Canonicalize with `find` before comparing (saturation merges it
    /// into a larger class).
    pub(super) true_id: egg::Id,
    /// Set on every mirrored `add`/`union` and every import; cleared by a
    /// saturation. Avoids re-saturating an unchanged scratch across consecutive
    /// obligations.
    pub(super) dirty: bool,
    /// Applier-memo scope for **this scratch graph**, resumed by each of its runs.
    /// The graph outlives a single run, so its memo must too — see the scoping
    /// notes in `rewrite`. A fresh scope per run would forget instances that are
    /// still in the graph and rebuild them on every obligation.
    pub(super) scope: u64,
    /// As `dirty`, but for the cheap reduce-only run used by **framing**
    /// ([`VerifyContext::reduce_scratch`]). A full saturation subsumes a reduce, so
    /// it clears both; a reduce clears only this one — otherwise every framing
    /// lookup would force the next obligation to re-saturate from scratch.
    pub(super) dirty_reduce: bool,
}

impl BlockScratch {
    /// Fast translate: `Some(scratch id)` for a mapped or pre-build id; `None`
    /// when a post-build ground id is unmapped and so must be imported by
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
    /// Translate a **ground** id into the current block scratch's id space.
    /// Identity outside a scratch. Fast path (a mapped or pre-build id) is O(1);
    /// otherwise the ground term at `g` is imported into the scratch (extract a
    /// representative e-node, translate its children, re-add), so the scratch is
    /// a valid clone base even when a lazy/structural ground has surfaced
    /// rule-derived ids as operands.
    pub(super) fn tr(&mut self, g: egg::Id) -> egg::Id {
        // NB: **no** ground `find` here. Ground canonicalization is not
        // meaning-preserving for the scratch: the first union that merges a term
        // into the `true` class makes ground `find` return the `true` leader for it,
        // so canonicalizing first would translate *every* such term to the scratch's
        // `true` — silently turning each mirrored union/assume into `true == true`.
        // Under invariant 3 (ground never saturates/reduces in-block) ground grows
        // only through mirrored ops, so the raw id a caller holds is a stable name.
        match self.scratch.as_ref() {
            None => return g,
            Some(sc) => {
                if let Some(s) = sc.fast_translate(g) {
                    return s;
                }
            }
        }
        // Miss: an unmirrored ground mint — recipe `build`s add straight to
        // `ctx.egraph`, bypassing the `add`/`union` hooks. Import it **faithfully**
        // via `id_to_node(g)`, the node *minted at that id*: `self.egraph[g].nodes[0]`
        // picks an arbitrary member of `g`'s canonical class, which is `Lit(true)`
        // once any union merged `g` into the `true` class.

        let node = self.egraph.id_to_node(g).clone();
        let kids: Vec<egg::Id> = node.children().to_vec();
        let tkids: Vec<egg::Id> = kids.iter().map(|c| self.tr(*c)).collect();
        let mut snode = node;
        for (slot, tk) in snode.children_mut().iter_mut().zip(tkids) {
            *slot = tk;
        }
        let sc = self.scratch.as_mut().unwrap();
        let s = sc.egraph.add(snode);
        sc.map.insert(g, s);
        sc.dirty = true;
        sc.dirty_reduce = true;
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

    /// Assume `fact` **unguarded** in the block scratch (invariant 4 of
    /// `design/block-vmir/82-two-egraph-block-model.md`): the scratch already bakes in
    /// the block PC and only ever discharges that block's goals, so a Viper
    /// `assume`/`inhale` fact holds there outright — no need to make `ite-reduce`
    /// release it from under a guard first. Ground keeps the PC-guarded implication, so
    /// the fact cannot leak to a sibling path — see [`Self::assume_guarded`].
    pub(super) fn scratch_assume_unguarded(&mut self, fact: egg::Id) {
        if self.scratch.is_none() {
            return;
        }
        let tf = self.tr(fact);
        let sc = self.scratch.as_mut().unwrap();
        let true_s = sc.true_id;
        sc.egraph.union(tf, true_s);
        sc.dirty = true;
        sc.dirty_reduce = true;
    }

    /// Mirror a ground `add` of `node` (which produced ground id `ground_id`)
    /// into the block scratch, translating its children through [`Self::tr`].
    /// Idempotent per ground class.
    pub(super) fn mirror_add(&mut self, ground_id: egg::Id, node: Symbolic) {
        // Key on the id `add` returned, raw — see the `find` note in [`Self::tr`].
        let key = ground_id;
        if self.scratch.as_ref().unwrap().map.contains_key(&key) {
            let sc = self.scratch.as_mut().unwrap();
            sc.dirty = true;
            sc.dirty_reduce = true;
            return;
        }

        let kids: Vec<egg::Id> = node.children().to_vec();
        let tkids: Vec<egg::Id> = kids.iter().map(|c| self.tr(*c)).collect();
        let mut snode = node;
        for (slot, tk) in snode.children_mut().iter_mut().zip(tkids) {
            *slot = tk;
        }
        let sc = self.scratch.as_mut().unwrap();
        let s = sc.egraph.add(snode);
        sc.map.insert(key, s);
        sc.dirty = true;
        sc.dirty_reduce = true;
    }

    /// Prove `pc ⇒ goal` against the live e-graph, escalating through named
    /// tiers (cheapest first) and **memoizing** the result. In execution order,
    /// each named by the stat it bumps:
    ///
    /// 1. `inconsistent` — the graph holds `true == false`, so everything is
    ///    vacuously provable. Two `find`s, hence first: it is both the cheapest
    ///    verdict and the one that makes all further work pointless.
    /// 2. `dead_block` — the current block's control cube is unsatisfiable
    ///    (`pc ⇒ false`), decided once when its scratch was built.
    /// 3. `goal_true` — the goal is *unconditionally* true, so the implication
    ///    holds whatever the pc is. Checked before the implication chain is
    ///    built, because building it allocates nodes and const-folding
    ///    obligations (`0 < 1/1`) are the bulk of the stream.
    /// 4. `memo` — the implication itself is already `true` (a prior identical
    ///    obligation merged it, or it is trivial).
    /// 5. `saturate` — saturate the live graph (only **unconditional** facts
    ///    live there) and re-check. No clone.
    /// 6. `probe` — the only tier that clones: the block scratch, or (functions
    ///    and resources, which have no CFG) a ground clone with the path condition
    ///    assumed and saturated.
    /// 7. `ite_decompose` — non-forking `ite`-goal decomposition on that probe,
    ///    see [`Self::prove_by_ite_decomposition`]. The last resort: there is
    ///    **no case split**, so a goal needing genuine reasoning-by-cases over
    ///    an opaque condition is reported unproven.
    ///
    /// In a block these last two are **one** step rather than two: the pc is not
    /// assumed by a loop of its own, it is handed to the decomposition *as* the
    /// implication and telescoped there — see [`Self::prove_via_scratch`].
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
        // the scratch was built, instead of rediscovered per obligation.
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

        // `memo`: already true (memoized / trivial).
        if self.egraph.find(imp) == self.egraph.find(true_) {
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

        // Inside a method block, discharge `probe`-tier proving against the per-block scratch
        // graph (reused across the block's obligations) instead of cloning ground
        // per obligation: it is warm, cube-assumed and kept in sync with ground, and
        // its `tr` imports any ground operand it is missing.
        //
        // Functions/resources have no CFG — a whole-body scratch would just equal
        // ground — so they keep the per-obligation clone path below.
        if self.in_block {
            let proven = self.prove_via_scratch(goal, pc_lits);
            if proven {
                // Export under the **full** path condition. `pc_lits` is only the
                // instruction's delta; the block's cube is ambient — assumed by the
                // scratch, absent from `imp`. Recording `imp` would publish a verdict
                // that only holds under the cube as if it held under the delta alone,
                // and for an empty delta that is the bare goal: a dead arm's
                // `assert false` merged `false` with `true` in ground and poisoned
                // every later block. Guarded by the cube it is exactly the fact the
                // arm licenses (`¬b ⇒ false`), true even where the cube is unsat.
                let cube = std::mem::take(&mut self.current_cube);
                let full_imp = if cube.is_empty() {
                    imp
                } else {
                    let lits: Vec<_> = cube.iter().chain(pc_lits).rev().copied().collect();
                    self.implication(goal, lits.into_iter())
                };
                self.current_cube = cube;
                self.record_proven(full_imp, true_);
            }
            return proven;
        }

        // Tier 3 shortcut: if every PC literal already carries its required polarity
        // in the just-saturated live graph, assuming the PC adds nothing — skip
        // straight to the `ite_decompose` tier. Only functions/resources reach
        // here, and their `saturate` tier always ran the full rule set, so the premise
        // (the live graph is fully saturated) holds.
        if pc_lits.iter().all(|(id, pol)| {
            matches!(
                self.egraph[*id].data.known(),
                Some(Literal::Bool(b)) if *b == matches!(pol, Polarity::Positive)
            )
        }) {
            let probe = self.egraph.clone();
            let proven = self.prove_by_ite_decomposition(&probe, goal, Rung::Saturated);
            if proven {
                self.record_proven(imp, true_);
            }
            return proven;
        }

        // Tier 3: clone, assume the path condition, saturate the clone, check.
        let mut probe = self.egraph.clone();
        let true_p = probe.add(Symbolic::Lit(Literal::Bool(true)));
        let false_p = probe.add(Symbolic::Lit(Literal::Bool(false)));
        let mut unsat_pc = false;
        for (id, pol) in pc_lits {
            let want_true = matches!(pol, Polarity::Positive);
            match probe[*id].data.known() {
                Some(Literal::Bool(b)) if *b != want_true => {
                    // PC literal contradicts its required polarity → off-path,
                    // so `pc ⇒ goal` is vacuously true. (Guard also avoids a
                    // `true == false` ConstFold conflict from the union below.)
                    unsat_pc = true;
                    break;
                }
                _ => {
                    probe.union(*id, if want_true { true_p } else { false_p });
                }
            }
        }
        stats::bump(|s| s.prove_probe += 1);
        let proven = if unsat_pc {
            true
        } else {
            let probe = self.run_probe_until(probe, goal);
            // Last resort is the non-forking ite-goal decomposition; there is no
            // case split beyond it.
            probe.find(goal) == probe.find(true_p)
                || self.prove_by_ite_decomposition(&probe, goal, Rung::Saturated)
        };

        // Persist the result so future identical obligations hit the `memo` tier.
        if proven {
            self.record_proven(imp, true_);
        }
        proven
    }

    /// Enter a method block: record its control cube (shared pc of all its
    /// insts) and drop any previous block's scratch (sibling cubes are mutually
    /// exclusive, so it cannot be reused). The scratch itself is built lazily, on
    /// the block's first `probe`-tier obligation — most blocks never reach the `probe` tier, and
    /// building at entry measured 1.8x slower on `structs_enums`.
    ///
    pub(crate) fn begin_block(&mut self, cube: Vec<(egg::Id, Polarity)>) {
        self.scratch = None;
        self.block_dead = false;
        self.current_cube = cube;
        self.in_block = true;
    }

    /// Leave the block walk (after the last block, or before a non-block walk):
    /// discard the scratch and clear block state.
    pub(crate) fn end_block(&mut self) {
        self.current_cube.clear();
        self.in_block = false;
        self.scratch = None;
        self.block_dead = false;
    }

    /// Build the block scratch if absent: clone ground, assume the block cube.
    /// Ids present now are identical in both graphs (clone preserves them), so
    /// the translation map starts empty.
    pub(super) fn ensure_scratch(&mut self) {
        if self.scratch.is_some() {
            return;
        }
        let true_id = expr!(self, true);
        let false_id = expr!(self, false);
        let cube = std::mem::take(&mut self.current_cube);
        let t_clone = std::time::Instant::now();
        let mut egraph = self.egraph.clone();
        // Id-space boundary = the size of the *id-indexed* node vector, i.e. the
        // number of ids ever minted. Not `total_size()` (= `memo.len()`, which
        // dedups and *shrinks* on a reduce): an under-count there pushes genuine
        // pre-clone ids onto `tr`'s import path, and importing rebuilds an
        // **unmerged** copy of a class whose ground merges the clone already had.
        let watermark = egraph.nodes().len();
        for (id, pol) in &cube {
            let lit = if matches!(pol, Polarity::Positive) {
                true_id
            } else {
                false_id
            };
            // Same-typed conflict ⇒ `Inconsistent` (not a panic): a contradictory
            // cube marks the block path infeasible, so its goals hold vacuously.
            egraph.union(*id, lit);
        }
        let t_rebuild = std::time::Instant::now();
        egraph.rebuild();
        // `pc ⇒ false`, in its cheap *assume* form: the cube literals are already
        // merged with their polarities, so a contradiction among them has surfaced
        // as `true == false`. Recording it here decides the block's reachability
        // once; every later obligation short-circuits at the top of
        // `prove_under_pc`. (The implication form would be strictly weaker —
        // const-fold does not collapse the `ite` chain of an unsatisfiable pc.)
        self.block_dead = graph_inconsistent(&egraph);
        if std::env::var_os("SILVER_OXIDE_TRACE_SCRATCH").is_some() {
            eprintln!(
                "[scratch-build] ground {}n/{}c ids {} cube {} | clone+union {:?} rebuild {:?}",
                self.egraph.total_number_of_nodes(),
                self.egraph.number_of_classes(),
                watermark,
                cube.len(),
                t_clone.elapsed() - t_rebuild.elapsed(),
                t_rebuild.elapsed(),
            );
        }
        self.current_cube = cube;
        self.scratch = Some(BlockScratch {
            egraph,
            map: HashMap::default(),
            watermark,
            true_id,
            dirty: true,
            dirty_reduce: true,
            scope: rewrite::new_scope_id(),
        });
        stats::bump(|s| s.graph_timing.0.scratch_clone += t_clone.elapsed().as_secs_f64());
        stats::bump(|s| s.block_scratch_clones += 1);
    }

    /// Saturate the block scratch under the full rule set if it changed since the
    /// last saturation. Counted separately from the live/probe saturations.
    pub(super) fn saturate_scratch(&mut self) {
        if !self.scratch.as_ref().is_some_and(|s| s.dirty) {
            return;
        }
        let mut sc = self.scratch.take().expect("scratch live");
        let _scope = rewrite::ScratchScope::resume(sc.scope);
        let _t = std::time::Instant::now();
        let before = stats::with_stats(|s| s.sat_iterations);
        let (n0, c0) = (
            sc.egraph.total_number_of_nodes(),
            sc.egraph.number_of_classes(),
        );
        sc.egraph = self.saturate_flat(sc.egraph);
        if std::env::var_os("SILVER_OXIDE_TRACE_SCRATCH").is_some() {
            eprintln!(
                "[scratch-sat] {n0}n/{c0}c -> {}n/{}c true={} (ground {}n/{}c true={}, {} iters)",
                sc.egraph.total_number_of_nodes(),
                sc.egraph.number_of_classes(),
                {
                    let t = sc.egraph.find(sc.true_id);
                    sc.egraph[t].nodes.len()
                },
                self.egraph.total_number_of_nodes(),
                self.egraph.number_of_classes(),
                {
                    let t = self.egraph.find(self.true_id_cached());
                    self.egraph[t].nodes.len()
                },
                stats::with_stats(|s| s.sat_iterations) - before,
            );
        }
        stats::bump(|s| s.block_scratch_saturations += 1);
        stats::bump(|s| s.block_scratch_iterations += s.sat_iterations - before);
        sc.dirty = false;
        sc.dirty_reduce = false;
        stats::bump(|s| s.graph_timing.0.scratch += _t.elapsed().as_secs_f64());
        self.scratch = Some(sc);
    }

    /// Run only the terminating structural reductions on the block scratch — the
    /// scratch counterpart of [`Self::reduce`], and what **framing** needs: address
    /// matching wants snapshot towers collapsed, not the full rule set. Keeping this
    /// separate from [`Self::saturate_scratch`] is what stops every heap lookup from
    /// paying a full saturation (the in-block equivalent of ground's cheap `reduce`).
    pub(super) fn reduce_scratch(&mut self) {
        if !self.scratch.as_ref().is_some_and(|s| s.dirty_reduce) {
            return;
        }
        let mut sc = self.scratch.take().expect("scratch live");
        let _scope = rewrite::ScratchScope::resume(sc.scope);
        let _t = std::time::Instant::now();
        let egraph = std::mem::take(&mut sc.egraph);
        let (n0, c0) = (egraph.total_number_of_nodes(), egraph.number_of_classes());
        let (egraph, iterations) = run_rules(
            egraph,
            self.static_reduce.iter().chain(self.alloc.rules()),
            None,
        );
        sc.egraph = egraph;
        if std::env::var_os("SILVER_OXIDE_TRACE_SCRATCH").is_some() {
            eprintln!(
                "[scratch-red] {n0}n/{c0}c -> {}n/{}c  (ground {}n/{}c)",
                sc.egraph.total_number_of_nodes(),
                sc.egraph.number_of_classes(),
                self.egraph.total_number_of_nodes(),
                self.egraph.number_of_classes(),
            );
        }
        stats::bump(|s| s.record_run(&iterations));
        // A reduce is not a saturation: leave `dirty` set so the next obligation
        // still runs the full rule set.
        sc.dirty_reduce = false;
        stats::bump(|s| s.graph_timing.0.scratch += _t.elapsed().as_secs_f64());
        self.scratch = Some(sc);
    }

    /// The `probe` tier against the per-block scratch. The scratch already has the block
    /// cube assumed and saturated, so an obligation whose pc is fully implied by
    /// the cube is discharged with no per-obligation clone (a "free hit").
    /// Obligations carrying extra pc literals (a perm-`Select` branch condition)
    /// clone the *warm* scratch and hand `pc ⇒ goal` — the extra literals only — to
    /// [`Self::prove_by_ite_decomposition`], which assumes them by telescoping the
    /// chain. Assuming them here in a loop and saturating would be the same work
    /// with a full saturation where a reduce per literal does.
    pub(super) fn prove_via_scratch(
        &mut self,
        goal: egg::Id,
        pc_lits: &[(egg::Id, Polarity)],
    ) -> bool {
        // Ground size at the moment the `probe` tier is reached, before the scratch is built or
        // touched — paired below with the scratch size the obligation actually
        // reasons over (`SILVER_OXIDE_TRACE_SCRATCH`).
        let trace = std::env::var_os("SILVER_OXIDE_TRACE_SCRATCH").is_some();
        let g0 = if trace {
            let t = self.egraph.find(self.true_id_cached());
            (
                self.egraph.total_number_of_nodes(),
                self.egraph.number_of_classes(),
                self.egraph[t].nodes.len(),
            )
        } else {
            (0, 0, 0)
        };
        let fresh = self.scratch.is_none();
        self.ensure_scratch();
        // Building the scratch is what decides `pc ⇒ false`; if the cube is
        // unsatisfiable there is nothing to translate or run.
        if self.block_dead {
            return true;
        }
        // Translate goal + pc into scratch space first: `tr` may *import* ground
        // operands (a lazy/structural ground surfaces rule-canonical leaders),
        // which dirties the scratch — so import before the saturation below.
        let tg = self.tr(goal);
        let tpc: Vec<(egg::Id, Polarity)> =
            pc_lits.iter().map(|(id, p)| (self.tr(*id), *p)).collect();
        // Tier the scratch the way the ground path is tiered: try the cheap
        // reductions first and saturate only if the goal is still open. Saturating
        // unconditionally is what made the scratch expensive — with the block PC
        // assumed *unguarded* (invariant 4) the `ite` guards no longer throttle
        // function-unfold / axiom-trigger / forall cascades, so a full run can grow
        // the graph 20-30× for a goal the reductions already close.
        self.reduce_scratch();
        {
            let sc = self.scratch.as_ref().expect("scratch live");
            // A cube that only contradicts once reduced still kills the block, and
            // latching it here spares every later obligation the same discovery.
            if graph_inconsistent(&sc.egraph) {
                self.block_dead = true;
                return true;
            }
            if sc.egraph.find(tg) == sc.egraph.find(sc.true_id) {
                stats::bump(|s| s.block_scratch_freehits += 1);
                if trace {
                    self.trace_probe(g0, fresh, "reduce");
                }
                return true;
            }
        }
        self.saturate_scratch();
        if trace {
            self.trace_probe(g0, fresh, "saturate");
        }

        // A contradictory cube (or a mirrored union that conflicts under it)
        // makes every goal vacuously provable — and kills the rest of the block.
        if graph_inconsistent(&self.scratch.as_ref().expect("scratch live").egraph) {
            self.block_dead = true;
            return true;
        }
        let sc = self.scratch.as_ref().expect("scratch live");
        if sc.egraph.find(tg) == sc.egraph.find(sc.true_id) {
            stats::bump(|s| s.block_scratch_freehits += 1);
            return true;
        }
        let all_sat = tpc.iter().all(|(tid, pol)| {
            matches!(
                sc.egraph[sc.egraph.find(*tid)].data.known(),
                Some(Literal::Bool(b)) if *b == matches!(pol, Polarity::Positive)
            )
        });
        let probe_base = sc.egraph.clone();

        // Free hit: the cube already implies the pc — no extra assumption needed,
        // go straight to the goal-structural decomposition on the warm scratch.
        if all_sat {
            stats::bump(|s| s.block_scratch_freehits += 1);
            return self.prove_by_ite_decomposition(&probe_base, tg, Rung::Saturated);
        }

        // Off-path: a literal that already folds against its required polarity
        // makes `pc ⇒ goal` vacuously true. O(k) folds, no allocation — kept ahead
        // of the chain below, which would only reach that literal after
        // telescoping the ones outside it.
        if tpc.iter().any(|(tid, pol)| {
            matches!(
                probe_base[probe_base.find(*tid)].data.known(),
                Some(Literal::Bool(b)) if *b != matches!(pol, Polarity::Positive)
            )
        }) {
            return true;
        }

        // Extra literals: rather than assume them and saturate, hand the
        // *implication* to the decomposition and let it telescope them — that is
        // the same work, one reduce per literal instead of one saturation for the
        // batch, and it shares the machinery that already had to handle a guarded
        // goal. The chain is built exactly as `VerifyContext::implication` builds
        // the ground one, but in scratch space, so the nodes die with the clone;
        // `prove_under_pc` still memoizes the ground `imp`.
        let mut probe = probe_base;
        let true_p = probe.add(Symbolic::Lit(Literal::Bool(true)));
        let mut imp = tg;
        for (id, pol) in tpc.iter().rev() {
            imp = match pol {
                Polarity::Positive => probe.add(Symbolic::Ite([*id, imp, true_p])),
                Polarity::Negative => probe.add(Symbolic::Ite([*id, true_p, imp])),
            };
        }
        probe.rebuild();
        // The scratch arrives saturated and the chain assumes nothing until its
        // first pick, so no saturation is owed up front.
        self.prove_by_ite_decomposition(&probe, imp, Rung::Saturated)
    }

    /// Persist a proven obligation so future identical ones hit the `memo` tier:
    /// union `imp` with `true`. This is also *productive* — a proven
    /// `Eq`/discriminator goal collapses its argument classes via `eq-true-union`
    /// / `contra-congruence`.
    fn record_proven(&mut self, imp: egg::Id, true_: egg::Id) {
        self.union(imp, true_);
        self.egraph.rebuild();
    }

    /// Tier 3.5 — **non-forking `ite`-goal decomposition**. When the goal's
    /// class holds an `ite` with a *`true` constant arm*, that arm's world is
    /// already discharged, so the goal reduces to proving the **other** arm
    /// under the corresponding condition polarity — **one** probe, never a fork:
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
    /// `assumed` set makes explicit.
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
    /// `entry` says how hard `probe` has already been run; every assumption drops
    /// the working graph back to [`Rung::Rebuilt`], since a union invalidates the
    /// fixpoint.
    fn prove_by_ite_decomposition(
        &mut self,
        probe: &egg::EGraph<Symbolic, ConstFold>,
        goal: egg::Id,
        entry: Rung,
    ) -> bool {
        // One working graph threaded across the chain, so its ids stay stable
        // and `assumed` (a set of condition classes) is a sound progress guard.
        let mut work = probe.clone();
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
                return true;
            }
            let g = work.find(goal);
            if Self::known_bool_class(&work, g, true) {
                stats::bump(|s| s.prove_ite_decompose += 1);
                return true;
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
            let usable = |work: &egg::EGraph<Symbolic, ConstFold>,
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
                    Rung::Saturated => return false,
                }
                continue;
            };
            // If the condition already can't take `want`, the constant-`true`
            // arm is the only reachable one — the goal holds outright.
            if Self::known_bool_class(&work, cond, !want) {
                stats::bump(|s| s.prove_ite_decompose += 1);
                return true;
            }
            // Progress guard: assuming a condition already assumed on this chain
            // would re-saturate an identical graph — give up instead of looping.
            if !assumed.insert(work.find(cond)) {
                return false;
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
    fn known_bool_class(probe: &egg::EGraph<Symbolic, ConstFold>, id: egg::Id, b: bool) -> bool {
        matches!(probe[probe.find(id)].data.known(), Some(Literal::Bool(v)) if *v == b)
    }

    /// Which held chunk addresses coincide with `addr` in the **block scratch** —
    /// the graph that already has this block's cube assumed and saturated, reused
    /// across the block's obligations. `None` outside a block (functions and
    /// resources have no CFG, so no scratch); the caller then falls back to its own
    /// clone-and-probe.
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
        self.ensure_scratch();
        if self.block_dead {
            return None;
        }
        // Translate everything first: `tr` can import nodes and dirty the scratch.
        let ta = self.tr(addr);
        let tcs: Vec<egg::Id> = chunk_addrs.iter().map(|c| self.tr(*c)).collect();
        self.saturate_scratch();
        let ground = self.egraph.find(addr);
        let sc = self.scratch.as_ref().unwrap();
        let canon = sc.egraph.find(ta);
        let mut out = Vec::new();
        for (c, tc) in chunk_addrs.iter().zip(tcs) {
            if sc.egraph.find(tc) == canon && self.egraph.find(*c) != ground {
                out.push(*c);
            }
        }
        Some(out)
    }

    /// Whether `pc_lits` *is* the current block's control cube — the precondition
    /// for reading an answer off the scratch, which has assumed exactly that cube.
    pub(crate) fn cube_matches(&self, pc_lits: &[(egg::Id, Polarity)]) -> bool {
        self.current_cube.len() == pc_lits.len()
            && self
                .current_cube
                .iter()
                .zip(pc_lits)
                .all(|(a, b)| self.egraph.find(a.0) == self.egraph.find(b.0) && a.1 == b.1)
    }

    /// Saturate a detached probe e-graph with the full rule set, inside a
    /// scratch memo scope (its instantiations die with the probe; the live
    /// base memo lets it skip rebuilding every already-live instance).
    pub(crate) fn run_probe(
        &mut self,
        probe: egg::EGraph<Symbolic, ConstFold>,
    ) -> egg::EGraph<Symbolic, ConstFold> {
        let _scope = crate::verify::rewrite::ScratchScope::enter();
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
    /// is a throwaway; see `run_rules_until`.
    pub(crate) fn run_probe_until(
        &mut self,
        probe: egg::EGraph<Symbolic, ConstFold>,
        goal: egg::Id,
    ) -> egg::EGraph<Symbolic, ConstFold> {
        let _scope = crate::verify::rewrite::ScratchScope::enter();
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

    /// Run only the terminating structural reductions over a detached probe —
    /// the `reduce` counterpart of [`Self::run_probe`], driving the same rule set
    /// as [`VerifyContext::reduce`] and [`Self::reduce_scratch`].
    ///
    /// This is what an `ite_decompose` link costs between picks: assuming a
    /// condition makes the surviving arm's `ite` collapse by const-fold plus these
    /// reductions, which is all the *telescope* needs — the full rule set is only
    /// required when the chain stalls (see [`Self::prove_by_ite_decomposition`]).
    fn run_reduce(
        &mut self,
        probe: egg::EGraph<Symbolic, ConstFold>,
    ) -> egg::EGraph<Symbolic, ConstFold> {
        let _scope = crate::verify::rewrite::ScratchScope::enter();
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

    /// Run `f` with `self.egraph` swapped for a scratch clone of the live
    /// graph, restoring the live graph — and its fixpoint cache, which `f`'s
    /// scratch runs would otherwise clobber — afterwards. The whole extent is
    /// a scratch memo scope.
    ///
    /// The block scratch is **detached** for the extent: inside `f` the "ground"
    /// graph is a throwaway whose ids are restored away afterwards, so mirroring
    /// would leave `map` entries keyed by ids that cease to mean anything, and a
    /// stale entry makes `tr` hand back an unrelated class. Detached, `f`'s
    /// obligations take the non-block path (saturate + clone the throwaway).
    pub(crate) fn with_scratch_graph<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let _scope = crate::verify::rewrite::ScratchScope::enter();
        let live = self.egraph.clone();
        let clean = self.clean;
        let scratch = self.scratch.take();
        let in_block = std::mem::replace(&mut self.in_block, false);
        let out = f(self);
        self.in_block = in_block;
        self.scratch = scratch;
        self.egraph = live;
        self.clean = clean;
        out
    }
}
