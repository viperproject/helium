pub(crate) mod prove;

use prove::BlockScratch;

use crate::dhash::HashMap;

use crate::{
    verify::{
        analysis::ConstFold,
        cert::FunctionDefinition,
        func_registry::FuncRegistry,
        lang::{FuncId, Symbolic},
        rewrite, stats,
    },
    vmir::{Declaration, Literal, MemberId, Polarity, Type},
};
use lasso::{Rodeo, Spur};
use typed_index_collections::TiVec;

/// Display name for a member id, off a bare `(interner, decls)` pair — the same
/// resolution [`VerifyContext::member_name`] does, callable where only the two
/// shared refs are at hand (e.g. under a `&mut ctx.alloc` borrow).
pub(crate) fn member_name_in(
    interner: &Rodeo,
    decls: &TiVec<MemberId, Declaration>,
    m: MemberId,
) -> String {
    if usize::from(m) < decls.len() {
        interner.resolve(&decls[m].name()).to_string()
    } else {
        format!("d{}", m.0)
    }
}

pub(crate) struct VerifyContext<'a> {
    pub(crate) egraph: egg::EGraph<Symbolic, ConstFold>,
    /// Static structural rules. The ADT cons/proj/tag reductions are pulled from
    /// the [`FuncRegistry`] at saturation time (it grows as ADT concepts are
    /// minted).
    static_rules: Vec<egg::Rewrite<Symbolic, ConstFold>>,
    /// Terminating structural reductions, run after heap-producing ops to
    /// normalize (collapse snapshot towers) without a full saturation.
    static_reduce: Vec<egg::Rewrite<Symbolic, ConstFold>>,
    /// Per-unit lazy rules, minted by `assume_axioms`: the single quantifier
    /// instantiation rule (`rewrite::forall_rule` — quantifiers are e-nodes;
    /// ground axioms are pre-added to the graph instead) and one unfold rule
    /// per verified function body (one per `fn_certs` entry — see
    /// `rewrite::function_rule`).
    /// Chained into full saturation (incl. the `probe` tier) but not `reduce`.
    pub(crate) axiom_rules: Vec<egg::Rewrite<Symbolic, ConstFold>>,
    /// Monotonic source of symbolic ids — both `Symbolic::Fresh(n)` and
    /// `Symbolic::Wildcard(n)`. A plain counter now that nothing mints either at
    /// saturation time: both certificate kinds are add-only recipes, so `transplant`
    /// (which threaded a shared counter) is gone, and a slot permission keeps its
    /// own `PermRecipe` steps rather than flattening a wildcard into a pure step.
    ///
    /// Shared between the two node kinds on purpose. They could not collide anyway
    /// (different constructors, and only `Fresh` is keyed into `fresh_types`), but one
    /// counter means no id is ever reused across them — so folding `Wildcard` into
    /// `Fresh` later needs no id migration.
    fresh_counter: u32,
    /// Cheap string repr for member/constructor names.
    pub(crate) interner: &'a Rodeo,
    /// Member names indexed by `MemberId` (for `member_name`/`func_name`).
    pub(crate) decls: &'a TiVec<MemberId, Declaration>,
    /// Location group tags (`Type::Addr.group`), for display resolution.
    pub(crate) groups: &'a Rodeo<Spur>,
    /// Shared verifier function-id registry (ADT cons/proj/tag ids + rules and
    /// builtin operators).
    /// Owned by `verify::verify`, threaded `&mut` through each unit so ids stay
    /// consistent across certificate grafts.
    pub(crate) alloc: &'a mut FuncRegistry,
    /// Type side-oracle: the irreducible type sources that the type-free
    /// e-graph nodes no longer carry. Keyed by stable node payloads (the
    /// `Fresh` counter and the `FuncApp` member id), so no union upkeep is
    /// needed — the visualization reads them directly to reconstruct types.
    pub(crate) fresh_types: HashMap<u32, Type>,
    pub(crate) func_ret_types: HashMap<FuncId, Type>,
    /// Verified non-recursive function bodies, keyed by `MemberId`. `assume_axioms`
    /// reads this to install one lazy unfold rule per entry into `axiom_rules`
    /// (see `rewrite::function_rule`). `None` in isolated contexts (unit tests)
    /// that never evaluate a `FunctionCall`.
    pub(crate) fn_certs: Option<&'a HashMap<MemberId, std::sync::Arc<FunctionDefinition>>>,
    /// The certificate recipe under construction, mirrored step-by-step by the
    /// eval walk of a **function or resource** body (single walk — see
    /// `cert::RecipeBuilder`). `None` for methods — they produce no
    /// certificate, so the recipe machinery costs them nothing.
    pub(crate) recipe: Option<crate::verify::cert::RecipeBuilder>,
    /// Fixpoint cache: the rule tier the live e-graph is known saturated under,
    /// with the rule-set sizes that saturation saw (ADT rules and axiom rules
    /// grow mid-unit; a grown set invalidates the fixpoint). `None` when any
    /// node/union landed since. Lets `saturate`/`reduce` skip whole runner
    /// invocations — most are re-runs on an unchanged graph.
    clean: Option<(CleanLevel, usize, usize)>,
    /// Whether the program uses `wildcard` permissions anywhere (any heap-op
    /// `Perm` with a wildcard leaf, in any body). Computed once from `decls`; a
    /// program without wildcards skips the per-subtract `contains_wildcard` scan
    /// entirely, so non-wildcard verification pays nothing for the feature.
    pub(crate) has_wildcard: bool,
    /// `SILVER_OXIDE_OOB_MEMO`: keep proven **conditional** obligations in an
    /// out-of-band set instead of unioning `pc ⇒ goal` into the `true` e-class.
    /// The union memoizes the proof but drags the whole `ite(pc.., goal, true)`
    /// chain permanently into `true` (the #1 growth driver — the graph has no GC),
    /// where the set memoizes the *verdict* alone. Empty-pc goals still union
    /// (productive: `eq-true-union`/congruence off a proven `Eq`).
    oob_memo: bool,
    /// Canonical class ids of implications already proven `true`, consulted at
    /// the `memo` tier when `oob_memo` is on. Keyed by `egraph.find(imp)`: two distinct
    /// obligations only share a class via congruence — which means their goals
    /// and pcs are pairwise equal, i.e. the *same* obligation — so a hit is
    /// sound; a stale leader after an unrelated merge only causes a safe
    /// re-prove.
    proven_imps: crate::dhash::HashSet<egg::Id>,
    /// The current method block's control cube (the shared pc of all its insts),
    /// as live-graph literal ids. Set by [`Self::begin_block`]; the scratch
    /// assumes it. Empty outside a method block (functions/resources don't use
    /// the scratch).
    current_cube: Vec<(egg::Id, Polarity)>,
    /// Whether we are inside a method block walk — gates the scratch on
    /// (functions/resources keep the per-obligation clone path).
    in_block: bool,
    /// The live per-block scratch, built lazily on the block's first `probe`-tier
    /// obligation and discarded at block exit. `None` when no obligation has
    /// needed it yet (or outside a method block).
    scratch: Option<BlockScratch>,
    /// Whether the current block's control cube is unsatisfiable — the block is
    /// unreachable, so every obligation in it holds vacuously. Set when the
    /// scratch is built (assuming the cube contradicts), cleared per block.
    /// Unlike [`Self::is_inconsistent`] this is a fact about the *scratch*, so it
    /// must never be consulted outside the block that established it.
    block_dead: bool,
}

/// Whether any declaration uses a `wildcard` permission. Scanned once per unit at
/// [`VerifyContext::new`]; lets a wildcard-free program skip the wildcard rules
/// entirely. A `wildcard` is either inline on a heap op or an arm of a permission
/// instruction, so this is a flat scan — no tree walk.
fn decls_have_wildcard(decls: &TiVec<MemberId, Declaration>) -> bool {
    use crate::vmir::{HeapInst, Inst, InstKind};
    fn insts_wild(insts: &[Inst]) -> bool {
        insts.iter().any(|i| match &i.kind {
            InstKind::Heap(
                HeapInst::Add { perm, .. }
                | HeapInst::Sub { perm, .. }
                | HeapInst::Inhale { perm, .. }
                | HeapInst::Exhale { perm, .. },
            ) => perm.is_wildcard(),
            InstKind::Perm(pi) => pi.has_wildcard_arm(),
            _ => false,
        })
    }
    decls.iter().any(|d| match d {
        Declaration::Method(m) => insts_wild(&m.flatten()),
        Declaration::Function(f) => f.body.as_ref().is_some_and(|b| insts_wild(&b.insts)),
        Declaration::Resource(r) => insts_wild(&r.body.insts),
        _ => false,
    })
}

/// How much of the rule set the live e-graph is saturated under. `Reduce`'s
/// set (terminating reductions + ADT) is a subset of `Full`'s, so `Full`
/// satisfies a `reduce()` request but not vice versa.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum CleanLevel {
    Reduce,
    Full,
}

impl<'a> VerifyContext<'a> {
    pub(crate) fn new(
        interner: &'a Rodeo,
        decls: &'a TiVec<MemberId, Declaration>,
        groups: &'a Rodeo<Spur>,
        alloc: &'a mut FuncRegistry,
    ) -> Self {
        Self {
            egraph: {
                // Fresh graph, fresh id space: remembered instantiations from
                // the previous unit are meaningless.
                rewrite::new_memo_unit();
                egg::EGraph::new(ConstFold::new(alloc.ctor_table()))
            },
            static_rules: rewrite::rules(),
            static_reduce: rewrite::reduce_rules(),
            axiom_rules: Vec::new(),
            fresh_counter: 0,
            interner,
            decls,
            groups,
            alloc,
            fresh_types: HashMap::default(),
            func_ret_types: HashMap::default(),
            fn_certs: None,
            recipe: None,
            clean: None,
            has_wildcard: decls_have_wildcard(decls),
            oob_memo: std::env::var_os("SILVER_OXIDE_OOB_MEMO").is_some(),
            proven_imps: crate::dhash::HashSet::default(),
            current_cube: Vec::new(),
            in_block: false,
            scratch: None,
            block_dead: false,
        }
    }

    /// The fixpoint-cache tag for the current rule sets (their sizes — growth
    /// invalidates a recorded fixpoint).
    fn clean_tag(&self, level: CleanLevel) -> (CleanLevel, usize, usize) {
        (level, self.alloc.rules().len(), self.axiom_rules.len())
    }

    /// Whether the live graph is already at a full-rule-set fixpoint, i.e. whether
    /// a `saturate()` right now would be a no-op. Callers that re-ask a question
    /// after saturating use this to tell "the graph changed, ask again" from
    /// "nothing moved, the answer cannot differ".
    pub(crate) fn is_saturated(&self) -> bool {
        self.is_clean(CleanLevel::Full)
    }

    /// Whether the live graph is known saturated at `level` (or stronger) under
    /// the *current* rule sets.
    fn is_clean(&self, level: CleanLevel) -> bool {
        matches!(self.clean, Some((l, adt, ax))
            if l >= level && adt == self.alloc.rules().len() && ax == self.axiom_rules.len())
    }

    /// Union two e-classes in the live graph — the sanctioned mutation path
    /// (invalidates the fixpoint cache). Callers still `rebuild()` after a
    /// batch of unions.
    pub(crate) fn union(&mut self, a: egg::Id, b: egg::Id) -> bool {
        let merged = self.egraph.union(a, b);
        if merged {
            self.clean = None;
        }
        if self.scratch.is_some() {
            // Mirror into the block scratch. A same-typed conflict there (the
            // cube made this union contradictory) folds to `Inconsistent`, not a
            // panic — that just makes the block's goals vacuously provable.
            let (ta, tb) = (self.tr(a), self.tr(b));
            let sc = self.scratch.as_mut().unwrap();
            sc.egraph.union(ta, tb);
            sc.dirty = true;
            sc.dirty_reduce = true;
        }
        merged
    }

    /// Whether the e-graph has reached a contradiction. Once inconsistent, every
    /// goal is vacuously provable — used by [`Self::prove_under_pc`] as the
    /// implicit channel through which an over-permissioned field location proves
    /// `false`.
    ///
    /// O(1): `ConstFold::modify` unions `true` with `false` the moment any class
    /// becomes `Data::Inconsistent`, so the contradiction is a fact *in* the
    /// graph rather than something to scan the classes for.
    pub(crate) fn is_inconsistent(&self) -> bool {
        graph_inconsistent(&self.egraph)
    }

    /// Display name for a member id. Registry-minted ids (outside the interner)
    /// resolve via the registry's name table.
    pub(crate) fn member_name(&self, m: MemberId) -> String {
        member_name_in(self.interner, self.decls, m)
    }

    /// Display name for an e-graph function id: a real declaration index resolves
    /// via the interner; an allocator-minted id via its name table.
    pub(crate) fn func_name(&self, f: FuncId) -> String {
        if f.0 < self.decls.len() {
            self.interner
                .resolve(&self.decls[MemberId::from(f.0)].name())
                .to_string()
        } else {
            self.alloc
                .name(f)
                .map(str::to_string)
                .unwrap_or_else(|| format!("f{}", f.0))
        }
    }

    /// Render a VMIR type using [`Self::member_name`] for `Domain` heads, so
    /// verifier-synthesised types (e.g. `Option[Int]`) print without panicking
    /// on the interner.
    pub(crate) fn type_name(&self, ty: &Type) -> String {
        match ty {
            Type::Int => "Int".to_string(),
            Type::Bool => "Bool".to_string(),
            Type::Real => "Real".to_string(),
            Type::Ref => "Ref".to_string(),
            Type::Addr {
                group,
                value,
                bound,
            } => format!(
                "&[{}] {} @ {bound}",
                self.groups.resolve(group),
                self.type_name(value)
            ),
            Type::Domain(id, args) => {
                let head = self.member_name(*id);
                if args.is_empty() {
                    head
                } else {
                    let inner: Vec<String> = args.iter().map(|a| self.type_name(a)).collect();
                    format!("{head}<{}>", inner.join(", "))
                }
            }
            Type::Snap(id) => format!("{}@snap", self.member_name(*id)),
            Type::Option(t) => format!("Option<{}>", self.type_name(t)),
            Type::Generic(i) => format!("?{i}"),
        }
    }

    /// Build a snapshot member `present ? Some(value) : None` over `Option[elem]`.
    /// When `present` const-folds to `true` (statically-positive permission) the
    /// `ite`/projection reductions peel it back to `value`.
    pub(crate) fn option_member(
        &mut self,
        elem: Type,
        present: egg::Id,
        value: egg::Id,
    ) -> egg::Id {
        // `Option` is a builtin parametric type; one polymorphic id each for
        // `Some`/`None` (variants 0/1). The element type is the application's ground
        // type instantiation (carried in the operator identity, not a child).
        let some_id = self.alloc.option_some();
        let none_id = self.alloc.option_none();
        let opt_ty = self.alloc.option_type(elem.clone());
        let tys: Box<[Type]> = Box::new([elem]);
        let some = self.add_func_app_id(some_id, tys.clone(), opt_ty.clone(), Box::new([value]));
        let none = self.add_func_app_id(none_id, tys, opt_ty, Box::new([]));
        self.add(Symbolic::Ite([present, some, none]))
    }

    /// Unwrap a snapshot member: `value(opt)`, the `Some` field accessor. With
    /// `opt = Some(v)` this reduces to `v`; on an opaque member it stays
    /// uninterpreted (correct — the value was never present).
    pub(crate) fn option_unwrap(&mut self, elem: Type, opt: egg::Id) -> egg::Id {
        let value_id = self.alloc.option_value();
        let tys: Box<[Type]> = Box::new([elem.clone()]);
        self.add_func_app_id(value_id, tys, elem, Box::new([opt]))
    }

    /// Run rewrite saturation over the e-graph in place. The rule set is the
    /// static rules plus the ADT reductions minted so far by the allocator
    /// plus the per-unit axiom/function rules.
    pub(crate) fn saturate(&mut self) {
        if self.is_clean(CleanLevel::Full) {
            return;
        }
        let t = std::time::Instant::now();
        let egraph = std::mem::take(&mut self.egraph);
        let (n0, c0) = (egraph.total_number_of_nodes(), egraph.number_of_classes());
        let it0 = stats::with_stats(|s| s.sat_iterations);
        self.egraph = self.saturate_flat(egraph);
        if std::env::var_os("SILVER_OXIDE_TRACE_SCRATCH").is_some() {
            eprintln!(
                "[ground-sat] {n0}n/{c0}c -> {}n/{}c true={} ({} iters)",
                self.egraph.total_number_of_nodes(),
                self.egraph.number_of_classes(),
                {
                    let t = self.egraph.find(self.true_id_cached());
                    self.egraph[t].nodes.len()
                },
                stats::with_stats(|s| s.sat_iterations) - it0,
            );
        }
        let secs = t.elapsed().as_secs_f64();
        stats::bump(|s| s.graph_timing.0.ground += secs);
        stats::bump(|s| s.saturations += 1);
        self.clean = Some(self.clean_tag(CleanLevel::Full));
    }

    /// One full-rule-set run, shared by [`Self::saturate`], [`Self::run_probe`],
    /// and the block scratch. Memo scoping is ambient (see `rewrite::Memo`): live
    /// runs write the persistent base, scratch scopes an overlay.
    fn saturate_flat(
        &mut self,
        egraph: egg::EGraph<Symbolic, ConstFold>,
    ) -> egg::EGraph<Symbolic, ConstFold> {
        let (egraph, iterations) = run_rules(
            egraph,
            self.static_rules
                .iter()
                .chain(self.alloc.rules())
                .chain(self.axiom_rules.iter()),
            None,
        );
        stats::bump(|s| s.record_run(&iterations));
        egraph
    }

    /// Run only the terminating structural reductions in place. Used after
    /// `fold`/`unfold` to collapse snapshot towers (so repeated round-trips
    /// don't grow the e-graph) without the cost/divergence risk of full
    /// saturation.
    pub(crate) fn reduce(&mut self) {
        if self.is_clean(CleanLevel::Reduce) {
            return;
        }
        let t = std::time::Instant::now();
        let egraph = std::mem::take(&mut self.egraph);
        let (egraph, iterations) = run_rules(
            egraph,
            self.static_reduce.iter().chain(self.alloc.rules()),
            None,
        );
        self.egraph = egraph;
        stats::bump(|s| s.graph_timing.0.ground += t.elapsed().as_secs_f64());
        stats::bump(|s| s.reduces += 1);
        stats::bump(|s| s.record_run(&iterations));
        self.clean = Some(self.clean_tag(CleanLevel::Reduce));
    }

    fn true_id_cached(&self) -> egg::Id {
        self.egraph
            .lookup(Symbolic::Lit(Literal::Bool(true)))
            .expect("true present")
    }

    /// One `[probe]` line per `probe`-tier obligation: ground size when the tier was
    /// reached versus the scratch size the obligation reasons over, and how far the
    /// scratch had to be run (`reduce` = the cheap reductions sufficed).
    fn trace_probe(&self, g0: (usize, usize, usize), fresh: bool, ran: &str) {
        let sc = self.scratch.as_ref().expect("scratch live");
        let st = sc.egraph.find(sc.true_id);
        eprintln!(
            "[probe] ground {}n/{}c true={} | scratch {}n/{}c true={} | ratio {:.2} | {} | {}",
            g0.0,
            g0.1,
            g0.2,
            sc.egraph.total_number_of_nodes(),
            sc.egraph.number_of_classes(),
            sc.egraph[st].nodes.len(),
            sc.egraph.total_number_of_nodes() as f64 / g0.0.max(1) as f64,
            ran,
            if fresh { "built" } else { "warm" },
        );
    }

    pub(crate) fn add(&mut self, node: Symbolic) -> egg::Id {
        let before = self.egraph.total_size();
        // Keep `node` for the scratch mirror only when a scratch is live (the
        // clone is not free — most adds happen with no scratch and pay nothing).
        if self.scratch.is_some() {
            let id = self.egraph.add(node.clone());
            if self.egraph.total_size() != before {
                self.clean = None;
            }
            self.mirror_add(id, node);
            id
        } else {
            let id = self.egraph.add(node);
            if self.egraph.total_size() != before {
                self.clean = None;
            }
            id
        }
    }

    /// Add a `FuncApp` over an already-allocated [`FuncId`] (a plain function,
    /// or an ADT constructor/projection/tag id from the allocator). Also used by
    /// grafting, which carries the id verbatim. `type_args` is the ground type
    /// instantiation — part of the node's operator identity (discriminant), not a
    /// child.
    pub(crate) fn add_func_app_id(
        &mut self,
        id: FuncId,
        type_args: Box<[Type]>,
        ret_ty: Type,
        args: Box<[egg::Id]>,
    ) -> egg::Id {
        self.func_ret_types.entry(id).or_insert(ret_ty);
        self.add(Symbolic::FuncApp(id, type_args, args))
    }

    pub(crate) fn fresh_symbolic_value(&mut self, ty: Type) -> egg::Id {
        let id = self.fresh_counter;
        self.fresh_counter += 1;
        self.fresh_types.insert(id, ty);
        self.add(Symbolic::Fresh(id))
    }

    /// Mint a fresh `wildcard` permission: an ordinary [`Symbolic::Fresh`] real
    /// assumed strictly positive (`0 < w`). Viper's `wildcard` — an unspecified
    /// positive share. Its upper bound (`w ≤ 1` for a field via the location axiom,
    /// `w < held` at exhale) is imposed elsewhere.
    ///
    /// No distinct node: nothing reads a wildcard off the graph any more. The two
    /// things that did are now answered where the answer actually lives —
    /// positivity by the `0 < w` fact this mints (`perm_sign`), and *origin* by
    /// `ChunkPerm::Leaf`'s `wild` flag, which an e-class could never have supplied
    /// since congruence puts wildcard-bearing terms into a literal's class.
    pub(crate) fn fresh_wildcard(&mut self) -> egg::Id {
        let id = self.fresh_counter;
        self.fresh_counter += 1;
        self.fresh_types.insert(id, crate::vmir::Type::Real);
        let w = self.add(Symbolic::Fresh(id));
        let pos = expr!(self, (0 / 1) < r { w });
        let true_ = expr!(self, true);
        self.union(pos, true_);
        // No eager `rebuild()`: the wildcard is minted mid-heap-op and every heap
        // op rebuilds downstream (obligation proving / `assume_location_axioms`)
        // before `0 < w` is queried. Rebuilding per mint dominated the cost.
        w
    }

    /// Build `antecedents ==> consequent` as a right-associative chain of `Ite`
    /// muxers with fallback `true` (vacuous truth). No boolean AND tree.
    /// `antecedents` must be in innermost-first fold order. A positive literal
    /// puts the running term in the true-branch (`true` in the false-branch); a
    /// negative literal swaps the branches.
    pub(crate) fn implication(
        &mut self,
        consequent: egg::Id,
        antecedents: impl Iterator<Item = (egg::Id, Polarity)>,
    ) -> egg::Id {
        let true_ = expr!(self, true);
        let mut imp = consequent;
        for (id, pol) in antecedents {
            imp = match pol {
                Polarity::Positive => self.add(Symbolic::Ite([id, imp, true_])),
                Polarity::Negative => self.add(Symbolic::Ite([id, true_, imp])),
            };
        }
        imp
    }

    /// Assume `fact` holds under `guards` — the **only** sanctioned way to record
    /// an assumption in the live e-graph. Merges `guards ==> fact` with `true`
    /// (via [`Self::implication`]), never `fact` itself: a raw `union(fact,
    /// true)` would assert `fact` on *every* path, letting the verifier assume what
    /// it must prove. With empty `guards` this degenerates to an unconditional
    /// assumption, correct only when the fact holds on all paths (a domain axiom).
    ///
    /// `guards` are in innermost-first fold order, matching [`Self::implication`].
    pub(crate) fn assume_guarded(
        &mut self,
        fact: egg::Id,
        guards: impl Iterator<Item = (egg::Id, Polarity)>,
    ) {
        let imp = self.implication(fact, guards);
        let true_ = expr!(self, true);
        self.union(imp, true_);
        // Invariant 4: ground guarded, scratch unguarded.
        self.scratch_assume_unguarded(fact);
        self.egraph.rebuild();
    }

    /// [`Self::assume_guarded`] for a **release key** rather than a fact about
    /// program state: a callee's `f%pre` token, minted at a call site under that
    /// call's path condition. Records the ground guarded implication exactly like
    /// [`Self::assume_guarded`], but deliberately **skips** the scratch-unguarded
    /// half of invariant 4.
    ///
    /// Why skip it: that invariant's licence is that the block scratch already
    /// bakes in the block PC — but it bakes in the block **cube**, and this token
    /// exists to carry the *finer* intra-block path condition (a call under a
    /// ternary or an implication; the same observation as the invariant-5 note in
    /// [`Self::prove_under_pc`]). Unioning `tok == true` in the warm scratch would
    /// make the callee's facts available on sibling intra-block paths there —
    /// precisely the leak this gating closes.
    ///
    /// Nothing is lost by skipping it: [`Self::union`] already mirrors the guarded
    /// implication into the scratch, so the cube prefix of the guard chain
    /// collapses there via `ite-reduce` (cube literals *are* unguarded in the
    /// scratch), and any residual intra-block guard collapses in the probe clone,
    /// which assumes the obligation's extra pc literals (see
    /// [`Self::prove_via_scratch`]). The token therefore ends up true in the
    /// scratch on exactly the paths an obligation is taken under.
    ///
    /// `guards` are in innermost-first fold order, matching [`Self::implication`].
    pub(crate) fn assume_token_guarded(
        &mut self,
        token: egg::Id,
        guards: impl Iterator<Item = (egg::Id, Polarity)>,
    ) {
        let imp = self.implication(token, guards);
        let true_ = expr!(self, true);
        self.union(imp, true_);
        self.egraph.rebuild();
    }

    /// [`Self::assume_guarded`] for several facts sharing one `guards`, with a
    /// single `rebuild()` at the end (rebuild dominates, so batching matters when
    /// a heap op assumes more than one fact — e.g. a wildcard exhale's
    /// `needed < held` and `0 < held − needed`).
    pub(crate) fn assume_all_guarded(
        &mut self,
        facts: impl IntoIterator<Item = egg::Id>,
        guards: &[(egg::Id, Polarity)],
    ) {
        let true_ = expr!(self, true);
        for fact in facts {
            let imp = self.implication(fact, guards.iter().copied());
            self.union(imp, true_);
            // Invariant 4: ground guarded, scratch unguarded.
            self.scratch_assume_unguarded(fact);
        }
        self.egraph.rebuild();
    }
}

/// One egg run over `rules`. Returns the graph and the iteration log — the
/// caller records the log into the stats once the rule borrow is released.
/// [`Context::is_inconsistent`] for a detached graph (a probe, or the block
/// scratch). `false` when either boolean literal is absent: a graph that never
/// mentioned one cannot have merged them.
pub(super) fn graph_inconsistent(egraph: &egg::EGraph<Symbolic, ConstFold>) -> bool {
    let (Some(t), Some(f)) = (
        egraph.lookup(Symbolic::Lit(Literal::Bool(true))),
        egraph.lookup(Symbolic::Lit(Literal::Bool(false))),
    ) else {
        return false;
    };
    egraph.find(t) == egraph.find(f)
}

pub(super) fn run_rules<'r>(
    egraph: egg::EGraph<Symbolic, ConstFold>,
    rules: impl IntoIterator<Item = &'r egg::Rewrite<Symbolic, ConstFold>>,
    iter_limit: Option<usize>,
) -> (egg::EGraph<Symbolic, ConstFold>, Vec<egg::Iteration<()>>) {
    run_rules_until(egraph, rules, iter_limit, None)
}

/// As [`run_rules`], but stopping as soon as `goal` is settled.
///
/// egg checks a `Runner`'s hooks at the start of every iteration and treats an
/// `Err` as a stop, so this costs one class lookup per iteration and saves every
/// iteration after the answer exists.
///
/// **`goal` is only sound on a throwaway graph.** The returned e-graph is *not*
/// saturated, so a caller that records it as clean — `VerifyContext::saturate`
/// (which sets `self.clean`) or `saturate_scratch` (which clears `dirty` on a graph
/// shared by every obligation in the block) — would make later obligations read a
/// fixpoint that was never reached. Pass `None` there and `Some` only for a probe
/// that dies with the call.
///
/// The *inconsistency* stop below carries no such condition and so is installed for
/// every caller: a contradictory graph proves everything, which is exactly why
/// `run_rules` refuses to start on one, so abandoning its saturation cannot change
/// a verdict either.
pub(super) fn run_rules_until<'r>(
    egraph: egg::EGraph<Symbolic, ConstFold>,
    rules: impl IntoIterator<Item = &'r egg::Rewrite<Symbolic, ConstFold>>,
    iter_limit: Option<usize>,
    goal: Option<egg::Id>,
) -> (egg::EGraph<Symbolic, ConstFold>, Vec<egg::Iteration<()>>) {
    // A contradictory graph proves everything, so no rule can change any verdict
    // it yields — stop running them. This is the single choke point for every
    // graph (ground saturate/reduce, scratch saturate/reduce, every probe), and
    // it is per-graph: a deliberately contradictory probe short-circuits without
    // touching a consistent ground graph.
    if graph_inconsistent(&egraph) {
        return (egraph, Vec::new());
    }
    // The observation cache describes the graph this run walks; a run on a
    // different graph must not read it (ground and its clones share ids).
    crate::verify::rewrite::diseq::new_scan_generation();
    // Explicit limits: egg's defaults (30 iterations, 10k nodes, **5 seconds**)
    // are SILENT truncation points — a run that hits one simply stops
    // mid-saturation and the caller sees an ordinary "not proven", which surfaced
    // as a false insufficient-permission at ~20 match arms (one tower level
    // collapses per iteration, so deep-but-terminating collapses need iterations
    // ∝ depth).
    //
    // The time limit is the worst of the three, because it makes a *verdict*
    // depend on wall clock: `enum_v8_p2::m_e_guarded` was measured failing on a
    // 119s run and verifying on a 149s one, same binary and input. That also
    // silently falsifies the premise `perf_regression` rests on ("egg is
    // deterministic for a fixed rule set + input") for any program whose
    // saturation approaches it. Disabled outright — the node and iteration limits
    // are the real backstops, and unlike a clock they are reproducible.
    let mut runner = egg::Runner::default()
        .with_scheduler(egg::SimpleScheduler)
        .with_node_limit(100_000)
        .with_iter_limit(iter_limit.unwrap_or(100))
        .with_time_limit(std::time::Duration::MAX)
        .with_egraph(egraph)
        // Contradiction reached mid-run: every remaining iteration is spent
        // elaborating a graph that already proves everything.
        .with_hook(|r| {
            if graph_inconsistent(&r.egraph) {
                stats::bump(|s| s.probe_early_stops += 1);
                return Err("inconsistent".to_owned());
            }
            Ok(())
        });
    if let Some(goal) = goal {
        runner = runner.with_hook(move |r| {
            // `ConstFold` settles a class without merging it into `true`, so ask
            // the analysis rather than comparing canonical ids.
            if matches!(
                r.egraph[r.egraph.find(goal)].data.known(),
                Some(Literal::Bool(true))
            ) {
                stats::bump(|s| s.probe_early_stops += 1);
                return Err("goal proven".to_owned());
            }
            Ok(())
        });
    }
    let runner = runner.run(rules);
    (runner.egraph, runner.iterations)
}
