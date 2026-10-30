//! Function unfolding: the lazy rule that installs `f(args) == body` and replays
//! a verified definition's postcondition at a call site.

use std::collections::BTreeSet;
use std::sync::Arc;

use egg::{Applier, EGraph, Id, PatternAst, Rewrite, SearchMatches, Searcher, Subst, Symbol, Var};

use crate::verify::analysis::ConstFold;
use crate::verify::cert::FunctionDefinition;
use crate::verify::lang::{Discriminant, FuncId, Symbolic};
use crate::vmir::{Literal, Polarity, Type, Val};

use super::*;

/// Applier for the **full** rule (keyed on `f`), paired with
/// [`FunctionCallSearcher`]: each match names one call `f(args)` and, when the
/// searcher found one on the clean graph, its `f%pre(args)` token. For that call
/// it rebuilds the function's **definition recipe** with `build_instance`
/// (params → args, `Generic(i)` → `tys`) and releases `f(args) == body` behind the
/// token, so the equation lands only on paths that call it.
/// **Imports no e-classes** (`build_instance` only adds), so merges derived while
/// verifying the body itself never ride along. It is not literally add-only: a
/// body's `AxiomInst::Assume` steps and its propagated callee tokens are merged
/// with `true` here, as they were before this rule existed. Works uniformly for
/// heap-free and heap-dependent `f`: the recipe already resolved every
/// `Deref`/`Unfold`/`Snap` into pure terms over the params + snapshot.
pub(super) struct FunctionUnfoldApplier {
    pub(super) func: FuncId,
    pub(super) def: Arc<FunctionDefinition>,
    /// Guards the releases behind the token (the definitional union and the
    /// post): done at most once per call. Only a call with a token is recorded,
    /// so a call first reached without one still releases both once
    /// propagation mints its token mid-saturation.
    pub(super) memo: Memo<CallKey>,
    /// The searcher's `?tok` binding: the call's `f%pre(args)` class, bound only
    /// when the token existed at search time (see [`FunctionCallSearcher`]).
    pub(super) tok_var: Var,
    /// The searcher's `?a0..?a{n-1}` bindings: the call's value args.
    pub(super) arg_vars: Vec<Var>,
}

/// Applier for a **post-only** rule (see [`post_rule`]): replays the
/// definition's post at each call in the matched e-class, and nothing else — no
/// body, no unions. Unfolding a recursive body yields `f'(smaller)`, and this is
/// what delivers the postcondition there (Silicon's `post` axiom,
/// `f%pre(x) ==> post(x)`, triggering on the limited symbol).
///
/// A postcondition holds only where the precondition does, and a node `f'(a)`
/// only says the term exists: unfolding a body builds the nodes of its dead
/// branches too. Replaying the post there ungated made Helium prove `false`
/// (`tests/cases/failing/functions/limited_twin_post_dead_branch.vpr`). So the
/// post is replayed behind the call's `f%pre(args)` token, which the unfolded
/// body propagates for its recursive call, and which a recursive batch mints at
/// the call under its path condition. A call without a token gets no post.
///
/// The induction hypothesis of a batch therefore holds only on the path that
/// makes the recursive call. A goal that needs it beyond that path needs a case
/// split on the path condition, which Helium does not make
/// (`tests/cases/known_limitations/recursive_post_induction_under_branch.vpr`).
pub(super) struct FunctionPostApplier {
    pub(super) func: FuncId,
    pub(super) def: Arc<FunctionDefinition>,
    /// Guards the post replay: done at most once per call. Only a call with a
    /// token is recorded, so one reached before its token still gets the post.
    pub(super) memo: Memo<CallKey>,
    /// The searcher's `?tok` binding (see [`FunctionUnfoldApplier::tok_var`]).
    pub(super) tok_var: Var,
    /// The searcher's `?a0..?a{n-1}` bindings: the call's value args.
    pub(super) arg_vars: Vec<Var>,
}

/// Searcher for the function rules: one match per distinct call `f(args)` in
/// each e-class, binding `?a0..` to the args and, when `f%pre(args)` exists,
/// `?tok` to its class. `f%pre` is the function's uniform pre-token
/// (`FuncRegistry::fn_pre_token`), a `forall`-style **presence** trigger: the
/// definitional `f(args)==body` release fires only for a genuine value-position
/// call. Mirrors Silicon's `f%pre ⟹ f==body`. The post-only rule keys it on the
/// limited twin `f'` with `f`'s token.
///
/// The token is looked up **here** because the runner searches on a freshly
/// rebuilt graph, where the hashcons is exact. An applier runs after earlier
/// rules of the same iteration have merged classes without a rebuild, so a
/// `lookup` there can miss a token that exists. A miss only delays the release
/// to a later iteration; it must never be read as "release ungated".
///
/// Nor may an applier *add* a missing token to gate on: the token's presence is
/// what makes [`FunctionUnfoldApplier`] build the body, so every occurrence
/// would then pay for a body unfolding.
pub(super) struct FunctionCallSearcher {
    func: FuncId,
    pre_token: FuncId,
    /// `?tok`, bound to the `f%pre(args)` class when one exists.
    tok_var: Var,
    /// `?a0..?a{n-1}`, one per param, bound to the call's args.
    arg_vars: Vec<Var>,
}

impl FunctionCallSearcher {
    fn new(func: FuncId, pre_token: FuncId, n_params: usize) -> Self {
        Self {
            func,
            pre_token,
            tok_var: "?tok".parse().unwrap(),
            arg_vars: (0..n_params)
                .map(|i| format!("?a{i}").parse().unwrap())
                .collect(),
        }
    }
}
impl Searcher<Symbolic, ConstFold> for FunctionCallSearcher {
    fn vars(&self) -> Vec<Var> {
        vec![]
    }

    fn search_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        mut limit: usize,
    ) -> Vec<SearchMatches<'_, Symbolic>> {
        let Some(classes) = egraph.classes_for_op(&Discriminant::FuncApp(self.func)) else {
            return vec![];
        };
        let mut out = vec![];
        for eclass in classes {
            if limit == 0 {
                break;
            }
            if let Some(m) = self.search_eclass_with_limit(egraph, eclass, limit) {
                limit = limit.saturating_sub(m.substs.len());
                out.push(m);
            }
        }
        out
    }
    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _limit: usize,
    ) -> Option<SearchMatches<'_, Symbolic>> {
        let mut seen: BTreeSet<&Box<[Id]>> = BTreeSet::new();
        let mut subst: Vec<Subst> = vec![];

        for node in &egraph[eclass].nodes {
            let Symbolic::FuncApp(f, _tys, args) = node else {
                continue;
            };
            if *f != self.func {
                continue;
            }

            if seen.contains(args) {
                continue;
            }
            seen.insert(args);

            let mut s = Subst::default();

            for (var, &arg) in self.arg_vars.iter().zip(args.iter()) {
                s.insert(*var, arg);
            }

            let probe = Symbolic::FuncApp(self.pre_token, Box::new([]), args.clone());

            if let Some(tok) = egraph.lookup(probe) {
                s.insert(self.tok_var, tok);
            }
            subst.push(s)
        }
        if subst.is_empty() {
            None
        } else {
            Some(SearchMatches {
                eclass,
                substs: subst,
                ast: None,
            })
        }
    }
}

/// Wrap `inner` in a guard chain over **live ids**: `guards ==> inner`, as nested
/// `Ite`s with `true` on the dead side. `guards` is outermost-first and folded
/// innermost-first, the same shape `VerifyContext::implication` builds. Empty
/// `guards` returns `inner`.
pub(crate) fn fold_guard_ids(
    egraph: &mut EGraph<Symbolic, ConstFold>,
    guards: &[(Id, Polarity)],
    inner: Id,
) -> Id {
    if guards.is_empty() {
        return inner;
    }
    let true_ = egraph.add(Symbolic::Lit(Literal::Bool(true)));
    let mut imp = inner;
    for (g, pol) in guards.iter().rev() {
        imp = match pol {
            Polarity::Positive => egraph.add(Symbolic::Ite([*g, imp, true_])),
            Polarity::Negative => egraph.add(Symbolic::Ite([*g, true_, imp])),
        };
    }
    imp
}

/// [`fold_guard_ids`] over **recipe-space** guards (matching
/// [`TokenStep::guards`]), resolving each against the built instance's slots.
pub(super) fn fold_guards(
    egraph: &mut EGraph<Symbolic, ConstFold>,
    vals: &[Id],
    guards: &[(Val, Polarity)],
    inner: Id,
) -> Id {
    if guards.is_empty() {
        return inner;
    }
    let resolved: Vec<(Id, Polarity)> = guards
        .iter()
        .map(|(g, pol)| (resolve_val(egraph, vals, g), *pol))
        .collect();
    fold_guard_ids(egraph, &resolved, inner)
}

/// Replay a definition's post at one call: build its standalone recipe off the
/// **arguments** and merge `token ⟹ post` (an `Ite`, the shape
/// `VerifyContext::implication` builds) with `true`. Guarded, so the
/// postcondition never fires outside the caller-side path condition the `token`
/// carries.
///
/// Off the arguments, not off a built body instance: [`PostRecipe`] mentions only
/// the params and `f(params)`, so nothing here materializes the callee's body.
/// That is what makes the definitional union's presence gate load-bearing.
///
/// `token` is the callee's `f%pre(args)` class at *this* call, as
/// [`FunctionCallSearcher`] found it on the clean graph. It is not optional: a
/// post replayed without it would hold at arguments that violate the
/// precondition. It is threaded separately rather than folded into the recipe
/// because the post's `Val`s live in the callee's recipe-temp space, whereas
/// this is already a caller-side `Id`.
pub(super) fn replay_post(
    egraph: &mut EGraph<Symbolic, ConstFold>,
    def: &FunctionDefinition,
    args: &[Id],
    token: Id,
    changed: &mut Vec<Id>,
) {
    let Some(post) = def.post.as_ref() else {
        return;
    };
    let vals = build_instance_vals(egraph, &post.steps, args, changed);
    let true_ = egraph.add(Symbolic::Lit(Literal::Bool(true)));
    // The truth (`f#ensures(args, f(args))`) and, beside it, the `#ensures`
    // member's own presence token — Silicon's `postPreconditionPropagationAxiom`.
    // Both are needed: the token alone licenses unfolding the postcondition but
    // never says it holds, and the truth alone leaves the callees the post names
    // opaque at this caller.
    for v in [Some(&post.res), post.token.as_ref()].into_iter().flatten() {
        let fact = resolve_val(egraph, &vals, v);
        let imp = expr!(egraph, {token} ==> {fact});
        if egraph.union(imp, true_) {
            changed.push(egraph.find(imp));
        }
    }
}

impl FunctionUnfoldApplier {
    /// Every release for one call `f[tys](args)` in `eclass`, with `tok_id` its
    /// `f%pre(args)` class (`None`: no token existed when the searcher ran).
    fn unfold_call(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        tys: Box<[Type]>,
        args: &[Id],
        tok_id: Option<Id>,
        changed: &mut Vec<Id>,
    ) {
        debug_assert_eq!(args.len(), self.def.n_params, "function unfold arity");
        // The definitional union and the post are both released behind the
        // token, so an occurrence that no value-position call produced — no
        // token — materializes no body and gets no post, which is what the
        // presence gate was always supposed to buy. It must NOT be memoized, so
        // both still release once propagation mints its token mid-saturation.
        let tok = tok_id.filter(|_| self.memo.insert((tys.clone(), args.to_vec())));
        if let Some(tok) = tok {
            let vals =
                build_instance_vals_guarded(egraph, &self.def.steps, args, changed, Some(tok));
            // Definitional axiom `tok ==> f(args) == body`. Absent for an abstract
            // function.
            //
            // This cannot be an e-class merge — a merge is unconditional and there
            // is no such thing as a conditional one — so it is stated as a guarded
            // *equality term*: `ite(tok, f(args) == body, true) == true`. Once the
            // token collapses to `true` (an unconditional call, or a conditional
            // one inside the probe clone that assumes its pc), `ite-reduce`
            // exposes `f(args) == body == true` and `eq-true-union` performs the
            // merge.
            if let Some(res) = &self.def.res {
                let result = resolve_val(egraph, &vals, res);
                let true_ = expr!(egraph, true);
                let rel = expr!(egraph, {tok} ==> ({eclass} == {result}));
                if egraph.union(rel, true_) {
                    changed.push(egraph.find(rel));
                }
            }
            // Propagated callee tokens: `tok ==> g%pre(gargs)` for each nested
            // callee this body calls (Silicon's `bodyPreconditionPropagation`).
            // Rebuilding the step above only *added* the node, which is what makes
            // `g` materializable here; this is what activates `g`'s own axioms, and
            // only where this body's own release fires. Never truer than the
            // release it accompanies, so it cannot lose a proof: if `tok` is not
            // true, this body's equality is not released either.
            //
            // Released with the **union**, never with a bare post replay: a
            // post-only occurrence built no body, so there is no `g(gargs)` node
            // here for a `g%pre` token to activate — and minting one would
            // reintroduce exactly the cascade the presence gate exists to stop.
            for ts in &self.def.token_steps {
                let tok_node = resolve_val(egraph, &vals, &ts.token);
                // `guards` is the body-internal condition guarding the nested
                // call; `tok` is this call's own token. Folding the former
                // inside the latter gives `f%pre(a) ==> (b[x:=a] ==> g%pre(..))`.
                let guarded = fold_guards(egraph, &vals, &ts.guards, tok_node);
                let true_ = egraph.add(Symbolic::Lit(Literal::Bool(true)));
                let rel = expr!(egraph, {tok} ==> {guarded});
                if egraph.union(rel, true_) {
                    changed.push(egraph.find(rel));
                }
            }
        }
        // Recursive function: frame the full occurrence to its limited twin
        // `f(args) == f'(args)`. `f'` has no unfold rule, so a limited call
        // produced by unfolding `f`'s body never re-unfolds (bounding
        // saturation); the frame lets a materialized `f(args)` value flow to
        // any `f'(args)` a sibling unfold produced.
        //
        // **Ungated**, matching Silicon's `limitedAxiom`. It must NOT sit
        // behind the `f%pre` presence token: the token is minted only at a
        // value-position call, so gating strands a recursive function's post
        // fact (stated on `f'`) in limited space whenever the occurrence
        // arrived via a callee's `ensures` rather than a direct call.
        //
        // Cannot reintroduce unbounded unfolding: this mints `f'` *from* an
        // existing `f` node, never the reverse, and the unfold rule filters
        // on `FuncApp(f)`.
        if let Some(lim) = self.def.limited {
            let twin = egraph.add(Symbolic::FuncApp(lim, tys, args.into()));
            if egraph.union(eclass, twin) {
                changed.push(egraph.find(eclass));
            }
        }
        // Replay the post once per call, behind its token.
        if let Some(tok) = tok {
            replay_post(egraph, &self.def, args, tok, changed);
        }
    }
}

impl Applier<Symbolic, ConstFold> for FunctionUnfoldApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        // The call this match names. Re-canonicalized: earlier rules of this
        // iteration may have merged classes since the search.
        let args: Vec<Id> = self
            .arg_vars
            .iter()
            .map(|&v| egraph.find(subst[v]))
            .collect();
        // The token (see `FunctionCallSearcher`) plays two distinct roles,
        // mirroring how a `Forall` e-class does: its **presence** triggers
        // materialization of the callee's body here, and its **truth** releases
        // the axioms that materialization produces.
        //
        // The token is assumed at each call site under that call's path
        // condition (`declaration.rs`, the `FunctionCall` arm), so its class is
        // an `ite(pc, tok, true)` shape and is unconditionally `true` only for
        // an unconditional call. `tok_id` therefore doubles as the release
        // guard.
        let tok_id = subst.get(self.tok_var).map(|&t| egraph.find(t));
        // The type instantiations applied at these args (a `Subst` cannot carry
        // types, so they are re-read from the class).
        let mut instantiations: Vec<Box<[Type]>> = Vec::new();
        for node in &egraph[eclass].nodes {
            let Symbolic::FuncApp(f, tys, nargs) = node else {
                continue;
            };
            if *f != self.func || nargs.len() != args.len() {
                continue;
            }
            let same = nargs.iter().zip(&args).all(|(&a, &b)| egraph.find(a) == b);
            if same && !instantiations.contains(tys) {
                instantiations.push(tys.clone());
            }
        }
        let mut changed = Vec::new();
        for tys in instantiations {
            self.unfold_call(egraph, eclass, tys, &args, tok_id, &mut changed);
        }
        changed
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

impl Applier<Symbolic, ConstFold> for FunctionPostApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        let mut changed = Vec::new();
        // No token, no post (see the struct doc). Not memoized, so a token
        // minted later still releases it.
        let Some(tok) = subst.get(self.tok_var).map(|&t| egraph.find(t)) else {
            return changed;
        };
        if self.def.post.is_none() {
            return changed;
        }
        // The call this match names, re-canonicalized as in
        // `FunctionUnfoldApplier`, and the type instantiations applied at it.
        let args: Vec<Id> = self
            .arg_vars
            .iter()
            .map(|&v| egraph.find(subst[v]))
            .collect();
        debug_assert_eq!(args.len(), self.def.n_params, "function post arity");
        let mut instantiations: Vec<Box<[Type]>> = Vec::new();
        for node in &egraph[eclass].nodes {
            let Symbolic::FuncApp(f, tys, nargs) = node else {
                continue;
            };
            if *f != self.func || nargs.len() != args.len() {
                continue;
            }
            let same = nargs.iter().zip(&args).all(|(&a, &b)| egraph.find(a) == b);
            if same && !instantiations.contains(tys) {
                instantiations.push(tys.clone());
            }
        }
        for tys in instantiations {
            if self.memo.insert((tys, args.clone())) {
                replay_post(egraph, &self.def, &args, tok, &mut changed);
            }
        }
        changed
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

/// Mint the lazy-unfolding rule for one verified function's definition recipe,
/// gated on its uniform pre-token `f%pre` (`FuncRegistry::fn_pre_token`).
pub(crate) fn function_rule(
    name: &str,
    func: FuncId,
    def: Arc<FunctionDefinition>,
    pre_token: FuncId,
) -> Rule {
    let searcher = FunctionCallSearcher::new(func, pre_token, def.n_params);
    let applier = FunctionUnfoldApplier {
        func,
        tok_var: searcher.tok_var,
        arg_vars: searcher.arg_vars.clone(),
        def,
        memo: Memo::new(),
    };
    timed(Rewrite::new(format!("fn-{name}"), searcher, applier).expect("function rule"))
}

/// Mint a post-only rule keyed on `func`: replays the definition's post behind
/// the call's `pre_token` (the full function's `f%pre`), no definitional union.
/// Two users:
/// - the limited-twin post rule of a recursive function ([`function_post_rule`]);
/// - the spec-derived post rule installed for each SCC member **during** a
///   recursive batch's own verification, which is what makes induction over a
///   recursive call work (Silicon's phase-1 `post` axiom). The recursive call
///   in the body mints its token under the call's path condition, so the
///   induction hypothesis holds only on the path that makes the call.
pub(crate) fn post_rule(
    name: &str,
    func: FuncId,
    def: Arc<FunctionDefinition>,
    pre_token: FuncId,
) -> Rule {
    let searcher = FunctionCallSearcher::new(func, pre_token, def.n_params);
    let applier = FunctionPostApplier {
        func,
        tok_var: searcher.tok_var,
        arg_vars: searcher.arg_vars.clone(),
        def,
        memo: Memo::new(),
    };
    timed(Rewrite::new(format!("fn-post-{name}"), searcher, applier).expect("function post rule"))
}

/// Mint the limited-twin post rule for a **recursive** function: keyed on
/// `f'(args)` occurrences (which unfolding a recursive body produces), replays
/// the definition's post behind `f%pre(args)` (no definitional union — that's
/// the point of the limited symbol). Registered alongside [`function_rule`] when
/// `def.limited` is set and a post fact exists.
pub(crate) fn function_post_rule(
    name: &str,
    def: Arc<FunctionDefinition>,
    pre_token: FuncId,
) -> Rule {
    let func = def.limited.expect("post rule requires a limited twin");
    post_rule(name, func, def, pre_token)
}

/// A canonicalized call key: the ground `(type_args, value_args)` an occurrence
/// of the function was applied to. Memoized so the recipe rebuilds once per call.
pub(super) type CallKey = (Box<[Type]>, Vec<Id>);
