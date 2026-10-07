//! Function unfolding: the lazy rule that installs `f(args) == body` and replays
//! a verified definition's postcondition at a call site.

use std::sync::Arc;

use egg::{Applier, EGraph, Id, PatternAst, Rewrite, Subst, Symbol, Var};

use crate::verify::analysis::ConstFold;
use crate::verify::cert::FunctionDefinition;
use crate::verify::lang::{FuncId, Symbolic};
use crate::vmir::{Literal, Polarity, Type, Val};

use super::*;

/// Applier: for each ground `FuncApp(self.func, tys, args)` node in the matched
/// e-class, rebuild the function's **definition recipe** with `build_instance`
/// (params → args, `Generic(i)` → `tys`) and release `f(args) == body` lazily as
/// occurrences are seen during saturation — behind the call's `f%pre` token when
/// the function has one, so the equation lands only on paths that call it.
/// **Imports no e-classes** (`build_instance` only adds), so merges derived while
/// verifying the body itself never ride along. It is not literally add-only: a
/// body's `AxiomInst::Assume` steps and its propagated callee tokens are merged
/// with `true` here, as they were before this rule existed. Works uniformly for
/// heap-free and heap-dependent `f`: the recipe already resolved every
/// `Deref`/`Unfold`/`Snap` into pure terms over the params + snapshot.
pub(super) struct FunctionUnfoldApplier {
    pub(super) func: FuncId,
    pub(super) def: Arc<FunctionDefinition>,
    /// `false`: the full rule (keyed on `f`) — definitional union, limited
    /// framing, and the post. `true`: the limited-post rule (keyed on `f'`) —
    /// the post and nothing else, no unions: unfolding a recursive body yields
    /// `f'(smaller)`, and this is what delivers the postcondition there
    /// (Silicon's `post` axiom triggering on the limited symbol).
    pub(super) limited_post: bool,
    /// Guards the (expensive) body build + post replay: done at most once per call.
    pub(super) memo: Memo<CallKey>,
    /// Guards the definitional union separately from the build: a call first built
    /// post-only (no token yet) and later reached by a propagation-minted token
    /// must still union `f==body`, which `memo` alone would suppress.
    pub(super) union_memo: Memo<CallKey>,
    /// The function's uniform pre-token `f%pre` (`FuncRegistry::fn_pre_token`),
    /// used as a `forall`-style **presence** trigger: the definitional
    /// `f(fargs)==body` union+build fires only when a `FuncApp(f%pre, fargs)` node
    /// is present for the *same* `fargs`, i.e. this occurrence came from a genuine
    /// value-position call. Mirrors Silicon's `f%pre ⟹ f==body`. `None` on the
    /// post-only rule, where there is no definitional union to gate.
    pub(super) pre_token: Option<FuncId>,
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
/// `token` is the callee's `f%pre(args)` class at *this* call, `None` when the
/// definition has no pre-token (a contract function) or when no token exists for
/// these args. It is threaded separately rather than folded into the recipe
/// because the post's `Val`s live in the callee's recipe-temp space, whereas
/// this is already a caller-side `Id`.
pub(super) fn replay_post(
    egraph: &mut EGraph<Symbolic, ConstFold>,
    def: &FunctionDefinition,
    args: &[Id],
    token: Option<Id>,
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
        let mut imp = resolve_val(egraph, &vals, v);
        if let Some(tok) = token {
            imp = expr!(egraph, {tok} ==> {imp});
        }
        if egraph.union(imp, true_) {
            changed.push(egraph.find(imp));
        }
    }
}

impl Applier<Symbolic, ConstFold> for FunctionUnfoldApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        // Candidate calls in this e-class (deduped locally; the persistent `memo`
        // is only consulted once we commit to building — see below).
        let mut calls: Vec<(Box<[Type]>, Vec<Id>)> = Vec::new();
        for node in &egraph[eclass].nodes {
            let Symbolic::FuncApp(f, tys, args) = node else {
                continue;
            };
            if *f != self.func {
                continue;
            }
            let args: Vec<Id> = args.iter().map(|&a| egraph.find(a)).collect();
            let key = (tys.clone(), args);
            if !calls.contains(&key) {
                calls.push(key);
            }
        }
        let mut changed = Vec::new();
        for (tys, args) in calls {
            debug_assert_eq!(args.len(), self.def.n_params, "function unfold arity");
            // The token (see `pre_token`) plays two distinct roles, mirroring how a
            // `Forall` e-class does: its **presence** triggers materialization of
            // the callee's body here, and its **truth** releases the axioms that
            // materialization produces. Type args are not in the `FuncApp`
            // congruence key, so `[]` matches any instantiation.
            //
            // The token is assumed at each call site under that call's path
            // condition (`declaration.rs`, the `FunctionCall` arm), so its class is
            // an `ite(pc, tok, true)` shape and is unconditionally `true` only for
            // an unconditional call. `tok_id` therefore doubles as the release
            // guard below. `pre_token = None` ⇒ released ungated (a contract
            // `#requires`/`#ensures` boolean, whose role is to inline its formula).
            let tok_id: Option<Id> = self.pre_token.and_then(|tok| {
                egraph.lookup(Symbolic::FuncApp(tok, Box::new([]), args.clone().into()))
            });
            let has_token = self.pre_token.is_none() || tok_id.is_some();
            // The body instance is needed for the definitional union, and for
            // nothing else: the post is a standalone recipe over the params. So an
            // occurrence that no value-position call produced — no token, hence no
            // union — materializes no body at all, which is what the presence gate
            // was always supposed to buy. It must NOT be union-memoized, so it can
            // still union later once propagation mints its token mid-saturation.
            let do_union = !self.limited_post && has_token;
            let has_post = self.def.post.is_some();
            if !do_union && !has_post {
                continue;
            }
            let key = (tys.clone(), args.clone());
            // Two independent memos, because the two releases happen at different
            // times: `memo` says the post has been replayed at this call, and
            // `do_union` is short-circuited before `union_memo` so the union memo is
            // only ever touched when the token is present — a fresh union, or a call
            // that replayed its post earlier and is now reached by a
            // propagation-minted token.
            let first_post = has_post && self.memo.insert(key.clone());
            let do_union_now = do_union && self.union_memo.insert(key);
            // Nothing new to do: post already replayed and no (new) union due.
            if !first_post && !do_union_now {
                continue;
            }
            // Seed slots for the body instance. Only built when the union needs it;
            // otherwise there is no body here and `vals` stays unused.
            let vals = match do_union_now {
                true => build_instance_vals_guarded(
                    egraph,
                    &self.def.steps,
                    &args,
                    &mut changed,
                    tok_id,
                ),
                false => Vec::new(),
            };
            // Definitional axiom `tok ==> f(args) == body`. Absent for an abstract
            // function.
            //
            // With a token this cannot be an e-class merge — a merge is
            // unconditional and there is no such thing as a conditional one — so it
            // is stated as a guarded *equality term*: `ite(tok, f(args) == body,
            // true) == true`. Once the token collapses to `true` (an unconditional
            // call, or a conditional one inside the probe clone that assumes its
            // pc), `ite-reduce` exposes `f(args) == body == true` and
            // `eq-true-union` performs the merge.
            //
            // Without a token (`pre_token = None`, a contract function) it stays a
            // direct union: the body is a total function of the args (Deref became
            // `unwrap∘proj`, div is total in the e-graph), so the equation holds
            // even at pre-violating args, and such a formula must inline freely.
            if do_union_now {
                if let Some(res) = &self.def.res {
                    let result = resolve_val(egraph, &vals, res);
                    match tok_id {
                        None => {
                            if egraph.union(eclass, result) {
                                changed.push(egraph.find(eclass));
                            }
                        }
                        Some(tok) => {
                            let true_ = expr!(egraph, true);
                            let rel = expr!(egraph, {tok} ==> ({eclass} == {result}));
                            if egraph.union(rel, true_) {
                                changed.push(egraph.find(rel));
                            }
                        }
                    }
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
            if do_union_now {
                for ts in &self.def.token_steps {
                    let tok_node = resolve_val(egraph, &vals, &ts.token);
                    // `guards` is the body-internal condition guarding the nested
                    // call; `tok_id` is this call's own token. Folding the former
                    // inside the latter gives `f%pre(a) ==> (b[x:=a] ==> g%pre(..))`.
                    let guarded = fold_guards(egraph, &vals, &ts.guards, tok_node);
                    let true_ = egraph.add(Symbolic::Lit(Literal::Bool(true)));
                    let rel = match tok_id {
                        Some(tok) => expr!(egraph, {tok} ==> {guarded}),
                        None => guarded,
                    };
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
            if !self.limited_post {
                if let Some(lim) = self.def.limited {
                    let twin = egraph.add(Symbolic::FuncApp(lim, tys.clone(), args.clone().into()));
                    if egraph.union(eclass, twin) {
                        changed.push(egraph.find(eclass));
                    }
                }
            }
            // Replay the post once per call (idempotent; guarded internally).
            if first_post {
                replay_post(egraph, &self.def, &args, tok_id, &mut changed);
            }
        }
        changed
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

/// Mint the lazy-unfolding rule for one verified function's definition recipe.
/// Reuses [`AxiomTriggerSearcher`] as-is — it only checks `FuncApp(f, ..)`
/// presence, independent of arity/type-args, exactly what's needed here too.
pub(crate) fn function_rule(
    name: &str,
    func: FuncId,
    def: Arc<FunctionDefinition>,
    pre_token: Option<FuncId>,
) -> Rule {
    let searcher = AxiomTriggerSearcher {
        func,
        token: pre_token,
    };
    let applier = FunctionUnfoldApplier {
        func,
        def,
        limited_post: false,
        memo: Memo::new(),
        union_memo: Memo::new(),
        pre_token,
    };
    timed(Rewrite::new(format!("fn-{name}"), searcher, applier).expect("function rule"))
}

/// Mint a post-only rule keyed on `func`: replays the definition's post, no
/// definitional union. Two users:
/// - the limited-twin post rule of a recursive function ([`function_post_rule`]);
/// - the spec-derived post rule installed for each SCC member **during** a
///   recursive batch's own verification, which is what makes induction over a
///   recursive call work (Silicon's phase-1 `post` axiom).
pub(crate) fn post_rule(name: &str, func: FuncId, def: Arc<FunctionDefinition>) -> Rule {
    let searcher = AxiomTriggerSearcher { func, token: None };
    let applier = FunctionUnfoldApplier {
        func,
        def,
        limited_post: true,
        memo: Memo::new(),
        union_memo: Memo::new(),
        // Post-only rule: no definitional union, so no token gate.
        pre_token: None,
    };
    timed(Rewrite::new(format!("fn-post-{name}"), searcher, applier).expect("function post rule"))
}

/// Mint the limited-twin post rule for a **recursive** function: keyed on
/// `f'(args)` occurrences (which unfolding a recursive body produces), replays
/// the definition's post (no definitional union — that's the point
/// of the limited symbol). Registered alongside [`function_rule`] when
/// `def.limited` is set and a post fact exists.
pub(crate) fn function_post_rule(name: &str, def: Arc<FunctionDefinition>) -> Rule {
    let func = def.limited.expect("post rule requires a limited twin");
    post_rule(name, func, def)
}

/// A canonicalized call key: the ground `(type_args, value_args)` an occurrence
/// of the function was applied to. Memoized so the recipe rebuilds once per call.
pub(super) type CallKey = (Box<[Type]>, Vec<Id>);
