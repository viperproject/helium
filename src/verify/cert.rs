//! Verified-body **definitions**: a function's or resource's body captured once
//! as an add-only **term recipe** and rebuilt at each call site with
//! [`build_instance`](crate::verify::rewrite::build_instance) (formal params →
//! actual args). Unlike the earlier e-graph certificates these import **no
//! e-classes**, so a merge proven during the body's own verification (e.g. a
//! precondition equality) never rides along into a caller — the caller re-derives
//! whatever it needs itself. See the plan's Findings B/C.

use egg::{EGraph, Id};

use crate::verify::analysis::ConstFold;
use crate::verify::error::VerifyError;
use crate::verify::heap::LocationKind;
use crate::verify::lang::{FuncId, Symbolic};
use crate::verify::rewrite::{AxiomInst, AxiomPure};
use crate::vmir::{MemberId, PermInst, PermVal, Polarity, Type, Val};
/// A (non-recursive) function's verified body as a **pure term recipe** — the
/// definition `f(params) == <steps>[res]`, add-only. Unlike a certificate this
/// imports **no e-classes**: the function unfold rule rebuilds `steps` at each
/// call site with [`build_instance`](crate::verify::rewrite::build_instance)
/// (params → args), so precondition-derived merges from the body's verification
/// never ride along (Finding B). `steps` is in dense recipe-temp space (params at
/// `Val::Temp(0..n_params)`, one slot per step); `res` is the result Val.
#[derive(Clone)]
pub(crate) struct FunctionDefinition {
    pub(crate) n_params: usize,
    pub(crate) steps: Vec<AxiomInst>,
    /// The body's result value — `None` for an **abstract** function, whose
    /// synthesized definition carries only `facts` (its guarded post axiom)
    /// and installs no definitional union.
    pub(crate) res: Option<Val>,
    /// The function's limited-twin id `f'`, `Some` iff the function is
    /// (mutually) recursive. When set, its unfold rule additionally frames
    /// `f(x) == f'(x)` at every full occurrence, and the recipe's own in-SCC
    /// recursive calls already target `f'` (uninterpreted) so unfolding halts
    /// after one level. `None` for a non-recursive function (unchanged behavior).
    pub(crate) limited: Option<FuncId>,
    /// Temps holding a nested callee's `g%pre(gargs)` token — Silicon's
    /// `bodyPreconditionPropagation`, emitted as an ordinary `App` step whose
    /// value is never read (see [`RecipeBuilder::token_steps`]). Rebuilding a step
    /// only *adds* the node, which makes `g` materializable here; the unfold rule
    /// additionally releases each of these under the enclosing `f%pre` truth, so a
    /// nested callee's axioms activate exactly when this body's own do, and only
    /// under the body-internal condition guarding the nested call.
    pub(crate) token_steps: Vec<TokenStep>,
    /// The **only** thing this function publishes to a caller:
    /// `f#ensures(params, f(params))` (`f'` when recursive), built from the
    /// declaration's `ensures` link — never from the body. `None` when the
    /// function declares no `ensures`.
    ///
    /// Replayed as `f%pre(args) ⟹ post` at every occurrence of `f(args)`
    /// (Silicon's `post` axiom), and at limited-twin occurrences too, which is
    /// what makes induction over a recursive call work. Carries no guards of its
    /// own: the call-site token is minted exactly where the precondition was
    /// checked, so a second conjunct restating it would add nothing.
    pub(crate) post: Option<PostRecipe>,
}

/// A function's exported postcondition as a **standalone** recipe over the
/// params alone: `Temp(0..n_params)` are the args, then one temp per step, and
/// `res` is `f#ensures(params, f(params))`.
///
/// Standalone, and not a slice of [`FunctionDefinition::steps`], is the whole
/// point: it mentions no body temp, so it can be replayed at an occurrence where
/// no body was materialized. That is what lets the definitional union's presence
/// gate actually gate — an occurrence no call produced pays for the post and
/// nothing else.
#[derive(Clone)]
pub(crate) struct PostRecipe {
    pub(crate) steps: Vec<AxiomInst>,
    pub(crate) res: Val,
    /// The `#ensures` member's own presence token,
    /// `f#ensures%pre(link_args, f(params))` — one more step of `steps`, so
    /// replaying the recipe *adds* the node, and the unfold rule additionally
    /// releases its truth under the call-site `f%pre`.
    ///
    /// A `#ensures` is an ordinary function whose only occurrence is here: no
    /// call ever mints its token, so without this it would never unfold and the
    /// callees it names would stay opaque at the caller. Silicon's
    /// `postPreconditionPropagationAxiom` (`FunctionData.scala:311`),
    /// `f%pre ⟹ tr(post)`, is the same release — our post *is* one opaque call,
    /// so `tr` of it is exactly this token.
    ///
    /// Presence, not knowledge: it licenses unfolding, it does not state that the
    /// postcondition holds. That is [`PostRecipe::res`], released beside it.
    /// `None` only for a declaration whose `ensures` link has no member.
    pub(crate) token: Option<Val>,
}

/// One propagated precondition token: a nested callee's `g%pre(gargs)` step plus
/// the **callee-internal** path condition guarding the call that produced it.
/// Released as `guards ==> token` (and, in a function definition, additionally
/// under the enclosing `f%pre` — see `FunctionUnfoldApplier`), giving
/// `f_pre(a) ==> (b[x:=a] ==> g_pre(gargs[x:=a]))` for a body
/// `f(x) { b ? g(x) : .. }`. Without the guards the release is too eager: `g`'s
/// own axioms would fire at args where this body never calls it.
///
/// Distinct from [`FunctionDefinition::post`]: that is a *fact* about the
/// callee's value, released once on the first build; this is a *presence* stamp,
/// re-released on every (re-)union so a propagation-minted token can still
/// reach a call that was first built without one.
///
/// `guards` is outermost-first and folded innermost-first at release.
#[derive(Clone)]
pub(crate) struct TokenStep {
    pub(crate) token: Val,
    pub(crate) guards: Vec<(Val, Polarity)>,
}

/// One seed slot of a [`BodyRecipe`]: resolved at graft time to an actual arg or
/// an earlier footprint slot's (caller-supplied) value.
#[derive(Clone, Debug)]
pub(crate) enum SeedRef {
    Param(usize),
    /// The value of footprint slot `i` — supplied at graft time (fresh on inhale,
    /// read from the heap on fold/exhale, projected from a snapshot on
    /// unfold/from_snap). A value-dependent inner address (`list(this.next)`)
    /// references an earlier slot's value this way.
    SlotValue(usize),
}

/// A self-contained pure recipe rebuilt at a call site with
/// [`build_instance`](crate::verify::rewrite::build_instance). Its `steps` live
/// in dense recipe-temp space: `Val::Temp(0..seed_refs.len())` are the seed
/// (resolved per `seed_refs`), then one temp per step. `res` is the output Val.
#[derive(Clone)]
pub(crate) struct BodyRecipe {
    pub(crate) seed_refs: Vec<SeedRef>,
    pub(crate) steps: Vec<AxiomInst>,
    pub(crate) res: Val,
    /// Temps holding an orphan `g%pre(gargs)` token kept by
    /// [`RecipeBuilder::slice_with_tokens`] (their values are never read, so the
    /// backward closure would otherwise prune them). Rebuilding a step only *adds*
    /// the node; [`Self::build`] additionally releases each token's truth, which is
    /// what lets a contract-introduced function application unfold at a client.
    /// Released under the **graft site's** path condition, which the caller passes
    /// to [`Self::build`]: a resource has no enclosing `f%pre` to inherit, but a
    /// graft on one arm of a branch must no more release its callees' axioms on the
    /// other arm than a function call's own token may. Each token's body-internal
    /// guards apply inside that (see [`TokenStep`]).
    pub(crate) token_steps: Vec<TokenStep>,
}

impl BodyRecipe {
    /// Rebuild this recipe in `egraph`, resolving each seed slot via `resolve`.
    /// Add-only (imports no e-classes). `changed` accumulates altered classes.
    ///
    /// `pc` is the graft site's path condition (outermost-first), applied to the
    /// [`Self::token_steps`] releases — the built terms themselves need no guard,
    /// being terms rather than facts.
    pub(crate) fn build(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        resolve: impl Fn(&SeedRef) -> Id,
        pc: &[(Id, Polarity)],
        changed: &mut Vec<Id>,
    ) -> Id {
        let seed: Vec<Id> = self.seed_refs.iter().map(resolve).collect();
        crate::verify::rewrite::build_instance_releasing_tokens(
            egraph,
            &self.steps,
            &self.res,
            &seed,
            &self.token_steps,
            pc,
            changed,
        )
    }
}

/// A footprint slot's permission in recipe space: the IR's `p`-temp steps with
/// every operand replaced by its own recipe. `res` names the result — an inline
/// leaf, or `PermVal::Temp(i)` indexing `steps`, which live in their own dense
/// space (`Temp(i)` refers to `steps[i]`, and a step only ever names earlier
/// ones).
///
/// Deliberately **not** flattened into [`AxiomInst`] steps. A permission amount
/// is a pure function of the params and earlier slot values, so it belongs in a
/// recipe — but a `wildcard` is not an amount, it is a symbolic picked when the
/// slot is grafted. Flattening used to force an `AxiomPure::Wildcard` *step* to
/// exist, and a step is rebuilt wherever the recipe is rebuilt — including inside
/// an egg applier, which has no `VerifyContext` and so minted from a
/// process-global counter. Keeping the permission in its own space means
/// `build_perm` does the picking, at a site that holds `ctx`.
#[derive(Clone)]
pub(crate) struct PermRecipe<A = BodyRecipe> {
    pub(crate) steps: Vec<PermInst<A>>,
    pub(crate) res: PermVal<A>,
}

impl<A> PermRecipe<A> {
    /// Whether any leaf of this permission is a `wildcard`. Shape-only, so it
    /// serves both spaces — the same predicate the IR side asks.
    pub(crate) fn has_wildcard(&self) -> bool {
        self.res.is_wildcard() || self.steps.iter().any(PermInst::has_wildcard_arm)
    }

    /// Rebuild with every operand mapped through `f` (one `slice` per operand
    /// when going from body temps to standalone recipes), shape preserved.
    pub(crate) fn try_map<B, E>(
        &self,
        f: &mut impl FnMut(&A) -> Result<B, E>,
    ) -> Result<PermRecipe<B>, E> {
        Ok(PermRecipe {
            steps: self
                .steps
                .iter()
                .map(|st| st.try_map(f))
                .collect::<Result<Vec<_>, E>>()?,
            res: self.res.try_map(f)?,
        })
    }
}

/// One footprint slot of a [`ResourceDefinition`]: its location kind and element
/// type, plus recipes for its address and permission (over the params and any
/// earlier slot values — see [`SeedRef`]).
#[derive(Clone)]
pub(crate) struct SlotRecipe {
    pub(crate) kind: LocationKind,
    pub(crate) elem: Type,
    pub(crate) addr: BodyRecipe,
    pub(crate) perm: PermRecipe,
}

/// A resource's verified body as a **pure term recipe**. Each call site rebuilds
/// the footprint addresses, permissions, and body boolean from these recipes
/// (params → args, slot values supplied by the caller), importing no e-classes;
/// the caller re-derives any merge it needs via its own
/// `heap_union`/`assume_location_axioms` (Finding C).
#[derive(Clone)]
pub(crate) struct ResourceDefinition {
    pub(crate) footprint: Vec<SlotRecipe>,
    /// The body boolean over the params and every footprint slot value.
    pub(crate) bool: BodyRecipe,
}

/// One step of a [`RecipeBuilder`] stream: either a **seed** (a footprint
/// slot's value placeholder, resolved at graft time — resources only) or an
/// ordinary pure step. Temps are dense over the combined stream, so a seed can
/// be introduced mid-body without renumbering.
enum RecipeStep {
    Seed(SeedRef),
    Inst(AxiomInst),
}

/// The recipe under construction during a function's or resource's **single**
/// eval walk: pure steps are mirrored into this stream as the body is
/// evaluated, so the certificate falls out of verification itself (no second
/// purify walk to keep in arm-parity). Recipe-temp space: `Temp(0..n_params)`
/// are the params (identity), then one temp per step.
pub(crate) struct RecipeBuilder {
    n_params: usize,
    steps: Vec<RecipeStep>,
    /// In-SCC callees (a recursion cycle's members), lowered to their limited
    /// twin so a downstream unfold halts after one level.
    recursive_scc: Option<crate::dhash::HashSet<MemberId>>,
    /// Footprint slots recorded by a resource body's `acc`s, in body order:
    /// the slot's location kind, element type, and the recipe temps of its
    /// address and permission (sliced into standalone [`SlotRecipe`]s at the
    /// end of the walk).
    pub(crate) pending_slots: Vec<(LocationKind, Type, Val, PermRecipe<Val>)>,
    /// Orphan steps that must survive [`Self::slice`]'s backward closure: the
    /// `g%pre` precondition tokens emitted alongside a callee application. Their
    /// value is never consumed, so reachability-from-the-result would prune them
    /// -- and then a resource recipe replayed at a client mints the callee
    /// application without its token, leaving the callee's body permanently
    /// un-unfolded at that occurrence (function bodies are unaffected: they keep
    /// every step via `into_function_parts`). Each carries the callee-internal pc
    /// of the call it accompanies (see [`TokenStep`]).
    token_steps: Vec<TokenStep>,
}

impl RecipeBuilder {
    pub(crate) fn new(
        n_params: usize,
        recursive_scc: Option<crate::dhash::HashSet<MemberId>>,
    ) -> Self {
        Self {
            n_params,
            steps: Vec::new(),
            recursive_scc,
            pending_slots: Vec::new(),
            token_steps: Vec::new(),
        }
    }

    /// Record an orphan `g%pre` token step so [`Self::slice`] keeps it (see
    /// [`Self::token_steps`]), together with the callee-internal path condition of
    /// the call it accompanies — empty when the walk has no recipe-space pc to
    /// offer, which reproduces the pre-guard behavior (see [`TokenStep`]).
    pub(crate) fn record_token_step(&mut self, token: Val, guards: Vec<(Val, Polarity)>) {
        self.token_steps.push(TokenStep { token, guards });
    }

    fn next_temp(&self) -> Val {
        Val::Temp(self.n_params + self.steps.len())
    }

    pub(crate) fn emit(&mut self, p: AxiomPure) -> Val {
        let v = self.next_temp();
        self.steps.push(RecipeStep::Inst(AxiomInst::Val(p)));
        v
    }

    pub(crate) fn emit_forall(
        &mut self,
        recipe: crate::verify::lang::RecipeId,
        caps: Vec<Val>,
    ) -> Val {
        let v = self.next_temp();
        self.steps
            .push(RecipeStep::Inst(AxiomInst::Forall { recipe, caps }));
        v
    }

    pub(crate) fn emit_seed(&mut self, s: SeedRef) -> Val {
        let v = self.next_temp();
        self.steps.push(RecipeStep::Seed(s));
        v
    }

    pub(crate) fn is_recursive_callee(&self, m: MemberId) -> bool {
        self.recursive_scc.as_ref().is_some_and(|s| s.contains(&m))
    }

    /// Consume the builder into a function definition's parts. A function
    /// stream contains no `Seed` steps (params are pre-seeded), so it converts
    /// 1:1 — `token_steps` index into the same shared stream, as does the post
    /// fact the caller appends afterwards.
    pub(crate) fn into_function_parts(
        self,
    ) -> Result<(Vec<AxiomInst>, Vec<TokenStep>), VerifyError> {
        let steps = self
            .steps
            .into_iter()
            .map(|s| match s {
                RecipeStep::Inst(i) => Ok(i),
                RecipeStep::Seed(_) => Err(VerifyError::Unimplemented(
                    "purify: seed step in a function recipe",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((steps, self.token_steps))
    }

    /// Slice the self-contained [`BodyRecipe`] computing `out` from the shared
    /// stream: backward closure over the step DAG, reached params/seeds become
    /// `seed_refs`, reached steps re-emit densely in original (topological)
    /// order. Unlike the old `RTree` flatten, shared subterms stay shared.
    pub(crate) fn slice(&self, out: &Val) -> Result<BodyRecipe, VerifyError> {
        self.slice_impl(out, false)
    }

    /// [`Self::slice`], additionally keeping the orphan `g%pre` token steps (see
    /// [`Self::token_steps`]). Only for a resource's **body boolean**: a footprint
    /// slot's recipe resolves `SeedRef::SlotValue(j)` against the slot values built
    /// so far, so pulling a token that mentions a later slot into a slot recipe
    /// would index past the end.
    pub(crate) fn slice_with_tokens(&self, out: &Val) -> Result<BodyRecipe, VerifyError> {
        self.slice_impl(out, true)
    }

    fn slice_impl(&self, out: &Val, keep_tokens: bool) -> Result<BodyRecipe, VerifyError> {
        let Val::Temp(root) = out else {
            return Ok(BodyRecipe {
                seed_refs: Vec::new(),
                steps: Vec::new(),
                res: out.clone(),
                token_steps: Vec::new(),
            });
        };
        let n = self.n_params + self.steps.len();
        let mut reach = vec![false; n];
        let mut stack = vec![*root];
        if keep_tokens {
            // Token steps are roots in their own right (see `token_steps`), and so
            // is each token's guard: a guard's *defining* steps are not reachable
            // from the result either, so without seeding them the backward closure
            // prunes them and the remap below finds no translation.
            for t in &self.token_steps {
                if let Val::Temp(i) = t.token {
                    stack.push(i);
                }
                for (g, _) in &t.guards {
                    if let Val::Temp(i) = g {
                        stack.push(*i);
                    }
                }
            }
        }
        while let Some(i) = stack.pop() {
            if reach[i] {
                continue;
            }
            reach[i] = true;
            if i < self.n_params {
                continue;
            }
            if let RecipeStep::Inst(inst) = &self.steps[i - self.n_params] {
                for_each_operand(inst, |v| {
                    if let Val::Temp(j) = v {
                        stack.push(*j);
                    }
                });
            }
        }
        // Seeds first (ascending stream order), then reached steps.
        let mut seed_refs: Vec<SeedRef> = Vec::new();
        let mut new_val: Vec<Option<Val>> = vec![None; n];
        for i in 0..n {
            if !reach[i] {
                continue;
            }
            let seed = if i < self.n_params {
                Some(SeedRef::Param(i))
            } else if let RecipeStep::Seed(s) = &self.steps[i - self.n_params] {
                Some(s.clone())
            } else {
                None
            };
            if let Some(s) = seed {
                new_val[i] = Some(Val::Temp(seed_refs.len()));
                seed_refs.push(s);
            }
        }
        let base = seed_refs.len();
        let mut steps: Vec<AxiomInst> = Vec::new();
        for i in self.n_params..n {
            if !reach[i] {
                continue;
            }
            let RecipeStep::Inst(inst) = &self.steps[i - self.n_params] else {
                continue;
            };
            // Operands precede their use (the stream is topologically sorted),
            // so they are translated before this step's slot is assigned.
            let translated = map_operands(inst, |v| match v {
                Val::Temp(j) => new_val[*j].clone().expect("operand precedes its use"),
                Val::Literal(l) => Val::Literal(l.clone()),
            });
            new_val[i] = Some(Val::Temp(base + steps.len()));
            steps.push(translated);
        }
        let res = new_val[*root].clone().expect("root is reached");
        // Token steps are reached only when `keep_tokens`; remap them into the
        // sliced temp space so `BodyRecipe::build` can release each one's truth.
        // A token that does not remap is dropped (it was never reached, so nothing
        // will mint it); a *guard* that does not remap must never be dropped — that
        // would strengthen the release, which is the unsound direction — so it is an
        // error instead, since the root seeding above is what makes it impossible.
        let mut token_steps: Vec<TokenStep> = Vec::new();
        for t in &self.token_steps {
            let Val::Temp(i) = t.token else { continue };
            let Some(token) = new_val[i].clone() else {
                continue;
            };
            let guards =
                t.guards
                    .iter()
                    .map(|(g, pol)| match g {
                        Val::Temp(j) => new_val[*j].clone().map(|v| (v, *pol)).ok_or(
                            VerifyError::Unimplemented("purify: token guard pruned by slice"),
                        ),
                        Val::Literal(l) => Ok((Val::Literal(l.clone()), *pol)),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            token_steps.push(TokenStep { token, guards });
        }
        Ok(BodyRecipe {
            seed_refs,
            steps,
            res,
            token_steps,
        })
    }
}

/// Visit every `Val` operand of a pure recipe step **in place**.
///
/// The mutating traversal is the one [`map_operands`] is built on: rebuilding a
/// step is a `clone` with its operands overwritten, never a re-construction of
/// every variant. That is what makes it impossible to silently drop a
/// *non*-operand field (`func`, `type_args`, `recipe`, a guard's `Polarity`) when
/// a step shape gains one — the old reconstructing form had to restate each of
/// them by hand at every arm.
fn for_each_operand_mut(inst: &mut AxiomInst, mut f: impl FnMut(&mut Val)) {
    match inst {
        AxiomInst::Val(p) => match p {
            AxiomPure::Binary(_, l, r) => {
                f(l);
                f(r);
            }
            AxiomPure::Ternary(c, t, e) => {
                f(c);
                f(t);
                f(e);
            }
            AxiomPure::RealCast(v) => f(v),
            AxiomPure::App { args, .. } => args.iter_mut().for_each(f),
        },
        AxiomInst::Forall { caps, .. } => caps.iter_mut().for_each(f),
        AxiomInst::Assume(v) => f(v),
        AxiomInst::Token { args, guards, .. } => {
            for v in args.iter_mut().chain(guards.iter_mut().map(|(g, _)| g)) {
                f(v);
            }
        }
    }
}

/// Visit each `Val` operand of a pure recipe step.
///
/// Read-only mirror of [`for_each_operand_mut`]: Rust cannot abstract one
/// traversal over mutability without a macro, so the two are kept adjacent and
/// arm-for-arm identical. A step shape that gains an operand must be added to
/// both — but only to the *walk*, never to a rebuild.
fn for_each_operand(inst: &AxiomInst, mut f: impl FnMut(&Val)) {
    match inst {
        AxiomInst::Val(p) => match p {
            AxiomPure::Binary(_, l, r) => {
                f(l);
                f(r);
            }
            AxiomPure::Ternary(c, t, e) => {
                f(c);
                f(t);
                f(e);
            }
            AxiomPure::RealCast(v) => f(v),
            AxiomPure::App { args, .. } => args.iter().for_each(f),
        },
        AxiomInst::Forall { caps, .. } => caps.iter().for_each(f),
        AxiomInst::Assume(v) => f(v),
        AxiomInst::Token { args, guards, .. } => {
            for v in args.iter().chain(guards.iter().map(|(g, _)| g)) {
                f(v);
            }
        }
    }
}

/// Rebuild a pure recipe step with each `Val` operand translated by `tr`.
pub(crate) fn map_operands(inst: &AxiomInst, tr: impl Fn(&Val) -> Val) -> AxiomInst {
    let mut out = inst.clone();
    for_each_operand_mut(&mut out, |v| {
        let mapped = tr(v);
        *v = mapped;
    });
    out
}
