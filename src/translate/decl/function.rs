use crate::dhash::HashMap;
use std::marker::PhantomData;

use lasso::Spur;

use crate::translate::{DeclSlot, Declarator, Definer};
use crate::translate::{
    Declared, Metaed, MethodContracts, TranslationContext, TranslationError, pure_exp, spatial,
};
use crate::viper::typed;
use crate::vmir;

/// A function's `#requires` slot is a boolean [`vmir::Function`] when
/// heap-free, or a self-framed [`vmir::Resource`] (footprint + bool) when the
/// precondition grants permission (`heap_dep`, decided at declare time via
/// [`spatial::spatial_contains_acc`]).
pub(crate) enum RequiresSlot {
    HeapFree(DeclSlot<vmir::Function>),
    HeapDep(DeclSlot<vmir::Resource>),
}

pub(crate) struct FunctionTranslator<'a, P = Declared> {
    src: &'a typed::Function,
    silver_name: Spur,
    fn_slot: DeclSlot<vmir::Function>,
    requires_slot: Option<RequiresSlot>,
    ensures_slot: Option<DeclSlot<vmir::Function>>,
    requires: Option<vmir::MemberId>,
    ensures: Option<vmir::MemberId>,
    /// Set when `requires` contains an `acc` — the `#requires` slot is a
    /// self-framed `Resource` (not a boolean function), and call sites pass its
    /// snapshot as an extra argument.
    heap_dep: bool,
    _p: PhantomData<P>,
}

impl<'a> FunctionTranslator<'a, Declared> {
    pub(crate) fn declare(
        f: &'a typed::Function,
        ctx: &mut TranslationContext<'_>,
        d: &mut impl Declarator,
    ) -> Self {
        let name = ctx.interner.resolve(&f.name.0).to_string();
        let (fn_id, fn_slot) = d.alloc_slot::<vmir::Function>(&name);

        let mut requires = None;
        let mut requires_slot = None;
        let mut heap_dep = false;
        if let Some(req_exp) = &f.requires {
            // An `acc` in the precondition makes the function heap-dependent:
            // its `#requires` slot is filled with a Resource (not a boolean
            // Function), and call sites pass its snapshot as an extra argument.
            heap_dep = spatial::spatial_contains_acc(req_exp);
            let req_name = format!("{name}#requires");
            if heap_dep {
                let (id, slot) = d.alloc_slot::<vmir::Resource>(&req_name);
                requires = Some(id);
                requires_slot = Some(RequiresSlot::HeapDep(slot));
            } else {
                let (id, slot) = d.alloc_slot::<vmir::Function>(&req_name);
                requires = Some(id);
                requires_slot = Some(RequiresSlot::HeapFree(slot));
            }
        }

        let mut ensures = None;
        let mut ensures_slot = None;
        if f.ensures.is_some() {
            let (id, slot) = d.alloc_slot::<vmir::Function>(&format!("{name}#ensures"));
            ensures = Some(id);
            ensures_slot = Some(slot);
        }

        ctx.name_map.insert(f.name.0, fn_id);
        ctx.contracts.insert(
            f.name.0,
            MethodContracts {
                requires,
                ensures,
                heap_dep,
            },
        );

        FunctionTranslator {
            src: f,
            silver_name: f.name.0,
            fn_slot,
            requires_slot,
            ensures_slot,
            requires,
            ensures,
            heap_dep,
            _p: PhantomData,
        }
    }

    /// No `name_map`-dependent metadata to publish (contracts + `heap_dep` were
    /// published at declare).
    pub(crate) fn meta(self, _ctx: &mut TranslationContext<'_>) -> FunctionTranslator<'a, Metaed> {
        FunctionTranslator {
            src: self.src,
            silver_name: self.silver_name,
            fn_slot: self.fn_slot,
            requires_slot: self.requires_slot,
            ensures_slot: self.ensures_slot,
            requires: self.requires,
            ensures: self.ensures,
            heap_dep: self.heap_dep,
            _p: PhantomData,
        }
    }
}

impl FunctionTranslator<'_, Metaed> {
    /// Fill this function's reserved slots (see [`Self::declare`]): the main
    /// function decl, plus `#requires` / `#ensures` contracts when present.
    ///
    /// Functions are **pure and heap-free** in VMIR. A **heap-free** function
    /// (no `acc` in its `requires`) gets boolean contract functions stitched
    /// as pure `assume`/`assert`:
    /// - `#requires` → a boolean [`vmir::Function`] `params -> Bool`.
    /// - `#ensures` → a boolean [`vmir::Function`] `(params ++ result) -> Bool`.
    /// - the main function → its lowered body (when present) which
    ///   **assumes** `#requires(params)` at entry and **asserts**
    ///   `#ensures(params, result)` at exit.
    ///
    /// A **heap-dependent** function (`acc` in its `requires`) instead
    /// receives its precondition **snapshot** as a trailing parameter:
    /// - `#requires` → a self-framed [`vmir::Resource`] (footprint + bool);
    ///   its pure conjuncts live in the resource bool, so there is no separate
    ///   boolean requires-function.
    /// - `#ensures` → a boolean [`vmir::Function`]
    ///   `(params ++ [result, s: Snap(req)]) -> Bool` whose body reconstructs
    ///   the precondition heap from `s` (a bound `inhale`) and reads it.
    /// - the main function → `(params ++ [s: Snap(req)]) -> ret`; its body
    ///   opens with the same bound `inhale` (which implicitly assumes the
    ///   resource bool — no entry `assume`) and asserts
    ///   `#ensures(params, result, s)` at exit. Call sites build `s` with
    ///   a frame-only `exhale` of its `#requires` (which implicitly asserts the precondition) — see
    ///   `pure_exp::lower_func_app`.
    pub(crate) fn define(
        self,
        ctx: &TranslationContext<'_>,
        definer: &mut impl Definer,
    ) -> Result<(), TranslationError> {
        let FunctionTranslator {
            src: f,
            silver_name,
            fn_slot,
            requires_slot,
            ensures_slot,
            requires: meta_requires,
            ensures: meta_ensures,
            heap_dep,
            _p,
        } = self;

        let fname = ctx.interner.resolve(&silver_name).to_string();
        let n_params = f.params.len();
        let params: Vec<vmir::Type> = f.params.iter().map(|p| ctx.lower_type(&p.ty)).collect();
        let ret = ctx.lower_type(&f.ret);

        // Params occupy `Val::Temp(0..n_params)` in every body lowered below.
        let mut env: HashMap<Spur, vmir::Val> = HashMap::default();
        for (i, p) in f.params.iter().enumerate() {
            env.insert(p.name.0, vmir::Val::Temp(i));
        }

        // #requires: heap-free → a boolean function `params -> Bool`;
        // heap-dependent → a self-framed Resource (footprint + bool).
        if let Some(requires) = f.requires.as_ref() {
            let name = definer.intern_name(&format!("{fname}#requires"));
            match requires_slot.expect("declared when f.requires is Some") {
                RequiresSlot::HeapDep(slot) => {
                    let body = spatial::lower_spatial_never(
                        ctx,
                        &env,
                        requires,
                        n_params,
                        vmir::HeapVal::Empty,
                        0,
                        // Function precondition: weaken permissions to wildcard.
                        true,
                    )?;
                    definer.define_resource(
                        slot,
                        vmir::Resource {
                            name,
                            params: params.clone(),
                            precond: vmir::Precond::SelfFramed,
                            body,
                        },
                    );
                }
                RequiresSlot::HeapFree(slot) => {
                    let body = spatial::lower_pure_precond_body(ctx, &env, requires, n_params)?;
                    definer.define_function(
                        slot,
                        vmir::Function {
                            name,
                            params: params.clone().into(),
                            ret: vmir::Type::Bool,
                            body: Some(body),
                            requires: None,
                            ensures: None,
                        },
                    );
                }
            }
        }

        // The trailing snapshot parameter of a heap-dependent function (and of
        // its ensures function, where it sits after `result`).
        let snap_ty = heap_dep.then(|| {
            let req_id = meta_requires.expect("heap-dep implies a requires");
            vmir::Type::Snap(req_id)
        });
        let param_vals: Vec<vmir::Val> = (0..n_params).map(vmir::Val::Temp).collect();
        // Contract link over the leading params — set on the main decl AND on
        // the `#ensures` decl itself (whose facts are proven under the entry
        // pre assumption, so the verifier must guard them by this pre-token).
        // A heap-dependent link additionally names the decl's own snapshot
        // param, which sits at a different index in the two decls (trailing on
        // the main one, after `result` on `#ensures`).
        let requires_link = |snap: Option<vmir::Val>| -> Option<vmir::Requires> {
            let m = meta_requires?;
            Some(match snap {
                Some(snap) => vmir::Requires::Framed {
                    resource: m,
                    args: param_vals.clone(),
                    snap,
                },
                None => vmir::Requires::Pure(vmir::ContractCall {
                    member: m,
                    args: param_vals.clone(),
                }),
            })
        };

        // #ensures: a boolean function `(params ++ result) -> Bool`, with the
        // snapshot appended for a heap-dependent function. `result` occupies
        // `Val::Temp(n_params)`, the snapshot (if any) `Temp(n_params + 1)`;
        // body temps start after them.
        if let Some(ensures) = &f.ensures {
            let slot = ensures_slot.expect("declared when f.ensures is Some");
            let mut ens_params = params.clone();
            ens_params.push(ret.clone());
            let result = vmir::Val::Temp(n_params);
            let mut val_base = n_params + 1;
            let mut snap_entry = None;
            let mut ens_snap = None;
            if let Some(snap_ty) = &snap_ty {
                ens_params.push(snap_ty.clone());
                let vmir::Type::Snap(req_id) = snap_ty else {
                    unreachable!()
                };
                let snap_val = vmir::Val::Temp(val_base);
                snap_entry = Some(pure_exp::SnapEntry {
                    resource: *req_id,
                    args: param_vals.clone(),
                    snap: snap_val.clone(),
                });
                ens_snap = Some(snap_val);
                val_base += 1;
            }
            // WF context (heap-free): the postcondition's side conditions may
            // rely on the precondition (`requires y != 0 ensures result == x/y`),
            // so its body opens with `assume #requires(params)` — mirroring what
            // the heap-dependent a bound `inhale` entry assumes implicitly.
            let wf_contract = (!heap_dep).then(|| pure_exp::FnContract {
                requires: meta_requires,
                ensures: None,
                params: param_vals.clone(),
                snap: None,
            });
            let body = pure_exp::lower_function_body(
                ctx,
                &env,
                ensures,
                val_base,
                vmir::HeapVal::Empty,
                Some(result),
                wf_contract,
                snap_entry,
            )?;
            let name = definer.intern_name(&format!("{fname}#ensures"));
            definer.define_function(
                slot,
                vmir::Function {
                    name,
                    params: ens_params.into(),
                    ret: vmir::Type::Bool,
                    body: Some(body),
                    requires: requires_link(ens_snap),
                    ensures: None,
                },
            );
        }

        // The main function: pure and heap-free at the call boundary. Its body
        // (when present) assumes the precondition at entry — as a pure
        // `assume #requires(params)` when heap-free, or implicitly via the
        // a bound `inhale` heap reconstruction when heap-dependent — and asserts
        // `#ensures(params, result[, s])` at exit.
        let mut fn_params = params;
        let mut val_base = n_params;
        let mut snap_entry = None;
        let mut contract_snap = None;
        if let Some(snap_ty) = &snap_ty {
            fn_params.push(snap_ty.clone());
            let vmir::Type::Snap(req_id) = snap_ty else {
                unreachable!()
            };
            let snap_val = vmir::Val::Temp(n_params);
            snap_entry = Some(pure_exp::SnapEntry {
                resource: *req_id,
                args: param_vals.clone(),
                snap: snap_val.clone(),
            });
            contract_snap = Some(snap_val);
            val_base += 1;
        }
        let contract = pure_exp::FnContract {
            // Heap-dependent: the precondition is assumed by a bound `inhale`, not
            // by a boolean entry stitch.
            requires: if heap_dep { None } else { meta_requires },
            // The exit `assert #ensures(params, result[, s])` — the snapshot
            // rides along for a heap-dependent function (`FnContract.snap`).
            ensures: meta_ensures,
            params: param_vals.clone(),
            snap: contract_snap.clone(),
        };
        let body = match &f.body {
            None => None,
            Some(body_exp) => Some(pure_exp::lower_function_body(
                ctx,
                &env,
                body_exp,
                val_base,
                vmir::HeapVal::Empty,
                None,
                Some(contract),
                snap_entry,
            )?),
        };
        let name = definer.intern_name(&fname);
        // The ensures link: `#ensures(params, result[, s])`. `Result` is the
        // one place `ContractArg::Result` is expressible.
        let ensures_link = meta_ensures.map(|m| {
            let mut args: Vec<vmir::ContractArg> = param_vals
                .iter()
                .cloned()
                .map(vmir::ContractArg::Val)
                .collect();
            args.push(vmir::ContractArg::Result);
            if heap_dep {
                args.push(vmir::ContractArg::Val(vmir::Val::Temp(n_params)));
            }
            vmir::ContractCall { member: m, args }
        });
        definer.define_function(
            fn_slot,
            vmir::Function {
                name,
                params: fn_params.into(),
                ret,
                body,
                requires: requires_link(contract_snap),
                ensures: ensures_link,
            },
        );
        Ok(())
    }
}
