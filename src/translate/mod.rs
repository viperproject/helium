//! Lowers Silver `typed::Program` to `vmir::Program`.
//!
//! The coordinator (`translate`) runs **exactly one translator per Viper
//! declaration** (`field`/`predicate`/`function`/`method`/`adt`/`domain`)
//! through **three type-enforced phases**:
//!
//! 1. **declare** — allocate every [`DeclSlot`] write-capability token and
//!    publish dependency-free metadata straight into `TranslationContext`
//!    (`name_map`, `contracts`, `fn_generic_sigs`). Needs nothing but the
//!    source node + the ids it just minted.
//! 2. **meta** — publish `name_map`-dependent metadata now that every id
//!    exists (`field_types`, ADT ctor/dtor info). No-op for all kinds but
//!    Field and ADT.
//! 3. **define** — lower bodies and consume every [`DeclSlot`], producing the
//!    actual `vmir::Declaration` payloads. Order across kinds is free: a body
//!    references any callee's contract by *id* (from `ctx.contracts`), never
//!    its filled slot.
//!
//! Phase order is enforced by the type system — a translator changes type
//! (`FooTranslator<'a, Declared>` → `<'a, Metaed>`), and `define` exists only
//! on the `Metaed` state. This is a second typestate axis, orthogonal to the
//! [`DeclSlot`] write capability (every reserved slot is filled exactly once;
//! checked in `Builder::finalize`).

use crate::viper::typed;
use crate::vmir;

/// Phase markers for the translator typestate (declare → meta → define). ZSTs:
/// they gate which methods exist, and carry no data.
pub(crate) struct Declared;
pub(crate) struct Metaed;

mod builder;
mod context;
mod decl;
pub mod errors;
mod pure_exp;
mod reach;
mod resource;
mod sink;
mod spatial;
mod types;

pub use errors::TranslationError;

pub(crate) use builder::{Builder, DeclSlot, Declarator, Definer};
pub(crate) use context::{MethodContracts, TranslationContext};

/// Build a `vmir::Program` from a typed `typed::Program`.
/// Lower a typed program, reporting any failure against the declaration it came
/// from. A declaration that cannot be lowered leaves its slot unfilled, so
/// there is no partial program to hand back: the caller drops the named
/// declarations and calls again. [`translate`] is the wrapper that discards the
/// names.
pub fn translate_reporting(
    program: &typed::Program,
) -> Result<vmir::Program, Vec<(String, TranslationError)>> {
    let decls = &program.decls;
    let mut builder = Builder::new();
    let mut ctx = TranslationContext::new(&program.interner);

    // ── Phase 1: declare ──
    // Allocate every slot and publish dependency-free meta into `ctx`. One
    // typed collection per Viper declaration kind; order within a kind and
    // across kinds is irrelevant (declare only mints ids + reads its source).
    // Each translator is carried with the name of the declaration it came from,
    // so a `define` failure can be reported against that declaration.
    let mut fields = Vec::new();
    let mut preds = Vec::new();
    let mut funcs = Vec::new();
    let mut methods = Vec::new();
    let mut adts = Vec::new();
    let mut domains = Vec::new();
    for decl_node in decls {
        let name = decl_name(decl_node, &program.interner);
        match decl_node {
            typed::Declaration::Field(f) => {
                fields.push((
                    name,
                    decl::FieldTranslator::declare(f, &mut ctx, &mut builder),
                ));
            }
            typed::Declaration::Predicate(p) => {
                preds.push((
                    name,
                    decl::PredicateTranslator::declare(p, &mut ctx, &mut builder),
                ));
            }
            typed::Declaration::Function(f) => {
                funcs.push((
                    name,
                    decl::FunctionTranslator::declare(f, &mut ctx, &mut builder),
                ));
            }
            typed::Declaration::Method(m) => {
                methods.push((
                    name,
                    decl::MethodTranslator::declare(m, &mut ctx, &mut builder),
                ));
            }
            typed::Declaration::Adt(a) => {
                adts.push((
                    name,
                    decl::AdtTranslator::declare(a, &mut ctx, &mut builder),
                ));
            }
            typed::Declaration::Domain(d) => {
                domains.push((
                    name,
                    decl::DomainTranslator::declare(d, &mut ctx, &mut builder),
                ));
            }
        }
    }

    let wands = decl::WandTranslator::declare(&program.wands, &mut ctx, &mut builder);

    // Barrier: `name_map` is complete. Snapshot the group interner into `ctx`
    // (owned, independent of `builder` from here on) — every group tag any
    // body-lowering helper looks up was registered during declare.
    ctx.groups = builder.clone_groups();

    // ── Phase 2: meta ──
    // Publish `name_map`-dependent meta now that every id exists. No-op for all
    // kinds but Field (`field_types`) and ADT (ctor/dtor info).
    let fields: Vec<_> = fields
        .into_iter()
        .map(|(n, t)| (n, t.meta(&mut ctx)))
        .collect();
    let preds: Vec<_> = preds
        .into_iter()
        .map(|(n, t)| (n, t.meta(&mut ctx)))
        .collect();
    let funcs: Vec<_> = funcs
        .into_iter()
        .map(|(n, t)| (n, t.meta(&mut ctx)))
        .collect();
    let methods: Vec<_> = methods
        .into_iter()
        .map(|(n, t)| (n, t.meta(&mut ctx)))
        .collect();
    let adts: Vec<_> = adts
        .into_iter()
        .map(|(n, t)| (n, t.meta(&mut ctx)))
        .collect();
    let domains: Vec<_> = domains
        .into_iter()
        .map(|(n, t)| (n, t.meta(&mut ctx)))
        .collect();

    // Barrier: `ctx` is complete.
    // ── Phase 3: define ──
    // Lower bodies and consume every `DeclSlot`, collecting any errors. Order
    // across kinds is free.
    let mut errors: Vec<(String, TranslationError)> = Vec::new();
    errors.extend(
        fields
            .into_iter()
            .filter_map(|(n, t)| t.define(&ctx, &mut builder).err().map(|e| (n, e))),
    );
    errors.extend(
        preds
            .into_iter()
            .filter_map(|(n, t)| t.define(&ctx, &mut builder).err().map(|e| (n, e))),
    );
    errors.extend(
        funcs
            .into_iter()
            .filter_map(|(n, t)| t.define(&ctx, &mut builder).err().map(|e| (n, e))),
    );
    errors.extend(
        adts.into_iter()
            .filter_map(|(n, t)| t.define(&ctx, &mut builder).err().map(|e| (n, e))),
    );
    errors.extend(
        methods
            .into_iter()
            .filter_map(|(n, t)| t.define(&ctx, &mut builder).err().map(|e| (n, e))),
    );
    wands.define(&ctx, &mut builder);
    // Domains take `&mut ctx`: axiom lowering scopes the axiom's type
    // parameters into `ctx.decl_generics` (cleared before returning).
    errors.extend(
        domains
            .into_iter()
            .filter_map(|(n, t)| t.define(&mut ctx, &mut builder).err().map(|e| (n, e))),
    );

    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(builder.finalize())
}

/// The name a typed declaration is reported under.
fn decl_name(decl: &typed::Declaration, interner: &crate::viper::Interner) -> String {
    let spur = match decl {
        typed::Declaration::Function(f) => f.name.0,
        typed::Declaration::Predicate(p) => p.name.0,
        typed::Declaration::Method(m) => m.name.0,
        typed::Declaration::Field(f) => f.0.name.0,
        typed::Declaration::Adt(a) => a.name.0,
        typed::Declaration::Domain(d) => d.name.0,
    };
    interner.resolve(&spur).to_string()
}

/// All-or-nothing translation, discarding the declaration names.
pub fn translate(program: &typed::Program) -> Result<vmir::Program, Vec<TranslationError>> {
    translate_reporting(program).map_err(|es| es.into_iter().map(|(_, e)| e).collect())
}

pub(crate) use types::lower_type;

#[cfg(test)]
mod tests;
