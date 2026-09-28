//! Lower `typed::ResourceExp` (the addressable resources `e.f` / `P(args)`) to
//! its VMIR address `Val`. An address is an ordinary heap-independent call to
//! the location's own function — the field's, or the predicate's — with the
//! `Type::Addr` return type published in `addr_types` during `declare`/`meta`.

use crate::dhash::HashMap;

use lasso::Spur;

use crate::translate::pure_exp::{self, HeapCtx, PureExt};
use crate::translate::sink::Sink;
use crate::translate::{TranslationContext, TranslationError};
use crate::viper::typed;
use crate::vmir::{self, PureInst, Val};

/// Lower a `ResourceExp` to its address: the location's own function applied to
/// the resource's base/arguments (`f(base)` or `P(args)`). The call is
/// heap-independent. Shared by `acc`, `perm`, and `new`.
pub(crate) fn lower_resource_addr<Ext: PureExt>(
    b: &TranslationContext<'_>,
    env: &HashMap<Spur, Val>,
    sink: &mut Sink,
    hctx: HeapCtx<'_>,
    res: &typed::ResourceExp<Ext>,
) -> Result<Val, TranslationError> {
    use typed::ResourceExpKind as R;
    match &*res.0 {
        R::Field(base, fname) => {
            let base_val = pure_exp::lower(b, env, sink, hctx, base)?;
            field_addr(b, sink, base_val, fname.0)
        }
        R::PredicateCall(call) => {
            let mut args = Vec::with_capacity(call.args.len());
            for a in &call.args {
                args.push(pure_exp::lower(b, env, sink, hctx, a)?);
            }
            addr_call(b, sink, call.name.0, args)
        }
    }
}

/// Emit a location's address call: the address function interned under `name`
/// applied to `args`, typed by the `Type::Addr` published in `addr_types`. The
/// one shape shared by every location — a field's `f(base)` and a predicate's
/// `P(args)` differ only in arity and in the permission cap carried by that
/// published type.
fn addr_call(
    b: &TranslationContext<'_>,
    sink: &mut Sink,
    name: Spur,
    args: Vec<Val>,
) -> Result<Val, TranslationError> {
    let unknown = || TranslationError::UnknownIdent(b.interner.resolve(&name).to_string());
    let id = *b.name_map.get(&name).ok_or_else(unknown)?;
    let ret_ty = b.addr_types.get(&name).cloned().ok_or_else(unknown)?;
    Ok(sink.emit_pure(
        ret_ty,
        PureInst::FunctionCall(vmir::FunctionCall {
            function: id,
            type_args: Vec::new(),
            args: args.into(),
            // An address function, not a Silver `function`.
            export: false,
        }),
    ))
}

/// Emit `field(base)`: the field's heap-independent address function applied to
/// the receiver, typed `Addr<field_ty>`. Its own entry point because field
/// assignment and `new` reach a field location without a `ResourceExp`.
pub(crate) fn field_addr(
    b: &TranslationContext<'_>,
    sink: &mut Sink,
    base: Val,
    fname: Spur,
) -> Result<Val, TranslationError> {
    addr_call(b, sink, fname, vec![base])
}

/// Lower `acc(base.fname, perm)` to its `(loc, perm)`: the field's address
/// function applied to `base`, paired with the permission amount. The caller
/// emits the `HeapInst::Add`. Shared by `new(...)` lowering.
pub(crate) fn field_acc(
    b: &TranslationContext<'_>,
    sink: &mut Sink,
    base: Val,
    fname: Spur,
    perm: Val,
) -> Result<(Val, Val), TranslationError> {
    let addr = field_addr(b, sink, base, fname)?;
    Ok((addr, perm))
}
