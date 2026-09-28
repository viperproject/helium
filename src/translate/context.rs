//! `TranslationContext` — the read-only state every body-lowering helper
//! (`resource.rs`/`method.rs`/`pure_exp.rs`/`spatial.rs`) consumes. It **owns**
//! its maps (`name_map`, `contracts`, ...), built up progressively by the
//! coordinator (`mod.rs`) folding each `Translator::declare`'s `Meta` as
//! members are declared. A genuinely separate value from `Builder` (the write
//! side), not a borrowed view of it, which is what lets a `Translator::define`
//! take `&TranslationContext` and `&mut impl Definer` in one call without an
//! aliasing conflict.

use crate::dhash::HashMap;

use lasso::{Rodeo, Spur};

use crate::viper::{Interner, typed};
use crate::vmir;

/// ADT shape metadata recorded in `declare`, consumed when lowering `AdtCons` /
/// `AdtProj` / `AdtTag` use sites.
#[derive(Default)]
pub(crate) struct AdtInfo {
    /// A constructor's `Spur` to `(owning ADT `Spur`, tag index)`.
    pub ctor_tag: HashMap<Spur, (Spur, usize)>,
    /// A destructor's `Spur` to the `(adt id, variant, field)` it projects.
    pub dtor_sem: HashMap<Spur, (vmir::MemberId, usize, usize)>,
}

/// A member's contract ids (`#requires` / `#ensures`), absent when the member
/// omits that clause. For a **method** both are Resource ids. For a **function**
/// they are boolean Function ids, except a heap-dependent function's
/// `requires` (`heap_dep == true`), which is a self-framed Resource id — the
/// footprint whose snapshot the function takes as its trailing parameter.
#[derive(Default)]
pub(crate) struct MethodContracts {
    pub requires: Option<vmir::MemberId>,
    pub ensures: Option<vmir::MemberId>,
    /// Set only for functions whose `requires` grants permission (`acc`):
    /// call sites pass a `Snap` of the `requires` resource as an extra argument.
    pub heap_dep: bool,
}

/// Read-only mid-translation state, owned and progressively folded by the
/// coordinator. Every body-lowering helper takes `&TranslationContext` — none
/// of them touch `Builder`'s write side (`decls`/`vmir_interner`/`decl_names`)
/// directly.
pub(crate) struct TranslationContext<'a> {
    pub interner: &'a Interner,
    /// Silver `Spur` names to VMIR `MemberId`s.
    pub name_map: HashMap<Spur, vmir::MemberId>,
    /// A location's `Spur` (field **or** predicate name) to the complete
    /// `Type::Addr` its address function returns — group tag, value type, and
    /// permission bound (`1/1` for a field, unbounded for a predicate). Read by
    /// `resource::lower_resource_addr`, which is thereby the same lookup for
    /// both kinds of location.
    pub addr_types: HashMap<Spur, vmir::Type>,
    /// A **concrete** predicate's `Spur` to its `vmir::Resource` id. An abstract
    /// predicate has no entry: it lowers to a `Function` + `Domain` pair, so a
    /// miss here *is* the "cannot fold an abstract predicate" error, with no
    /// explicit abstractness test anywhere.
    pub pred_resources: HashMap<Spur, vmir::MemberId>,
    /// A method's `Spur` to its contract resource ids.
    pub contracts: HashMap<Spur, MethodContracts>,
    /// ADT constructor/destructor metadata.
    pub adt: AdtInfo,
    /// Location **group** tags (`Type::Addr.group`) — field/predicate names.
    /// A clone of `Builder`'s `groups` interner, taken once `declare`
    /// finishes (fields/predicates are the only ones that register groups,
    /// all during `declare`; nothing registers one afterwards).
    pub(crate) groups: Rodeo<Spur>,
}

impl<'a> TranslationContext<'a> {
    pub(crate) fn new(interner: &'a Interner) -> Self {
        Self {
            interner,
            name_map: HashMap::default(),
            addr_types: HashMap::default(),
            pred_resources: HashMap::default(),
            contracts: HashMap::default(),
            adt: AdtInfo::default(),
            groups: Rodeo::new(),
        }
    }

    /// The `#requires` contract resource of method `m`, if it has one.
    pub(crate) fn method_requires(&self, m: Spur) -> Option<vmir::MemberId> {
        self.contracts.get(&m).and_then(|c| c.requires)
    }

    /// The `#ensures` contract resource of method `m`, if it has one.
    pub(crate) fn method_ensures(&self, m: Spur) -> Option<vmir::MemberId> {
        self.contracts.get(&m).and_then(|c| c.ensures)
    }

    /// Lower a type in a monomorphic context — every body VMIR lowers is one
    /// (domains are monomorphic, methods/functions/resources are too). For
    /// ADT-declaration field types (which may mention the ADT's type parameters)
    /// call the free [`super::lower_type`] with the owning ADT's parameter list
    /// instead.
    pub(crate) fn lower_type(&self, ty: &typed::Type) -> vmir::Type {
        super::lower_type(&self.name_map, &[], ty)
    }

    /// The interned group tag for a field/predicate name (registered in the
    /// declare phase).
    pub fn group_tag(&self, name: Spur) -> Spur {
        let s = self.interner.resolve(&name);
        self.groups
            .get(s)
            .unwrap_or_else(|| panic!("group tag `{s}` not registered"))
    }
}
