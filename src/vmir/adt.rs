use crate::vmir::Type;
use crate::vmir::display::VmirDisplay;
use lasso::Spur;
use std::fmt::{self, Display, Formatter};

/// The type-parameter arity of a generic declaration. ADTs are the only generic
/// declarations: domains, functions, methods and resources are monomorphic.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TyParams(usize);

impl From<usize> for TyParams {
    fn from(n: usize) -> Self {
        Self(n)
    }
}

impl TyParams {
    /// The type-parameter arity.
    pub fn count(&self) -> usize {
        self.0
    }
}

impl Display for TyParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // A declaration's generic parameters are positional (`Generic(n)` → `?n`),
        // so the binder only states the **arity** (`<2>`); the params are referred
        // to as `?0`, `?1`, … Angle brackets match type-argument instantiation
        // (`[..]` is reserved for heaps / addr groups). Nothing is printed for a
        // non-generic declaration.
        if self.0 == 0 {
            return Ok(());
        }
        write!(f, "<{}>", self.0)
    }
}

/// An algebraic data type: a list of variants (constructors), identified by
/// their index. Purely semantic — no synthetic tester / accessor member ids.
/// The verifier mints its own ids and reduction rules from this structure (see
/// `verify::func_registry`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Adt {
    pub name: Spur,
    pub ty_params: TyParams,
    pub variants: Vec<AdtVariant>,
}

/// One variant (constructor) of an [`Adt`]: an optional interned constructor name
/// (kept from the source for display; only needs to be distinct within the ADT —
/// `None` for synthetic ADTs like a predicate snapshot) plus its field types in
/// order. A constructor is not a declaration, so its name is an interner `Spur`,
/// not a `MemberId`. The constructor / projection / tester operations over it are
/// the semantic `PureInst::{AdtCons,AdtProj,AdtIs}` nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdtVariant {
    pub name: Option<Spur>,
    pub field_types: Vec<Type>,
}

impl<'a> Display for VmirDisplay<'a, &'a Adt> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let name = self.interner.resolve(&self.item.name);
        let ty_params = &self.item.ty_params;
        write!(f, "adt {name}{ty_params} {{ ")?;
        for (v, ctor) in self.item.variants.iter().enumerate() {
            if v > 0 {
                write!(f, " | ")?;
            }
            match ctor.name {
                Some(id) => write!(f, "{}(", self.interner.resolve(&id))?,
                None => write!(f, "#{v}(")?,
            }
            for (i, ty) in ctor.field_types.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.with(ty))?;
            }
            write!(f, ")")?;
        }
        write!(f, " }}")
    }
}
