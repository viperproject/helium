use itertools::Itertools;
use crate::verify::lang::Symbolic;

use z3::{DatatypeAccessor, DatatypeBuilder, DatatypeSort, Sort};
use crate::verify::context::VerifyContext;
use crate::{vmir, HashMap, HashSet};
use crate::vmir::Type;


#[derive(Default)]
pub(crate) struct Z3Sorts {
    sorts: HashMap<Type, Sort>,
    adts: HashMap<Type, DatatypeSort>,
    // which adts we are currently building
    building: HashSet<Type>,
}

impl Z3Sorts {
    pub(crate) fn option(&mut self, cx: &VerifyContext, ty: &Type, inner: &Type) -> &DatatypeSort {
        if !self.adts.contains_key(ty) {
            let inner_sort = self.sort(cx, inner);
            let name = format!("adt@{ty}"); // `adt@Option<Int>`
            let dt = DatatypeBuilder::new(name.as_str())
                .variant(
                    &format!("{name}.Some"),
                    vec![(&format!("{name}.Some.0"), DatatypeAccessor::sort(inner_sort))],
                )
                .variant(&format!("{name}.None"), vec![])
                .finish();
            self.adts.insert(ty.clone(), dt);
        }
        &self.adts[ty]
    }
    pub(crate) fn sort(&mut self, cx: &VerifyContext, ty: &Type) -> Sort {
        if let Some(s) = self.sorts.get(ty) {
            return s.clone();
        }
        let sort = match ty {
            Type::Int => Sort::int(),
            Type::Bool => Sort::bool(),
            Type::Real => Sort::real(),
            Type::Ref => Sort::uninterpreted("Ref".into()),
            Type::Domain(id, args) => match &cx.decls[*id] {
                vmir::Declaration::Domain(d) => {
                    debug_assert!(args.is_empty(), "domains are monomorphic");
                    Sort::uninterpreted(format!("dom@{}", cx.interner.resolve(&d.name)).into())
                }
                vmir::Declaration::Adt(adt) => self.adt(cx, ty, adt, args).sort.clone(),
                _ => unreachable!("Type::Domain must name a domain or an ADT"),
            },
            Type::Snap(id) => {
                let vmir::Declaration::Resource(r) = &cx.decls[*id] else {
                    unreachable!("Type::Snap must name a resource");
                };

                Sort::uninterpreted(format!("snap@{}", cx.interner.resolve(&r.name)).into())
            },
            Type::Option(inner) => self.option(cx, ty, inner).sort.clone(),
            // `addr@[g] T` (translation.typ). The bound is not part of the name: the
            // group already fixes it (fields are 1/1, predicates unbounded).
            Type::Addr { group, value, .. } => Sort::uninterpreted(
                format!("addr@[{}] {value}", cx.groups.resolve(group)).into(),
            ),
            Type::Generic(_) => unreachable!("generics are substituted before reaching Z3"),
        };
        self.sorts.insert(ty.clone(), sort.clone());
        sort
    }

    pub(crate) fn adt(
        &mut self,
        cx: &VerifyContext,
        ty: &Type,
        adt: &vmir::Adt,
        args: &[Type],
    ) -> &DatatypeSort {
        if !self.adts.contains_key(ty) {
            self.building.insert(ty.clone());
            let base = format!("adt@{}", cx.interner.resolve(&adt.name));
            let name = if args.is_empty() {
                base.to_string()
            } else {
                format!("{base}<{}>", args.iter().join(","))
            };
            let mut builder = DatatypeBuilder::new(name.as_str());
            for (v, variant) in adt.variants.iter().enumerate() {
                let ctor = match variant.name {
                    Some(n) => format!("{name}.{}", cx.interner.resolve(&n)),
                    None => format!("{name}#{v}"),
                };
                let mut acc_names = Vec::new();
                let mut accs = Vec::new();
                for (i, field) in variant.field_types.iter().enumerate() {
                    let field = field.subst_generics(args);
                    acc_names.push(format!("{ctor}.{i}"));
                    accs.push(if field == *ty {
                        // Self-reference: the sort doesn't exist yet, refer by name.
                        DatatypeAccessor::datatype(name.as_str())
                    } else if self.building.contains(&field) {
                        todo!("mutually recursive ADTs need datatype_builder::create_datatypes")
                    } else {
                        DatatypeAccessor::sort(self.sort(cx, &field))
                    });
                }
                let fields = acc_names.iter().map(String::as_str).zip(accs).collect();
                builder = builder.variant(&ctor, fields);
            }
            self.building.swap_remove(ty);
            self.adts.insert(ty.clone(), builder.finish());
        }
        &self.adts[ty]
    }
}
fn symbol_to_z3(context: &VerifyContext, symbol : &Symbolic) {

    match symbol {
        Symbolic::Fresh(i) => {
            let ty = context.fresh_types.get(i).expect("fresh type not found");



        }
        Symbolic::Lit(_) => {}
        Symbolic::Binary(_, _) => {}
        Symbolic::Ite(_) => {}
        Symbolic::FuncApp(_, _, _) => {}
        Symbolic::RealCast(_) => {}
        Symbolic::Forall(_, _) => {}
    }
}