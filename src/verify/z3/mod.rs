use itertools::Itertools;
use crate::verify::lang::{FuncId, Symbolic};

use z3::{DatatypeAccessor, DatatypeBuilder, DatatypeSort, FuncDecl, Sort};
use z3::ast::{Ast, Bool, Dynamic, Int};
use crate::verify::context::VerifyContext;
use crate::{vmir, HashMap, HashSet};
use crate::vmir::{BinOp, Literal, MemberId, Type};
use crate::verify::func_registry::{FuncKind, BUILTIN_OPTION_NONE, BUILTIN_OPTION_SOME, BUILTIN_OPTION_IS_SOME, BUILTIN_OPTION_IS_NONE, BUILTIN_OPTION_VALUE, BUILTIN_OPTION_HEAD};
use egg::Id;

/// The translator's caches, kept in `VerifyContext::z3translator` so sorts and
/// datatypes are created once per context. The translation itself is the free
/// functions below, which reach these through the context.
#[derive(Default)]
pub(crate) struct Z3Translation {
    sorts: HashMap<Type, Sort>,
    adts: HashMap<Type, DatatypeSort>,
    // which adts we are currently building
    building: HashSet<Type>,
}

/// Parameter and return types of function declaration `func` at instantiation
/// `tys`. Every declared `FuncId` is a `vmir::Function`: ordinary, domain, and
/// field/predicate address functions alike.
fn function_signature(ctxt: &VerifyContext, func: MemberId, tys: &[Type]) -> (Vec<Type>, Type) {
    let vmir::Declaration::Function(fun) = &ctxt.decls[func] else {
        unreachable!("FuncApp of declaration {} that is not a function", func.0);
    };
    let params = fun.params.iter().map(|t| t.subst_generics(tys)).collect();
    (params, fun.ret.subst_generics(tys))
}

/// The type of e-class `id` (translation.typ, "Class sorts"). Exact or a panic:
/// every node whose type is certain must agree, and at least one must exist.
/// Unlike `types::infer_type` this never guesses.
fn class_type(ctxt: &VerifyContext, id: Id) -> Type {
    let mut memo = crate::dhash::HashMap::default();
    exact_class_type(ctxt, id, &mut memo)
        .unwrap_or_else(|| panic!("e-class {id} has no node with a certain type"))
}

/// `None` when no node of the class has a certain type. `memo` maps a class to
/// its result; `None` is pre-seeded so an `Ite` cycle back into a class still
/// being resolved counts as uncertain rather than looping.
fn exact_class_type(
    ctxt: &VerifyContext,
    id: Id,
    memo: &mut crate::dhash::HashMap<Id, Option<Type>>,
) -> Option<Type> {
    let id = ctxt.egraph.find(id);
    if let Some(ty) = memo.get(&id) {
        return ty.clone();
    }
    memo.insert(id, None);
    let mut found: Option<Type> = None;
    for node in &ctxt.egraph[id].nodes {
        let Some(ty) = exact_node_type(ctxt, node, memo) else {
            continue;
        };
        match &found {
            None => found = Some(ty),
            Some(prev) if *prev == ty => {}
            Some(prev) => panic!("e-class {id} mixes types {prev} and {ty}"),
        }
    }
    memo.insert(id, found.clone());
    found
}

/// The type of one e-node, or `None` when it is not certain.
fn exact_node_type(
    ctxt: &VerifyContext,
    node: &Symbolic,
    memo: &mut crate::dhash::HashMap<Id, Option<Type>>,
) -> Option<Type> {
    match node {
        Symbolic::Lit(lit) => Some(crate::verify::types::lit_type(lit)),
        Symbolic::Fresh(u) => ctxt.fresh_types.get(u).cloned(),
        Symbolic::Binary(op, _) => match op {
            BinOp::Eq | BinOp::LtI | BinOp::LtR => Some(Type::Bool),
            // Arithmetic yields its operand type.
            _ => op.sort(),
        },
        Symbolic::RealCast(_) => Some(Type::Real),
        Symbolic::Forall(..) => Some(Type::Bool),
        Symbolic::Ite([_, then, _]) => exact_class_type(ctxt, *then, memo),
        Symbolic::FuncApp(f, tys, _) => match *f {
            BUILTIN_OPTION_SOME | BUILTIN_OPTION_NONE => Some(Type::Option(Box::new(tys[0].clone()))),
            BUILTIN_OPTION_VALUE => Some(tys[0].clone()),
            BUILTIN_OPTION_IS_SOME | BUILTIN_OPTION_IS_NONE => Some(Type::Bool),
            // The return type is a function of `(f, tys)`. With no type
            // arguments `f` has exactly one instantiation, so the type
            // `func_ret_types` recorded for it is the right one. With type
            // arguments it may be another instantiation's: uncertain.
            _ if tys.is_empty() => ctxt.func_ret_types.get(f).cloned(),
            _ => None,
        },
    }
}

/// One equation `c_{class} == ⟦node⟧` per e-node (translation.typ, "The e-graph").
pub(crate) fn build_facts(ctxt: &mut VerifyContext) -> Vec<Bool> {
    // Snapshot the classes first: translating needs `ctxt` mutably (the sort
    // cache), so the e-graph cannot stay borrowed while we iterate it.
    let classes: Vec<(Id, Vec<Symbolic>)> = ctxt
        .egraph
        .classes()
        .map(|class| (class.id, class.nodes.clone()))
        .collect();
    let mut facts = Vec::new();
    for (id, nodes) in classes {
        let class_const = class_const(ctxt, id);
        // A `Forall` gets no equation: its class constant stays a free
        // `Bool`, and its instances are already in the e-graph as
        // `ite(forall, body, true) == true` (translation.typ, "The e-graph").
        for node in nodes.iter().filter(|node| !matches!(node, Symbolic::Forall(..))) {
            let node_expression = symbol_to_z3(ctxt, node);
            facts.push(class_const.eq(node_expression));
        }
    }
    facts
}

pub(crate) fn option<'c>(ctxt: &'c mut VerifyContext, ty: &Type, inner: &Type) -> &'c DatatypeSort {
    if !ctxt.z3translator.adts.contains_key(ty) {
        let inner_sort = sort(ctxt, inner);
        let name = format!("adt@{ty}"); // `adt@Option<Int>`
        let dt = DatatypeBuilder::new(name.as_str())
            .variant(
                &format!("{name}.Some"),
                vec![(&format!("{name}.Some.0"), DatatypeAccessor::sort(inner_sort))],
            )
            .variant(&format!("{name}.None"), vec![])
            .finish();
        ctxt.z3translator.adts.insert(ty.clone(), dt);
    }
    &ctxt.z3translator.adts[ty]
}

fn class_const_given_sort(ctxt: &VerifyContext, id: Id, sort: &Sort) -> Dynamic {
    Dynamic::new_const(format!("c_{}", ctxt.egraph.find(id)), sort)
}

fn class_const(ctxt: &mut VerifyContext, id: Id) -> Dynamic {
    let typ = class_type(ctxt, id);
    let sort = sort(ctxt, &typ);
    class_const_given_sort(ctxt, id, &sort)
}

pub(crate) fn sort(ctxt: &mut VerifyContext, ty: &Type) -> Sort {
    if let Some(s) = ctxt.z3translator.sorts.get(ty) {
        return s.clone();
    }
    // Copy the `&'a` references out, so what they point to is not borrowed
    // through `ctxt` while `ctxt` is passed on mutably.
    let (decls, interner) = (ctxt.decls, ctxt.interner);
    let sort = match ty {
        Type::Int => Sort::int(),
        Type::Bool => Sort::bool(),
        Type::Real => Sort::real(),
        Type::Ref => Sort::uninterpreted("Ref".into()),
        Type::Domain(id, args) => match &decls[*id] {
            vmir::Declaration::Domain(d) => {
                debug_assert!(args.is_empty(), "domains are monomorphic");
                Sort::uninterpreted(format!("dom@{}", interner.resolve(&d.name)).into())
            }
            vmir::Declaration::Adt(a) => adt(ctxt, ty, a, args).sort.clone(),
            _ => unreachable!("Type::Domain must name a domain or an ADT"),
        },
        Type::Snap(id) => {
            let vmir::Declaration::Resource(r) = &decls[*id] else {
                unreachable!("Type::Snap must name a resource");
            };

            Sort::uninterpreted(format!("snap@{}", interner.resolve(&r.name)).into())
        },
        Type::Option(inner) => option(ctxt, ty, inner).sort.clone(),
        // `addr@[g] T` (translation.typ). The bound is not part of the name: the
        // group already fixes it (fields are 1/1, predicates unbounded).
        Type::Addr { group, value, .. } => Sort::uninterpreted(
            format!("addr@[{}] {value}", ctxt.groups.resolve(group)).into(),
        ),
        Type::Generic(_) => unreachable!("generics are substituted before reaching Z3"),
    };
    ctxt.z3translator.sorts.insert(ty.clone(), sort.clone());
    sort
}

fn symbol_to_z3(ctxt: &mut VerifyContext, symbol: &Symbolic) -> Dynamic {
    match symbol {
        Symbolic::Fresh(i) => {
            let ty = ctxt.fresh_types.get(i).expect("fresh type not found").clone();
            Dynamic::new_const(format!("fresh{i}"), &sort(ctxt, &ty))
        }
        Symbolic::Lit(literal) => match literal {
            Literal::Null => Dynamic::new_const("null", &sort(ctxt, &Type::Ref)),
            Literal::Bool(value) => z3::ast::Bool::from_bool(*value).into(),
            Literal::Int(value) => z3::ast::Int::from_big_int(value).into(),
            Literal::Real(value) => z3::ast::Real::from_big_rational(value).into(),
        },
        Symbolic::Binary(op, [l, r]) => {
            let operand_ty = op.sort().unwrap_or_else(|| class_type(ctxt, *l));
            let sort = sort(ctxt, &operand_ty);
            let l = class_const_given_sort(ctxt, *l, &sort);
            let r = class_const_given_sort(ctxt, *r, &sort);
            match op {
                BinOp::Eq => l.eq(&r).into(),
                BinOp::AddI | BinOp::SubI | BinOp::MulI | BinOp::DivI | BinOp::Mod | BinOp::LtI => {
                    let (l, r) = (l.as_int().unwrap(), r.as_int().unwrap());
                    match op {
                        BinOp::AddI => (l + r).into(),
                        BinOp::SubI => (l - r).into(),
                        BinOp::MulI => (l * r).into(),
                        BinOp::DivI => l.div(&r).into(),
                        BinOp::Mod => l.modulo(&r).into(),
                        BinOp::LtI => l.lt(&r).into(),
                        _ => unreachable!(),
                    }
                }
                BinOp::AddR | BinOp::SubR | BinOp::MulR | BinOp::DivR | BinOp::LtR => {
                    let (l, r) = (l.as_real().unwrap(), r.as_real().unwrap());
                    match op {
                        BinOp::AddR => (l + r).into(),
                        BinOp::SubR => (l - r).into(),
                        BinOp::MulR => (l * r).into(),
                        BinOp::DivR => l.div(&r).into(),
                        BinOp::LtR => l.lt(&r).into(),
                        _ => unreachable!(),
                    }
                }
            }
        }
        Symbolic::Ite([cond, then, els]) => {
            let then_ty = class_type(ctxt, *then);
            let sort = sort(ctxt, &then_ty);
            let cond = class_const_given_sort(ctxt, *cond, &Sort::bool()).as_bool().unwrap();
            let then = class_const_given_sort(ctxt, *then, &sort);
            let els = class_const_given_sort(ctxt, *els, &sort);
            cond.ite(&then, &els)
        },
        Symbolic::FuncApp(id, ty, args) => translate_function(ctxt, id, ty, args),
        Symbolic::RealCast(id) => {
            let ty = class_type(ctxt, *id);
            let sort = sort(ctxt, &ty);
            let class = class_const_given_sort(ctxt, *id, &sort).as_int().unwrap();
            class.to_real().into()
        },
        Symbolic::Forall(_, _) => todo!(),
    }
}


// the type build by heads constructor at instantiation typs
fn adt_type(ctxt: &VerifyContext, head: MemberId, tys: &[Type]) -> Type {
    if head == BUILTIN_OPTION_HEAD {
        return Type::Option(Box::new(tys[0].clone()))
    }

    match &ctxt.decls[head] {
        vmir::Declaration::Adt(_) => Type::Domain(head, tys.into()),
        vmir::Declaration::Resource(_) => Type::Snap(head),
        _ => unreachable!("an ADT head is an ADT, a predicate (snapshot), or Option")
    }
}

fn field_types(ctxt: &VerifyContext, head: MemberId, variant: usize, tys: &[Type]) -> Vec<Type> {
    if head == BUILTIN_OPTION_HEAD {
        return if variant == 0 { vec![tys[0].clone()] } else { vec![] }; // Some(T) / None
    }
    match &ctxt.decls[head] {
        vmir::Declaration::Adt(adt) => adt.variants[variant]
            .field_types
            .iter()
            .map(|t| t.subst_generics(tys))
            .collect(),
        vmir::Declaration::Resource(r) => r
            .derive_snapshot()
            .expect("snapshot of a resource that is not self-framed")
            .variants[variant]
            .field_types
            .clone(),
        _ => unreachable!("an ADT head is an ADT, a predicate (snapshot), or Option"),
    }
}
/// `f[tys](args)` as an application of the uninterpreted Z3 function
/// `fn{f}<tys>` with signature `params -> ret`. The name carries the type
/// arguments because one `FuncId` serves every instantiation
/// (translation.typ, "Function applications").
fn uninterpreted_app(
    ctxt: &mut VerifyContext,
    f: FuncId,
    tys: &[Type],
    params: &[Type],
    ret: &Type,
    args: &[Id],
) -> Dynamic {
    assert_eq!(params.len(), args.len(), "arity mismatch applying fn{}", f.0);
    let domain: Vec<Sort> = params.iter().map(|t| sort(ctxt, t)).collect();
    let arg_consts: Vec<Dynamic> = args
        .iter()
        .zip(&domain)
        .map(|(a, s)| class_const_given_sort(ctxt, *a, s))
        .collect();
    let range = sort(ctxt, ret);
    let name = if tys.is_empty() {
        format!("fn{}", f.0)
    } else {
        format!("fn{}<{}>", f.0, tys.iter().join(","))
    };
    let domain_refs: Vec<&Sort> = domain.iter().collect();
    let arg_refs: Vec<&dyn Ast> = arg_consts.iter().map(|a| a as &dyn Ast).collect();
    FuncDecl::new(name, &domain_refs, &range).apply(&arg_refs)
}

fn translate_function(ctxt: &mut VerifyContext, f_id: &FuncId, ty: &Box<[Type]>, args: &Box<[Id]>) -> Dynamic {

    let function_type = ctxt.alloc.kind(*f_id);

    match function_type {
        // A function, a domain function, or a field/predicate address
        // function: all are `vmir::Function`s.
        FuncKind::Decl(id) => {
            let (params, ret) = function_signature(ctxt, id, ty);
            uninterpreted_app(ctxt, *f_id, ty, &params, &ret, args)
        },
        // ADT construction
        FuncKind::Cons {
            variant,
            head
        } => {
            let adt_ty = adt_type(ctxt, head, ty);
            let field_tys = field_types(ctxt, head, variant, ty);

            // Snapshots use the uninterpreted encoding (translation.typ,
            // "ADTs and snapshots"); ADTs and Option are datatypes.
            if let Type::Snap(_) = adt_ty {
                uninterpreted_app(ctxt, *f_id, ty, &field_tys, &adt_ty, args)
            } else {
                let fields = args.iter().zip(&field_tys).map(|(arg, ty)| {
                    let s = sort(ctxt, ty);
                    class_const_given_sort(ctxt, *arg, &s)
                }).collect::<Vec<_>>();

                let field_refs = fields.iter().map(|f| f as &dyn Ast).collect::<Vec<&dyn Ast>>();

                sort(ctxt, &adt_ty); // builds and caches the datatype
                ctxt.z3translator.adts[&adt_ty].variants[variant].constructor.apply(&field_refs)
            }
        },
        // ADT projection
        FuncKind::Proj {
            head,
            variant,
            field
        } => {
            let adt_ty = adt_type(ctxt, head, ty);
            // TODO: bad performance building the whole vec for nothing
            let field_ty = field_types(ctxt, head, variant, ty).swap_remove(field);

            // Snapshots use the uninterpreted encoding (translation.typ,
            // "ADTs and snapshots"); ADTs and Option are datatypes.
            if let Type::Snap(_) = adt_ty {
                uninterpreted_app(ctxt, *f_id, ty, &[adt_ty.clone()], &field_ty, args)
            } else {
                let adt_sort = sort(ctxt, &adt_ty); // builds and caches the datatype
                let base = class_const_given_sort(ctxt, args[0], &adt_sort);
                ctxt.z3translator.adts[&adt_ty].variants[variant].accessors[field].apply(&[&base])
            }
        },
        // ADT tester `Adt::Ctor?`
        FuncKind::IsVariant {
            head,
            variant
        } => {
            let adt_ty = adt_type(ctxt, head, ty);
            if let Type::Snap(_) = adt_ty {
                // Snapshot: uninterpreted tester, `snap -> Bool`.
                uninterpreted_app(ctxt, *f_id, ty, &[adt_ty.clone()], &Type::Bool, args)
            } else {
                let adt_sort = sort(ctxt, &adt_ty); // builds and caches the datatype
                let base = class_const_given_sort(ctxt, args[0], &adt_sort);
                ctxt.z3translator.adts[&adt_ty].variants[variant].tester.apply(&[&base])
            }
        },
        // `f#lim`: same signature as `f`, only the id (and so the name) differs.
        // `f(x) == f#lim(x)` reaches Z3 only where the e-graph already has it.
        FuncKind::Limited(func) => {
            let (params, ret) = function_signature(ctxt, func, ty);
            uninterpreted_app(ctxt, *f_id, ty, &params, &ret, args)
        },
        // `f%pre`: over `f`'s own parameters, returning `Bool`.
        FuncKind::PreToken(func) => {
            let (params, _) = function_signature(ctxt, func, ty);
            uninterpreted_app(ctxt, *f_id, ty, &params, &Type::Bool, args)
        },
    }
}

pub(crate) fn adt<'c>(
    ctxt: &'c mut VerifyContext,
    ty: &Type,
    adt: &vmir::Adt,
    args: &[Type],
) -> &'c DatatypeSort {
    if !ctxt.z3translator.adts.contains_key(ty) {
        ctxt.z3translator.building.insert(ty.clone());
        let base = format!("adt@{}", ctxt.interner.resolve(&adt.name));
        let name = if args.is_empty() {
            base.to_string()
        } else {
            format!("{base}<{}>", args.iter().join(","))
        };
        let mut builder = DatatypeBuilder::new(name.as_str());
        for (v, variant) in adt.variants.iter().enumerate() {
            let ctor = match variant.name {
                Some(n) => format!("{name}.{}", ctxt.interner.resolve(&n)),
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
                } else if ctxt.z3translator.building.contains(&field) {
                    todo!("mutually recursive ADTs need datatype_builder::create_datatypes")
                } else {
                    DatatypeAccessor::sort(sort(ctxt, &field))
                });
            }
            let fields = acc_names.iter().map(String::as_str).zip(accs).collect();
            builder = builder.variant(&ctor, fields);
        }
        ctxt.z3translator.building.swap_remove(ty);
        ctxt.z3translator.adts.insert(ty.clone(), builder.finish());
    }
    &ctxt.z3translator.adts[ty]
}
