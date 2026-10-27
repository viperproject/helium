use crate::verify::lang::{FuncId, Symbolic};
use itertools::Itertools;

use crate::verify::context::VerifyContext;
use crate::verify::func_registry::{BUILTIN_OPTION_HEAD, FuncKind};
use crate::vmir::{BinOp, Literal, MemberId, Polarity, Type};
use crate::{HashMap, vmir};
use egg::Id;
use std::borrow::Cow;
use z3::ast::{Ast, Bool, Dynamic};
use z3::datatype_builder::create_datatypes;
use z3::{DatatypeAccessor, DatatypeBuilder, DatatypeSort, FuncDecl, Solver, Sort};

/// The translator's caches, kept in `VerifyContext::z3translator` so sorts and
/// datatypes are created once per context. The translation itself is the free
/// functions below, which reach these through the context.
#[derive(Default)]
pub(crate) struct Z3Translation {
    sorts: HashMap<Type, Sort>,
    adts: HashMap<Type, DatatypeSort>,
    class_types: HashMap<Id, Type>,
    func_cache : HashMap<FuncId, HashMap<Box<[Type]>, FuncDecl>>,
    max_node_id: usize,
    max_merges_is: usize,
    solver : Solver,
}

/// Parameter and return types of function declaration `func` at instantiation
/// `tys`. Every declared `FuncId` is a `vmir::Function`: ordinary, domain, and
/// field/predicate address functions alike.
fn function_signature(ctxt: &VerifyContext, func: MemberId, tys: &[Type]) -> (Vec<Type>, Type) {
    match &ctxt.decls[func] {
        vmir::Declaration::Domain(_) => {
            unreachable!(
                "the domain declaration only holds the name of the declaration. It can never be a function."
            )
        }
        vmir::Declaration::Axiom(_) => {
            unreachable!("the axiom declaration only holds the axiom. It can never be a function.")
        }
        vmir::Declaration::Function(fun) => {
            let params = fun.params.iter().map(|t| t.subst_generics(tys)).collect();
            (params, fun.ret.subst_generics(tys))
        }
        vmir::Declaration::Method(_) => {
            unreachable!("methods are lowered to exhale/inhale. A method is never called.")
        }
        vmir::Declaration::Resource(r) => {
            let params = r.params.iter().map(|t| t.subst_generics(tys)).collect();
            let group = ctxt
                .groups
                .get(ctxt.interner.resolve(&r.name))
                .expect("predicate group tag");
            let ret = r.derive_location(func, group).ret;
            (params, ret)
        }
        vmir::Declaration::Adt(_) => {
            unreachable!("adt are lowered to specific pure instructions and not to a function call")
        }
    }
}

/// The type of e-class `id` (translation.typ, "Class sorts"), or `None` when no
/// node of the class has a certain type. Unlike `types::infer_type` this never
/// guesses: without a sort the class cannot be declared to Z3, so its facts are
/// left out instead.
///
/// Cached per canonical id. Only `Some` is cached: a type survives merges,
/// "unknown" may not.
fn class_type(ctxt: &mut VerifyContext, id: Id) -> Option<Type> {
    // Take the cache out so `ctxt` can be borrowed immutably (and the class's
    // nodes iterated in place) while the cache is filled.
    let mut cache = std::mem::take(&mut ctxt.z3translator.class_types);
    let ty = cached_class_type(ctxt, &mut cache, &mut Vec::new(), id);
    ctxt.z3translator.class_types = cache;
    ty
}

/// `visiting` holds the classes whose `Ite` nodes are being followed, so an
/// `Ite` cycle back into one of them counts as uncertain instead of looping.
fn cached_class_type(
    ctxt: &VerifyContext,
    cache: &mut HashMap<Id, Type>,
    visiting: &mut Vec<Id>,
    id: Id,
) -> Option<Type> {
    let id = ctxt.egraph.find(id);
    if let Some(ty) = cache.get(&id) {
        return Some(ty.clone());
    }
    if visiting.contains(&id) {
        return None;
    }
    let nodes = &ctxt.egraph[id].nodes;
    // Nodes with a type of their own first: no recursion, and nearly every
    // class has one.
    let ty = nodes
        .iter()
        .find_map(|n| local_node_type(ctxt, n))
        .or_else(|| {
            visiting.push(id);
            let ty = nodes.iter().find_map(|n| match n {
                Symbolic::Ite([_, then, els]) => cached_class_type(ctxt, cache, visiting, *then)
                    .or_else(|| cached_class_type(ctxt, cache, visiting, *els)),
                _ => None,
            });
            visiting.pop();
            ty
        });
    debug_assert!(
        ty.as_ref().is_none_or(|ty| nodes
            .iter()
            .filter_map(|n| local_node_type(ctxt, n))
            .all(|other| other == *ty)),
        "e-class {id} mixes types"
    );
    if let Some(ty) = &ty {
        cache.insert(id, ty.clone());
    }
    ty
}

/// The type of one e-node from the node alone, or `None` when it is not
/// certain. An `Ite` takes its branches' type, so it is `None` here and left to
/// [`cached_class_type`].
fn local_node_type(ctxt: &VerifyContext, node: &Symbolic) -> Option<Type> {
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
        Symbolic::Ite(_) => None,
        Symbolic::FuncApp(f, tys, _) => match ctxt.alloc.kind(f) {
            FuncKind::Decl(m) | FuncKind::Limited(m) => Some(function_signature(ctxt, m, tys).1),
            FuncKind::PreToken(_) => Some(Type::Bool),
            FuncKind::Cons { head, .. } => Some(adt_type(ctxt, head, tys)),
            FuncKind::Proj {
                head,
                variant,
                field,
            } => Some(field_types(ctxt, head, variant, tys).swap_remove(field)),
            FuncKind::IsVariant { .. } => Some(Type::Bool),
        },
    }
}

/// One equation `c_{class} == ⟦node⟧` per e-node (translation.typ, "The e-graph").
/// An equation that needs a sort we cannot determine is dropped with a warning:
/// leaving out an assumption only weakens what Z3 knows, so it stays sound.
pub(crate) fn sync_facts(ctxt: &mut VerifyContext) {

    let start = ctxt.z3translator.max_node_id;
    let end = ctxt.egraph.nodes().len();
    for i in start..end {
        let id = Id::from(i);
        // Clone just this node: translating needs `ctxt` mutably (the caches),
        // so the node list cannot stay borrowed across the call.
        let node = ctxt.egraph.id_to_node(id).clone();
        if matches!(node, Symbolic::Forall(..)) {
            continue;
        }
        let Some(expr) = symbol_to_z3(ctxt, &node) else {
            eprintln!("[z3] warning: an operand of `{node}` has no certain type; dropping its fact");
            continue;
        };
        let class = ctxt.egraph.find(id);
        let Some(c) = class_const(ctxt, class) else {
            eprintln!("[z3] warning: could not create the class_const for {class}");
            continue;
        };
        ctxt.z3translator.solver.assert(c.eq(expr));
    }
    ctxt.z3translator.max_node_id = end;

    let start = ctxt.z3translator.max_merges_is;
    let end = ctxt.egraph.union_log().len();
    for i in start..end {
        let (a,b) = ctxt.egraph.union_log()[i];
        match (logged_class_const(ctxt, a), logged_class_const(ctxt, b)) {
            (Some(c_a), Some(c_b)) => ctxt.z3translator.solver.assert(c_a.eq(c_b)),
            (_, _) => eprintln!("[z3] warning: could not find class_const for {a} or {b}"),
        }
    }
    ctxt.z3translator.max_merges_is = end;
}

pub(crate) fn build_goal(ctxt: &mut VerifyContext, id: Id) -> Bool {
    class_const_given_sort(ctxt, id, &Sort::bool())
        .as_bool()
        .unwrap()
}

pub(crate) fn build_pc(ctxt: &mut VerifyContext, pc: &[(egg::Id, Polarity)]) -> Vec<Bool> {
    pc.iter()
        .map(|(id, p)| {
            let c_const = class_const_given_sort(ctxt, *id, &Sort::bool())
                .as_bool()
                .unwrap();
            match p {
                Polarity::Positive => c_const,
                Polarity::Negative => !c_const,
            }
        })
        .collect()
}

/// The constant for `id` itself, *not* its current root.
fn raw_class_const(id: Id, sort: &Sort) -> Dynamic {
    Dynamic::new_const(format!("c_{id}"), sort)
}

fn class_const_given_sort(ctxt: &VerifyContext, id: Id, sort: &Sort) -> Dynamic {
    raw_class_const(ctxt.egraph.find(id), sort)
}

/// The constant of a root recorded in the union log, named by that id and not
/// by its current root: by now both sides of a merge share one root, so
/// `find` would turn every logged merge into `c_r == c_r`. Facts were asserted
/// under the root at sync time, and the log links every former root to its
/// absorber, so these equations connect them to the current root.
fn logged_class_const(ctxt: &mut VerifyContext, id: Id) -> Option<Dynamic> {
    let typ = class_type(ctxt, id)?;
    let sort = sort(ctxt, &typ);
    Some(raw_class_const(id, &sort))
}

/// `None` when the class's type is not certain (see [`class_type`]).
fn class_const(ctxt: &mut VerifyContext, id: Id) -> Option<Dynamic> {
    let typ = class_type(ctxt, id)?;
    let sort = sort(ctxt, &typ);
    Some(class_const_given_sort(ctxt, id, &sort))
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
            vmir::Declaration::Adt(_) => datatype(ctxt, ty).sort.clone(),
            _ => unreachable!("Type::Domain must name a domain or an ADT"),
        },
        Type::Snap(_) | Type::Option(_) => datatype(ctxt, ty).sort.clone(),
        // `addr@[g] T` (translation.typ). The bound is not part of the name: the
        // group already fixes it (fields are 1/1, predicates unbounded).
        Type::Addr { group, value, .. } => {
            Sort::uninterpreted(format!("addr@[{}] {value}", ctxt.groups.resolve(group)).into())
        }
        Type::Generic(_) => unreachable!("generics are currently not supported"),
    };
    ctxt.z3translator.sorts.insert(ty.clone(), sort.clone());
    sort
}

/// `⟦symbol⟧`, or `None` when an operand's sort is not certain.
fn symbol_to_z3(ctxt: &mut VerifyContext, symbol: &Symbolic) -> Option<Dynamic> {
    Some(match symbol {
        Symbolic::Fresh(i) => {
            let ty = ctxt
                .fresh_types
                .get(i)
                .expect("fresh type not found")
                .clone();
            Dynamic::new_const(format!("fresh{i}"), &sort(ctxt, &ty))
        }
        Symbolic::Lit(literal) => match literal {
            Literal::Null => Dynamic::new_const("null", &sort(ctxt, &Type::Ref)),
            Literal::Bool(value) => z3::ast::Bool::from_bool(*value).into(),
            Literal::Int(value) => z3::ast::Int::from_big_int(value).into(),
            Literal::Real(value) => z3::ast::Real::from_big_rational(value).into(),
        },
        Symbolic::Binary(op, [l, r]) => {
            // `Eq` is the only operator without a fixed operand sort; either
            // side fixes it.
            let operand_ty = match op.sort() {
                Some(ty) => ty,
                None => class_type(ctxt, *l).or_else(|| class_type(ctxt, *r))?,
            };
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
                        _ => unreachable!("excluded by outer matching"),
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
                        _ => unreachable!("excluded by outer matching"),
                    }
                }
            }
        }
        Symbolic::Ite([cond, then, els]) => {
            let then_ty = class_type(ctxt, *then).or_else(|| class_type(ctxt, *els))?;
            let sort = sort(ctxt, &then_ty);
            let cond = class_const_given_sort(ctxt, *cond, &Sort::bool())
                .as_bool()
                .unwrap();
            let then = class_const_given_sort(ctxt, *then, &sort);
            let els = class_const_given_sort(ctxt, *els, &sort);
            cond.ite(&then, &els)
        }
        Symbolic::FuncApp(id, ty, args) => translate_function(ctxt, id, ty, args),
        Symbolic::RealCast(id) => {
            let class = class_const(ctxt, *id)?.as_int().unwrap();
            class.to_real().into()
        }
        Symbolic::Forall(_, _) => todo!("implement translation for forall nodes"),
    })
}

// the type build by heads constructor at instantiation typs
fn adt_type(ctxt: &VerifyContext, head: MemberId, tys: &[Type]) -> Type {
    if head == BUILTIN_OPTION_HEAD {
        return Type::Option(Box::new(tys[0].clone()));
    }

    match &ctxt.decls[head] {
        vmir::Declaration::Adt(_) => Type::Domain(head, tys.into()),
        vmir::Declaration::Resource(_) => Type::Snap(head),
        _ => unreachable!("an ADT head is an ADT, a predicate (snapshot), or Option"),
    }
}

fn field_types(ctxt: &VerifyContext, head: MemberId, variant: usize, tys: &[Type]) -> Vec<Type> {
    if head == BUILTIN_OPTION_HEAD {
        return if variant == 0 {
            vec![tys[0].clone()]
        } else {
            vec![]
        }; // Some(T) / None
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
    assert_eq!(
        params.len(),
        args.len(),
        "arity mismatch applying fn{}",
        f.0
    );

    let domain: Vec<Sort> = params.iter().map(|t| sort(ctxt, t)).collect();
    let arg_consts: Vec<Dynamic> = args
        .iter()
        .zip(&domain)
        .map(|(a, s)| class_const_given_sort(ctxt, *a, s))
        .collect();
    let arg_refs: Vec<&dyn Ast> = arg_consts.iter().map(|a| a as &dyn Ast).collect();
    if let Some(decl) = ctxt.z3translator.func_cache.get(&f).and_then(|m| m.get(tys)) {
        decl.apply(&arg_refs)
    } else {
        let range = sort(ctxt, ret);
        let name = if tys.is_empty() {
            format!("fn{}", f.0)
        } else {
            format!("fn{}<{}>", f.0, tys.iter().join(","))
        };
        let domain_refs: Vec<&Sort> = domain.iter().collect();
        let func_decl = FuncDecl::new(name, &domain_refs, &range);
        let res = func_decl.apply(&arg_refs);
        ctxt.z3translator.func_cache.entry(f).or_default().insert(tys.into(), func_decl);
        res
    }
}

fn translate_function(
    ctxt: &mut VerifyContext,
    f_id: &FuncId,
    ty: &[Type],
    args: &[Id],
) -> Dynamic {
    let function_type = ctxt.alloc.kind(f_id);

    match function_type {
        // A function, a domain function, or a field/predicate address
        // or a resource function
        // function: all are `vmir::Function`s.
        FuncKind::Decl(id) => {
            let (params, ret) = function_signature(ctxt, id, ty);
            uninterpreted_app(ctxt, *f_id, ty, &params, &ret, args)
        }
        // ADT construction
        FuncKind::Cons { variant, head } => {
            let adt_ty = adt_type(ctxt, head, ty);
            let field_tys = field_types(ctxt, head, variant, ty);
            let fields = args
                .iter()
                .zip(&field_tys)
                .map(|(arg, ty)| {
                    let s = sort(ctxt, ty);
                    class_const_given_sort(ctxt, *arg, &s)
                })
                .collect::<Vec<_>>();

            let field_refs = fields
                .iter()
                .map(|f| f as &dyn Ast)
                .collect::<Vec<&dyn Ast>>();

            sort(ctxt, &adt_ty); // builds and caches the datatype
            ctxt.z3translator.adts[&adt_ty].variants[variant]
                .constructor
                .apply(&field_refs)
        }
        // ADT projection
        FuncKind::Proj {
            head,
            variant,
            field,
        } => {
            let adt_ty = adt_type(ctxt, head, ty);
            let adt_sort = sort(ctxt, &adt_ty); // builds and caches the datatype
            let base = class_const_given_sort(ctxt, args[0], &adt_sort);
            ctxt.z3translator.adts[&adt_ty].variants[variant].accessors[field].apply(&[&base])
        }
        // ADT tester `Adt::Ctor?`: the datatype's own recognizer for `variant`.
        FuncKind::IsVariant { head, variant } => {
            let adt_ty = adt_type(ctxt, head, ty);
            let adt_sort = sort(ctxt, &adt_ty); // builds and caches the datatype
            let base = class_const_given_sort(ctxt, args[0], &adt_sort);
            ctxt.z3translator.adts[&adt_ty].variants[variant]
                .tester
                .apply(&[&base])
        }
        // `f#lim`: same signature as `f`, only the id (and so the name) differs.
        // `f(x) == f#lim(x)` reaches Z3 only where the e-graph already has it.
        FuncKind::Limited(func) => {
            let (params, ret) = function_signature(ctxt, func, ty);
            uninterpreted_app(ctxt, *f_id, ty, &params, &ret, args)
        }
        // `f%pre`: over `f`'s own parameters, returning `Bool`.
        FuncKind::PreToken(func) => {
            let (params, _) = function_signature(ctxt, func, ty);
            uninterpreted_app(ctxt, *f_id, ty, &params, &Type::Bool, args)
        }
    }
}

/// A datatype's declaration before Z3 sees it: the builder name and, per
/// constructor, its name and its `(accessor name, field type)`s.
struct DatatypeShape {
    name: String,
    variants: Vec<(String, Vec<(String, Type)>)>,
}

/// The shape of `ty` if it is a datatype (an ADT, a snapshot, or `Option`);
/// `None` for every other sort.
fn datatype_shape(ctxt: &VerifyContext, ty: &Type) -> Option<DatatypeShape> {
    let interner = ctxt.interner;
    let (adt, args): (Cow<vmir::Adt>, &[Type]) = match ty {
        Type::Option(inner) => {
            let name = format!("adt@{ty}"); // `adt@Option<Int>`
            return Some(DatatypeShape {
                variants: vec![
                    (
                        format!("{name}.Some"),
                        vec![(format!("{name}.Some.0"), (**inner).clone())],
                    ),
                    (format!("{name}.None"), vec![]),
                ],
                name,
            });
        }
        Type::Domain(id, args) => match &ctxt.decls[*id] {
            vmir::Declaration::Adt(a) => (Cow::Borrowed(a), &args[..]),
            _ => return None,
        },
        Type::Snap(id) => match &ctxt.decls[*id] {
            vmir::Declaration::Resource(r) => (
                Cow::Owned(r.derive_snapshot().expect("adt of resource")),
                &[],
            ),
            _ => unreachable!("snapshot can only be taken from resources."),
        },
        _ => return None,
    };
    let base = format!("adt@{}", interner.resolve(&adt.name));
    let name = if args.is_empty() {
        base
    } else {
        format!("{base}<{}>", args.iter().join(","))
    };
    let mut variants = Vec::new();
    for (v, variant) in adt.variants.iter().enumerate() {
        let ctor = match variant.name {
            Some(n) => format!("{name}.{}", interner.resolve(&n)),
            None => format!("{name}#{v}"),
        };
        let fields = variant
            .field_types
            .iter()
            .enumerate()
            .map(|(i, field)| (format!("{ctor}.{i}"), field.subst_generics(args)))
            .collect();
        variants.push((ctor, fields));
    }
    Some(DatatypeShape { name, variants })
}

/// The Z3 datatype for `ty`, declared on first use.
pub(crate) fn datatype<'c>(ctxt: &'c mut VerifyContext, ty: &Type) -> &'c DatatypeSort {
    if !ctxt.z3translator.adts.contains_key(ty) {
        declare_datatypes(ctxt, ty);
    }
    &ctxt.z3translator.adts[ty]
}

/// Declare `root` together with every not-yet-declared datatype reachable from
/// it through constructor fields, in one `create_datatypes` call. Inside the
/// group a field refers to another member by name, so recursion through another
/// datatype (a snapshot holding `Option` of itself, mutually recursive ADTs)
/// never needs a sort that does not exist yet.
fn declare_datatypes(ctxt: &mut VerifyContext, root: &Type) {
    let mut group: HashMap<Type, DatatypeShape> = HashMap::default();
    let mut pending = vec![root.clone()];
    while let Some(ty) = pending.pop() {
        if group.contains_key(&ty) || ctxt.z3translator.adts.contains_key(&ty) {
            continue;
        }
        let Some(shape) = datatype_shape(ctxt, &ty) else {
            continue;
        };
        for (_, fields) in &shape.variants {
            pending.extend(fields.iter().map(|(_, field)| field.clone()));
        }
        group.insert(ty, shape);
    }

    // A field outside the group is not a datatype or is already declared, so
    // `sort` on it never comes back here.
    let mut builders = Vec::new();
    for shape in group.values() {
        let mut builder = DatatypeBuilder::new(shape.name.as_str());
        for (ctor, fields) in &shape.variants {
            let mut accs = Vec::new();
            for (acc, field) in fields {
                accs.push((
                    acc.as_str(),
                    match group.get(field) {
                        Some(member) => DatatypeAccessor::datatype(member.name.as_str()),
                        None => DatatypeAccessor::sort(sort(ctxt, field)),
                    },
                ));
            }
            builder = builder.variant(ctor, accs);
        }
        builders.push(builder);
    }

    // `create_datatypes` returns the sorts in builder order.
    for (ty, dt) in group.into_keys().zip(create_datatypes(builders)) {
        ctxt.z3translator.adts.insert(ty, dt);
    }
}
