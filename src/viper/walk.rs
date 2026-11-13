use lasso::Spur;
use num::{BigInt, BigRational};

use crate::viper::parsed::ast::*;

type ExpCall = Call<ExpCallKind>;
type StmtCall = Call<StmtCallKind>;

macro_rules! walk_children {
    ($name:ident, $l:lifetime, $ty:ident) => {
        fn $name(&mut self, ast: &$l $ty) {
            ast.walk_children(self);
        }
    };
}

pub trait AstWalker<'a>: Sized {
    walk_children!(walk_program, 'a, Program);
    walk_children!(walk_pre_post_dec, 'a, PrePostDec);
    walk_children!(walk_decreases, 'a, Decreases);
    walk_children!(walk_decreases_kind, 'a, DecreasesKind);
    walk_children!(walk_ident, 'a, Ident);
    walk_children!(walk_idn_decl, 'a, IdnDecl);
    walk_children!(walk_declaration, 'a, Declaration);
    walk_children!(walk_domain_element, 'a, DomainElement);
    walk_children!(walk_domain_element_kind, 'a, DomainElementKind);
    walk_children!(walk_axiom, 'a, Axiom);
    walk_children!(walk_import, 'a, Import);
    walk_children!(walk_define, 'a, Define);
    walk_children!(walk_exp_or_block, 'a, ExpOrBlock);
    walk_children!(walk_idn_decl_typed, 'a, IdnDeclTyped);
    walk_children!(walk_arg_or_type, 'a, ArgOrType);
    walk_children!(walk_exp_block, 'a, ExpBlock);
    walk_children!(walk_exp, 'a, Exp);
    walk_children!(walk_exp_kind, 'a, ExpKind);
    walk_children!(walk_exp_call_kind, 'a, ExpCallKind);
    walk_children!(walk_stmt_call_kind, 'a, StmtCallKind);
    walk_children!(walk_exp_call, 'a, ExpCall);
    walk_children!(walk_stmt_call, 'a, StmtCall);
    walk_children!(walk_const, 'a, ConstKind);
    walk_children!(walk_heap_op_kind, 'a, HeapUpdateOp);
    walk_children!(walk_quantifier_kind, 'a, QuantifierKind);
    walk_children!(walk_acc_exp, 'a, AccExp);
    walk_children!(walk_bin_op, 'a, BinOp);
    walk_children!(walk_un_op, 'a, UnOp);
    walk_children!(walk_trigger, 'a, Trigger);
    walk_children!(walk_res_access, 'a, ResAccess);
    walk_children!(walk_block, 'a, StmtBlock);
    walk_children!(walk_statement, 'a, Statement);
    walk_children!(walk_assign_lhs, 'a, AssignLhs);
    walk_children!(walk_assign_rhs, 'a, AssignRhs);
    walk_children!(walk_star_or_names, 'a, StarOrNames);
    walk_children!(walk_index_op, 'a, IndexOp);
    walk_children!(walk_invariant, 'a, Invariant);
    walk_children!(walk_field, 'a, Field);
    walk_children!(walk_domain, 'a, Domain);
    walk_children!(walk_domain_function, 'a, DomainFunction);
    walk_children!(walk_signature, 'a, Signature);
    walk_children!(walk_type, 'a, Type);
    walk_children!(walk_predicate, 'a, Predicate);
    walk_children!(walk_function, 'a, Function);
    walk_children!(walk_contract, 'a, Contract);
    walk_children!(walk_method, 'a, Method);
    walk_children!(walk_adt, 'a, Adt);
    walk_children!(walk_variant, 'a, Variant);
    walk_children!(walk_adt_constructor, 'a, AdtConstructor);

    walk_children!(walk_string, 'a, String);
    walk_children!(walk_bool, 'a, bool);
    walk_children!(walk_big_int, 'a, BigInt);
    walk_children!(walk_big_rational, 'a, BigRational);
    walk_children!(walk_spur, 'a, Spur);
    walk_children!(walk_usize, 'a, usize);
}

macro_rules! walk_mut_children {
    ($name:ident, $l:lifetime, $ty:ident) => {
        fn $name(&mut self, ast: &'a mut $ty) {
            ast.walk_mut_children(self);
        }
    };
}

pub trait AstWalkerMut<'a>: Sized {
    walk_mut_children!(walk_mut_program, 'a, Program);
    walk_mut_children!(walk_mut_pre_post_dec, 'a, PrePostDec);
    walk_mut_children!(walk_mut_decreases, 'a, Decreases);
    walk_mut_children!(walk_mut_decreases_kind, 'a, DecreasesKind);
    walk_mut_children!(walk_mut_ident, 'a, Ident);
    walk_mut_children!(walk_mut_idn_decl, 'a, IdnDecl);
    walk_mut_children!(walk_mut_declaration, 'a, Declaration);
    walk_mut_children!(walk_mut_domain_element, 'a, DomainElement);
    walk_mut_children!(walk_mut_domain_element_kind, 'a, DomainElementKind);
    walk_mut_children!(walk_mut_axiom, 'a, Axiom);
    walk_mut_children!(walk_mut_import, 'a, Import);
    walk_mut_children!(walk_mut_define, 'a, Define);
    walk_mut_children!(walk_mut_exp_or_block, 'a, ExpOrBlock);
    walk_mut_children!(walk_mut_idn_decl_typed, 'a, IdnDeclTyped);
    walk_mut_children!(walk_mut_arg_or_type, 'a, ArgOrType);
    walk_mut_children!(walk_mut_exp_block, 'a, ExpBlock);
    walk_mut_children!(walk_mut_exp, 'a, Exp);
    walk_mut_children!(walk_mut_exp_kind, 'a, ExpKind);
    walk_mut_children!(walk_mut_exp_call_kind, 'a, ExpCallKind);
    walk_mut_children!(walk_mut_stmt_call_kind, 'a, StmtCallKind);
    walk_mut_children!(walk_mut_exp_call, 'a, ExpCall);
    walk_mut_children!(walk_mut_stmt_call, 'a, StmtCall);
    walk_mut_children!(walk_mut_const, 'a, ConstKind);
    walk_mut_children!(walk_mut_heap_op_kind, 'a, HeapUpdateOp);
    walk_mut_children!(walk_mut_quantifier_kind, 'a, QuantifierKind);
    walk_mut_children!(walk_mut_acc_exp, 'a, AccExp);
    walk_mut_children!(walk_mut_bin_op, 'a, BinOp);
    walk_mut_children!(walk_mut_un_op, 'a, UnOp);
    walk_mut_children!(walk_mut_trigger, 'a, Trigger);
    walk_mut_children!(walk_mut_res_access, 'a, ResAccess);
    walk_mut_children!(walk_mut_block, 'a, StmtBlock);
    walk_mut_children!(walk_mut_statement, 'a, Statement);
    walk_mut_children!(walk_mut_assign_lhs, 'a, AssignLhs);
    walk_mut_children!(walk_mut_assign_rhs, 'a, AssignRhs);
    walk_mut_children!(walk_mut_star_or_names, 'a, StarOrNames);
    walk_mut_children!(walk_mut_index_op, 'a, IndexOp);
    walk_mut_children!(walk_mut_invariant, 'a, Invariant);
    walk_mut_children!(walk_mut_field, 'a, Field);
    walk_mut_children!(walk_mut_domain, 'a, Domain);
    walk_mut_children!(walk_mut_domain_function, 'a, DomainFunction);
    walk_mut_children!(walk_mut_signature, 'a, Signature);
    walk_mut_children!(walk_mut_type, 'a, Type);
    walk_mut_children!(walk_mut_predicate, 'a, Predicate);
    walk_mut_children!(walk_mut_function, 'a, Function);
    walk_mut_children!(walk_mut_contract, 'a, Contract);
    walk_mut_children!(walk_mut_method, 'a, Method);
    walk_mut_children!(walk_mut_adt, 'a, Adt);
    walk_mut_children!(walk_mut_variant, 'a, Variant);
    walk_mut_children!(walk_mut_adt_constructor, 'a, AdtConstructor);

    walk_mut_children!(walk_mut_string, 'a, String);
    walk_mut_children!(walk_mut_bool, 'a, bool);
    walk_mut_children!(walk_mut_big_int, 'a, BigInt);
    walk_mut_children!(walk_mut_big_rational, 'a, BigRational);
    walk_mut_children!(walk_mut_spur, 'a, Spur);
    walk_mut_children!(walk_mut_usize, 'a, usize);
}

pub trait AstWalkable {
    fn walk<'a>(&'a self, walker: &mut impl AstWalker<'a>);
    fn walk_mut<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>);
    fn walk_children<'a>(&'a self, walker: &mut impl AstWalker<'a>);
    fn walk_mut_children<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>);
}

macro_rules! walk_struct {
    ($name:ident, $walk:ident, $walk_mut:ident$(, $field:tt)*) => {
        impl AstWalkable for $name {
            fn walk<'a>(&'a self, walker: &mut impl AstWalker<'a>) {
                walker.$walk(self);
            }
            fn walk_mut<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>) {
                walker.$walk_mut(self);
            }
            #[allow(unused_variables)]
            fn walk_children<'a>(&'a self, walker: &mut impl AstWalker<'a>) {
                $(self.$field.walk(walker));*
            }
            #[allow(unused_variables)]
            fn walk_mut_children<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>) {
                $(self.$field.walk_mut(walker));*
            }
        }
    };
}

macro_rules! walk_enum {
    ($name:ident, $walk:ident, $walk_mut:ident$(, $variant:ident$(($($field:ident),+))?)*) => {
        impl AstWalkable for $name {
            fn walk<'a>(&'a self, walker: &mut impl AstWalker<'a>) {
                walker.$walk(self);
            }
            fn walk_mut<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>) {
                walker.$walk_mut(self);
            }
            #[allow(unused_variables)]
            fn walk_children<'a>(&'a self, walker: &mut impl AstWalker<'a>) {
                match self {
                    $($name::$variant$(($($field),+))? => { $($($field.walk(walker));+)? })*
                }
            }
            #[allow(unused_variables)]
            fn walk_mut_children<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>) {
                match self {
                    $($name::$variant$(($($field),+))? => { $($($field.walk_mut(walker));+)? })*
                }
            }
        }
    };
}

#[allow(unused_macros)]
macro_rules! walk_box {
    ($name:ident, $walk:ident, $walk_mut:ident) => {
        impl AstWalkable for $name {
            fn walk<'a>(&'a self, walker: &mut impl AstWalker<'a>) {
                walker.$walk(self);
            }
            fn walk_mut<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>) {
                walker.$walk_mut(self);
            }
            fn walk_children<'a>(&'a self, walker: &mut impl AstWalker<'a>) {
                (**self).walk(walker);
            }
            fn walk_mut_children<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>) {
                (**self).walk_mut(walker);
            }
        }
    };
}

walk_struct!(Program, walk_program, walk_mut_program, 0);
walk_enum!(
    PrePostDec,
    walk_pre_post_dec,
    walk_mut_pre_post_dec,
    Pre(e),
    Post(e),
    Decreases(d)
);
walk_struct!(Decreases, walk_decreases, walk_mut_decreases, kind, guard);
walk_enum!(
    DecreasesKind,
    walk_decreases_kind,
    walk_mut_decreases_kind,
    Star,
    Underscore,
    Exp(e)
);
walk_enum!(Ident, walk_ident, walk_mut_ident, Raw(s), Interned(spur));
walk_struct!(IdnDecl, walk_idn_decl, walk_mut_idn_decl, 0);
walk_enum!(
    Declaration,
    walk_declaration,
    walk_mut_declaration,
    Import(i),
    Define(d),
    Domain(d),
    DomainElement(f),
    Field(f),
    Function(f),
    Predicate(p),
    Method(m),
    Adt(a),
    AdtConstructor(c)
);
walk_struct!(
    DomainElement,
    walk_domain_element,
    walk_mut_domain_element,
    domain,
    kind
);
walk_enum!(
    DomainElementKind,
    walk_domain_element_kind,
    walk_mut_domain_element_kind,
    Function(f),
    Axiom(a)
);
walk_struct!(Axiom, walk_axiom, walk_mut_axiom, name, exp);
walk_struct!(Import, walk_import, walk_mut_import, path);
walk_struct!(Define, walk_define, walk_mut_define, name, args, body);
walk_enum!(
    ExpOrBlock,
    walk_exp_or_block,
    walk_mut_exp_or_block,
    Exp(e),
    Block(b)
);
walk_struct!(
    IdnDeclTyped,
    walk_idn_decl_typed,
    walk_mut_idn_decl_typed,
    idn,
    ty
);
walk_enum!(
    ArgOrType,
    walk_arg_or_type,
    walk_mut_arg_or_type,
    Arg(a),
    Type(t)
);
walk_struct!(ExpBlock, walk_exp_block, walk_mut_exp_block, 0);
walk_struct!(Exp, walk_exp, walk_mut_exp, kind);
walk_enum!(
    ExpKind,
    walk_exp_kind,
    walk_mut_exp_kind,
    Const(c),
    Ident(i),
    Result,
    Old(i, e),
    Ascribe(e, t),
    UnOp(op, e),
    BinOp(op, l, r),
    Ternary(c, t, e),
    LetIn(i, e1, e2),
    Index(e, op),
    Acc(a),
    Field(e, i),
    Call(a),
    AdtDestructor(e, i),
    AdtDiscriminator(e, i),
    HeapUpdate(kind, acc_exp, e),
    Quantifier(kind, vars, triggers, e),
    ForPerm(vars, p, e),
    MagicWand(l, r),
    SeqLit(t, es),
    SetLit(t, es),
    Unsupported(what)
);
walk_enum!(
    ExpCallKind,
    walk_exp_call_kind,
    walk_mut_exp_call_kind,
    Function,
    DomainFunction,
    Predicate,
    AdtConstructor,
    Macro
);
walk_enum!(
    StmtCallKind,
    walk_stmt_call_kind,
    walk_mut_stmt_call_kind,
    Method,
    Macro
);
walk_struct!(ExpCall, walk_exp_call, walk_mut_exp_call, kind, name, args);
walk_struct!(
    StmtCall,
    walk_stmt_call,
    walk_mut_stmt_call,
    kind,
    name,
    args
);
walk_enum!(
    ConstKind,
    walk_const,
    walk_mut_const,
    Bool(b),
    Int(i),
    Real(r),
    Null,
    Epsilon,
    Wildcard
);
walk_enum!(
    HeapUpdateOp,
    walk_heap_op_kind,
    walk_mut_heap_op_kind,
    Unfold,
    Fold,
    Apply,
    Package
);
walk_enum!(
    QuantifierKind,
    walk_quantifier_kind,
    walk_mut_quantifier_kind,
    Forall,
    Exists
);
walk_struct!(AccExp, walk_acc_exp, walk_mut_acc_exp, loc, perm);
walk_enum!(
    BinOp,
    walk_bin_op,
    walk_mut_bin_op,
    Implies,
    Iff,
    And,
    Or,
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    Plus,
    Minus,
    Mult,
    Div,
    IntDiv,
    Mod,
    Union,
    SetMinus,
    Intersection,
    Subset,
    Concat,
    Range,
    InhaleExhale
);
walk_enum!(
    UnOp,
    walk_un_op,
    walk_mut_un_op,
    Not,
    Neg,
    Perm,
    Cardinality
);
walk_struct!(Trigger, walk_trigger, walk_mut_trigger, exp);
walk_enum!(
    ResAccess,
    walk_res_access,
    walk_mut_res_access,
    Loc(l),
    Exp(e)
);
walk_struct!(StmtBlock, walk_block, walk_mut_block, 0);
walk_enum!(
    Statement,
    walk_statement,
    walk_mut_statement,
    Assume(e),
    Assert(e),
    Refute(e),
    Inhale(e),
    Exhale(e),
    Fold(e),
    Unfold(e),
    Goto(i),
    Label(i, invariants),
    Var(vars, e),
    While(e, invs, decs, b),
    If(e, then, else_),
    Assign(lhs, rhs),
    Block(b),
    Unsupported(what)
);
walk_enum!(
    AssignLhs,
    walk_assign_lhs,
    walk_mut_assign_lhs,
    Ident(i),
    Field(e, i)
);
walk_enum!(
    AssignRhs,
    walk_assign_rhs,
    walk_mut_assign_rhs,
    New(n),
    Call(k),
    Exp(e)
);
walk_enum!(
    StarOrNames,
    walk_star_or_names,
    walk_mut_star_or_names,
    Star,
    Names(names)
);
walk_enum!(
    IndexOp,
    walk_index_op,
    walk_mut_index_op,
    Index(e),
    LowerBound(e),
    UpperBound(e),
    Range(e1, e2),
    Assign(e1, e2)
);
walk_struct!(Invariant, walk_invariant, walk_mut_invariant, 0);
walk_struct!(Field, walk_field, walk_mut_field, 0);
walk_struct!(
    Domain,
    walk_domain,
    walk_mut_domain,
    name,
    params,
    interpretation
);
walk_struct!(
    Function,
    walk_function,
    walk_mut_function,
    signature,
    contract,
    body
);
walk_struct!(
    Contract,
    walk_contract,
    walk_mut_contract,
    precondition,
    postcondition,
    decreases
);
walk_struct!(
    DomainFunction,
    walk_domain_function,
    walk_mut_domain_function,
    unique,
    signature,
    interpretation
);
walk_struct!(
    Signature,
    walk_signature,
    walk_mut_signature,
    name,
    args,
    ret
);
walk_enum!(
    Type,
    walk_type,
    walk_mut_type,
    Bool,
    Int,
    Real,
    Ref,
    Generic(g),
    Domain(i, ts),
    Seq(t),
    Set(t)
);
walk_struct!(
    Predicate,
    walk_predicate,
    walk_mut_predicate,
    signature,
    body
);
walk_struct!(
    Method,
    walk_method,
    walk_mut_method,
    signature,
    contract,
    body
);
walk_struct!(Adt, walk_adt, walk_mut_adt, name, params, variants, derives);
walk_struct!(Variant, walk_variant, walk_mut_variant, name, fields);
walk_struct!(
    AdtConstructor,
    walk_adt_constructor,
    walk_mut_adt_constructor,
    signature
);

impl<T: AstWalkable> AstWalkable for Vec<T> {
    fn walk<'a>(&'a self, walker: &mut impl AstWalker<'a>) {
        self.walk_children(walker);
    }
    fn walk_mut<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>) {
        self.walk_mut_children(walker);
    }
    fn walk_children<'a>(&'a self, walker: &mut impl AstWalker<'a>) {
        for item in self {
            item.walk(walker);
        }
    }
    fn walk_mut_children<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>) {
        for item in self {
            item.walk_mut(walker);
        }
    }
}

impl<T: AstWalkable> AstWalkable for Option<T> {
    fn walk<'a>(&'a self, walker: &mut impl AstWalker<'a>) {
        self.walk_children(walker);
    }
    fn walk_mut<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>) {
        self.walk_mut_children(walker);
    }
    fn walk_children<'a>(&'a self, walker: &mut impl AstWalker<'a>) {
        if let Some(item) = self {
            item.walk(walker);
        }
    }
    fn walk_mut_children<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>) {
        if let Some(item) = self {
            item.walk_mut(walker);
        }
    }
}

impl<T: AstWalkable, U: AstWalkable> AstWalkable for Result<T, U> {
    fn walk<'a>(&'a self, walker: &mut impl AstWalker<'a>) {
        self.walk_children(walker);
    }
    fn walk_mut<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>) {
        self.walk_mut_children(walker);
    }
    fn walk_children<'a>(&'a self, walker: &mut impl AstWalker<'a>) {
        match self {
            Ok(item) => item.walk(walker),
            Err(item) => item.walk(walker),
        }
    }
    fn walk_mut_children<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>) {
        match self {
            Ok(item) => item.walk_mut(walker),
            Err(item) => item.walk_mut(walker),
        }
    }
}

impl<T: AstWalkable, U: AstWalkable> AstWalkable for (T, U) {
    fn walk<'a>(&'a self, walker: &mut impl AstWalker<'a>) {
        self.walk_children(walker);
    }
    fn walk_mut<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>) {
        self.walk_mut_children(walker);
    }
    fn walk_children<'a>(&'a self, walker: &mut impl AstWalker<'a>) {
        self.0.walk(walker);
        self.1.walk(walker);
    }
    fn walk_mut_children<'a>(&'a mut self, walker: &mut impl AstWalkerMut<'a>) {
        self.0.walk_mut(walker);
        self.1.walk_mut(walker);
    }
}

/// Leaf: the `&'static str` payload of the `Unsupported` variants. Carries no
/// AST children, so both walks stop here.
impl AstWalkable for &'static str {
    fn walk<'a>(&'a self, _walker: &mut impl AstWalker<'a>) {}
    fn walk_mut<'a>(&'a mut self, _walker: &mut impl AstWalkerMut<'a>) {}
    fn walk_children<'a>(&'a self, _walker: &mut impl AstWalker<'a>) {}
    fn walk_mut_children<'a>(&'a mut self, _walker: &mut impl AstWalkerMut<'a>) {}
}

walk_struct!(String, walk_string, walk_mut_string);
walk_struct!(bool, walk_bool, walk_mut_bool);
walk_struct!(BigInt, walk_big_int, walk_mut_big_int);
walk_struct!(BigRational, walk_big_rational, walk_mut_big_rational);
walk_struct!(Spur, walk_spur, walk_mut_spur);
walk_struct!(usize, walk_usize, walk_mut_usize);
