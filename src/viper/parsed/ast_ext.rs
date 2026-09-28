use super::*;

impl Program {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, &Declaration)> {
        self.0.iter().enumerate()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut Declaration)> {
        self.0.iter_mut().enumerate()
    }
}

impl Declaration {
    pub fn contract(&self) -> Option<&Contract> {
        match self {
            Declaration::Function(f) => Some(&f.contract),
            Declaration::Method(m) => Some(&m.contract),
            _ => None,
        }
    }

    pub fn signature(&self) -> Option<&Signature> {
        use Declaration::*;
        match self {
            Function(f) => Some(&f.signature),
            Method(m) => Some(&m.signature),
            Predicate(p) => Some(&p.signature),
            DomainElement(e) => match &e.kind {
                DomainElementKind::Function(f) => Some(&f.signature),
                DomainElementKind::Axiom(_) => None,
            },
            AdtConstructor(a) => Some(&a.signature),
            Field(_) | Import(_) | Define(_) | Domain(_) | Adt(_) => None,
        }
    }

    pub fn idn_decl(&self) -> Option<&IdnDecl> {
        use Declaration::*;
        match self {
            Function(f) => Some(&f.signature.name),
            Method(m) => Some(&m.signature.name),
            Predicate(p) => Some(&p.signature.name),
            DomainElement(e) => match &e.kind {
                DomainElementKind::Function(f) => Some(&f.signature.name),
                DomainElementKind::Axiom(a) => a.name.as_ref(),
            },
            Import(_) => None,
            Define(d) => Some(&d.name),
            Domain(d) => Some(&d.name),
            Field(f) => Some(&f.0.idn),
            Adt(a) => Some(&a.name),
            AdtConstructor(a) => Some(&a.signature.name),
        }
    }
}

impl Adt {
    pub fn identity(&self) -> Type {
        Type::Domain(
            self.name.0.clone(),
            self.params
                .iter()
                .map(|p| Type::Domain(p.0.clone(), Vec::new()))
                .collect(),
        )
    }
}

impl Variant {
    pub fn destructors<'a>(&'a self) -> impl Iterator<Item = &'a IdnDeclTyped> + 'a {
        expect_args(&self.fields)
    }
}

impl AdtConstructor {
    pub fn destructors<'a>(&'a self) -> impl Iterator<Item = &'a IdnDeclTyped> + 'a {
        expect_args(&self.signature.args)
    }

    pub fn adt(&self) -> &Ident {
        let Type::Domain(adt, ..) = &self.signature.ret[0].ty() else {
            unreachable!()
        };
        adt
    }
}

// impl HeapExp {
//     pub(crate) fn new(exp: Exp) -> Self {
//         Self {
//             kind: HeapExpKind::Pure(exp),
//         }
//     }
//
//     pub(super) fn conjoin(exps: Vec<HeapExp>) -> Option<Self> {
//         if exps.is_empty() {
//             None
//         } else {
//             Some(HeapExp {
//                 kind: HeapExpKind::Conjunction(exps),
//             })
//         }
//     }
// }

// impl From<Vec<PrePostDec>> for Contract {
//     fn from(value: Vec<PrePostDec>) -> Self {
//         let mut precondition = None;
//         let mut decreases = vec![];
//         for p in value {
//             match p {
//                 PrePostDec::Pre(e) => precondition = ExpKind::conjoin(precondition, e),
//                 PrePostDec::Decreases(d) => decreases.push(d),
//                 _ => {}
//             }
//         }
//         Self {
//             precondition: precondition.map(HeapExp::new),
//             postcondition: None,
//             decreases,
//         }
//     }
// }
//
// impl Contract {
//     pub(super) fn add_posts(mut self, posts: Vec<PrePostDec>) -> Self {
//         for p in posts {
//             match p {
//                 PrePostDec::Post(e) => {
//                     let new = match self.postcondition {
//                         Some(post) => {
//                             HeapExp::new(Box::new(ExpKind::BinOp(BinOp::And, post.into_exp(), e)))
//                         }
//                         None => HeapExp::new(e),
//                     };
//                     self.postcondition = Some(new);
//                 }
//                 PrePostDec::Decreases(d) => self.decreases.push(d),
//                 _ => {}
//             }
//         }
//         self
//     }
// }

impl Field {
    pub fn ty(&self) -> &Type {
        &self.0.ty
    }
}

impl ExpKind {
    pub fn is_true(&self) -> bool {
        matches!(self, ExpKind::Const(ConstKind::Bool(true)))
    }
}

// impl HeapExp {
//     pub fn into_exp(self) -> Exp {
//         match self.kind {
//             HeapExpKind::Pure(exp) => exp,
//             HeapExpKind::Acc(_) => {
//                 panic!("HeapExpKind::Acc cannot be converted into a pure ExpKind")
//             }
//             HeapExpKind::Conjunction(heap_exps) => heap_exps
//                 .into_iter()
//                 .fold(None, |acc, heap_exp| {
//                     ExpKind::conjoin(acc, heap_exp.into_exp())
//                 })
//                 .unwrap_or_else(|| Box::new(ExpKind::Const(ConstKind::Bool(true)))),
//             HeapExpKind::MagicWand(mut heap_exps) => {
//                 let rhs = heap_exps
//                     .pop()
//                     .expect("magic wand must contain rhs heap expression");
//                 let lhs = heap_exps
//                     .pop()
//                     .expect("magic wand must contain lhs heap expression");
//                 assert!(
//                     heap_exps.is_empty(),
//                     "magic wand must contain exactly two heap expressions"
//                 );
//                 Box::new(ExpKind::MagicWand(lhs, rhs))
//             }
//             HeapExpKind::Ternary(cond, then_heap, else_heap) => Box::new(ExpKind::Ternary(
//                 cond,
//                 then_heap.into_exp(),
//                 else_heap.into_exp(),
//             )),
//         }
//     }
// }
//
// impl ResourceExp {
//     pub fn loc(&self) -> Result<&Ident, (&HeapExp, &HeapExp)> {
//         match &*self.acc.acc.loc {
//             ExpKind::Call(callee, ..) => Ok(callee),
//             ExpKind::MagicWand(lhs, rhs) => Err((lhs, rhs)),
//             _ => unreachable!(),
//         }
//     }
// }
//
fn expect_args<'a>(args: &'a [ArgOrType]) -> impl Iterator<Item = &'a IdnDeclTyped> + 'a {
    args.iter().map(|arg| {
        let ArgOrType::Arg(arg) = arg else {
            panic!("expected argument, got type");
        };
        arg
    })
}
