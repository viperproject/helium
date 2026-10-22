use std::fmt;

use crate::dhash::{HashMap, HashSet};

use lasso::Spur;
use nonmax::NonMaxU32;
use typed_index_collections::TiVec;

use crate::viper::{IdnDecl, interner::Interner, typed::Type, walk::AstWalker};

/// Rewrite references to a bound type parameter into `Type::Generic`. The parser
/// emits every named type as `Type::Domain(name, args)`, so a type-parameter use
/// like `T` arrives as `Domain(T, [])`. Within a generic domain/ADT, such a
/// nullary domain whose name is one of the declared parameters is actually a
/// type variable; this turns it into `Generic(T)` so type inference can
/// instantiate it. Recurses into type arguments (e.g. `List[T]`).
fn genericize(ty: Type, params: &HashSet<Spur>) -> Type {
    match ty {
        Type::Domain(id, args) if args.is_empty() && params.contains(&id.0) => Type::Generic(id),
        Type::Domain(id, args) => Type::Domain(
            id,
            args.into_iter().map(|a| genericize(a, params)).collect(),
        ),
        other => other,
    }
}

#[derive(Debug, Clone)]
pub struct FunctionSig {
    pub params: Vec<Type>,
    pub ret: Type,
}

#[derive(Debug, Clone)]
pub struct MethodSig {
    pub params: Vec<Type>,
    pub rets: Vec<Type>,
}

#[derive(Debug, Clone)]
pub struct PredicateSig {
    pub params: Vec<Type>,
    /// Whether the declaration supplied a body. A bodyless (abstract) predicate
    /// is a bare location: it cannot be folded, unfolded, or used in
    /// `unfolding`, and lowers to a `function` + `domain` pair rather than a
    /// `vmir::Resource`.
    pub has_body: bool,
}

#[derive(Debug, Clone)]
pub struct DomainSig {
    pub type_arity: usize,
    /// Type-parameter names, in declaration order (length == `type_arity`).
    pub params: Vec<Spur>,
}

#[derive(Debug, Clone)]
pub struct AdtSig {
    pub type_arity: usize,
    /// Type-parameter names, in declaration order (length == `type_arity`).
    /// Lets a destructor's `Generic` field type be mapped to the scrutinee's
    /// corresponding type argument.
    pub params: Vec<Spur>,
}

#[derive(Debug, Clone)]
pub struct AdtConstructorSig {
    pub params: Vec<Type>,
    pub ret: Type,
    /// Name `Spur` of the owning ADT.
    pub adt: Spur,
    /// Position of this constructor among its ADT's variants (its tag index).
    pub tag: usize,
}

/// A destructor (constructor field accessor), e.g. `head` of `Cons`.
#[derive(Debug, Clone)]
pub struct DtorInfo {
    /// Name `Spur` of the owning ADT.
    pub adt: Spur,
    /// Name `Spur` of the constructor this field belongs to.
    pub ctor: Spur,
    /// Field position within the constructor.
    pub index: usize,
    /// The field's type.
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct MacroSig {
    pub arity: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlobalKind {
    Field,
    Predicate,
    Function,
    DomainFunction,
    Method,
    Domain,
    Adt,
    AdtConstructor,
    ExpMacro,
    StmtMacro,
    /// The kinds below have no [`GlobalSignature`]: they only take part in
    /// Silver's duplicate-identifier check.
    AdtField,
    Axiom,
    TypeParam,
}

impl fmt::Display for GlobalKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Field => "field",
            Self::Predicate => "predicate",
            Self::Function => "function",
            Self::DomainFunction => "domain function",
            Self::Method => "method",
            Self::Domain => "domain",
            Self::Adt => "ADT",
            Self::AdtConstructor => "ADT constructor",
            Self::ExpMacro => "macro",
            Self::StmtMacro => "macro",
            Self::AdtField => "ADT field",
            Self::Axiom => "axiom",
            Self::TypeParam => "type parameter",
        };
        write!(f, "{}", name)
    }
}

#[derive(Debug, Clone)]
pub enum GlobalSignature {
    Field(Type),
    Predicate(PredicateSig),
    Function(FunctionSig),
    DomainFunction(FunctionSig),
    Method(MethodSig),
    Domain(DomainSig),
    Adt(AdtSig),
    AdtConstructor(AdtConstructorSig),
    ExpMacro(MacroSig),
    StmtMacro(MacroSig),
}

impl GlobalSignature {
    pub fn kind(&self) -> GlobalKind {
        match self {
            Self::Field(_) => GlobalKind::Field,
            Self::Predicate(_) => GlobalKind::Predicate,
            Self::Function(_) => GlobalKind::Function,
            Self::DomainFunction(_) => GlobalKind::DomainFunction,
            Self::Method(_) => GlobalKind::Method,
            Self::Domain(_) => GlobalKind::Domain,
            Self::Adt(_) => GlobalKind::Adt,
            Self::AdtConstructor(_) => GlobalKind::AdtConstructor,
            Self::ExpMacro(_) => GlobalKind::ExpMacro,
            Self::StmtMacro(_) => GlobalKind::StmtMacro,
        }
    }
}

impl From<GlobalSignature> for GlobalKind {
    fn from(sig: GlobalSignature) -> Self {
        sig.kind()
    }
}

impl GlobalSignature {
    pub fn as_function(&self) -> Option<&FunctionSig> {
        match self {
            Self::Function(sig) => Some(sig),
            _ => None,
        }
    }

    pub fn as_method(&self) -> Option<&MethodSig> {
        match self {
            Self::Method(sig) => Some(sig),
            _ => None,
        }
    }

    pub fn as_predicate(&self) -> Option<&PredicateSig> {
        match self {
            Self::Predicate(sig) => Some(sig),
            _ => None,
        }
    }

    pub fn as_field(&self) -> Option<&Type> {
        match self {
            Self::Field(ty) => Some(ty),
            _ => None,
        }
    }

    pub fn as_domain(&self) -> Option<&DomainSig> {
        match self {
            Self::Domain(sig) => Some(sig),
            _ => None,
        }
    }

    pub fn as_adt(&self) -> Option<&AdtSig> {
        match self {
            Self::Adt(sig) => Some(sig),
            _ => None,
        }
    }

    pub fn as_adt_constructor(&self) -> Option<&AdtConstructorSig> {
        match self {
            Self::AdtConstructor(sig) => Some(sig),
            _ => None,
        }
    }

    pub fn as_macro(&self) -> Option<&MacroSig> {
        match self {
            Self::ExpMacro(sig) | Self::StmtMacro(sig) => Some(sig),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DuplicateGlobalError {
    pub name: String,
    pub this: GlobalKind,
    pub other: GlobalKind,
}

impl fmt::Display for DuplicateGlobalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Duplicate global definition: `{}` is defined as {}, but was already defined as {}",
            self.name, self.this, self.other
        )
    }
}

impl std::error::Error for DuplicateGlobalError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemberId(NonMaxU32);

impl From<MemberId> for usize {
    fn from(val: MemberId) -> Self {
        val.0.get() as usize
    }
}

impl From<usize> for MemberId {
    fn from(value: usize) -> Self {
        Self(NonMaxU32::new(value as u32).expect("MemberId overflow"))
    }
}

#[derive(Debug, Clone, Default)]
pub struct Globals {
    pub signatures: TiVec<MemberId, GlobalSignature>,
    pub symbol_table: HashMap<Spur, MemberId>,
    /// ADT constructor name (resolved string) → its name `Spur`. Lets passes
    /// that only hold a `RodeoResolver` (no string lookup) resolve a
    /// discriminator `is<Ctor>` back to the constructor.
    pub ctor_by_name: HashMap<String, Spur>,
    /// Destructor (constructor field) name `Spur` → one entry per ADT
    /// declaring it. Lets `e.f` be classified as an ADT destructor. Different
    /// ADTs may share a field name; the receiver's type picks the entry.
    pub dtor_by_name: HashMap<Spur, Vec<DtorInfo>>,
}

/// A lightweight view into a successfully resolved global symbol.
pub struct ResolvedSymbol<'a> {
    globals: &'a Globals,
    mid: MemberId,
}

// A throwaway view: the `as_*` accessors consume it and borrow from `globals`.
#[allow(clippy::wrong_self_convention)]
impl<'a> ResolvedSymbol<'a> {
    pub fn id(self) -> MemberId {
        self.mid
    }

    pub fn signature(self) -> &'a GlobalSignature {
        &self.globals.signatures[self.mid]
    }

    pub fn kind(self) -> GlobalKind {
        self.signature().kind()
    }

    // These now naturally return Option<&T>
    // because they delegate to the updated GlobalSignature methods.
    pub fn as_function(self) -> Option<&'a FunctionSig> {
        self.signature().as_function()
    }
    pub fn as_method(self) -> Option<&'a MethodSig> {
        self.signature().as_method()
    }
    pub fn as_predicate(self) -> Option<&'a PredicateSig> {
        self.signature().as_predicate()
    }
    pub fn as_field(self) -> Option<&'a Type> {
        self.signature().as_field()
    }
    pub fn as_domain(self) -> Option<&'a DomainSig> {
        self.signature().as_domain()
    }
    pub fn as_adt(self) -> Option<&'a AdtSig> {
        self.signature().as_adt()
    }
    pub fn as_adt_constructor(self) -> Option<&'a AdtConstructorSig> {
        self.signature().as_adt_constructor()
    }
    pub fn as_macro(self) -> Option<&'a MacroSig> {
        self.signature().as_macro()
    }
}

impl Globals {
    pub fn resolve(&self, id: Spur) -> Option<ResolvedSymbol<'_>> {
        self.symbol_table
            .get(&id)
            .map(|&mid| ResolvedSymbol { globals: self, mid })
    }
}

#[derive(Debug, Clone)]
pub struct GlobalsCollector<'i> {
    interner: &'i Interner,
    signatures: TiVec<MemberId, GlobalSignature>,
    symbol_table: HashMap<Spur, MemberId>,
    ctor_by_name: HashMap<String, Spur>,
    /// ADT field names are global identifiers in Silver, with one exception:
    /// different ADTs may share one (resolved through the receiver's type).
    dtor_by_name: HashMap<Spur, Vec<DtorInfo>>,
    /// Running per-ADT constructor counter, for assigning tag indices.
    adt_ctor_count: HashMap<Spur, usize>,
    /// Named domain axioms. Global identifiers in Silver, unique program-wide.
    axiom_names: HashSet<Spur>,
    /// Every ADT's type parameters. Scoped to their ADT, but like a local they
    /// may not reuse a global name.
    adt_type_params: Vec<Spur>,
    errors: Vec<DuplicateGlobalError>,
}

impl<'i> GlobalsCollector<'i> {
    pub fn new(interner: &'i Interner) -> Self {
        Self {
            interner,
            signatures: TiVec::new(),
            symbol_table: HashMap::default(),
            ctor_by_name: HashMap::default(),
            dtor_by_name: HashMap::default(),
            adt_ctor_count: HashMap::default(),
            axiom_names: HashSet::default(),
            adt_type_params: Vec::new(),
            errors: Vec::new(),
        }
    }

    pub fn finalize(mut self) -> Result<Globals, Vec<DuplicateGlobalError>> {
        // The names outside `symbol_table` clash with any global declared
        // anywhere in the file, so they can only be checked once all are in.
        let mut late: Vec<(Spur, GlobalKind)> = Vec::new();
        late.extend(self.dtor_by_name.keys().map(|&f| (f, GlobalKind::AdtField)));
        late.extend(self.axiom_names.iter().map(|&a| (a, GlobalKind::Axiom)));
        late.extend(
            self.adt_type_params
                .iter()
                .map(|&t| (t, GlobalKind::TypeParam)),
        );
        for (id, this) in late {
            let other = match self.symbol_table.get(&id) {
                Some(&mid) => self.signatures[mid].kind(),
                None if this == GlobalKind::Axiom && self.dtor_by_name.contains_key(&id) => {
                    GlobalKind::AdtField
                }
                None => continue,
            };
            self.duplicate(id, this, other);
        }
        if self.errors.is_empty() {
            Ok(Globals {
                signatures: self.signatures,
                symbol_table: self.symbol_table,
                ctor_by_name: self.ctor_by_name,
                dtor_by_name: self.dtor_by_name,
            })
        } else {
            Err(self.errors)
        }
    }

    fn duplicate(&mut self, id: Spur, this: GlobalKind, other: GlobalKind) {
        self.errors.push(DuplicateGlobalError {
            name: self.interner.resolve(&id).to_string(),
            this,
            other,
        });
    }

    fn register(&mut self, name: &IdnDecl, sig: GlobalSignature) -> Option<MemberId> {
        let id = name.0.id();
        if let Some(&mid) = self.symbol_table.get(&id) {
            let other = self.signatures[mid].kind();
            self.duplicate(id, sig.kind(), other);
            None
        } else {
            let mid = self.signatures.push_and_get_key(sig);
            self.symbol_table.insert(id, mid);
            Some(mid)
        }
    }
}

impl<'ast, 'i> AstWalker<'ast> for GlobalsCollector<'i> {
    fn walk_field(&mut self, field: &'ast super::Field) {
        self.register(
            &field.0.idn,
            GlobalSignature::Field(Type::from(&field.0.ty)),
        );
    }

    fn walk_predicate(&mut self, pred: &'ast super::Predicate) {
        let sig = PredicateSig {
            params: pred
                .signature
                .args
                .iter()
                .map(|p| Type::from(p.ty()))
                .collect(),
            has_body: pred.body.is_some(),
        };
        self.register(&pred.signature.name, GlobalSignature::Predicate(sig));
    }

    fn walk_function(&mut self, func: &'ast super::Function) {
        let sig = FunctionSig {
            params: func
                .signature
                .args
                .iter()
                .map(|p| Type::from(p.ty()))
                .collect(),
            ret: Type::from(func.signature.ret[0].ty()),
        };
        self.register(&func.signature.name, GlobalSignature::Function(sig));
    }

    fn walk_method(&mut self, method: &'ast super::Method) {
        let sig = MethodSig {
            params: method
                .signature
                .args
                .iter()
                .map(|p| Type::from(p.ty()))
                .collect(),
            rets: method
                .signature
                .ret
                .iter()
                .map(|r| Type::from(r.ty()))
                .collect(),
        };
        self.register(&method.signature.name, GlobalSignature::Method(sig));
    }

    fn walk_domain(&mut self, domain: &'ast super::Domain) {
        let sig = DomainSig {
            type_arity: domain.params.len(),
            params: domain.params.iter().map(|p| p.0.id()).collect(),
        };
        self.register(&domain.name, GlobalSignature::Domain(sig));
    }

    fn walk_adt(&mut self, adt: &'ast super::Adt) {
        let mut seen = HashSet::default();
        for p in &adt.params {
            let id = p.0.id();
            if !seen.insert(id) {
                self.duplicate(id, GlobalKind::TypeParam, GlobalKind::TypeParam);
            }
            self.adt_type_params.push(id);
        }
        let sig = AdtSig {
            type_arity: adt.params.len(),
            params: adt.params.iter().map(|p| p.0.id()).collect(),
        };
        self.register(&adt.name, GlobalSignature::Adt(sig));
    }

    fn walk_adt_constructor(&mut self, adt_cons: &'ast super::AdtConstructor) {
        let adt = adt_cons.adt().id();
        let tag = *self.adt_ctor_count.entry(adt).or_insert(0);
        self.adt_ctor_count.insert(adt, tag + 1);
        // The ADT (registered before its constructors) supplies the set of bound
        // type parameters, so generic field/return types lower to `Generic`.
        let type_params: HashSet<Spur> = self
            .symbol_table
            .get(&adt)
            .and_then(|mid| self.signatures[*mid].as_adt())
            .map(|s| s.params.iter().copied().collect())
            .unwrap_or_default();
        let sig = AdtConstructorSig {
            params: adt_cons
                .signature
                .args
                .iter()
                .map(|p| genericize(Type::from(p.ty()), &type_params))
                .collect(),
            ret: genericize(Type::from(adt_cons.signature.ret[0].ty()), &type_params),
            adt,
            tag,
        };
        let name_spur = adt_cons.signature.name.0.id();
        self.ctor_by_name
            .insert(self.interner.resolve(&name_spur).to_string(), name_spur);
        // Register each field as a destructor. A name repeated within one ADT
        // is a duplicate; one shared with another ADT is legal Silver.
        for (index, field) in adt_cons.destructors().enumerate() {
            let field_spur = field.idn.0.id();
            let owners = self.dtor_by_name.entry(field_spur).or_default();
            if owners.iter().any(|d| d.adt == adt) {
                self.duplicate(field_spur, GlobalKind::AdtField, GlobalKind::AdtField);
                continue;
            }
            owners.push(DtorInfo {
                adt,
                ctor: name_spur,
                index,
                ty: genericize(Type::from(&field.ty), &type_params),
            });
        }
        self.register(
            &adt_cons.signature.name,
            GlobalSignature::AdtConstructor(sig),
        );
    }

    fn walk_domain_element(&mut self, elem: &'ast super::DomainElement) {
        let func = match &elem.kind {
            super::DomainElementKind::Function(func) => func,
            // An axiom is not a callable global, but its name is an identifier.
            super::DomainElementKind::Axiom(ax) => {
                if let Some(name) = &ax.name {
                    let id = name.0.id();
                    if !self.axiom_names.insert(id) {
                        self.duplicate(id, GlobalKind::Axiom, GlobalKind::Axiom);
                    }
                }
                return;
            }
        };
        // The owning domain (registered before its elements) supplies the bound
        // type parameters, so generic arg/return types lower to `Generic`.
        let type_params: HashSet<Spur> = self
            .symbol_table
            .get(&elem.domain.id())
            .and_then(|mid| self.signatures[*mid].as_domain())
            .map(|s| s.params.iter().copied().collect())
            .unwrap_or_default();
        let sig = FunctionSig {
            params: func
                .signature
                .args
                .iter()
                .map(|p| genericize(Type::from(p.ty()), &type_params))
                .collect(),
            ret: genericize(Type::from(func.signature.ret[0].ty()), &type_params),
        };
        self.register(&func.signature.name, GlobalSignature::DomainFunction(sig));
    }

    fn walk_define(&mut self, define: &'ast super::Define) {
        let sig = MacroSig {
            arity: define.args.len(),
        };

        let sig_enum = match define.body {
            super::ExpOrBlock::Exp(_) => GlobalSignature::ExpMacro(sig),
            super::ExpOrBlock::Block(_) => GlobalSignature::StmtMacro(sig),
        };

        self.register(&define.name, sig_enum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viper::{IdentCollector, viper_parser, walk::AstWalkable};

    fn collect(input: &str) -> Result<Globals, Vec<DuplicateGlobalError>> {
        let mut program = viper_parser::vpr_program(input).expect("parse failed");
        let mut ident_collector = IdentCollector::default();
        program.walk_mut(&mut ident_collector);
        let interner = ident_collector.finalize();
        let mut collector = GlobalsCollector::new(&interner);
        program.walk(&mut collector);
        collector.finalize()
    }

    fn assert_duplicate(input: &str, name: &str, this: GlobalKind) {
        let errs = collect(input).expect_err("expected a duplicate identifier");
        assert!(
            errs.iter().any(|e| e.name == name && e.this == this),
            "{input}: unexpected errors {errs:?}"
        );
    }

    #[test]
    fn adt_field_repeated_within_one_adt_is_duplicate() {
        // futures-util: two constructors of one (merged) ADT share a field name,
        // which Silicon reports as a duplicate and, at each use, as ambiguous.
        assert_duplicate(
            "adt s_Projection_dis26 {\n\
             s_Projection_dis26_cons(s_Projection_dis26_0: Int, s_Projection_dis26_1: Bool)\n\
             s_Projection_dis26_cons__a1(s_Projection_dis26_0: Bool)\n}",
            "s_Projection_dis26_0",
            GlobalKind::AdtField,
        );
        assert_duplicate("adt A { A1(f: Int, f: Int) }", "f", GlobalKind::AdtField);
    }

    #[test]
    fn adt_field_clashing_with_a_global_is_duplicate() {
        for (src, name, other) in [
            ("adt A { A1(f: Int) }\nfield f: Int", "f", GlobalKind::Field),
            (
                "adt A { A1(f: Int) }\nfunction f(): Int",
                "f",
                GlobalKind::Function,
            ),
            (
                "adt A { A1(f: Int) }\npredicate f(x: Ref)",
                "f",
                GlobalKind::Predicate,
            ),
            (
                "domain D { function f(): Int }\nadt A { A1(f: Int) }",
                "f",
                GlobalKind::DomainFunction,
            ),
            ("adt A { A1(A: Int) }", "A", GlobalKind::Adt),
            ("adt A { f(f: Int) }", "f", GlobalKind::AdtConstructor),
        ] {
            let errs = collect(src).expect_err("expected a duplicate identifier");
            let expected = DuplicateGlobalError {
                name: name.to_string(),
                this: GlobalKind::AdtField,
                other,
            };
            assert!(
                errs.contains(&expected),
                "{src}: unexpected errors {errs:?}"
            );
        }
    }

    #[test]
    fn adt_field_shared_between_adts_is_legal() {
        assert!(collect("adt A { A1(f: Int) }\nadt B { B1(f: Bool) }").is_ok());
    }

    #[test]
    fn axiom_names_are_global() {
        assert_duplicate(
            "domain D { function d(): Int  axiom a { d() == 0 } }\n\
             domain E { function e(): Int  axiom a { e() == 0 } }",
            "a",
            GlobalKind::Axiom,
        );
        assert_duplicate(
            "function f(): Int\ndomain D { function d(): Int  axiom f { d() == 0 } }",
            "f",
            GlobalKind::Axiom,
        );
        // Unnamed axioms take no name.
        assert!(collect("domain D { function d(): Int  axiom { true } axiom { true } }").is_ok());
    }

    #[test]
    fn adt_type_parameter_clashes_are_duplicates() {
        assert_duplicate("adt A[T, T] { A1(x: T) }", "T", GlobalKind::TypeParam);
        assert_duplicate(
            "function T(): Int\nadt A[T] { A1(x: T) }",
            "T",
            GlobalKind::TypeParam,
        );
        // Each ADT scopes its own parameters.
        assert!(collect("adt A[T] { A1(x: T) }\nadt B[T] { B1(y: T) }").is_ok());
    }
}
