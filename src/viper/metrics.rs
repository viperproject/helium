//! Shape metrics of a parsed Viper program, for `verify --viper-metrics`.
//!
//! These describe the *input* the verifier is handed — how big Prusti's
//! encoding is and what it is made of — so the benchmark runner can relate
//! verification cost to program shape. They are counted on the parsed AST,
//! before any pass rewrites it, so they measure exactly what is in the file.

use crate::dhash::HashSet;
use crate::json::Json;
use crate::viper::parsed::ast::*;
use crate::viper::units::unit_name;
use crate::viper::walk::{AstWalkable, AstWalker};

/// Node counts over some part of a program (one member, or the whole file).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Counts {
    /// Statements of any kind, nested ones included.
    pub stmts: u64,
    /// Expression nodes.
    pub exps: u64,
    pub folds: u64,
    pub unfolds: u64,
    /// `unfolding ... in`, `folding ... in` and friends inside expressions.
    pub heap_update_exps: u64,
    pub inhales: u64,
    pub exhales: u64,
    pub asserts: u64,
    pub assumes: u64,
    /// `acc(...)` occurrences in expressions (contracts, inhale/exhale bodies).
    pub accs: u64,
    pub quantifiers: u64,
    /// `label` and `goto`: together, the size of a Prusti method's CFG.
    pub labels: u64,
    pub gotos: u64,
    pub ifs: u64,
    pub whiles: u64,
    pub assigns: u64,
    pub method_calls: u64,
    /// Calls in expressions: functions, domain functions, predicate instances.
    pub function_apps: u64,
    pub old_exps: u64,
    /// Pre- and postconditions.
    pub contract_clauses: u64,
}

impl Counts {
    pub fn to_json(&self) -> Json {
        let Counts {
            stmts,
            exps,
            folds,
            unfolds,
            heap_update_exps,
            inhales,
            exhales,
            asserts,
            assumes,
            accs,
            quantifiers,
            labels,
            gotos,
            ifs,
            whiles,
            assigns,
            method_calls,
            function_apps,
            old_exps,
            contract_clauses,
        } = self;
        Json::obj([
            ("stmts", Json::from(*stmts)),
            ("exps", Json::from(*exps)),
            ("folds", Json::from(*folds)),
            ("unfolds", Json::from(*unfolds)),
            ("heap_update_exps", Json::from(*heap_update_exps)),
            ("inhales", Json::from(*inhales)),
            ("exhales", Json::from(*exhales)),
            ("asserts", Json::from(*asserts)),
            ("assumes", Json::from(*assumes)),
            ("accs", Json::from(*accs)),
            ("quantifiers", Json::from(*quantifiers)),
            ("labels", Json::from(*labels)),
            ("gotos", Json::from(*gotos)),
            ("ifs", Json::from(*ifs)),
            ("whiles", Json::from(*whiles)),
            ("assigns", Json::from(*assigns)),
            ("method_calls", Json::from(*method_calls)),
            ("function_apps", Json::from(*function_apps)),
            ("old_exps", Json::from(*old_exps)),
            ("contract_clauses", Json::from(*contract_clauses)),
        ])
    }
}

/// Counts nodes into `counts`. The parser cannot tell `x := f(a)` (a function
/// application) from `x := m(a)` (a method call) — both come out as
/// [`AssignRhs::Call`] until disambiguation — so the walker is handed the
/// program's method names to decide.
struct Counter<'c, 'm> {
    counts: &'c mut Counts,
    methods: &'m HashSet<String>,
}

impl Counter<'_, '_> {
    fn call_rhs(&mut self, call: &Call<StmtCallKind>) {
        let is_method = match &call.name {
            Ident::Raw(name) => self.methods.contains(name),
            Ident::Interned(_) => true,
        };
        if is_method {
            self.counts.method_calls += 1;
        } else {
            // Walking `args` does not visit the call node itself.
            self.counts.assigns += 1;
            self.counts.function_apps += 1;
            self.counts.exps += 1;
        }
    }
}

impl<'a> AstWalker<'a> for Counter<'_, '_> {
    fn walk_statement(&mut self, stmt: &'a Statement) {
        let c = &mut *self.counts;
        c.stmts += 1;
        match stmt {
            Statement::Assume(_) => c.assumes += 1,
            Statement::Assert(_) | Statement::Refute(_) => c.asserts += 1,
            Statement::Inhale(_) => c.inhales += 1,
            Statement::Exhale(_) => c.exhales += 1,
            Statement::Fold(_) => c.folds += 1,
            Statement::Unfold(_) => c.unfolds += 1,
            Statement::Goto(_) => c.gotos += 1,
            Statement::Label(..) => c.labels += 1,
            Statement::While(..) => c.whiles += 1,
            Statement::If(..) => c.ifs += 1,
            Statement::Assign(_, AssignRhs::Call(call))
            | Statement::Var(_, Some(AssignRhs::Call(call))) => self.call_rhs(call),
            Statement::Assign(..) | Statement::Var(_, Some(_)) => c.assigns += 1,
            Statement::Var(_, None) | Statement::Block(_) | Statement::Unsupported(_) => {}
        }
        stmt.walk_children(self);
    }

    fn walk_exp_kind(&mut self, kind: &'a ExpKind) {
        let c = &mut *self.counts;
        c.exps += 1;
        match kind {
            ExpKind::Acc(_) => c.accs += 1,
            ExpKind::Call(_) => c.function_apps += 1,
            ExpKind::Quantifier(..) | ExpKind::ForPerm(..) => c.quantifiers += 1,
            ExpKind::HeapUpdate(..) => c.heap_update_exps += 1,
            ExpKind::Old(..) => c.old_exps += 1,
            _ => {}
        }
        kind.walk_children(self);
    }

    fn walk_contract(&mut self, contract: &'a Contract) {
        self.counts.contract_clauses +=
            (contract.precondition.len() + contract.postcondition.len()) as u64;
        contract.walk_children(self);
    }
}

/// Metrics for one file: its size, its declarations, node counts over the whole
/// program, and the same counts per method, function and predicate.
#[derive(Debug, Default, Clone)]
pub struct ViperMetrics {
    /// Lines that are neither blank nor `//` comments.
    pub loc: u64,
    pub lines: u64,
    pub bytes: u64,
    pub methods: u64,
    pub functions: u64,
    pub predicates: u64,
    pub domains: u64,
    pub domain_functions: u64,
    pub axioms: u64,
    pub fields: u64,
    pub adts: u64,
    pub totals: Counts,
    /// `(name, kind, counts)` for each method, function and predicate, in
    /// source order.
    pub members: Vec<(String, &'static str, Counts)>,
}

impl ViperMetrics {
    pub fn of(source: &str, program: &Program) -> Self {
        let mut m = ViperMetrics {
            lines: source.lines().count() as u64,
            loc: source
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with("//"))
                .count() as u64,
            bytes: source.len() as u64,
            ..Default::default()
        };
        let methods: HashSet<String> = program
            .0
            .iter()
            .filter_map(|d| match d {
                Declaration::Method(_) => Some(unit_name(d)),
                _ => None,
            })
            .collect();
        program.walk(&mut Counter {
            counts: &mut m.totals,
            methods: &methods,
        });
        for decl in &program.0 {
            let kind = match decl {
                Declaration::Method(_) => {
                    m.methods += 1;
                    "method"
                }
                Declaration::Function(_) => {
                    m.functions += 1;
                    "function"
                }
                Declaration::Predicate(_) => {
                    m.predicates += 1;
                    "predicate"
                }
                Declaration::Domain(_) => {
                    m.domains += 1;
                    continue;
                }
                Declaration::DomainElement(DomainElement { kind, .. }) => {
                    match kind {
                        DomainElementKind::Function(_) => m.domain_functions += 1,
                        DomainElementKind::Axiom(_) => m.axioms += 1,
                    }
                    continue;
                }
                Declaration::Field(_) => {
                    m.fields += 1;
                    continue;
                }
                Declaration::Adt(_) => {
                    m.adts += 1;
                    continue;
                }
                Declaration::Import(_)
                | Declaration::Define(_)
                | Declaration::AdtConstructor(_) => {
                    continue;
                }
            };
            let mut counts = Counts::default();
            decl.walk(&mut Counter {
                counts: &mut counts,
                methods: &methods,
            });
            m.members.push((unit_name(decl), kind, counts));
        }
        m
    }

    pub fn to_json(&self) -> Json {
        Json::obj([
            ("loc", Json::from(self.loc)),
            ("lines", Json::from(self.lines)),
            ("bytes", Json::from(self.bytes)),
            ("methods", Json::from(self.methods)),
            ("functions", Json::from(self.functions)),
            ("predicates", Json::from(self.predicates)),
            ("domains", Json::from(self.domains)),
            ("domain_functions", Json::from(self.domain_functions)),
            ("axioms", Json::from(self.axioms)),
            ("fields", Json::from(self.fields)),
            ("adts", Json::from(self.adts)),
            ("totals", self.totals.to_json()),
            (
                "members",
                Json::Arr(
                    self.members
                        .iter()
                        .map(|(name, kind, counts)| {
                            Json::obj([
                                ("name", Json::from(name.as_str())),
                                ("kind", Json::from(*kind)),
                                ("counts", counts.to_json()),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viper_parser;

    #[test]
    fn counts_statements_per_member() {
        let src = "\
field f: Int
predicate P(x: Ref) { acc(x.f) }
function g(x: Int): Int { x + 1 }
// a comment
method m(x: Ref)
  requires acc(P(x))
  ensures acc(P(x))
{
  unfold P(x)
  x.f := g(x.f)
  fold P(x)
  label l
  goto l
  inhale forall i: Int :: i == i
  n(x)
}
method n(x: Ref)
";
        let program = viper_parser::vpr_program(src).unwrap();
        let m = ViperMetrics::of(src, &program);
        assert_eq!(
            (m.methods, m.functions, m.predicates, m.fields),
            (2, 1, 1, 1)
        );
        assert_eq!(m.loc, 16);
        let (name, kind, c) = &m.members[2];
        assert_eq!((name.as_str(), *kind), ("m", "method"));
        assert_eq!((c.folds, c.unfolds, c.labels, c.gotos), (1, 1, 1, 1));
        assert_eq!(
            (c.inhales, c.quantifiers, c.assigns, c.method_calls),
            (1, 1, 1, 1)
        );
        assert_eq!(c.contract_clauses, 2);
        assert_eq!(c.accs, 2);
        // `g(x.f)`, and the predicate instance `P(x)` in both `acc`s, the
        // `unfold` and the `fold`.
        assert_eq!(c.function_apps, 5);
        assert_eq!(m.totals.accs, 3);
    }
}
