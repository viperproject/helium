//! The declarative arithmetic and equality identities — plain `rw!` patterns with
//! no hand-written searcher or applier.

use egg::{
    Applier, EGraph, Id, PatternAst, SearchMatches, Searcher, Subst, Symbol, Var, rewrite as rw,
};

use crate::verify::analysis::ConstFold;
use crate::verify::lang::{Discriminant, Symbolic};
use crate::vmir::{BinOp, Literal};

use super::*;

/// The static (ADT-independent) rule set run during saturation.
pub(super) fn static_rules() -> Vec<Rule> {
    let mut rules = terminating_ite_rules();
    rules.extend(vec![
        // Arithmetic identities, written once per operand sort (`+i` / `+r`) so
        // that a rule producing a literal produces the right one: an integer `0`
        // and a permission `0/1` are different literals, and merging them into
        // one e-class is a type error the analysis panics on.
        //
        // x + 0 => x
        rw!("add-zero-int-r"; "(+i ?x 0)" => "?x"),
        rw!("add-zero-int-l"; "(+i 0 ?x)" => "?x"),
        rw!("add-zero-real-r"; "(+r ?x 0/1)" => "?x"),
        rw!("add-zero-real-l"; "(+r 0/1 ?x)" => "?x"),
        // x - 0 => x
        rw!("sub-zero-int"; "(-i ?x 0)" => "?x"),
        rw!("sub-zero-real"; "(-r ?x 0/1)" => "?x"),
        // x * 1 => x
        rw!("mul-one-real-r"; "(*r ?x 1/1)" => "?x"),
        rw!("mul-one-real-l"; "(*r 1/1 ?x)" => "?x"),
        rw!("mul-one-int-r"; "(*i ?x 1)" => "?x"),
        rw!("mul-one-int-l"; "(*i 1 ?x)" => "?x"),
        // x * 0 => 0
        rw!("mul-zero-real-r"; "(*r ?x 0/1)" => "0/1"),
        rw!("mul-zero-real-l"; "(*r 0/1 ?x)" => "0/1"),
        rw!("mul-zero-int-r"; "(*i ?x 0)" => "0"),
        rw!("mul-zero-int-l"; "(*i 0 ?x)" => "0"),
        // x / 1 => x. No `x / x => 1` or `0 / x => 0`: division by zero is
        // unspecified (see `eval_binary`), so neither holds at `x = 0`.
        rw!("div-one-real"; "(/r ?x 1/1)" => "?x"),
        rw!("div-one-int"; "(/i ?x 1)" => "?x"),
        // (x - p) + p => x, and mirrors. A consume/produce cycle at one location
        // leaves the chunk's permission in this shape. Both operand orders are
        // spelled out because there is no commutativity rule (it blows the graph
        // up).
        rw!("add-sub-cancel-int-r"; "(+i (-i ?x ?p) ?p)" => "?x"),
        rw!("add-sub-cancel-int-l"; "(+i ?p (-i ?x ?p))" => "?x"),
        rw!("add-sub-cancel-real-r"; "(+r (-r ?x ?p) ?p)" => "?x"),
        rw!("add-sub-cancel-real-l"; "(+r ?p (-r ?x ?p))" => "?x"),
        rw!("sub-add-cancel-int-r"; "(-i (+i ?x ?p) ?p)" => "?x"),
        rw!("sub-add-cancel-int-l"; "(-i (+i ?p ?x) ?p)" => "?x"),
        rw!("sub-add-cancel-real-r"; "(-r (+r ?x ?p) ?p)" => "?x"),
        rw!("sub-add-cancel-real-l"; "(-r (+r ?p ?x) ?p)" => "?x"),
        // x - x => 0. The cancel rules above need nested `-`/`+` and so miss a
        // bare self-subtraction, which is what a give-back leaves when the
        // returned share is not the outer addend (`(p - p) + 1/1`).
        rw!("sub-self-int"; "(-i ?x ?x)" => "0"),
        rw!("sub-self-real"; "(-r ?x ?x)" => "0/1"),
        // x < x => false   (irreflexivity)
        rw!("lt-irrefl-real"; "(<r ?x ?x)" => "false"),
        rw!("lt-irrefl-int"; "(<i ?x ?x)" => "false"),
        // x == x => true   (reflexivity; also fires once congruence has merged
        // the two operands into one e-class)
        rw!("eq-refl"; "(== ?x ?x)" => "true"),
        // (a == b) proven true  =>  a ≡ b   (congruence)
        rw!("eq-true-union"; "(== ?a ?b)" => {
            UnionEqArgs { a: var("?a"), b: var("?b") }
        }),
        // `b == false` is `!b`, which lowers to `ite(b, false, true)`
        // (`translate/pure_exp.rs`, `UnOp::Not`). An equivalence, so no `Known`
        // gate. It also normalizes goals into the `Ite` spelling that
        // `Context::prove_by_ite_decomposition` decomposes — Prusti emits MIR asserts as `== false`,
        // which that tier would otherwise not recognize.
        rw!("eq-false-is-not-r"; "(== ?b false)" => "(ite ?b false true)"),
        rw!("eq-false-is-not-l"; "(== false ?b)" => "(ite ?b false true)"),
        // `b == true` is `b` itself — a pure union, minting no node.
        rw!("eq-true-is-self-r"; "(== ?b true)" => "?b"),
        rw!("eq-true-is-self-l"; "(== true ?b)" => "?b"),
        // The boolean decompositions (and-true, or-false, not-true) live in the
        // fused `ite-reduce` pass — they are `Ite`-bucket shapes conditioned on
        // the class's proven boolean, exactly what its applier already inspects.
    ]);
    // Disequality reasoning over disproven `==` classes — standalone rules that
    // share the `Eq` bucket + the `Known(false)` gate.
    rules.push(bucket_rule(
        "eq-false-mirror",
        EqBucketSearcher,
        EqFalseMirrorApplier,
    ));
    rules.push(bucket_rule(
        "contra-congruence",
        EqBucketSearcher,
        ContraCongruenceApplier { memo: Memo::new() },
    ));
    rules.push(bucket_rule(
        "distinguishing-observation",
        EqBucketSearcher,
        DistinguishingObsApplier { memo: Memo::new() },
    ));
    rules.extend(disequality_unit_prop_rules());
    rules.extend(lt_asymmetry_rules());
    rules
}

/// Asymmetry of the strict orders: `a < b` proven true refutes `b < a`.
///
/// This is the one order fact the rule set needs to take a *strict* inequality
/// to the *non-strict* one a side condition asks for. An assumed `p > none`
/// lowers to `0/1 <r p` (`translate::pure_exp`, `B::Gt` swaps the operands), and
/// the non-negativity obligation on a slot op is `not (p <r 0/1)`, i.e. the
/// mirror class pinned false — exactly what this supplies.
///
/// **Lookup-only**: the mirror node is *found*, never minted. Minting it would
/// add a node per `<` in the program whether or not anything asks about the
/// reverse direction; a lookup fires only when the goal (or another assumption)
/// has already put the mirror in the graph, which is the only case where the
/// derived `false` can be read back out.
fn lt_asymmetry_rules() -> Vec<Rule> {
    [BinOp::LtI, BinOp::LtR]
        .into_iter()
        .map(|op| {
            let name = match op {
                BinOp::LtI => "lt-asymmetry-int",
                _ => "lt-asymmetry-real",
            };
            bucket_rule(name, LtBucketSearcher(op), LtAsymmetryApplier(op))
        })
        .collect()
}

/// Searcher over the `<` op bucket for one operand sort. One empty subst per
/// class; the applier re-reads the nodes.
#[derive(Clone)]
struct LtBucketSearcher(BinOp);

impl Searcher<Symbolic, ConstFold> for LtBucketSearcher {
    fn search_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        limit: usize,
    ) -> Vec<SearchMatches<'_, Symbolic>> {
        let Some(classes) = egraph.classes_for_op(&Discriminant::Binary(self.0)) else {
            return vec![];
        };
        classes
            .take(limit)
            // Only a class already *proven* true can refute anything, and that is
            // a rare shape — filtering here keeps the applier off every `<` node.
            .filter(|eclass| known_bool(egraph, *eclass) == Some(true))
            .map(|eclass| SearchMatches {
                eclass,
                substs: vec![Subst::default()],
                ast: None,
            })
            .collect()
    }

    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _limit: usize,
    ) -> Option<SearchMatches<'_, Symbolic>> {
        (known_bool(egraph, eclass) == Some(true)
            && egraph[eclass]
                .nodes
                .iter()
                .any(|n| matches!(n, Symbolic::Binary(op, _) if *op == self.0)))
        .then(|| SearchMatches {
            eclass,
            substs: vec![Subst::default()],
            ast: None,
        })
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

struct LtAsymmetryApplier(BinOp);

impl Applier<Symbolic, ConstFold> for LtAsymmetryApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        _subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        // Collect first: the lookup needs `&egraph`, the union `&mut`.
        let mirrors: Vec<Id> = egraph[eclass]
            .nodes
            .iter()
            .filter_map(|node| match node {
                Symbolic::Binary(op, [a, b]) if *op == self.0 => {
                    egraph.lookup(Symbolic::Binary(*op, [*b, *a]))
                }
                _ => None,
            })
            .collect();
        if mirrors.is_empty() {
            return vec![];
        }
        let f = egraph.add(Symbolic::Lit(Literal::Bool(false)));
        let mut changed = Vec::new();
        for mirror in mirrors {
            if egraph.union(mirror, f) {
                changed.push(egraph.find(mirror));
            }
        }
        changed
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}
