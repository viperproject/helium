//! The declarative arithmetic and equality identities — plain `rw!` patterns with
//! no hand-written searcher or applier.

use egg::{Applier, EGraph, Id, Pattern, PatternAst, Rewrite, Subst, Symbol, Var};

use crate::verify::analysis::ConstFold;
use crate::verify::lang::Symbolic;
use crate::vmir::{BinOp, Literal};

use super::*;

fn pattern(p: &str) -> Pattern<Symbolic> {
    p.parse().expect("valid pattern")
}

/// A rewrite to a pattern. Its application reads no analysis data, so only its
/// e-nodes can make a match new (see `Pattern::with_data_reads`).
fn rw(name: &str, lhs: &str, rhs: &str) -> Rule {
    let lhs = pattern(lhs).with_data_reads(false, &[]);
    Rewrite::new(name, lhs, pattern(rhs)).expect("valid rewrite")
}

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
        rw("add-zero-int-r", "(+i ?x 0)", "?x"),
        rw("add-zero-int-l", "(+i 0 ?x)", "?x"),
        rw("add-zero-real-r", "(+r ?x 0/1)", "?x"),
        rw("add-zero-real-l", "(+r 0/1 ?x)", "?x"),
        // x - 0 => x
        rw("sub-zero-int", "(-i ?x 0)", "?x"),
        rw("sub-zero-real", "(-r ?x 0/1)", "?x"),
        // x * 1 => x
        rw("mul-one-real-r", "(*r ?x 1/1)", "?x"),
        rw("mul-one-real-l", "(*r 1/1 ?x)", "?x"),
        rw("mul-one-int-r", "(*i ?x 1)", "?x"),
        rw("mul-one-int-l", "(*i 1 ?x)", "?x"),
        // x * 0 => 0
        rw("mul-zero-real-r", "(*r ?x 0/1)", "0/1"),
        rw("mul-zero-real-l", "(*r 0/1 ?x)", "0/1"),
        rw("mul-zero-int-r", "(*i ?x 0)", "0"),
        rw("mul-zero-int-l", "(*i 0 ?x)", "0"),
        // x / 1 => x. No `x / x => 1` or `0 / x => 0`: division by zero is
        // unspecified (see `eval_binary`), so neither holds at `x = 0`.
        rw("div-one-real", "(/r ?x 1/1)", "?x"),
        rw("div-one-int", "(/i ?x 1)", "?x"),
        // (x - p) + p => x, and mirrors. A consume/produce cycle at one location
        // leaves the chunk's permission in this shape. Both operand orders are
        // spelled out because there is no commutativity rule (it blows the graph
        // up).
        rw("add-sub-cancel-int-r", "(+i (-i ?x ?p) ?p)", "?x"),
        rw("add-sub-cancel-int-l", "(+i ?p (-i ?x ?p))", "?x"),
        rw("add-sub-cancel-real-r", "(+r (-r ?x ?p) ?p)", "?x"),
        rw("add-sub-cancel-real-l", "(+r ?p (-r ?x ?p))", "?x"),
        rw("sub-add-cancel-int-r", "(-i (+i ?x ?p) ?p)", "?x"),
        rw("sub-add-cancel-int-l", "(-i (+i ?p ?x) ?p)", "?x"),
        rw("sub-add-cancel-real-r", "(-r (+r ?x ?p) ?p)", "?x"),
        rw("sub-add-cancel-real-l", "(-r (+r ?p ?x) ?p)", "?x"),
        // x - x => 0. The cancel rules above need nested `-`/`+` and so miss a
        // bare self-subtraction, which is what a give-back leaves when the
        // returned share is not the outer addend (`(p - p) + 1/1`).
        rw("sub-self-int", "(-i ?x ?x)", "0"),
        rw("sub-self-real", "(-r ?x ?x)", "0/1"),
        // x < x => false   (irreflexivity)
        rw("lt-irrefl-real", "(<r ?x ?x)", "false"),
        rw("lt-irrefl-int", "(<i ?x ?x)", "false"),
        // x == x => true   (reflexivity; also fires once congruence has merged
        // the two operands into one e-class)
        rw("eq-refl", "(== ?x ?x)", "true"),
        // (a == b) proven true  =>  a ≡ b   (congruence)
        Rewrite::new(
            "eq-true-union",
            pattern("(== ?a ?b)").with_data_reads(true, &[]),
            UnionEqArgs {
                a: var("?a"),
                b: var("?b"),
            },
        )
        .expect("eq-true-union rule"),
        // `b == false` is `!b`, which lowers to `ite(b, false, true)`
        // (`translate/pure_exp.rs`, `UnOp::Not`). An equivalence, so no `Known`
        // gate. It also normalizes goals into the `Ite` spelling that
        // `Context::prove_by_ite_decomposition` decomposes — Prusti emits MIR asserts as `== false`,
        // which that tier would otherwise not recognize.
        rw("eq-false-is-not-r", "(== ?b false)", "(ite ?b false true)"),
        rw("eq-false-is-not-l", "(== false ?b)", "(ite ?b false true)"),
        // `b == true` is `b` itself — a pure union, minting no node.
        rw("eq-true-is-self-r", "(== ?b true)", "?b"),
        rw("eq-true-is-self-l", "(== true ?b)", "?b"),
        // The boolean decompositions (and-true, or-false, not-true) live in the
        // fused `ite-reduce` pass — they are `Ite`-bucket shapes conditioned on
        // the class's proven boolean, exactly what its applier already inspects.
    ]);
    // Disequality reasoning over disproven `==` nodes — standalone rules that
    // share the `Eq` matches + the `Known(false)` gate.
    rules.push(local_fixpoint_rule(
        "eq-false-mirror",
        EqSearcher::new(EqReads::Node, EqGate::Disproven),
        EqFalseMirrorApplier,
    ));
    rules.push(local_fixpoint_rule(
        "contra-congruence",
        EqSearcher::new(EqReads::OperandApps, EqGate::Disproven),
        ContraCongruenceApplier { memo: Memo::new() },
    ));
    rules.push(local_fixpoint_rule(
        "distinguishing-observation",
        EqSearcher::new(EqReads::Observations, EqGate::Undecided),
        DistinguishingObsApplier,
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
///
/// A match is one `<` node, and the applier looks both ways: a proven `a < b`
/// refutes its mirror, and a mirror proven true refutes `a < b`. So whichever of
/// the two appears or is proven last, its own match makes the derivation, and the
/// rule reads nothing outside its match but the mirror's data.
fn lt_asymmetry_rules() -> Vec<Rule> {
    [
        ("lt-asymmetry-int", "(<i ?a ?b)", BinOp::LtI),
        ("lt-asymmetry-real", "(<r ?a ?b)", BinOp::LtR),
    ]
    .into_iter()
    .map(|(name, pattern, op)| {
        // the applier reads its own class's data (and the mirror's, which is the
        // root of its own match)
        let pattern = self::pattern(pattern).with_data_reads(true, &[]);
        local_fixpoint_rule(name, pattern, LtAsymmetryApplier(op))
    })
    .collect()
}

struct LtAsymmetryApplier(BinOp);

impl Applier<Symbolic, ConstFold> for LtAsymmetryApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Symbolic, ConstFold>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<Symbolic>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        let (a, b) = (egraph.find(subst[var("?a")]), egraph.find(subst[var("?b")]));
        let Some(mirror) = egraph.lookup(Symbolic::Binary(self.0, [b, a])) else {
            return vec![];
        };
        // The refuted side: the mirror of whichever of the two is proven true.
        let refuted = if known_bool(egraph, eclass) == Some(true) {
            mirror
        } else if known_bool(egraph, mirror) == Some(true) {
            eclass
        } else {
            return vec![];
        };
        let f = egraph.add(Symbolic::Lit(Literal::Bool(false)));
        if egraph.union(refuted, f) {
            vec![egraph.find(refuted)]
        } else {
            vec![]
        }
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}
