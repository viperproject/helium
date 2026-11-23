//! A small DSL for building e-graph terms.
//!
//! `Symbolic` has no `not`, `and`, `or` or `implies` node — [`BinOp`] carries only
//! arithmetic, `<` and `==`, so every boolean connective is spelled as an
//! [`Symbolic::Ite`]. That is fine for the engine and terrible for the reader:
//!
//! ```ignore
//! let le     = ctx.add(Symbolic::Ite([gt, false_, true_]));   // perm <= b
//! let nonpos = ctx.add(Symbolic::Ite([pos, false_, true_]));  // !(0 < p)
//! imp        = ctx.add(Symbolic::Ite([g, imp, true_]));       // g ==> imp
//! ```
//!
//! Three call sites, identical syntax, three different intents — and each needs
//! `true_`/`false_` hoisted into scope first. [`expr!`] states the intent instead:
//!
//! ```ignore
//! let le     = expr!(ctx, not ({b} <r {leaf}));
//! let nonpos = expr!(ctx, not {pos});
//! imp        = expr!(ctx, {g} ==> {imp});
//! ```
//!
//! # Grammar
//!
//! Operands are `{rust_expr}` (an already-built [`egg::Id`]), a nested
//! parenthesised term, or a literal keyword. Braces are what make the macro
//! unambiguous: it never has to guess whether a token is a term or a value.
//!
//! ```text
//! true | false                 boolean literals
//! 1                            integer literal
//! 1/1                          rational literal (`Real`); `0/1` is zero
//! {e}                          a Rust expression evaluating to an `egg::Id`
//! (..)                         a nested term
//!
//! not A                        ite(A, false, true)
//! A and B                      ite(A, B, false)
//! A or B                       ite(A, true, B)
//! A ==> B                      ite(A, B, true)
//! if A then B else C           ite(A, B, C)
//!
//! A == B                       Binary(Eq)
//! A <i B   |  A <r B           Binary(LtI / LtR)
//! A +i B   |  A +r B           Binary(AddI / AddR)
//! A -i B   |  A -r B           Binary(SubI / SubR)
//! ```
//!
//! # What it deliberately does not cover
//!
//! Chains whose shape is decided at **runtime** stay functions:
//! [`VerifyContext::implication`] and `rewrite::function::fold_guards` fold a
//! `Polarity`-tagged list, so their `ite` arms swap based on a value, not on
//! syntax. A macro there would only obscure the branch.

use crate::verify::analysis::ConstFold;
use crate::verify::context::VerifyContext;
use crate::verify::lang::Symbolic;

/// Anything that can intern a node. Lets [`expr!`] serve both the verifier
/// (where `add` also mirrors into the working graph — see
/// [`VerifyContext::add`]) and the rewrite appliers, which hold a bare `EGraph`
/// and would otherwise have to keep spelling nodes by hand.
pub(crate) trait NodeSink {
    fn node(&mut self, n: Symbolic) -> egg::Id;
}

impl NodeSink for VerifyContext<'_> {
    /// Routes through [`VerifyContext::add`], **not** `egraph.add` — the
    /// working-graph mirror and the fixpoint-cache invalidation live there.
    fn node(&mut self, n: Symbolic) -> egg::Id {
        self.add(n)
    }
}

impl NodeSink for egg::EGraph<Symbolic, ConstFold> {
    fn node(&mut self, n: Symbolic) -> egg::Id {
        self.add(n)
    }
}

/// Build an e-graph term. See the module docs for the grammar.
macro_rules! expr {
    // ---- leaves ----------------------------------------------------------
    ($s:expr, true) => {
        $crate::verify::expr::NodeSink::node(
            $s,
            $crate::verify::lang::Symbolic::Lit($crate::vmir::Literal::Bool(true)),
        )
    };
    ($s:expr, false) => {
        $crate::verify::expr::NodeSink::node(
            $s,
            $crate::verify::lang::Symbolic::Lit($crate::vmir::Literal::Bool(false)),
        )
    };
    // Rationals are written `n/d` -- must precede the bare-integer rule, and
    // both must follow `true`/`false` (which are also `literal`s).
    ($s:expr, $n:literal / $d:literal) => {
        $crate::verify::expr::NodeSink::node(
            $s,
            $crate::verify::lang::Symbolic::Lit($crate::vmir::Literal::Real(
                ::num::BigRational::new(::num::BigInt::from($n), ::num::BigInt::from($d)),
            )),
        )
    };
    ($s:expr, $n:literal) => {
        $crate::verify::expr::NodeSink::node(
            $s,
            $crate::verify::lang::Symbolic::Lit($crate::vmir::Literal::Int(
                ::num::BigInt::from($n),
            )),
        )
    };
    ($s:expr, {$e:expr}) => { $e };
    ($s:expr, ($($inner:tt)+)) => { expr!($s, $($inner)+) };

    // ---- boolean connectives (all desugar to Ite) ------------------------
    ($s:expr, not $a:tt) => {{
        let a = expr!($s, $a);
        let f = expr!($s, false);
        let t = expr!($s, true);
        $crate::verify::expr::NodeSink::node($s, $crate::verify::lang::Symbolic::Ite([a, f, t]))
    }};
    ($s:expr, $a:tt and $b:tt) => {{
        let a = expr!($s, $a);
        let b = expr!($s, $b);
        let f = expr!($s, false);
        $crate::verify::expr::NodeSink::node($s, $crate::verify::lang::Symbolic::Ite([a, b, f]))
    }};
    ($s:expr, $a:tt or $b:tt) => {{
        let a = expr!($s, $a);
        let b = expr!($s, $b);
        let t = expr!($s, true);
        $crate::verify::expr::NodeSink::node($s, $crate::verify::lang::Symbolic::Ite([a, t, b]))
    }};
    ($s:expr, $a:tt ==> $b:tt) => {{
        let a = expr!($s, $a);
        let b = expr!($s, $b);
        let t = expr!($s, true);
        $crate::verify::expr::NodeSink::node($s, $crate::verify::lang::Symbolic::Ite([a, b, t]))
    }};
    ($s:expr, if $c:tt then $t:tt else $e:tt) => {{
        let c = expr!($s, $c);
        let t = expr!($s, $t);
        let e = expr!($s, $e);
        $crate::verify::expr::NodeSink::node($s, $crate::verify::lang::Symbolic::Ite([c, t, e]))
    }};

    // ---- binary operators ------------------------------------------------
    ($s:expr, $a:tt == $b:tt) => { expr!(@bin $s, Eq, $a, $b) };
    ($s:expr, $a:tt <i $b:tt) => { expr!(@bin $s, LtI, $a, $b) };
    ($s:expr, $a:tt <r $b:tt) => { expr!(@bin $s, LtR, $a, $b) };
    ($s:expr, $a:tt +i $b:tt) => { expr!(@bin $s, AddI, $a, $b) };
    ($s:expr, $a:tt +r $b:tt) => { expr!(@bin $s, AddR, $a, $b) };
    ($s:expr, $a:tt -i $b:tt) => { expr!(@bin $s, SubI, $a, $b) };
    ($s:expr, $a:tt -r $b:tt) => { expr!(@bin $s, SubR, $a, $b) };

    (@bin $s:expr, $op:ident, $a:tt, $b:tt) => {{
        let a = expr!($s, $a);
        let b = expr!($s, $b);
        $crate::verify::expr::NodeSink::node(
            $s,
            $crate::verify::lang::Symbolic::Binary($crate::vmir::BinOp::$op, [a, b]),
        )
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::test_support::fresh_ctx;
    use crate::vmir::{BinOp, Literal};

    /// Every form builds the node it claims to. Checked structurally against the
    /// e-graph rather than by round-tripping, so a wrong `Ite` arm order fails.
    #[test]
    fn forms_build_the_documented_nodes() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);

        let t = expr!(&mut ctx, true);
        let f = expr!(&mut ctx, false);
        assert_eq!(t, ctx.add(Symbolic::Lit(Literal::Bool(true))));
        assert_eq!(f, ctx.add(Symbolic::Lit(Literal::Bool(false))));

        let p = ctx.fresh_symbolic_value(crate::vmir::Type::Bool);
        let q = ctx.fresh_symbolic_value(crate::vmir::Type::Bool);

        // not p  ==  ite(p, false, true)
        assert_eq!(
            expr!(&mut ctx, not { p }),
            ctx.add(Symbolic::Ite([p, f, t]))
        );
        // p and q  ==  ite(p, q, false)
        assert_eq!(
            expr!(&mut ctx, {p} and {q}),
            ctx.add(Symbolic::Ite([p, q, f]))
        );
        // p or q  ==  ite(p, true, q)
        assert_eq!(
            expr!(&mut ctx, {p} or {q}),
            ctx.add(Symbolic::Ite([p, t, q]))
        );
        // p ==> q  ==  ite(p, q, true)
        assert_eq!(
            expr!(&mut ctx, {p} ==> {q}),
            ctx.add(Symbolic::Ite([p, q, t]))
        );
        // if p then q else p
        assert_eq!(
            expr!(&mut ctx, if {p} then {q} else {p}),
            ctx.add(Symbolic::Ite([p, q, p]))
        );
        assert_eq!(
            expr!(&mut ctx, { p } == { q }),
            ctx.add(Symbolic::Binary(BinOp::Eq, [p, q]))
        );
    }

    /// Nesting composes, and a parenthesised term is an operand.
    #[test]
    fn nested_terms_compose() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);
        let z = expr!(&mut ctx, 0 / 1);
        let p = ctx.fresh_symbolic_value(crate::vmir::Type::Real);

        // not (0 < p) — the `perm <= b` / `nonpos` shape, in one line.
        let built = expr!(&mut ctx, not({ z } < r { p }));
        let lt = ctx.add(Symbolic::Binary(BinOp::LtR, [z, p]));
        let (f, t) = (
            ctx.add(Symbolic::Lit(Literal::Bool(false))),
            ctx.add(Symbolic::Lit(Literal::Bool(true))),
        );
        assert_eq!(built, ctx.add(Symbolic::Ite([lt, f, t])));
    }

    /// Numeric literals land in the same class as the hand-built node, and
    /// `real 0` is distinct from `int 0` (different `Literal` variants, so
    /// different e-classes -- the sorts must not collapse).
    #[test]
    fn numeric_literals_match_hand_built_nodes() {
        let interner = lasso::Rodeo::new();
        let mut ctx = fresh_ctx(&interner);
        let r0 = expr!(&mut ctx, 0 / 1);
        let i0 = expr!(&mut ctx, 0);
        assert_eq!(
            r0,
            ctx.add(Symbolic::Lit(Literal::Real(num::BigInt::from(0).into())))
        );
        assert_eq!(
            i0,
            ctx.add(Symbolic::Lit(Literal::Int(num::BigInt::from(0))))
        );
        assert_ne!(ctx.egraph.find(r0), ctx.egraph.find(i0));
    }
}
