//! Viper source the verifier adds to a program itself, rather than the user.
//!
//! `sequences.vpr` holds the axioms of the built-in `Seq[T]` type as a generic
//! domain `$Seq[E]`. It is added only to a program that can contain a sequence.
//! Like any generic domain, its axioms are instantiated through their type
//! triggers, once per `Seq[T]` in a verification unit.

use std::sync::LazyLock;

use crate::viper::parsed::ast::{BinOp, ExpKind, Program, Type};
use crate::viper::walk::{AstWalkable, AstWalker};
use crate::viper_parser;

/// The name of the domain in `sequences.vpr`.
pub const SEQ_DOMAIN: &str = "$Seq";

/// Parsed once per process. Each lowering pass gets its own copy, because
/// interning rewrites the identifiers in place.
static SEQUENCES: LazyLock<Program> = LazyLock::new(|| {
    viper_parser::vpr_program(include_str!("sequences.vpr"))
        .unwrap_or_else(|e| panic!("the sequence prelude does not parse: {e}"))
});

/// Add the sequence prelude to `program` if it can contain a sequence.
///
/// Call this before identifiers are interned.
pub fn add_collections_to_if_appearing(program: &mut Program) {
    if mentions_sequences(program) {
        program.0.extend(SEQUENCES.0.iter().cloned());
    }
}

/// Whether `program` writes a `Seq[..]` type, a `Seq(..)` literal or `[a..b)`.
/// Every sequence value starts from one of these, so this cannot miss one.
fn mentions_sequences(program: &Program) -> bool {
    let mut finder = SeqFinder { found: false };
    program.walk(&mut finder);
    finder.found
}

struct SeqFinder {
    found: bool,
}

impl<'a> AstWalker<'a> for SeqFinder {
    fn walk_type(&mut self, ty: &'a Type) {
        self.found |= matches!(ty, Type::Seq(_));
        ty.walk_children(self);
    }

    fn walk_exp_kind(&mut self, exp: &'a ExpKind) {
        self.found |= matches!(exp, ExpKind::SeqLit(..));
        exp.walk_children(self);
    }

    fn walk_bin_op(&mut self, op: &'a BinOp) {
        self.found |= matches!(op, BinOp::Range);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viper::parsed::ast::{Ident, IdnDeclTyped};
    use crate::viper::{
        GlobalsCollector, IdentCollector, disambiguate, inline_macros, typecheck_program,
    };

    fn parse(input: &str) -> Program {
        viper_parser::vpr_program(input).expect("parse failed")
    }

    #[test]
    fn the_prelude_parses() {
        assert!(!SEQUENCES.0.is_empty());
    }

    #[test]
    fn only_programs_that_can_contain_a_sequence_get_it() {
        for used in [
            "method m(s: Seq[Int]) {}",
            "method m(x: D[Seq[Int]]) {}",
            "method m() { assert |Seq(1)| == 1 }",
            "method m() { assert 1 in [0..2) }",
        ] {
            assert!(mentions_sequences(&parse(used)), "{used}");
        }
        assert!(!mentions_sequences(&parse("method m(x: Int) {}")));
    }

    #[test]
    fn every_binder_and_parameter_starts_with_a_dollar() {
        struct Names(Vec<String>);
        impl<'a> AstWalker<'a> for Names {
            fn walk_idn_decl_typed(&mut self, decl: &'a IdnDeclTyped) {
                if let Ident::Raw(name) = &decl.idn.0
                    && !name.starts_with('$')
                {
                    self.0.push(name.clone());
                }
                decl.walk_children(self);
            }
        }
        let mut names = Names(Vec::new());
        SEQUENCES.walk(&mut names);
        assert!(names.0.is_empty(), "{:?}", names.0);
    }

    #[test]
    fn a_global_may_share_a_name_with_what_the_prelude_binds() {
        let mut program = parse(
            "method m(q: Seq[Int]) {} method s() {} method x() {} method max() {} method E() {}",
        );
        add_collections_to_if_appearing(&mut program);
        let mut ident_collector = IdentCollector::default();
        program.walk_mut(&mut ident_collector);
        let interner = ident_collector.finalize();
        let mut globals_collector = GlobalsCollector::new(&interner);
        program.walk(&mut globals_collector);
        let globals = globals_collector.finalize().expect("globals error");
        disambiguate(&mut program, &interner, &globals).expect("disambiguation failed");
        inline_macros(&mut program, &interner).expect("macro inlining failed");
        if let Err(errors) = typecheck_program(&mut program, interner, &globals) {
            panic!("{errors:?}");
        }
    }
}
