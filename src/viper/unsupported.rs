//! Naming the Viper constructs this verifier does not implement.
//!
//! Everything that reaches here is *parseable*: the grammar accepts the whole
//! surface syntax it can recognise, and turns the parts we do not implement
//! into `ExpKind::Unsupported` / `Statement::Unsupported` markers or into
//! ordinary nodes we then refuse here (a magic wand, `epsilon`).
//! One scan per declaration is what lets the pipeline report "this declaration
//! uses `package`" and carry on with the rest of the file, instead of dying at
//! the first construct with a parse error that names something else.
//!
//! The rule this module encodes: **if we cannot give a construct its Viper
//! meaning, the declaration that mentions it does not verify.** Constructs we
//! merely lower differently (`decreases`, dropped, with termination checked
//! nowhere) belong in the documentation, not here — a scan hit means the
//! declaration is rejected.

use crate::viper::parsed::ast::*;
use crate::viper::walk::{AstWalkable, AstWalker};

/// The first unsupported construct `decl` mentions, named as it is written in
/// the source, or `None` if the declaration is entirely within the fragment we
/// implement.
pub fn scan_declaration(decl: &Declaration) -> Option<&'static str> {
    let mut scan = Scan { found: None };
    decl.walk(&mut scan);
    scan.found
}

struct Scan {
    /// First hit wins: the walk keeps running (there is no early exit in the
    /// walker) but never overwrites what it already found, so the reported
    /// construct is the first one in source order.
    found: Option<&'static str>,
}

impl Scan {
    fn hit(&mut self, what: &'static str) {
        self.found.get_or_insert(what);
    }
}

/// Whether an expression mentions `acc(..)` anywhere — what makes a `forall`
/// body a quantified permission rather than a pure fact.
fn mentions_acc(exp: &Exp) -> bool {
    struct FindAcc(bool);
    impl<'a> AstWalker<'a> for FindAcc {
        fn walk_exp_kind(&mut self, exp: &'a ExpKind) {
            if matches!(exp, ExpKind::Acc(_)) {
                self.0 = true;
            }
            exp.walk_children(self);
        }
    }
    let mut find = FindAcc(false);
    exp.walk(&mut find);
    find.0
}

/// The collection types other than `Seq` and `Set`, which are parsed as plain
/// named types so that a declaration mentioning one can be reported rather than
/// failing the parse. (`Seq` and `Set` have their own `Type::Seq`/`Type::Set`.)
const COLLECTION_TYPES: [&str; 2] = ["Multiset", "Map"];

impl<'a> AstWalker<'a> for Scan {
    fn walk_exp_kind(&mut self, exp: &'a ExpKind) {
        match exp {
            ExpKind::Unsupported(what) => self.hit(what),
            ExpKind::MagicWand(..) => self.hit("magic wand `--*`"),
            ExpKind::ForPerm(..) => self.hit("forperm"),
            ExpKind::Quantifier(QuantifierKind::Exists, ..) => self.hit("exists"),
            // `forall x: Ref :: acc(x.f)` — a quantified permission. The
            // quantifier's body has to stay heap-free.
            ExpKind::Quantifier(QuantifierKind::Forall, _, _, body) if mentions_acc(body) => {
                self.hit("quantified permission")
            }
            ExpKind::HeapUpdate(HeapUpdateOp::Fold, ..) => self.hit("folding"),
            ExpKind::HeapUpdate(HeapUpdateOp::Apply, ..) => self.hit("applying"),
            ExpKind::HeapUpdate(HeapUpdateOp::Package, ..) => self.hit("packaging"),
            _ => {}
        }
        exp.walk_children(self);
    }

    fn walk_statement(&mut self, stmt: &'a Statement) {
        match stmt {
            Statement::Unsupported(what) => self.hit(what),
            // Viper has no multi-target assignment from an expression; only a
            // method call may bind several targets.
            Statement::Assign(targets, rhs)
                if targets.len() > 1 && !matches!(rhs, AssignRhs::Call(_)) =>
            {
                self.hit("multi-target assignment from an expression")
            }
            _ => {}
        }
        stmt.walk_children(self);
    }

    fn walk_assign_rhs(&mut self, rhs: &'a AssignRhs) {
        if let AssignRhs::New(StarOrNames::Star) = rhs {
            self.hit("new(*)");
        }
        rhs.walk_children(self);
    }

    fn walk_const(&mut self, c: &'a ConstKind) {
        if matches!(c, ConstKind::Epsilon) {
            // Lowering `epsilon` to any concrete fraction changes what the
            // program means; Viper's `epsilon` is an unspecified positive
            // amount.
            self.hit("epsilon");
        }
        c.walk_children(self);
    }

    fn walk_bin_op(&mut self, op: &'a BinOp) {
        // `[A, B]`: the body is given `A` and the caller is charged `B`.
        // Nothing here can hold the two apart.
        if matches!(op, BinOp::InhaleExhale) {
            self.hit("inhale-exhale expression `[A, B]`");
        }
        op.walk_children(self);
    }

    fn walk_type(&mut self, ty: &'a Type) {
        if let Type::Domain(name, _) = ty
            && let Ident::Raw(name) = name
            && let Some(collection) = COLLECTION_TYPES.iter().find(|c| *c == name)
        {
            self.hit(collection);
        }
        ty.walk_children(self);
    }

    fn walk_import(&mut self, import: &'a Import) {
        // The path is never read, so every name the imported file would have
        // provided is simply missing.
        self.hit("import");
        import.walk_children(self);
    }
}
