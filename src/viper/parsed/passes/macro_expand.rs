use crate::dhash::HashMap;
use std::fmt;

use lasso::Spur;

use crate::viper::{
    AssignLhs, AssignRhs, Define, ExpCallKind, ExpKind, ExpOrBlock, Program, Statement,
    StmtCallKind,
    interner::Interner,
    walk::{AstWalkable, AstWalkerMut},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MacroResolutionError {
    CyclicMacroExpansion(String),
    ArityMismatch {
        name: String,
        expected: usize,
        found: usize,
    },
    AssigningBlockMacro(String),
}

impl fmt::Display for MacroResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MacroResolutionError::CyclicMacroExpansion(name) => {
                write!(f, "cyclic macro expansion detected for `{name}`")
            }
            MacroResolutionError::ArityMismatch {
                name,
                expected,
                found,
            } => {
                write!(
                    f,
                    "macro `{name}` expected {expected} argument(s), found {found}"
                )
            }
            MacroResolutionError::AssigningBlockMacro(name) => {
                write!(f, "cannot assign statement macro `{name}` to a target")
            }
        }
    }
}

impl std::error::Error for MacroResolutionError {}

struct ParameterSubstitutor<'a> {
    bindings: &'a HashMap<Spur, ExpKind>,
}

impl<'a> AstWalkerMut<'_> for ParameterSubstitutor<'a> {
    fn walk_mut_exp_kind(&mut self, exp: &mut ExpKind) {
        if let ExpKind::Ident(ident) = exp
            && let Some(arg_exp) = self.bindings.get(&ident.id())
        {
            *exp = arg_exp.clone();
            return;
        }
        exp.walk_mut_children(self);
    }
}

pub fn inline_macros(
    program: &mut Program,
    interner: &Interner,
) -> Result<(), Vec<MacroResolutionError>> {
    let mut macro_dict = HashMap::default();

    // Extract and remove all Define declarations from the AST
    program.0.retain(|decl| {
        if let crate::viper::Declaration::Define(define) = decl {
            macro_dict.insert(define.name.0.id(), define.clone());
            false // Remove from AST
        } else {
            true // Keep
        }
    });

    let mut inliner = MacroInliner {
        interner,
        macro_dict,
        expansion_stack: Vec::new(),
        errors: Vec::new(),
    };

    program.walk_mut(&mut inliner);

    if inliner.errors.is_empty() {
        Ok(())
    } else {
        Err(inliner.errors)
    }
}

struct MacroInliner<'i> {
    interner: &'i Interner,
    macro_dict: HashMap<Spur, Define>,
    expansion_stack: Vec<Spur>,
    errors: Vec<MacroResolutionError>,
}

impl<'i> MacroInliner<'i> {
    fn macro_name(&self, macro_id: Spur) -> String {
        self.interner.resolve(&macro_id).to_string()
    }

    fn try_enter_expansion(&mut self, macro_id: Spur) -> Result<(), MacroResolutionError> {
        if self.expansion_stack.contains(&macro_id) {
            Err(MacroResolutionError::CyclicMacroExpansion(
                self.macro_name(macro_id),
            ))
        } else {
            self.expansion_stack.push(macro_id);
            Ok(())
        }
    }

    fn exit_expansion(&mut self) {
        self.expansion_stack.pop();
    }

    fn check_arity(
        &self,
        macro_id: Spur,
        expected: usize,
        found: usize,
    ) -> Result<(), MacroResolutionError> {
        if expected != found {
            Err(MacroResolutionError::ArityMismatch {
                name: self.macro_name(macro_id),
                expected,
                found,
            })
        } else {
            Ok(())
        }
    }

    /// Helper to expand a statement macro into a list of statements
    fn expand_stmt_macro(
        &mut self,
        lhs: &[AssignLhs],
        call: &mut crate::viper::Call<StmtCallKind>,
    ) -> Option<Vec<Statement>> {
        let macro_id = call.name.id();

        if !lhs.is_empty() {
            self.errors.push(MacroResolutionError::AssigningBlockMacro(
                self.macro_name(macro_id),
            ));
            return None;
        }

        let macro_def = self
            .macro_dict
            .get(&macro_id)
            .expect("Macro definition to be present");

        if let Err(e) = self.check_arity(macro_id, macro_def.args.len(), call.args.len()) {
            self.errors.push(e);
            return None;
        }

        let mut expanded_block = match &macro_def.body {
            ExpOrBlock::Block(block) => block.clone(),
            ExpOrBlock::Exp(_) => {
                unreachable!("CallResolver should prevent exp macros in statement calls")
            }
        };

        let mut bindings = HashMap::default();
        for (param, arg) in macro_def.args.iter().zip(std::mem::take(&mut call.args)) {
            bindings.insert(param.0.id(), *arg.kind);
        }

        let mut substitutor = ParameterSubstitutor {
            bindings: &bindings,
        };

        if let Err(e) = self.try_enter_expansion(macro_id) {
            self.errors.push(e);
            return None;
        }

        // 1. Substitute the parameters
        expanded_block.walk_mut(&mut substitutor);

        // 2. Walk the expanded block with the inliner itself!
        // This recursively flattens and inlines any nested macro calls perfectly.
        expanded_block.walk_mut(self);

        self.exit_expansion();

        Some(expanded_block.0)
    }
}

impl<'i> AstWalkerMut<'_> for MacroInliner<'i> {
    // NEW: Intercept blocks to splice expanded macros directly into the statement list
    fn walk_mut_block(&mut self, block: &mut crate::viper::StmtBlock) {
        let mut new_stmts = Vec::new();

        for mut stmt in std::mem::take(&mut block.0) {
            // Check if the statement is a macro call
            if let Statement::Assign(lhs, AssignRhs::Call(call)) = &mut stmt
                && call.kind == Some(StmtCallKind::Macro)
            {
                // Walk the macro's arguments BEFORE expanding (matches the original order)
                for arg in &mut call.args {
                    arg.walk_mut(self);
                }

                if let Some(expanded_stmts) = self.expand_stmt_macro(lhs, call) {
                    // INLINE the statements directly into the current block!
                    new_stmts.extend(expanded_stmts);
                    continue;
                }
            }

            // If it's not a macro (or expansion failed), walk normally and keep it
            stmt.walk_mut(self);
            new_stmts.push(stmt);
        }

        block.0 = new_stmts;
    }

    fn walk_mut_statement(&mut self, stmt: &mut Statement) {
        stmt.walk_mut_children(self);

        // Fallback: If a statement macro is called in a weird AST position where it
        // wasn't inside a block (e.g. if you allow single-statement `if` branches without `{}`),
        // we wrap it in a block here. Usually, `walk_mut_block` will intercept it first!
        if let Statement::Assign(lhs, AssignRhs::Call(call)) = stmt
            && call.kind == Some(StmtCallKind::Macro)
        {
            if let Some(expanded_stmts) = self.expand_stmt_macro(lhs, call) {
                *stmt = Statement::Block(crate::viper::Block(expanded_stmts));
            }
        }
    }

    fn walk_mut_exp_kind(&mut self, exp: &mut ExpKind) {
        exp.walk_mut_children(self);

        if let ExpKind::Call(call) = exp
            && let Some(ExpCallKind::Macro) = call.kind
        {
            let macro_id = call.name.id();
            let macro_def = self
                .macro_dict
                .get(&macro_id)
                .expect("Macro definition to be present");

            if let Err(e) = self.check_arity(macro_id, macro_def.args.len(), call.args.len()) {
                self.errors.push(e);
                return;
            }

            let mut expanded_exp = match &macro_def.body {
                ExpOrBlock::Exp(exp) => exp.clone(),
                ExpOrBlock::Block(_) => {
                    unreachable!("CallResolver should prevent block macros in expression calls")
                }
            };

            let mut bindings = HashMap::default();
            for (param, arg) in macro_def.args.iter().zip(std::mem::take(&mut call.args)) {
                bindings.insert(param.0.id(), *arg.kind);
            }

            let mut substitutor = ParameterSubstitutor {
                bindings: &bindings,
            };
            expanded_exp.walk_mut(&mut substitutor);

            if let Err(e) = self.try_enter_expansion(macro_id) {
                self.errors.push(e);
                return;
            }
            expanded_exp.walk_mut(self);
            self.exit_expansion();

            *exp = *expanded_exp.kind;
        }
    }
}
