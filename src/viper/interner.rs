use crate::viper::{Ident, walk::AstWalkerMut};

pub type Interner = lasso::RodeoReader;

#[derive(Debug, Clone, Default)]
pub struct IdentCollector {
    interner: lasso::Rodeo,
}

impl IdentCollector {
    /// `result` is interned up front: a function postcondition binds it as an
    /// ordinary variable (see `typecheck::lower_pure_kind`).
    pub fn finalize(mut self) -> Interner {
        self.interner.get_or_intern_static("result");
        self.interner.into_reader()
    }
}

impl AstWalkerMut<'_> for IdentCollector {
    fn walk_mut_ident(&mut self, ident: &mut Ident) {
        if let Ident::Raw(raw) = ident {
            let interned = self.interner.get_or_intern(&raw);
            *ident = Ident::Interned(interned);
        }
    }
}
