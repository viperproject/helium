//! Per-declaration translators — exactly one per Viper `typed::Declaration`
//! (field, predicate, function, method, adt, domain). Each runs the
//! `declare → meta → define` typestate phases described in the parent module,
//! self-publishing its metadata into `TranslationContext` and consuming its
//! `DeclSlot`s (from the sibling `builder` module) to emit `vmir::Declaration`s.
//! The program's magic-wand structures, which are not declarations, have one
//! translator between them (`wand`).

mod adt;
mod domain;
mod field;
mod function;
mod method;
mod predicate;
mod wand;

pub(crate) use adt::AdtTranslator;
pub(crate) use domain::DomainTranslator;
pub(crate) use field::FieldTranslator;
pub(crate) use function::FunctionTranslator;
pub(crate) use method::MethodTranslator;
pub(crate) use predicate::PredicateTranslator;
pub(crate) use wand::WandTranslator;
