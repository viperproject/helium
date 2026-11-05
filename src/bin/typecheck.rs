//! Reads a Silver program from a file, runs the full pre-processing pipeline
//! (intern → globals → resolve calls → inline macros → typecheck) and
//! debug-dumps the resulting typed `typed::Program`, so the inferred
//! types can be inspected.
//!
//! Usage: `cargo run --bin typecheck -- cases/foo.vpr`

use silver_oxide::viper::{
    GlobalsCollector, IdentCollector, disambiguate, inline_macros, prelude, show,
    typecheck_program, viper_parser, walk::AstWalkable,
};
use std::{error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let file = std::env::args()
        .nth(1)
        .ok_or("usage: typecheck <file.vpr>")?;
    let input = fs::read_to_string(&file)?;

    let mut program = viper_parser::vpr_program(&input)?;
    prelude::add_collections_to_if_appearing(&mut program);

    let mut ident_collector = IdentCollector::default();
    program.walk_mut(&mut ident_collector);
    let interner = ident_collector.finalize();

    let mut globals_collector = GlobalsCollector::new(&interner);
    program.walk(&mut globals_collector);
    let globals = globals_collector.finalize().expect("globals error");

    disambiguate(&mut program, &interner, &globals).expect("disambiguation failed");
    inline_macros(&mut program, &interner).expect("macro inlining failed");

    match typecheck_program(&mut program, interner, &globals) {
        Ok(typed) => {
            eprintln!("typecheck OK");
            println!("{}", show(&typed, &typed.interner));
        }
        Err(errors) => {
            eprintln!("typecheck FAILED with {} error(s):", errors.len());
            for e in &errors {
                eprintln!("  - {e}");
            }
            std::process::exit(1);
        }
    }

    Ok(())
}
