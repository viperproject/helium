//! Parse a Silver file, typecheck it, lower to VMIR, and print the resulting
//! `vmir::Program`.
//!
//! Usage: `cargo run --bin translate -- cases/foo.vpr [--derived]`
//!
//! With `--derived`, also dump each resource's derived members (its address
//! location and snapshot) — which are *not* stored in plain VMIR but computed on
//! demand via `Resource::derive_location` / `derive_snapshot`.

use silver_oxide::translate;
use silver_oxide::viper::{
    GlobalsCollector, IdentCollector, disambiguate, inline_macros, prelude, typecheck_program,
    viper_parser, walk::AstWalkable,
};
use std::{error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let derived = args.iter().any(|a| a == "--derived");
    let file = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .ok_or("usage: translate <file.vpr> [--derived]")?;
    let input = fs::read_to_string(file)?;

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

    let typed = typecheck_program(&mut program, interner, &globals)
        .map_err(|e| format!("typecheck failed: {e:?}"))?;

    let vmir = translate::translate(&typed).map_err(|e| format!("translation failed: {e:?}"))?;

    println!("{}", vmir);
    if derived {
        println!("\n; --- derived members ---");
        print!("{}", vmir.derived_dump());
    }
    Ok(())
}
