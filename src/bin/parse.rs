use silver_oxide::viper::prelude;
use silver_oxide::viper_parser;
use std::{error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let file = std::env::args().nth(1).unwrap();
    let input = fs::read_to_string(file)?;
    let mut program = viper_parser::vpr_program(&input)?;
    prelude::add_collections_to_if_appearing(&mut program);

    dbg!(program);

    Ok(())
}
