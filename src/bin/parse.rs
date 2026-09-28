use silver_oxide::viper_parser;
use std::{error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let file = std::env::args().nth(1).unwrap();
    let input = fs::read_to_string(file)?;
    let program = viper_parser::vpr_program(&input)?;

    dbg!(program);

    Ok(())
}
