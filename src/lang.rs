use crate::shapes::rect::Rect;
use crate::Transform;
use anyhow::{Error, Result};
use lalrpop_util::lalrpop_mod;

pub mod ast;

lalrpop_mod!(pub parser, "/lang/grammar.rs");

pub fn parse(input: String) -> Result<Vec<Transform>> {
    let expr = parser::StatementsParser::new().parse(&input).unwrap();
    println!("{expr:?}");

    // TODO: convert TransformExprs to Transforms
    let mut transforms = Vec::new();

    Ok(transforms)
}

#[test]
fn basic_command() -> Result<()> {
    let stuff = r#"
    leye: translate(100, -80)
    mouth#1: swap_with(mouth#0)
    mouth#0: scale(2.5), write_to(leye_region), swap_with(reye)
    "#;

    let res = parse(stuff.into())?;
    println!("{res:?}");
    assert_eq!(res.len(), 3);
    // let expr = parser::StatementParser::new()
    //     .parse("mouth#0: scale(2.5), write_to(leye_region), swap_with(reye)")
    //     .unwrap();
    // println!("{expr:?}");

    Ok(())
}
