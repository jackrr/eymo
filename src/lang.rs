use crate::pipeline::{Detection, Face};
use crate::shapes::shape::Shape;
use crate::Transform;
use anyhow::{Error, Result};
use ast::{Operation, Statement};
use lalrpop_util::lalrpop_mod;
use std::ptr;
use tracing::{trace, warn};

pub mod ast;

lalrpop_mod!(pub parser, "/lang/grammar.rs");

pub fn parse(input: &str) -> Result<Interpreter> {
    match parser::StatementsParser::new().parse(input) {
        Ok(res) => Ok(Interpreter::new(res)),
        Err(e) => Err(Error::msg(format!("{e:?}"))),
    }
}

// TODO: maintain mapping of statements and current state,
// allowing transforms to "save" state into mapping,
// and forwarding that state into the transform for next iteration
// (for spin + drift)
#[derive(Debug)]
pub struct Interpreter {
    statements: Vec<Statement>,
}

impl Interpreter {
    pub fn new(statements: Vec<Statement>) -> Self {
        Self { statements }
    }

    // pub fn load(&mut self, input: &'prog str) -> Result<()> {
    //     let statements = parser::StatementsParser::new().parse(input)?;
    //     Ok(Self {
    //         statements: &statements,
    //     })
    // }

    pub fn transforms(&self, detection: &Detection) -> Vec<Transform> {
        let mut transforms = Vec::new();

        for statement in &self.statements {
            trace!("Handling statement {statement:?}");
            match statement {
                Statement::Transform(t) => {
                    match build_transforms(t, detection) {
                        Ok(mut ts) => transforms.append(&mut ts),
                        Err(e) => warn!("{e:?}"),
                    };
                }
                Statement::Clear(idxs) => {
                    warn!("TODO: Clear handler not implemented!");
                }
            }
        }

        transforms
    }
}

fn build_transforms(statement: &ast::Transform, detection: &Detection) -> Result<Vec<Transform>> {
    match &statement.shape {
        ast::Shape::Rect(r) => {
            let mut t = Transform::new(r.clone());
            apply_operations(&mut t, statement, detection, None);
            Ok(Vec::from([t]))
        }
        ast::Shape::FaceRef(fr) => match fr.face_idx {
            Some(idx) => match detection.get(idx as usize) {
                Some(face) => {
                    let mut t = Transform::new(face_shape(&fr.part, face));
                    apply_operations(&mut t, statement, detection, Some(face));
                    Ok(Vec::from([t]))
                }
                None => Err(Error::msg(format!("No matching faces found for {fr:?}"))),
            },
            None => {
                let mut transforms = Vec::new();
                for face in detection {
                    let mut t = Transform::new(face_shape(&fr.part, face));
                    apply_operations(&mut t, statement, detection, Some(face));
                    transforms.push(t);
                }
                Ok(transforms)
            }
        },
    }
}

fn apply_operations(
    t: &mut Transform,
    statement: &ast::Transform,
    detection: &Detection,
    face: Option<&Face>,
) {
    for o in &statement.operations {
        match o {
            Operation::Tile => t.set_tiling(true),
            Operation::Scale(s) => t.set_scale(*s),
            Operation::Rotate(r) => t.set_rot_degrees(*r),
            Operation::WriteTo(others) => {
                let others = others
                    .iter()
                    .map(|s| match s {
                        ast::Shape::FaceRef(fr) => shapes(&fr, detection, face),
                        ast::Shape::Rect(r) => Vec::from([r.clone().into()]),
                    })
                    .collect::<Vec<Vec<_>>>()
                    .concat();
                t.write_to(others)
            }
            Operation::CopyTo(others) => {
                let others = others
                    .iter()
                    .map(|s| match s {
                        ast::Shape::FaceRef(fr) => shapes(&fr, detection, face),
                        ast::Shape::Rect(r) => Vec::from([r.clone().into()]),
                    })
                    .collect::<Vec<Vec<_>>>()
                    .concat();
                t.copy_to(others)
            }
            Operation::SwapWith(other) => match other {
                ast::Shape::FaceRef(fr) => {
                    let shapes = shapes(&fr, detection, face);
                    if shapes.len() == 0 {
                        warn!("No swap target found in {statement:?}");
                    } else {
                        if shapes.len() > 1 {
                            warn!("Ambiguous swap target found in {statement:?}");
                        }

                        t.swap_with(shapes[0].clone())
                    }
                }
                ast::Shape::Rect(r) => t.swap_with(r.clone().into()),
            },
            Operation::Translate(x, y) => t.translate_by(*x, *y),
            Operation::Flip(v) => t.set_flip(*v),
            // TODO: state for drift and spin
            Operation::Drift(velocity, angle) => t.set_drift(velocity, angle),
            Operation::Spin(velocity, counter_clockwise) => t.set_spin(velocity, counter_clockwise),
        }
    }
}

fn shapes(fr: &ast::FaceRef, d: &Detection, f: Option<&Face>) -> Vec<Shape> {
    match fr.face_idx {
        Some(idx) => match d.get(idx as usize) {
            Some(face) => match f {
                Some(target_face) => {
                    if ptr::eq(target_face, face) {
                        Vec::from([face_shape(&fr.part, face)])
                    } else {
                        Vec::new()
                    }
                }
                None => Vec::from([face_shape(&fr.part, face)]),
            },
            None => Vec::new(),
        },
        None => d
            .iter()
            .filter_map(|face| match f {
                Some(target_face) => {
                    if ptr::eq(target_face, face) {
                        Some(face_shape(&fr.part, face))
                    } else {
                        None
                    }
                }
                None => Some(face_shape(&fr.part, face)),
            })
            .collect::<Vec<_>>(),
    }
}

fn face_shape(p: &ast::FacePart, f: &Face) -> Shape {
    match p {
        ast::FacePart::LEye => f.l_eye.clone(),
        ast::FacePart::LEyeRegion => f.l_eye_region.clone(),
        ast::FacePart::REye => f.r_eye.clone(),
        ast::FacePart::REyeRegion => f.r_eye_region.clone(),
        ast::FacePart::Face => f.face.clone(),
        ast::FacePart::Mouth => f.mouth.clone(),
        ast::FacePart::Nose => f.nose.clone(),
    }
    .into()
}

#[test]
fn parse_integration_test() -> Result<()> {
    let stuff = r#"leye: translate(100, -80)
    mouth#1: swap_with(mouth#0)
    mouth#0: scale(2.5), write_to(leye_region, nose), swap_with(reye)
    "#;

    let res = parse(stuff.into())?;
    // println!("{res:?}");
    assert_eq!(res.statements.len(), 3);
    Ok(())
}

#[test]
fn write_to_multiple() -> Result<()> {
    let stmt = "mouth#0: scale(2.5), write_to(leye_region, nose), swap_with(reye)";
    let _res = parser::StatementParser::new().parse(&stmt)?;
    Ok(())
}
