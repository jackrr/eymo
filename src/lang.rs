use crate::imggpu::gpu::GpuExecutor;
use crate::pipeline::{Detection, Face};
use crate::shapes::shape::Shape;
use crate::transform::Transform;
use anyhow::{Error, Result};
use ast::{Operation, Statement};
use lalrpop_util::lalrpop_mod;
use std::collections::HashMap;
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

type TransformCache = HashMap<String, Transform>;

#[derive(Debug)]
pub struct Interpreter {
    statements: Vec<Statement>,
    transforms: TransformCache,
}

impl Interpreter {
    pub fn new(statements: Vec<Statement>) -> Self {
        Self {
            statements,
            transforms: HashMap::new(),
        }
    }

    pub fn execute<F>(
        &mut self,
        detection: &Detection,
        tex: wgpu::Texture,
        gpu: &mut GpuExecutor,
        timeout_check: F,
    ) -> Result<wgpu::Texture>
    where
        F: Fn(&str) -> Result<()>,
    {
        let mut output = tex;
        // FIXME: this will cause state loss on timeout
        let mut prev_transforms = std::mem::replace(&mut self.transforms, HashMap::new());

        for (idx, statement) in self.statements.iter().enumerate() {
            match timeout_check(&format!("Transform {idx}")) {
                Ok(_) => trace!("Handling statement {statement:?}"),
                Err(e) => {
                    warn!("{e:?}");
                    return Ok(output);
                }
            };

            match statement {
                Statement::Transform(s) => {
                    match build_transforms(&s, detection, idx, &mut prev_transforms) {
                        Ok(mut ts) => {
                            while ts.len() > 0 {
                                let mut t = ts.swap_remove(0);
                                output = t.execute(gpu, &output)?;
                                self.transforms.insert(t.id.clone(), t);
                            }
                        }
                        Err(e) => warn!("{e:?}"),
                    };
                }
                Statement::Clear(idxs) => {
                    warn!("TODO: Clear handler not implemented!");
                }
            }
        }

        Ok(output)
    }
}

fn get_or_create_transform(
    transform_cache: &mut TransformCache,
    id: String,
    s: impl Into<Shape>,
) -> Transform {
    let transform = transform_cache.remove(&id);
    match transform {
        Some(mut transform) => {
            transform.set_shape(s);
            transform
        }
        None => Transform::new(s, id),
    }
}

fn build_transforms(
    statement: &ast::Transform,
    detection: &Detection,
    statement_idx: usize,
    transform_cache: &mut TransformCache,
) -> Result<Vec<Transform>> {
    match &statement.shape {
        ast::Shape::Rect(r) => {
            let mut t = get_or_create_transform(
                transform_cache,
                format!("rect-{statement_idx}"),
                r.clone(),
            );
            apply_operations(&mut t, statement, detection, None);
            Ok(Vec::from([t]))
        }
        ast::Shape::FaceRef(fr) => match fr.face_idx {
            Some(idx) => match detection.get(idx as usize) {
                Some(face) => {
                    let mut t = get_or_create_transform(
                        transform_cache,
                        format!("face-{idx}-{statement_idx}"),
                        face_shape(&fr.part, face),
                    );
                    apply_operations(&mut t, statement, detection, Some(face));
                    Ok(Vec::from([t]))
                }
                None => Err(Error::msg(format!("No matching faces found for {fr:?}"))),
            },
            None => {
                let mut transforms = Vec::new();
                for (idx, face) in detection.iter().enumerate() {
                    let mut t = get_or_create_transform(
                        transform_cache,
                        format!("face-{idx}-{statement_idx}"),
                        face_shape(&fr.part, face),
                    );
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
            Operation::Drift(velocity, angle) => t.set_drift(*velocity, *angle),
            Operation::Spin(velocity) => t.set_spin(*velocity),
            Operation::Brightness(b) => t.set_brightness(*b),
            Operation::Saturation(s) => t.set_saturation(*s),
            Operation::Chans(r, g, b) => t.set_chans(*r, *g, *b),
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
