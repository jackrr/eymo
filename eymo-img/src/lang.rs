use crate::imggpu::gpu::GpuExecutor;
use crate::pipeline::{Detection, Face};
use crate::shapes::shape::Shape;
use crate::transform::{ShapeOp, Transform};
use anyhow::{Error, Result};
use ast::{Operation, Statement};
use lalrpop_util::lalrpop_mod;
use tracing::warn;

pub mod ast;

lalrpop_mod!(pub parser, "/lang/grammar.rs");

pub fn parse(input: &str, gpu: &mut GpuExecutor) -> Result<Interpreter> {
    // HACK: append newline for parser happiness
    match parser::StatementsParser::new().parse(&(input.to_owned() + "\n")) {
        Ok(res) => Ok(Interpreter::new(res, gpu)),
        Err(e) => Err(Error::msg(format!("{e:?}"))),
    }
}

#[derive(Debug)]
pub struct Interpreter {
    transforms: Vec<(Transform, ast::Transform)>,
}

impl Interpreter {
    pub fn new(statements: Vec<Statement>, gpu: &mut GpuExecutor) -> Self {
        Self {
            transforms: statements
                .into_iter()
                .map(|s| match s {
                    ast::Statement::Transform(t) => (build_transform(&t, gpu), t),
                })
                .collect::<Vec<_>>(),
        }
    }

    pub fn execute<F>(
        &mut self,
        detection: &Detection,
        tex: wgpu::Texture,
        gpu: &mut GpuExecutor,
        timeout_check: F,
    ) -> wgpu::Texture
    where
        F: Fn(&str) -> Result<()>,
    {
        let mut output = tex;

        for (idx, (transform, cmd)) in self.transforms.iter_mut().enumerate() {
            match timeout_check(&format!("Transform {idx}")) {
                Ok(_) => {}
                Err(e) => {
                    warn!("{e:?}");
                    return output;
                }
            };

            let ops = shape_ops(idx.to_string(), cmd, detection);
            if ops.len() > 0 {
                output = transform.execute(gpu, &output, ops);
            }
        }

        output
    }
}

fn build_transform(cmd: &ast::Transform, gpu: &mut GpuExecutor) -> Transform {
    let mut t = Transform::new(gpu);
    apply_shape_agnostic_operations(&mut t, cmd);
    t
}

fn apply_shape_agnostic_operations(t: &mut Transform, cmd: &ast::Transform) {
    for o in &cmd.operations {
        match o {
            Operation::Brightness(b) => t.set_brightness(*b),
            Operation::Chans(r, g, b) => t.set_chans(*r, *g, *b),
            Operation::Reshape(dxl, dxr, dyt, dyb) => t.set_reshape(*dxl, *dxr, *dyt, *dyb),
            Operation::Drift(velocity, angle) => t.set_drift(*velocity, *angle),
            Operation::Flip(v) => t.set_flip(*v),
            Operation::Rotate(r) => t.set_rot_degrees(*r),
            Operation::Saturation(s) => t.set_saturation(*s),
            Operation::Scale(s) => t.set_scale(*s),
            Operation::Spin(velocity) => t.set_spin(*velocity),
            Operation::Tile => t.set_tiling(true),
            Operation::Translate(x, y) => t.translate_by(*x, *y),
            _ => {}
        }
    }
}

fn shape_ops(
    cache_key_prefix: String,
    cmd: &ast::Transform,
    detection: &Detection,
) -> Vec<ShapeOp> {
    match &cmd.shape {
        ast::Shape::Rect(r) => shape_ops_for_src_shape(
            cache_key_prefix,
            r.clone(),
            &cmd.operations,
            detection,
            None,
        ),
        ast::Shape::FaceRef(fr) => match fr.face_idx {
            Some(ast::FaceIdx::Absolute(abs)) => match detection.get(abs as usize) {
                Some(face) => shape_ops_for_src_shape(
                    cache_key_prefix,
                    face_shape(&fr.part, face),
                    &cmd.operations,
                    detection,
                    Some(abs as usize),
                ),
                None => {
                    warn!("No matching face found for {cmd:?}");
                    Vec::new()
                }
            },
            Some(ast::FaceIdx::Relative(rel)) => {
                let mut ops = Vec::new();
                for (idx, face) in detection.iter().enumerate() {
                    ops.append(&mut shape_ops_for_src_shape(
                        format!("{cache_key_prefix}-{idx}"),
                        face_shape(&fr.part, face),
                        &cmd.operations,
                        detection,
                        Some(((idx as i32 - rel) % detection.len() as i32) as usize),
                    ));
                }
                ops
            }
            None => {
                let mut ops = Vec::new();
                for (idx, face) in detection.iter().enumerate() {
                    ops.append(&mut shape_ops_for_src_shape(
                        format!("{cache_key_prefix}-{idx}"),
                        face_shape(&fr.part, face),
                        &cmd.operations,
                        detection,
                        Some(idx),
                    ));
                }
                ops
            }
        },
    }
}

fn shape_ops_for_src_shape(
    cache_key_prefix: String,
    src: impl Into<Shape> + Clone,
    ops: &Vec<ast::Operation>,
    detection: &Detection,
    target_face_idx: Option<usize>,
) -> Vec<ShapeOp> {
    let mut sops = Vec::new();

    for o in ops {
        match o {
            Operation::CopyTo(others) => {
                for (idx, other) in others.iter().enumerate() {
                    match other {
                        ast::Shape::FaceRef(fr) => {
                            for (sidx, s) in
                                shapes(&fr, detection, target_face_idx).iter().enumerate()
                            {
                                sops.push(ShapeOp::copy(
                                    format!("{cache_key_prefix}-{idx}-{sidx}"),
                                    src.clone(),
                                    s.clone(),
                                ));
                            }
                        }
                        ast::Shape::Rect(r) => sops.push(ShapeOp::copy(
                            format!("{cache_key_prefix}-{idx}"),
                            src.clone(),
                            r.clone(),
                        )),
                    }
                }
            }
            Operation::SwapWith(other) => match other {
                ast::Shape::FaceRef(fr) => {
                    for (sidx, s) in shapes(&fr, detection, target_face_idx).iter().enumerate() {
                        sops.push(ShapeOp::swap(
                            format!("{cache_key_prefix}-{sidx}"),
                            src.clone(),
                            s.clone(),
                        ));
                    }
                }
                ast::Shape::Rect(r) => sops.push(ShapeOp::swap(
                    format!("{cache_key_prefix}-rect"),
                    src.clone(),
                    r.clone(),
                )),
            },
            _ => {}
        }
    }

    if sops.len() == 0 {
        sops.push(ShapeOp::on_shape(cache_key_prefix, src));
    }

    sops
}

fn shapes(fr: &ast::FaceRef, d: &Detection, target_idx: Option<usize>) -> Vec<Shape> {
    match fr.face_idx {
        Some(ast::FaceIdx::Absolute(abs)) => match d.get(abs as usize) {
            Some(face) => Vec::from([face_shape(&fr.part, face)]),
            None => Vec::new(),
        },
        Some(ast::FaceIdx::Relative(rel)) => {
            let idx = (match target_idx {
                Some(target_idx) => rel + target_idx as i32,
                None => rel,
            } % d.len() as i32) as usize;

            Vec::from([face_shape(&fr.part, &d[idx])])
        }
        None => match target_idx {
            Some(target_idx) => match d.get(target_idx) {
                Some(face) => Vec::from([face_shape(&fr.part, face)]),
                None => Vec::new(),
            },
            None => d
                .iter()
                .map(|face| face_shape(&fr.part, face))
                .collect::<Vec<_>>(),
        },
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
        ast::FacePart::Forehead => f.forehead.clone(),
    }
    .into()
}

#[test]
fn parse_test() -> Result<()> {
    let stuff = r#"leye: translate(100, -80)
    mouth#1: swap_with(mouth#0)
    mouth#0: scale(2.5), write_to(leye_region, nose), swap_with(reye)
    "#;

    let res = parser::StatementsParser::new().parse(&stuff)?;
    assert_eq!(res.len(), 3);
    Ok(())
}

#[test]
fn write_to_multiple() -> Result<()> {
    let stmt = "mouth#0: scale(2.5), write_to(leye_region, nose), swap_with(reye)";
    let _res = parser::StatementParser::new().parse(&stmt)?;
    Ok(())
}
