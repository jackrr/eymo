use super::detection;
use super::model::{initialize_model, Session};
use super::Face;
use crate::imggpu::resize::{CachedResizer, ResizeAlgo};
use crate::imggpu::rotate::{rotate, GpuExecutor};
use crate::shapes::point::Point;
use crate::shapes::polygon::Polygon;
use crate::shapes::rect::Rect;
use anyhow::{Error, Result};
use image::{GenericImage, GenericImageView, RgbImage};
use ndarray::Array;
use ort::session::SessionOutputs;
use ort::value::Tensor;
use tracing::{debug, span, Level};

pub struct FaceLandmarker {
    model: Session,
    resizer: CachedResizer,
    gpu: GpuExecutor,
}

const HEIGHT: u32 = 192;
const WIDTH: u32 = 192;

// TODO: get a "canonical" detection to dial in these landmark indices
const MOUTH_IDXS: [usize; 20] = [
    212, 186, 92, 165, 167, 164, 393, 391, 322, 410, 287, 422, 424, 418, 421, 200, 201, 194, 204,
    202,
];
const L_EYE_IDXS: [usize; 15] = [
    226, 247, 30, 29, 27, 28, 56, 190, 243, 233, 121, 120, 119, 118, 31,
];
const R_EYE_IDXS: [usize; 16] = [
    6, 417, 441, 442, 443, 444, 445, 353, 356, 261, 448, 449, 450, 451, 452, 351,
];
const NOSE_IDXS: [usize; 18] = [
    189, 193, 168, 6, 197, 195, 5, 275, 274, 393, 164, 167, 165, 203, 129, 126, 114, 244,
];

impl FaceLandmarker {
    pub fn new(threads: usize) -> Result<FaceLandmarker> {
        Ok(FaceLandmarker {
            model: initialize_model("mediapipe_face_landmark.onnx", threads)?,
            resizer: CachedResizer::new()?,
            gpu: GpuExecutor::new()?,
        })
    }

    pub fn run_gpu(
        &mut self,
        face: &detection::Face,
        tex: &wgpu::Texture,
        gpu: &mut GpuExecutor,
    ) -> Result<Face> {
        let span = span!(Level::INFO, "face_landmarker");
        let _guard = span.enter();

        let mut bounds = face.bounds.clone();
        // pad 25% on each side
        bounds.scale(1.5, tex.width(), tex.height());

        // 1. crop texture to bounds
        // 2. resize image to WIDTHxHEIGHT
        // 3. rotate image
        // 4. output HEIGHTxWIDTH rgb f32 0->1 BUFFER
        // 5. quick tensor from buffer
        // 6. interpret outputs
        Err(Error::msg("run_gpu Not implemented!"))
    }

    // ~80-90ms
    pub fn run(&mut self, img: &RgbImage, face: &detection::Face) -> Result<Face> {
        let span = span!(Level::INFO, "face_landmarker");
        let _guard = span.enter();

        let mut bounds = face.bounds.clone();
        // pad 25% on each side
        bounds.scale(1.5, img.width(), img.height());

        let view = *img.view(bounds.left(), bounds.top(), bounds.w, bounds.h);

        // face_img - img cropped to face and and rotated
        let mut face_img = RgbImage::new(bounds.w, bounds.h);
        face_img.copy_from(&view, 0, 0)?;

        let input_img = self
            .resizer
            .run(&face_img, WIDTH, HEIGHT, ResizeAlgo::Nearest);

        face_img = rotate(
            &mut self.gpu,
            &face_img,
            -face.rot_theta(),
            [0f32, 0f32, 0f32, 0f32],
        );

        // ~18ms
        let input_img_height = input_img.height();
        let input_img_width = input_img.width();

        let input_arr =
            Array::from_shape_fn((1, HEIGHT as usize, WIDTH as usize, 3), |(_, y, x, c)| {
                let x: u32 = x as u32;
                let y: u32 = y as u32;

                if y >= input_img_height {
                    0.
                } else if x >= input_img_width {
                    0.
                } else {
                    input_img.get_pixel(x, y)[c] as f32 / 255. // 0. - 1. range
                }
            });

        let input = Tensor::from_array(input_arr)?;

        // ~3-10ms
        let outputs = self.model.run(ort::inputs!["input_1" => input]?)?;

        extract_results(
            outputs,
            input_img_width,
            input_img_height,
            bounds,
            face.rot_theta(),
        )
    }
}

fn extract_results(
    outputs: SessionOutputs,
    input_width: u32,
    input_height: u32,
    run_bounds: Rect,
    run_rot: f32,
) -> Result<Face> {
    let output = outputs["conv2d_21"].try_extract_tensor::<f32>()?;
    let mesh = output.squeeze().squeeze().squeeze();

    let r = mesh.as_slice().unwrap();

    let x_scale = run_bounds.w as f32 / input_width as f32;
    let y_scale = run_bounds.h as f32 / input_height as f32;
    let x_offset = run_bounds.left() as f32;
    let y_offset = run_bounds.top() as f32;
    let origin = run_bounds.center();

    Ok(Face {
        mouth: extract_feature(
            &r,
            &MOUTH_IDXS,
            x_offset,
            y_offset,
            x_scale,
            y_scale,
            &origin,
            run_rot,
        ),
        l_eye: extract_feature(
            &r,
            &L_EYE_IDXS,
            x_offset,
            y_offset,
            x_scale,
            y_scale,
            &origin,
            run_rot,
        ),
        r_eye: extract_feature(
            &r,
            &R_EYE_IDXS,
            x_offset,
            y_offset,
            x_scale,
            y_scale,
            &origin,
            run_rot,
        ),
        nose: extract_feature(
            &r, &NOSE_IDXS, x_offset, y_offset, x_scale, y_scale, &origin, run_rot,
        ),
    })
}

fn extract_feature(
    mesh: &[f32],
    kpt_idxs: &[usize],
    x_offset: f32,
    y_offset: f32,
    x_scale: f32,
    y_scale: f32,
    origin: &Point,
    rotation: f32,
) -> Polygon {
    let mut points = Vec::new();

    for i in kpt_idxs {
        let idx = i * 3;
        let x = x_offset + mesh[idx] * x_scale;
        let y = y_offset + mesh[idx + 1] * y_scale;

        let mut p = Point::new(x.round() as u32, y.round() as u32);

        p.rotate(*origin, rotation);

        points.push(p)
    }

    Polygon::new(points)
}
