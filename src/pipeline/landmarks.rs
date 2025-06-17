use super::detection;
use super::model::{initialize_model, Session};
use super::Face;
use crate::imggpu::resize::{CachedResizer, ResizeAlgo};
use crate::imggpu::rotate::{rotate, GpuExecutor};
use crate::shapes::point::Point;
use crate::shapes::polygon::Polygon;
use anyhow::Result;
use image::{GenericImage, GenericImageView, RgbImage};
use ndarray::Array;
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

    // ~80-90ms
    pub fn run(&mut self, img: &RgbImage, face: &detection::Face) -> Result<Face> {
        let span = span!(Level::INFO, "face_landmarker");
        let _guard = span.enter();

        // ~25-60ms
        let span_rotate = span!(Level::INFO, "face_landmarker_rotate");
        let rotate_guard = span_rotate.enter();
        let mut bounds = face.bounds.clone();
        // pad 25% on each side
        bounds.scale(1.5, img.width(), img.height());

        let view = *img.view(bounds.left(), bounds.top(), bounds.w, bounds.h);

        // face_img - img cropped to face and rotated
        let mut face_img = RgbImage::new(bounds.w, bounds.h);
        face_img.copy_from(&view, 0, 0)?;
        face_img = rotate(
            &self.gpu,
            &face_img,
            -face.rot_theta(),
            [0f32, 0f32, 0f32, 0f32],
        );
        drop(rotate_guard);

        // ~33-50ms
        let span_resize = span!(Level::INFO, "face_landmarker_resize");
        let resize_guard = span_resize.enter();
        let input_img = self
            .resizer
            .run(&face_img, WIDTH, HEIGHT, ResizeAlgo::Nearest);
        drop(resize_guard);

        // ~18ms
        let span_tensor = span!(Level::INFO, "face_landmarker_tensor");
        let tensor_guard = span_tensor.enter();
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
        drop(tensor_guard);

        // ~3-10ms
        let span_inference = span!(Level::INFO, "face_landmarker_inference");
        let inference_guard = span_inference.enter();
        let outputs = self.model.run(ort::inputs!["input_1" => input]?)?;

        let output = outputs["conv2d_21"].try_extract_tensor::<f32>()?;
        let mesh = output.squeeze().squeeze().squeeze();
        drop(inference_guard);

        // ~20micros
        let span_results = span!(Level::INFO, "face_landmarker_results");
        let results_guard = span_results.enter();

        let r = mesh.as_slice().unwrap();

        let x_scale = face_img.width() as f32 / input_img_width as f32;
        let y_scale = face_img.height() as f32 / input_img_height as f32;
        let x_offset = bounds.left() as f32;
        let y_offset = bounds.top() as f32;
        let origin = bounds.center();
        let rotation = face.rot_theta();
        let result = Face {
            mouth: extract_feature(
                &r,
                &MOUTH_IDXS,
                x_offset,
                y_offset,
                x_scale,
                y_scale,
                &origin,
                rotation,
            ),
            l_eye: extract_feature(
                &r,
                &L_EYE_IDXS,
                x_offset,
                y_offset,
                x_scale,
                y_scale,
                &origin,
                rotation,
            ),
            r_eye: extract_feature(
                &r,
                &R_EYE_IDXS,
                x_offset,
                y_offset,
                x_scale,
                y_scale,
                &origin,
                rotation,
            ),
            nose: extract_feature(
                &r, &NOSE_IDXS, x_offset, y_offset, x_scale, y_scale, &origin, rotation,
            ),
        };

        drop(results_guard);

        Ok(result)
    }
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
