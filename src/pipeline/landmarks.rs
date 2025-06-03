use super::detection;
use super::model::{initialize_model, Session};
use super::Face;
use crate::shapes::npoint::NPoint;
use crate::shapes::point::Point;
use anyhow::Result;
use image::imageops::{resize, FilterType};
use image::{GenericImage, GenericImageView, Rgb, RgbImage};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use log::debug;
use ndarray::Array;
use ort::value::Tensor;

pub struct FaceLandmarker {
    model: Session,
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
        })
    }

    pub fn run(&self, img: &RgbImage, face: &detection::Face) -> Result<Face> {
        let mut bounds = face.bounds.clone();
        // pad 25% on each side
        bounds.scale(1.5);

        let view = *img.view(bounds.left(), bounds.top(), bounds.w, bounds.h);

        // face_img - img cropped to face and rotated
        let mut face_img = RgbImage::new(bounds.w, bounds.h);
        face_img.copy_from(&view, 0, 0)?;
        face_img = rotate_about_center(
            &face_img,
            -face.rot_theta(),
            Interpolation::Nearest,
            Rgb([0u8, 0u8, 0u8]),
        );

        let input_img = resize(&face_img, WIDTH, HEIGHT, FilterType::Nearest);
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
        let outputs = self.model.run(ort::inputs!["input_1" => input]?)?;

        let output = outputs["conv2d_21"].try_extract_tensor::<f32>()?;
        let mesh = output.squeeze().squeeze().squeeze();

        let r = mesh.as_slice().unwrap();

        let x_scale = face_img.width() as f32 / input_img_width as f32;
        let y_scale = face_img.height() as f32 / input_img_height as f32;
        let x_offset = bounds.left() as f32;
        let y_offset = bounds.top() as f32;
        let origin = bounds.center();
        let rotation = face.rot_theta();

        Ok(Face {
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
        })
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
) -> NPoint {
    let mut points = Vec::new();

    for i in kpt_idxs {
        let idx = i * 3;
        let x = x_offset + mesh[idx] * x_scale;
        let y = y_offset + mesh[idx + 1] * y_scale;

        let mut p = Point::new(x.round() as u32, y.round() as u32);

        p.rotate(*origin, rotation);

        points.push(p)
    }

    NPoint { points }
}
