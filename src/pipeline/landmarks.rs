use super::detection::Face;
use super::model::{initialize_model, Session};
use super::rect::Point;
use ab_glyph::FontRef;
use anyhow::Result;
use image::imageops::{resize, FilterType};
use image::{GenericImage, GenericImageView, Rgb, RgbImage};
use imageproc::drawing;
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use log::debug;
use ndarray::Array;
use ort::value::Tensor;

pub struct FaceLandmarker {
    model: Session,
    font: FontRef<'static>,
}

const HEIGHT: u32 = 192;
const WIDTH: u32 = 192;

const MOUTH_IDXS: [usize; 5] = [
    171, 558, 276, 495, 501, 492, 1179, 1173, 966, 1230, 861, 1266,
];
// const L_EYE_IDXS: [usize; 5] = [];
// const R_EYE_IDXS: [usize; 5] = [];
// const NOSE_IDXS: [usize; 5] = [];

impl FaceLandmarker {
    pub fn new(threads: usize) -> Result<FaceLandmarker> {
        Ok(FaceLandmarker {
            model: initialize_model("mediapipe_face_landmark.onnx", threads)?,
            font: FontRef::try_from_slice(include_bytes!(
                "/usr/share/fonts/fira-code/FiraCode-Bold.ttf"
            ))
            .unwrap(),
        })
    }

    pub fn run(&self, img: &mut RgbImage, face: &Face) -> Result<()> {
        let mut bounds = face.bounds.clone();
        // pad 25% on each side
        bounds.w = (bounds.w as f32 * 1.5).round() as u32;
        bounds.h = (bounds.h as f32 * 1.5).round() as u32;

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

        // TODO: make image MASSIVE, then draw
        let scale = 10;
        let old_width = img.width();
        let old_height = img.height();
        *img = resize(
            img,
            scale * old_width,
            scale * old_height,
            FilterType::Nearest,
        );

        let scale_x = img.width() as f32 / old_width as f32;
        let scale_y = img.height() as f32 / old_height as f32;

        // TODO: find indices that correspond to mouth, nose, eye boundaries

        for i in 0..mesh.len() / 3 {
            // 1. rescale
            let idx = i * 3;
            let x = (bounds.left() as f32 + r[idx] * x_scale) * scale_x;
            let y = (bounds.top() as f32 + r[idx + 1] * y_scale) * scale_y;
            let mut p = Point::new(x.round() as u32, y.round() as u32);

            let origin = bounds.center();

            // 2. unrotate
            p.rotate(origin, face.rot_theta());

            drawing::draw_filled_circle_mut(
                img,
                (p.x as i32, p.y as i32),
                3,
                Rgb([255u8, 255u8, 255u8]),
            );

            drawing::draw_text_mut(
                img,
                Rgb([0u8, 0u8, 0u8]),
                p.x as i32,
                p.y as i32,
                24.,
                &self.font,
                &format!("{idx}"),
            );
        }

        Ok(())
    }
}
