use super::detection::Face;
use super::model::{initialize_model, Session};
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
}

#[derive(Debug, Copy, Clone)]
pub struct Point {
    pub x: u32,
    pub y: u32,
}

impl Point {
    pub fn new(x: u32, y: u32) -> Point {
        Point { x, y }
    }
}

const HEIGHT: u32 = 192;
const WIDTH: u32 = 192;

impl FaceLandmarker {
    pub fn new(threads: usize) -> Result<FaceLandmarker> {
        Ok(FaceLandmarker {
            model: initialize_model("mediapipe_face_landmark.onnx", threads)?,
        })
    }

    pub fn run(&self, img: &mut RgbImage, face: &Face) -> Result<()> {
        let mut bounds = face.bounds.clone();
        // pad 25% on each side
        bounds.w = (bounds.w as f32 * 1.5).round() as u32;
        bounds.h = (bounds.h as f32 * 1.5).round() as u32;

        let rotated = rotate_about_center(
            img,
            -face.rot_theta(),
            Interpolation::Nearest,
            Rgb([0u8, 0u8, 0u8]),
        );

        let view = *rotated.view(bounds.left(), bounds.top(), bounds.w, bounds.h);
        let mut face_img = RgbImage::new(bounds.w, bounds.h);
        face_img.copy_from(&view, 0, 0)?;

        let resized = resize(&face_img, WIDTH, HEIGHT, FilterType::Nearest);
        let resized_height = resized.height();
        let resized_width = resized.width();

        resized.save("tmp/face.jpg")?;
        let input_arr =
            Array::from_shape_fn((1, HEIGHT as usize, WIDTH as usize, 3), |(_, y, x, c)| {
                let x: u32 = x as u32;
                let y: u32 = y as u32;

                if y >= resized_height {
                    0.
                } else if x >= resized_width {
                    0.
                } else {
                    resized.get_pixel(x, y)[c] as f32 / 255. // 0. - 1. range
                }
            });

        let input = Tensor::from_array(input_arr)?;
        let outputs = self.model.run(ort::inputs!["input_1" => input]?)?;

        let output = outputs["conv2d_21"].try_extract_tensor::<f32>()?;
        let mesh = output.squeeze().squeeze().squeeze();
        debug!("{:?}", mesh);
        let r = mesh.as_slice().unwrap();
        let x_scale = img.width() as f32 / resized_width as f32;
        let y_scale = img.height() as f32 / resized_height as f32;

        // TODO: unrotate output
        // FIXME: strange scale issue
        // TODO: find indices that correspond to mouth, nose, eye boundaries

        for i in 0..mesh.len() / 3 {
            let idx = i * 3;
            let x = bounds.left() as f32 + r[idx] * x_scale;
            let y = bounds.top() as f32 + r[idx + 1] * y_scale;
            let p = Point::new(x.round() as u32, y.round() as u32);
            debug!("{}x{}", p.x, p.y);

            drawing::draw_filled_circle_mut(
                img,
                (p.x as i32, p.y as i32),
                2,
                Rgb([255u8, 255u8, 255u8]),
            );
        }

        Ok(())
    }
}
