use anyhow::Result;
use image::{Rgb, RgbImage};

use detection::FaceDetector;
use imageproc::drawing;
use landmarks::FaceLandmarker;
use log::debug;

mod detection;
mod landmarks;
mod model;
mod rect;

pub struct Pipeline {
    face_detector: FaceDetector,
    face_landmarker: FaceLandmarker,
}

#[derive(Debug, Clone, Copy)]
pub struct PointF32 {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: u32,
    pub y: u32,
}

#[derive(Debug, Clone)]
pub struct Feature {
    pub points: Vec<Point>,
}

#[derive(Debug, Clone)]
pub struct FaceFeatures {
    pub left_eye: Option<Feature>,
    pub right_eye: Option<Feature>,
    pub mouth: Option<Feature>,
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub faces: Vec<FaceFeatures>,
}

impl Pipeline {
    pub fn new(max_threads: usize) -> Result<Pipeline> {
        Ok(Pipeline {
            face_detector: FaceDetector::new(max_threads)?,
            face_landmarker: FaceLandmarker::new(max_threads)?,
        })
    }

    pub fn run(&self, img: &RgbImage) -> Result<()> {
        println!("TODO: implement me!");
        let res = self.face_detector.run(img)?;
        Ok(())
    }

    pub fn run_trace(&self, img: &mut RgbImage) -> Result<()> {
        let res = self.face_detector.run(img)?;
        for face in res {
            let res = self.face_landmarker.run(img, &face);
            debug!("{face:?}");
            drawing::draw_hollow_rect_mut(img, face.bounds.into(), Rgb([255u8, 0u8, 0u8]));
            drawing::draw_filled_circle_mut(
                img,
                (face.l_eye.x as i32, face.l_eye.y as i32),
                10,
                Rgb([0u8, 255u8, 0u8]),
            );
            drawing::draw_filled_circle_mut(
                img,
                (face.r_eye.x as i32, face.r_eye.y as i32),
                10,
                Rgb([255u8, 0u8, 0u8]),
            );
        }

        Ok(())
    }
}
