use anyhow::Result;
use image::{Rgb, RgbImage};

use detection::FaceDetector;
use imageproc::drawing;
use landmarks::FaceLandmarker;

mod detection;
mod landmarks;
mod model;
mod rect;

pub struct Pipeline {
    face_detector: FaceDetector,
    face_landmarker: FaceLandmarker,
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
pub struct Face {
    pub left_eye: Option<Feature>,
    pub right_eye: Option<Feature>,
    pub mouth: Option<Feature>,
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub faces: Vec<Face>,
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
            drawing::draw_hollow_rect_mut(img, face.into(), Rgb([255u8, 0u8, 0u8]));
            let res = self.face_landmarker.run(img, &face);
        }

        Ok(())
    }
}
