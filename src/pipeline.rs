use std::time::Instant;

use anyhow::Result;
use image::{Rgb, RgbImage};

use crate::shapes::npoint::NPoint;
use detection::FaceDetector;
use imageproc::drawing;
use landmarks::FaceLandmarker;
use log::{debug, info};

mod detection;
mod landmarks;
mod model;

pub struct Pipeline {
    face_detector: FaceDetector,
    face_landmarker: FaceLandmarker,
}

#[derive(Debug, Clone)]
pub struct Face {
    pub l_eye: NPoint,
    pub r_eye: NPoint,
    pub mouth: NPoint,
    pub nose: NPoint,
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

    pub fn run(&self, img: &RgbImage) -> Result<Detection> {
        let start = Instant::now();
        let face_bounds = self.face_detector.run(img)?;
        info!("Detector took {}ms", start.elapsed().as_millis());
        let mut faces = Vec::new();

        for face_bound in face_bounds {
            debug!("Face bound: {face_bound:?}");
            let start = Instant::now();
            let face = self.face_landmarker.run(img, &face_bound)?;
            info!("Landmarker took {}ms", start.elapsed().as_millis());
            debug!("Face features: {face:?}");

            faces.push(face);
        }

        Ok(Detection { faces })
    }

    pub fn run_trace(&self, img: &mut RgbImage) -> Result<Detection> {
        let face_bounds = self.face_detector.run(img)?;
        let mut faces = Vec::new();

        for face_bound in face_bounds {
            debug!("Face bound: {face_bound:?}");
            let face = self.face_landmarker.run(img, &face_bound)?;
            debug!("Face features: {face:?}");
            faces.push(face);
            drawing::draw_hollow_rect_mut(img, face_bound.bounds.into(), Rgb([255u8, 0u8, 0u8]));
            drawing::draw_filled_circle_mut(
                img,
                (face_bound.l_eye.x as i32, face_bound.l_eye.y as i32),
                3,
                Rgb([0u8, 255u8, 0u8]),
            );
            drawing::draw_filled_circle_mut(
                img,
                (face_bound.r_eye.x as i32, face_bound.r_eye.y as i32),
                3,
                Rgb([255u8, 0u8, 0u8]),
            );
        }

        Ok(Detection { faces })
    }
}
