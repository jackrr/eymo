use anyhow::Result;
use image::{Rgb, RgbImage};

use crate::shapes::polygon::Polygon;
use detection::FaceDetector;
use imageproc::drawing;
use landmarks::FaceLandmarker;
use tracing::{debug, info, span, trace, Level};

mod detection;
mod landmarks;
mod model;

pub struct Pipeline {
    face_detector: FaceDetector,
    face_landmarker: FaceLandmarker,
}

#[derive(Debug, Clone)]
pub struct Face {
    pub l_eye: Polygon,
    pub r_eye: Polygon,
    pub mouth: Polygon,
    pub nose: Polygon,
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
        let span = span!(Level::INFO, "pipeline");
        let _guard = span.enter();

        let face_bounds = self.face_detector.run(img)?;

        let mut faces = Vec::new();
        for face_bound in face_bounds {
            trace!("Face bound: {face_bound:?}");

            let face = self.face_landmarker.run(img, &face_bound)?;
            info!("Landmarker done");
            trace!("Face features: {face:?}");

            faces.push(face);
        }

        Ok(Detection { faces })
    }

    pub fn run_trace(&self, img: &mut RgbImage) -> Result<Detection> {
        let face_bounds = self.face_detector.run(img)?;
        let mut faces = Vec::new();

        for face_bound in face_bounds {
            trace!("Face bound: {face_bound:?}");
            let face = self.face_landmarker.run(img, &face_bound)?;
            trace!("Face features: {face:?}");
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
