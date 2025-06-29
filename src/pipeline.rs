use anyhow::Result;

use crate::{
    imggpu::gpu::GpuExecutor,
    shapes::{point::Point, polygon::Polygon},
};
use detection::FaceDetector;
use landmarks::{FaceLandmarker, Landmark};
use tracing::{info, span, trace, Level};

mod detection;
mod landmarks;
mod model;

pub struct Pipeline {
    face_detector: FaceDetector,
    face_landmarker: FaceLandmarker,
}

#[derive(Debug, Clone)]
pub struct Face {
    pub face: Polygon,
    pub mouth: Polygon,
    pub nose: Polygon,
    pub l_eye: Polygon,
    pub l_eye_region: Polygon,
    pub r_eye: Polygon,
    pub r_eye_region: Polygon,
    pub corner_sus: Vec<Landmark>,
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

    pub fn run_gpu(&mut self, tex: &wgpu::Texture, gpu: &mut GpuExecutor) -> Result<Detection> {
        let span = span!(Level::INFO, "pipeline");
        let _guard = span.enter();

        let face_bounds = self.face_detector.run_gpu(tex, gpu)?;
        let mut faces = Vec::new();
        for face_bound in face_bounds {
            trace!("Face bound: {face_bound:?}");

            let face = self.face_landmarker.run_gpu(&face_bound, tex, gpu)?;
            trace!("Face features: {face:?}");

            faces.push(face);
        }

        Ok(Detection { faces })
    }
}
