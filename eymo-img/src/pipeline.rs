use anyhow::Result;

use crate::shapes::rect::Rect;
use crate::{imggpu::gpu::GpuExecutor, shapes::polygon::Polygon};
use detection::FaceDetector;
use landmarks::FaceLandmarker;
use tracing::{info, span, trace, Level};

mod detection;
mod landmarks;
mod model;

pub struct Pipeline {
    face_detector: FaceDetector,
    face_landmarker: FaceLandmarker,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Face {
    pub face: Polygon,
    pub mouth: Polygon,
    pub nose: Polygon,
    pub l_eye: Polygon,
    pub l_eye_region: Polygon,
    pub r_eye: Polygon,
    pub r_eye_region: Polygon,
    pub bound: Rect,
}

pub type Detection = Vec<Face>;

impl Pipeline {
    pub fn new() -> Result<Pipeline> {
        Ok(Pipeline {
            face_detector: FaceDetector::new()?,
            face_landmarker: FaceLandmarker::new()?,
        })
    }

    pub async fn run_gpu(&mut self, tex: &wgpu::Texture, gpu: &mut GpuExecutor) -> Result<Detection> {
        let span = span!(Level::DEBUG, "pipeline");
        let _guard = span.enter();

        info!("Starting face detector..");
        let face_bounds = self.face_detector.run_gpu(tex, gpu).await?;
        let mut faces = Vec::new();
        for face_bound in face_bounds {
            trace!("Face bound: {face_bound:?}");

            let face = self.face_landmarker.run_gpu(&face_bound, tex, gpu).await?;
            trace!("Face features: {face:?}");

            faces.push(face);
        }

        Ok(faces)
    }
}
