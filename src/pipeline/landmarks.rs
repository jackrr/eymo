use crate::pipeline::model::{initialize_model, Session};
use anyhow::Result;

pub struct FaceLandmarker {
    model: Session,
}

impl FaceLandmarker {
    pub fn new(threads: usize) -> Result<FaceLandmarker> {
        Ok(FaceLandmarker {
            model: initialize_model("mediapipe_face_landmark.onnx", threads)?,
        })
    }
}
