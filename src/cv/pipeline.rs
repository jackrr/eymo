use anyhow::Result;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

pub struct Pipeline {
    face_detector: Session,
    face_landmarker: Session,
}

impl Pipeline {
    pub fn new(max_threads: usize) -> Result<Pipeline> {
        Ok(Pipeline {
            face_detector: initialize_model(
                "mediapipe_face_detection_short_range.onnx",
                max_threads,
            )?,
            face_landmarker: initialize_model(
                "mediapipe_face_landmark_attention.onnx",
                max_threads,
            )?,
        })
    }

    pub fn run(&self, img: &DynamicImage) {}

    pub fn run_trace(&self, img: &mut DynamicImage) {}
}

fn initialize_model(model_file_path: &str, threads: usize) -> Result<Session> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(threads)?
        .commit_from_file(format!("./models/{model_file_path:}"))?;

    Ok(model)
}
