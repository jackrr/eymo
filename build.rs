use burn_import::onnx::ModelGen;

fn main() {
    for model in [
        "./models/mediapipe_face_detection_short_range.onnx",
        "./models/mediapipe_face_landmark.onnx",
    ] {
        ModelGen::new()
            .input(model)
            .out_dir("models/")
            .run_from_script();
    }
}
