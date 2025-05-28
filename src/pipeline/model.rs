use anyhow::Result;
use ort::session::builder::GraphOptimizationLevel;
pub use ort::session::Session;

pub fn initialize_model(model_file_path: &str, threads: usize) -> Result<Session> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(threads)?
        .commit_from_file(format!("./models/{model_file_path:}"))?;

    Ok(model)
}
