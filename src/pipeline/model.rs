use std::num::NonZero;

use anyhow::Result;
use ort::execution_providers;
use ort::session::builder::GraphOptimizationLevel;
pub use ort::session::Session;

pub fn initialize_model(model_file_path: &str, threads: usize) -> Result<Session> {
    ort::init()
        .with_execution_providers([execution_providers::XNNPACKExecutionProvider::default()
            .with_intra_op_num_threads(NonZero::new(threads - 2).unwrap())
            .build()
            .error_on_failure()])
        .commit()?;

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_parallel_execution(true)?
        // .with_intra_threads(threads - 2)?
        .with_inter_threads(2)?
        .commit_from_file(format!("./models/{model_file_path:}"))?;

    Ok(model)
}
