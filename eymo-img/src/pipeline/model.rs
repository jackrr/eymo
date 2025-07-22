use anyhow::Result;
use tract_onnx::prelude::*;

pub type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub fn initialize_model(filename: &str, threads: usize) -> Result<Model> {
    let model = tract_onnx::onnx()
        .model_for_path(format!(
            // TODO: package model into binary
            "/home/jack/projects/eymo/eymo-img/models/{filename}"
        ))?
        .into_optimized()?
        .into_runnable()?;

    Ok(model)
}
