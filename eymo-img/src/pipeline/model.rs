use anyhow::Result;
use tract_onnx::prelude::*;

pub type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub fn initialize_model(model: &[u8]) -> Result<Model> {
    let mut model_dup = model;
    let model = tract_onnx::onnx()
        .model_for_read(&mut model_dup)?
        .into_optimized()?
        .into_runnable()?;

    Ok(model)
}
