mod util;

use eymo_img::imggpu::gpu::GpuExecutor;
use eymo_img::lang;
use eymo_img::pipeline::Pipeline;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct ImgProcessor {
    interpreter: lang::Interpreter,
    gpu: GpuExecutor,
    pipeline: Pipeline,
}

#[wasm_bindgen]
impl ImgProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(command: &str) -> Result<ImgProcessor, JsValue> {
        let mut gpu = anyhow_result_to_js_result(GpuExecutor::new())?;
        let interpreter = anyhow_result_to_js_result(lang::parse(command, &mut gpu))?;
        let pipeline = anyhow_result_to_js_result(Pipeline::new(1))?;

        Ok(Self {
            interpreter,
            gpu,
            pipeline,
        })
    }

    // TODO: add a function to specify config
    // TODO: add a function to register a target dom element to display video
    // TODO: add a function to start video
    // TODO: add a function to stop video
}

fn anyhow_result_to_js_result<T>(r: anyhow::Result<T>) -> Result<T, JsValue> {
    match r {
        Ok(t) => Ok(t),
        Err(e) => Err(JsValue::from_str(&format!("{e:?}"))),
    }
}
