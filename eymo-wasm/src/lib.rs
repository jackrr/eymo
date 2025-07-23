// What follows liberally draws from:
// https://sotrh.github.io/learn-wgpu/beginner/tutorial1-window/#the-code

mod util;

use eymo_img::imggpu::gpu::GpuExecutor;
use eymo_img::lang;
use eymo_img::pipeline::Pipeline;
use log::{error, info};
use nokhwa::pixel_format::RgbAFormat;
use nokhwa::utils::{RequestedFormat, RequestedFormatType};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct State {
    interpreter: lang::Interpreter,
    gpu: GpuExecutor,
    pipeline: Pipeline,
    // canvas: web_sys::HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    camera: nokhwa::Camera,
}

#[wasm_bindgen(start)]
fn main() -> Result<(), JsValue> {
    info!("Loaded! Setting panic hook...");
    console_log::init_with_level(log::Level::Info).unwrap_throw();
    util::set_panic_hook();
    Ok(())
}

#[wasm_bindgen]
impl State {
    fn new_anyhow() -> anyhow::Result<Self> {
        let browser_window = wgpu::web_sys::window().unwrap_throw();
        let document = browser_window.document().unwrap_throw();
        let canvas = document.get_element_by_id("canvas").unwrap_throw();
        let html_canvas_element = canvas.unchecked_into();

        let command = "mouth: swap_with(leye_region), scale(2)";
        let (mut gpu, surface, config) = GpuExecutor::new_wasm(html_canvas_element)?;

        // NOTE: to resize canvas, just call configure again with updated width + height on config
        surface.configure(&gpu.device, &config);

        let interpreter = lang::parse(command, &mut gpu)?;
        let pipeline = Pipeline::new()?;

        let cameras = nokhwa::query(nokhwa::utils::ApiBackend::Browser).unwrap();

        let mut camera = nokhwa::Camera::new(
            cameras.last().unwrap().index().clone(),
            RequestedFormat::new::<RgbAFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
        )?;
        camera.set_frame_rate(30)?;
        camera.open_stream().unwrap();

        Ok(Self {
            interpreter,
            gpu,
            pipeline,
            surface,
            config,
            camera,
        })
    }

    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<Self, JsValue> {
        wrap_err(Self::new_anyhow())
    }

    #[wasm_bindgen]
    pub fn start(&mut self) -> Result<(), JsValue> {
        loop {
            match self.process_frame() {
                Ok(_) => {}
                Err(e) => {
                    error!("Failed to process frame: {}", e.to_string());
                }
            }
        }

        Ok(())
    }

    fn process_frame(&mut self) -> anyhow::Result<()> {
        let frame = self.camera.frame()?;
        let input_image = frame.decode_image::<RgbAFormat>()?;
        let input = self.gpu.rgba_buffer_to_texture(
            input_image.as_raw(),
            input_image.width(),
            input_image.height(),
        );
        let detection = self.pipeline.run_gpu(&input, &mut self.gpu)?;

        let result = self
            .interpreter
            .execute(&detection, input, &mut self.gpu, |_| Ok(()));

        let output = self.surface.get_current_texture()?;

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });
        // TODO: likely panics due to size mismatch. ensure targets canvas size properly
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                texture: &result,
            },
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                texture: &output.texture,
            },
            output.texture.size(),
        );

        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        output.present();

        Ok(())
    }

    // TODO: add a function to specify transforms string
    // TODO: add a function to register a target dom element to display video
    // TODO: add a function to stop video
}

fn wrap_err<T>(r: anyhow::Result<T>) -> Result<T, JsValue> {
    match r {
        Ok(t) => Ok(t),
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}
