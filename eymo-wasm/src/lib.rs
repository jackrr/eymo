// What follows liberally draws from:
// https://sotrh.github.io/learn-wgpu/beginner/tutorial1-window/#the-code

mod img;
mod util;

use eymo_img::imggpu::gpu::GpuExecutor;
use eymo_img::lang;
use eymo_img::pipeline::Pipeline;
use tracing::{warn, debug, error, info};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::*;

#[wasm_bindgen]
pub struct State {
    interpreter: lang::Interpreter,
    gpu: GpuExecutor,
    pipeline: Pipeline,
    // canvas: web_sys::HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

#[wasm_bindgen(start)]
fn main() -> Result<(), JsValue> {
    tracing_wasm::set_as_global_default_with_config(tracing_wasm::WASMLayerConfigBuilder::default().set_max_level(tracing::Level::INFO).build());
    debug!("Loaded! Setting panic hook...");
    util::set_panic_hook();
    Ok(())
}

#[wasm_bindgen]
impl State {
    async fn new_anyhow() -> anyhow::Result<Self> {
        // TODO: better error handling
        debug!("Loading gpu executor...");
        let browser_window = wgpu::web_sys::window().unwrap_throw();
        let document = browser_window.document().unwrap_throw();
        let canvas = document.get_element_by_id("canvas").unwrap_throw();
        let html_canvas_element = canvas.unchecked_into();
        let (mut gpu, surface, config) = GpuExecutor::new_wasm(html_canvas_element).await?;

        // TODO: detect canvas resize and call configure
        // NOTE: to resize canvas, just call configure again with updated width + height on config
        surface.configure(&gpu.device, &config);

        // TODO: don't hardcode command
        debug!("Loading image transformer...");
        let command = "mouth: swap_with(leye_region), scale(2)\n";
        let interpreter = lang::parse(command, &mut gpu)?;

        debug!("Loading detection pipeline...");
        let pipeline = Pipeline::new()?;

        debug!("State init complete!");
        Ok(Self {
            interpreter,
            gpu,
            pipeline,
            surface,
            config,
        })
    }

    #[wasm_bindgen(constructor)]
    pub async fn new() -> Result<Self, JsValue> {
        debug!("Constructor for State...");
        wrap_err(Self::new_anyhow().await)
    }

    #[wasm_bindgen]
    pub async fn start(&mut self) -> Result<(), JsValue> {
        debug!("Loading camera...");
        let browser_window = wgpu::web_sys::window().unwrap_throw();
        let devices = browser_window.navigator().media_devices().unwrap_throw();
        let constraints = MediaStreamConstraints::new();
        constraints.set_video(&js_sys::Boolean::from(true));
        let stream = JsFuture::from(devices.get_user_media_with_constraints(&constraints).unwrap_throw()).await.unwrap().unchecked_into::<MediaStream>();
        let vid = stream.get_video_tracks().get(0).unchecked_into::<MediaStreamTrack>();
        let proc = MediaStreamTrackProcessor::new(&MediaStreamTrackProcessorInit::new(&vid))?;
        let reader = proc.readable().get_reader().unchecked_into::<ReadableStreamDefaultReader>();

        loop {
            match JsFuture::from(reader.read()).await {
                Ok(js_frame) => {
                    let video_frame =
                        js_sys::Reflect::get(&js_frame, &js_sys::JsString::from("value"))
                            .unwrap()
                            .unchecked_into::<VideoFrame>();
                    info!("{}x{}", video_frame.coded_width(), video_frame.coded_height());
                    let img = img::from_frame(&video_frame).await?;
                    match self.process_frame(img).await {
                        Ok(_) => {}
                        Err(e) => {
                            error!("Failed to process frame: {}", e.to_string());
                        }
                    }
                    video_frame.close();
                },
                Err(e) => {
                    error!("Failed to read frame: {e:?}");
                }
            }
            
        }

        Ok(())
    }

    async fn process_frame(&mut self, input_image: image::RgbaImage) -> anyhow::Result<()> {
        let input = self.gpu.rgba_buffer_to_texture(
            input_image.as_raw(),
            input_image.width(),
            input_image.height(),
        );
        debug!("Running detection..");
        let detection = self.pipeline.run_gpu(&input, &mut self.gpu).await?;

        debug!("Running transforms..");
        let result = self
            .interpreter
            .execute(&detection, input, &mut self.gpu, |_| Ok(()));

        debug!("Copying result to 'surface'...");
        let output = self.surface.get_current_texture()?;

        // TODO: scale texture to canvas dimensions (goal: result.size == output.size)

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

        debug!("Presenting!...");
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
