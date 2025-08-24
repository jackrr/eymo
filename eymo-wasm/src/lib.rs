mod img;
mod util;

use eymo_img::imggpu::gpu::GpuExecutor;
use eymo_img::imggpu::resize::resize_texture;
use eymo_img::lang;
use eymo_img::pipeline::{Detection, Pipeline};
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tracing::{Level, debug, error, info, span, trace};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::*;

#[wasm_bindgen]
pub struct State {
    inner_state: Mutex<InnerState>,
}

struct InnerState {
    interpreter: lang::Interpreter,
    gpu: GpuExecutor,
    pipeline: Pipeline,
    canvas: web_sys::HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    detection_cache: Option<Detection>,
    stop_tx: Option<oneshot::Sender<()>>,
    stop_rx: oneshot::Receiver<()>,
    resize_rx: mpsc::Receiver<()>,
}

#[wasm_bindgen(start)]
fn main() -> Result<(), JsValue> {
    tracing_wasm::set_as_global_default_with_config(
        tracing_wasm::WASMLayerConfigBuilder::default()
            .set_max_level(Level::WARN)
            .build(),
    );
    debug!("Loaded! Setting panic hook...");
    util::set_panic_hook();
    Ok(())
}

impl InnerState {
    async fn process_frame(&mut self, input_image: image::RgbaImage) -> anyhow::Result<()> {
        let span = span!(Level::DEBUG, "process_frame");
        let _guard = span.enter();
        let input = self.gpu.rgba_buffer_to_texture(
            input_image.as_raw(),
            input_image.width(),
            input_image.height(),
        );

        let input = resize_texture(&mut self.gpu, &input, self.config.width, self.config.height)?;

        let mut replace_detection = false;

        let detection = match self.detection_cache.take() {
            Some(detection) => detection,
            None => {
                debug!("Running detection..");
                replace_detection = true;
                self.pipeline.run_gpu(&input, &mut self.gpu).await?
            }
        };

        debug!("Running transforms..");
        let result = self
            .interpreter
            .execute(&detection, input, &mut self.gpu, |_| Ok(()));

        if replace_detection {
            self.detection_cache.replace(detection);
        }

        debug!("Copying result to 'surface'...");
        let output = self.surface.get_current_texture()?;

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

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
}

#[wasm_bindgen]
impl State {
    async fn new_anyhow(canvas_id: &str, cmd: &str) -> anyhow::Result<Self> {
        // TODO: better error handling
        debug!("Loading gpu executor...");
        let browser_window = wgpu::web_sys::window().unwrap_throw();
        let document = browser_window.document().unwrap_throw();
        let canvas = document.get_element_by_id(canvas_id).unwrap_throw();
        let canvas: web_sys::HtmlCanvasElement = canvas
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .map_err(|_| ())
            .unwrap();
        let (mut gpu, surface, config) = GpuExecutor::new_wasm(canvas.clone()).await?;

        surface.configure(&gpu.device, &config);

        debug!("Loading image transformer...");
        let interpreter = lang::parse(cmd, &mut gpu)?;

        debug!("Loading detection pipeline...");
        let pipeline = Pipeline::new()?;

        let (resize_tx, resize_rx) = mpsc::channel(1);
        let c: Closure<dyn FnMut()> = wasm_bindgen::closure::Closure::new(move || {
            match resize_tx.blocking_send(()) {
                Ok(_) => {
                    debug!("Sent resize");
                }
                Err(e) => {
                    error!("Failed to send resize: {e:?}");
                }
            };
        });

        let resize_fn = c.as_ref().unchecked_ref();
        let observer = web_sys::ResizeObserver::new(resize_fn).unwrap_throw();
        c.forget();

        observer.observe(&canvas);

        let (stop_tx, stop_rx) = oneshot::channel();
        let inner_state = Mutex::new(InnerState {
            interpreter,
            gpu,
            pipeline,
            surface,
            config,
            canvas,
            detection_cache: None,
            stop_tx: Some(stop_tx),
            stop_rx,
            resize_rx,
        });

        debug!("State init complete!");

        Ok(Self { inner_state })
    }

    #[wasm_bindgen(constructor)]
    pub async fn new(canvas_id: &str, cmd: &str) -> Result<Self, JsValue> {
        debug!("Constructor for State...");
        wrap_err(Self::new_anyhow(canvas_id, cmd).await)
    }

    #[wasm_bindgen]
    pub async fn set_cmd(&self, cmd: &str) -> Result<(), JsValue> {
        debug!("Setting command to {cmd}");

        let mut s = self.inner_state.lock().await;
        let next_interpreter = wrap_err(lang::parse(cmd, &mut s.gpu))?;
        s.interpreter = next_interpreter;
        Ok(())
    }

    #[wasm_bindgen]
    pub async fn stop(&self) -> Result<(), JsValue> {
        let mut is = self.inner_state.lock().await;
        let stop_tx = is.stop_tx.take();
        stop_tx.unwrap().send(()).unwrap_throw();

        Ok(())
    }

    #[wasm_bindgen]
    pub async fn start(&self) -> Result<(), JsValue> {
        debug!("Loading camera...");
        let browser_window = wgpu::web_sys::window().unwrap_throw();
        let devices = browser_window.navigator().media_devices().unwrap_throw();
        let constraints = MediaStreamConstraints::new();
        constraints.set_video(&js_sys::Boolean::from(true));
        let stream = JsFuture::from(
            devices
                .get_user_media_with_constraints(&constraints)
                .unwrap_throw(),
        )
        .await
        .unwrap()
        .unchecked_into::<MediaStream>();
        let vid = stream
            .get_video_tracks()
            .get(0)
            .unchecked_into::<MediaStreamTrack>();
        let proc = MediaStreamTrackProcessor::new(&MediaStreamTrackProcessorInit::new(&vid))?;
        let reader = proc
            .readable()
            .get_reader()
            .unchecked_into::<ReadableStreamDefaultReader>();

        // Re-init termination channel
        let mut is = self.inner_state.lock().await;
        let (stop_tx, stop_rx) = oneshot::channel();
        is.stop_tx = Some(stop_tx);
        is.stop_rx = stop_rx;
        drop(is);

        'frame_loop: loop {
            match JsFuture::from(reader.read()).await {
                Ok(js_frame) => {
                    let video_frame =
                        js_sys::Reflect::get(&js_frame, &js_sys::JsString::from("value"))
                            .unwrap()
                            .unchecked_into::<VideoFrame>();
                    info!(
                        "Got frame with dims {}x{}",
                        video_frame.coded_width(),
                        video_frame.coded_height()
                    );
                    let img = img::from_frame(&video_frame).await?;

                    let mut is = self.inner_state.lock().await;
                    if !is.stop_rx.is_empty() {
                        // terminate message sent
                        video_frame.close();
                        break 'frame_loop;
                    }

                    match is.resize_rx.try_recv() {
                        Ok(_) => {
                            debug!("Resizing!");
                            is.config.width = is.canvas.width();
                            is.config.height = is.canvas.height();
                            is.surface.configure(&is.gpu.device, &is.config);
                        }
                        Err(_) => {
                            trace!("No resize queued, continuing...");
                        }
                    }

                    match is.process_frame(img).await {
                        Ok(_) => {}
                        Err(e) => {
                            error!("Failed to process frame: {}", e.to_string());
                        }
                    }
                    drop(is);
                    video_frame.close();
                }
                Err(e) => {
                    error!("Failed to read frame: {e:?}");
                }
            }
        }

        Ok(())
    }
}

fn wrap_err<T>(r: anyhow::Result<T>) -> Result<T, JsValue> {
    match r {
        Ok(t) => Ok(t),
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}
