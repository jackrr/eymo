mod img;
mod util;

use eymo_img::imggpu::gpu::GpuExecutor;
use eymo_img::imggpu::resize::resize_texture;
use eymo_img::lang;
use eymo_img::pipeline::{Detection, Pipeline};
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tracing::{Level, debug, error, info, span, trace, warn};
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
    stop_rx: Option<oneshot::Receiver<()>>,
    resize_rx: mpsc::Receiver<()>,
}

#[wasm_bindgen(start)]
fn main() -> Result<(), JsValue> {
    tracing_wasm::set_as_global_default_with_config(
        tracing_wasm::WASMLayerConfigBuilder::default()
            .set_max_level(Level::DEBUG)
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

        let inner_state = Mutex::new(InnerState {
            interpreter,
            gpu,
            pipeline,
            surface,
            config,
            canvas,
            detection_cache: None,
            stop_tx: None,
            stop_rx: None,
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
        debug!("Stopping...");

        let mut is = self.inner_state.lock().await;
        let stop_tx = is.stop_tx.take();
        drop(is);

        match stop_tx {
            Some(stop_tx) => stop_tx.send(()).unwrap_throw(),
            None => warn!("Cannot stop when already stopped"),
        };

        Ok(())
    }

    async fn process_with_reader(
        &self,
        reader: ReadableStreamDefaultReader,
    ) -> Result<(), JsValue> {
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
                    if is.stop_rx.as_ref().is_some_and(|rx| !rx.is_empty()) {
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

    // FIXME: video is frozen on first load
    async fn process_with_video_element(&self, stream: MediaStream) -> Result<(), JsValue> {
        let browser_window = wgpu::web_sys::window().unwrap_throw();
        let document = browser_window.document().unwrap_throw();

        // Create a video element to capture frames
        let video = document
            .create_element("video")?
            .unchecked_into::<HtmlVideoElement>();
        video.set_src_object(Some(&stream));
        video.set_autoplay(true);
        video.set_muted(true);

        // Wait for video to be ready
        let video_ready = js_sys::Promise::new(&mut |resolve, _| {
            let video_clone = video.clone();
            let onloadeddata: Closure<dyn FnMut()> = Closure::new(move || {
                resolve.call0(&JsValue::NULL).unwrap_throw();
            });
            video_clone.set_onloadeddata(Some(onloadeddata.as_ref().unchecked_ref()));
            onloadeddata.forget();
        });
        JsFuture::from(video_ready).await?;
        debug!("Video ready.");

        // Create canvas for frame capture
        let capture_canvas = document
            .create_element("canvas")?
            .unchecked_into::<HtmlCanvasElement>();

        let canvas_ctx_opts = js_sys::Object::new();
        js_sys::Reflect::set(&canvas_ctx_opts, &"willReadFrequently".into(), &true.into())?;

        let canvas_ctx = capture_canvas
            .get_context_with_context_options("2d", &canvas_ctx_opts)?
            .unwrap()
            .unchecked_into::<CanvasRenderingContext2d>();

        'frame_loop: loop {
            let mut is = self.inner_state.lock().await;
            if is.stop_rx.as_ref().is_some_and(|rx| !rx.is_empty()) {
                break 'frame_loop;
            }

            // Capture frame from video
            let video_width = video.video_width();
            let video_height = video.video_height();

            capture_canvas.set_width(video_width);
            capture_canvas.set_height(video_height);

            canvas_ctx.draw_image_with_html_video_element(&video, 0.0, 0.0)?;

            let image_data =
                canvas_ctx.get_image_data(0.0, 0.0, video_width as f64, video_height as f64)?;
            let img =
                image::RgbaImage::from_raw(video_width, video_height, image_data.data().to_vec())
                    .ok_or_else(|| JsValue::from_str("Failed to create image from canvas data"))?;

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

            debug!("Frame processed, doing delay thing...");

            // Small delay to prevent overwhelming the browser
            let delay = js_sys::Promise::new(&mut |resolve, _| {
                let window = web_sys::window().unwrap();
                let after_timeout: Closure<dyn FnMut()> =
                    wasm_bindgen::closure::Closure::new(move || {
                        debug!("Calling resolve from within closure");
                        resolve.call0(&JsValue::NULL).unwrap_throw();
                    });
                window
                    .set_timeout_with_callback_and_timeout_and_arguments_0(
                        after_timeout.as_ref().unchecked_ref(),
                        16, // ~60fps
                    )
                    .unwrap();
                after_timeout.forget();
            });
            JsFuture::from(delay).await?;
        }

        Ok(())
    }

    #[wasm_bindgen]
    pub async fn start(&self) -> Result<(), JsValue> {
        debug!("Starting...");

        // Re-init termination channel
        let mut is = self.inner_state.lock().await;
        let prev = is.stop_tx.as_ref();
        if prev.is_some() && !prev.unwrap().is_closed() {
            return Err(JsValue::from_str(
                "Cannot start when already running. Invoke .stop on instance before calling start again.",
            ));
        }

        let (stop_tx, stop_rx) = oneshot::channel();
        is.stop_tx = Some(stop_tx);
        is.stop_rx = Some(stop_rx);
        drop(is);

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

        // Check if MediaStreamTrackProcessor is available
        let has_processor = js_sys::Reflect::has(
            &js_sys::global(),
            &js_sys::JsString::from("MediaStreamTrackProcessor"),
        )
        .unwrap_or(false);

        if has_processor {
            debug!("Using MediaStreamTrackProcessor");
            let proc = MediaStreamTrackProcessor::new(&MediaStreamTrackProcessorInit::new(&vid))?;
            let reader = proc
                .readable()
                .get_reader()
                .unchecked_into::<ReadableStreamDefaultReader>();

            self.process_with_reader(reader).await?;
        } else {
            debug!("MediaStreamTrackProcessor not available, using video element fallback");
            self.process_with_video_element(stream).await?;
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
