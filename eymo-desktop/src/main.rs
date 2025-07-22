#![warn(unused_extern_crates)]
use anyhow::{Error, Result};
use clap::{Args, Parser};
use eymo_img::imggpu::gpu::GpuExecutor;
use eymo_img::imggpu::rgb;
use eymo_img::lang;
use eymo_img::pipeline::{Detection, Pipeline};
use image::RgbaImage;
use nokhwa::pixel_format::RgbAFormat;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, error, span, trace, warn, Level};
use tracing_subscriber;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::EnvFilter;
use video::{create_input_stream, OutputVideoStream};

mod video;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CmdArgs {
    /// Max delay (ms) before timing out processing a thread
    #[arg(short = 'l', long)]
    max_frame_lag_ms: Option<u32>,

    /// Target frame rate
    #[arg(long, default_value = "30")]
    fps: u32,

    /// Config file to read from
    #[arg(
        short,
        long,
        value_name = "FILE",
        default_value = "../eymo-img/examples/rotate-face.eymo"
    )]
    config: PathBuf,

    #[command(flatten)]
    out: Out,

    /// Process single input frame, reading from input path
    #[arg(short, long, requires = "output")]
    input: Option<PathBuf>,
}

#[derive(Args, Debug)]
#[group(multiple = false)]
struct Out {
    /// Loopback device to write to. Displays in window if unset
    #[arg(group = "dest", short, long)]
    device: Option<String>,

    /// Process single input frame, writing to output path
    #[arg(group = "dest", short, long, requires = "input")]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let filter = EnvFilter::from_default_env();
    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::CLOSE)
        .with_target(false)
        .with_env_filter(filter)
        .init();

    let args = CmdArgs::parse();

    let mut pipeline = Pipeline::new()?;
    let mut gpu = GpuExecutor::new()?;
    let mut interpreter = lang::parse(&std::fs::read_to_string(args.config)?, &mut gpu)?;

    if args.out.output.is_some() {
        // Process single image at file and exit
        return process_image(
            args.input.unwrap(),
            args.out.output.unwrap(),
            &mut gpu,
            &mut pipeline,
            &mut interpreter,
            args.max_frame_lag_ms,
        );
    }

    let mut camera = create_input_stream(args.fps)?;
    let resolution = camera.resolution();
    let mut output_stream =
        OutputVideoStream::new(resolution.width(), resolution.height(), args.out.device)?;

    let mut detection_cache = None;
    loop {
        let span = span!(Level::INFO, "frame_loop_iter");
        let _guard = span.enter();

        let get_frame_span = span!(Level::DEBUG, "get_frame");
        let get_frame_guard = get_frame_span.enter();
        let result = camera.frame();
        let frame = match result {
            Ok(frame) => frame,
            Err(e) => {
                error!("Failed to pull frame from webcam: {e:?}");
                break;
            }
        };
        drop(get_frame_guard);

        // TODO: Is there a faster camera format/decode solution
        let decode_nokwha_buff_span = span!(Level::DEBUG, "decode_nokwha_buff");
        let decode_nokwha_buff_guard = decode_nokwha_buff_span.enter();
        let input_img: RgbaImage = frame.decode_image::<RgbAFormat>()?;
        drop(decode_nokwha_buff_guard);

        match process_frame(
            input_img,
            &mut gpu,
            &mut pipeline,
            &mut interpreter,
            &mut detection_cache,
            args.max_frame_lag_ms,
        ) {
            Ok(img) => {
                // ~1-2ms
                let write_frame_span = span!(Level::DEBUG, "write_frame");
                let write_frame_guard = write_frame_span.enter();
                match output_stream.write_frame(img) {
                    Ok(_) => trace!("Rendered frame."),
                    Err(e) => error!("Failed to render frame: {e:?}"),
                }
                drop(write_frame_guard);
            }
            Err(e) => error!("Failed to process frame: {e:?}"),
        }
    }

    camera.stop_stream()?;

    Ok(())
}

fn process_image(
    src: PathBuf,
    dest: PathBuf,
    gpu: &mut GpuExecutor,
    pipeline: &mut Pipeline,
    interpreter: &mut lang::Interpreter,
    within_ms: Option<u32>,
) -> Result<()> {
    let img: RgbaImage = image::open(src)?.into();
    let result = process_frame(img, gpu, pipeline, interpreter, &mut None, within_ms)?;
    result.save(dest)?;
    Ok(())
}

fn process_frame(
    input_img: RgbaImage,
    gpu: &mut GpuExecutor,
    pipeline: &mut Pipeline,
    interpreter: &mut lang::Interpreter,
    detection_cache: &mut Option<Detection>,
    within_ms: Option<u32>,
) -> Result<RgbaImage> {
    let span = span!(Level::DEBUG, "process_frame");
    let _guard = span.enter();
    let start = Instant::now();

    let texture =
        gpu.rgba_buffer_to_texture(input_img.as_raw(), input_img.width(), input_img.height());

    let mut store_detection = false;
    let detection = match detection_cache.take() {
        Some(d) => d,
        None => {
            store_detection = true;
            pipeline.run_gpu(&texture, gpu)?
        }
    };

    match check_time(within_ms, start, "Face Detection") {
        Ok(_) => {}
        Err(e) => {
            error!("{e:?}");
            return Ok(rgb::texture_to_rgba(gpu, &texture));
        }
    };

    let output = interpreter.execute(&detection, texture, gpu, |waypoint| {
        check_time(within_ms, start, waypoint)
    });

    let img = rgb::texture_to_rgba(gpu, &output);
    if store_detection {
        detection_cache.replace(detection);
    }

    Ok(img)
}

fn check_time(within_ms: Option<u32>, start: Instant, waypoint: &str) -> Result<()> {
    let elapsed_ms = start.elapsed().as_millis();
    debug!("{elapsed_ms}ms at {waypoint}");

    match within_ms {
        Some(within_ms) => {
            if elapsed_ms >= within_ms.into() {
                return Err(Error::msg(format!(
                    "{elapsed_ms}ms exceeds allowed time of {within_ms}ms at {waypoint}",
                )));
            }
        }
        None => {}
    }

    Ok(())
}
