#![warn(unused_extern_crates)]
use crate::imggpu::gpu::GpuExecutor;
use crate::pipeline::Pipeline;
use anyhow::{Error, Result};
use clap::Parser;
use image::RgbaImage;
use imggpu::rgb;
use nokhwa::pixel_format::RgbAFormat;
use nokhwa::Buffer;
use num_cpus::get as get_cpu_count;
use std::time::Instant;
use tracing::{debug, error, span, trace, warn, Level};
use tracing_subscriber::fmt;
use tracing_subscriber::fmt::format::FmtSpan;
use transform::Transform;
use video::{create_input_stream, OutputVideoStream};
mod imggpu;
mod pipeline;
mod shapes;
mod transform;
mod triangulate;
mod video;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    threads: Option<usize>,

    #[arg(short, long, default_value = "500")]
    max_frame_lag_ms: u32,

    #[arg(short, long, default_value = "30")]
    fps: u32,
}

fn main() -> Result<()> {
    fmt::fmt()
        .with_span_events(FmtSpan::CLOSE)
        .with_target(false)
        .init();

    let args = Args::parse();

    let total_threads = get_cpu_count();
    let total_threads = args.threads.unwrap_or(total_threads).min(total_threads);
    let mut pipeline = Pipeline::new(total_threads / 2)?;

    // TODO: allow arg to specify a video source
    let mut camera = create_input_stream(args.fps)?;

    let resolution = camera.resolution();
    let mut output_stream = OutputVideoStream::new(resolution.width(), resolution.height())?;
    let mut gpu = GpuExecutor::new()?;

    loop {
        let span = span!(Level::INFO, "frame_loop_iter");
        let _guard = span.enter();

        let get_frame_span = span!(Level::INFO, "get_frame");
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

        match process_frame(frame, &mut gpu, &mut pipeline, args.max_frame_lag_ms) {
            Ok(img) => {
                // ~1-2ms
                let write_frame_span = span!(Level::INFO, "write_frame");
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

    output_stream.close()?;
    camera.stop_stream()?;

    Ok(())
}

fn process_frame(
    frame: Buffer,
    gpu: &mut GpuExecutor,
    pipeline: &mut Pipeline,
    within_ms: u32,
) -> Result<RgbaImage> {
    let span = span!(Level::INFO, "process_frame");
    let _guard = span.enter();
    let start = Instant::now();

    // WOAH: 24ms
    // TODO: Is there a faster camera format/decode solution
    let decode_nokwha_buff_span = span!(Level::INFO, "decode_nokwha_buff");
    let decode_nokwha_buff_guard = decode_nokwha_buff_span.enter();
    let input_img: RgbaImage = frame.decode_image::<RgbAFormat>()?;

    let texture =
        gpu.rgba_buffer_to_texture(input_img.as_raw(), input_img.width(), input_img.height());
    drop(decode_nokwha_buff_guard);

    let detection = pipeline.run_gpu(&texture, gpu)?;

    // TODO: make check time return image early instead of erroring
    check_time(within_ms, start, "Face Detection")?;

    let mut output = texture;

    for face in detection.faces {
        trace!("Handling face {:?}", face);

        let mut t = Transform::new(face.l_eye.clone());
        t.set_translation(100, -80);
        t.set_scale(2.);

        output = t.execute(gpu, &output)?;

        let mut t = Transform::new(face.mouth.clone());
        t.set_scale(1.3);
        t.copy_to([face.r_eye_region.into()], false);
        t.swap_with(face.l_eye_region.into());
        output = t.execute(gpu, &output)?;

        check_time(within_ms, start, &format!("Image Manipulation TODO: index"))?;
    }

    let img = rgb::texture_to_rgba(gpu, &output);

    Ok(img)
}

fn check_time(within_ms: u32, start: Instant, waypoint: &str) -> Result<()> {
    let elapsed_ms = start.elapsed().as_millis();
    if elapsed_ms >= within_ms.into() {
        return Err(Error::msg(format!(
            "{elapsed_ms}ms exceeds allowed time of {within_ms}ms at {waypoint}",
        )));
    }

    debug!("{elapsed_ms}ms at {waypoint}");

    Ok(())
}
