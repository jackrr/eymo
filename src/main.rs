#![warn(unused_extern_crates)]

use crate::manipulation::{Copy, Operation, OperationTree, Rotate, Scale, Swap, Tile};
use crate::pipeline::Pipeline;
use anyhow::{Error, Result};
use clap::Parser;
use image::{DynamicImage, RgbImage, RgbaImage};
use imggpu::resize::GpuExecutor;
use imggpu::rgb;
use nokhwa::pixel_format::RgbAFormat;
use nokhwa::{Buffer, FormatDecoder};
use num_cpus::get as get_cpu_count;
use std::time::Instant;
use tracing::{debug, error, info, span, trace, warn, Level};
use tracing_subscriber::fmt;
use tracing_subscriber::fmt::format::FmtSpan;
use video::{create_input_stream, OutputVideoStream};
mod imggpu;
mod manipulation;
mod pipeline;
mod shapes;
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

        let result = camera.frame();
        let frame = match result {
            Ok(frame) => frame,
            Err(e) => {
                error!("Failed to pull frame from webcam: {e:?}");
                break;
            }
        };

        match process_frame(frame, &mut gpu, &mut pipeline, args.max_frame_lag_ms) {
            Ok(img) => match output_stream.write_frame(img) {
                Ok(_) => trace!("Rendered frame."),
                Err(e) => error!("Failed to render frame: {e:?}"),
            },
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
) -> Result<RgbImage> {
    let span = span!(Level::INFO, "process_frame");
    let _guard = span.enter();
    let start = Instant::now();

    trace!(
        "Byte len: {}, res: {}, format {:?}",
        frame.buffer().len(),
        frame.resolution(),
        frame.source_frame_format()
    );
    // TODO: can we get nokwha to give us rgba byte buffer to prevent need for decoding?
    let input_img: RgbaImage = frame.decode_image::<RgbAFormat>()?;
    // let input_img = image::open("./tmp/input_img.jpg")?;
    // let input_img: RgbaImage = input_img.into();
    // DynamicImage::ImageRgba8(input_img.clone())
    //     .to_rgb8()
    //     .save("tmp/input_img.jpg")?;

    let texture =
        gpu.rgba_buffer_to_texture(input_img.as_raw(), input_img.width(), input_img.height());

    let detection = pipeline.run_gpu(&texture, gpu)?;
    // TODO: make check time return image early instead of erroring

    let mut img = rgb::texture_to_img(gpu, &texture)?;
    check_time(within_ms, start, "Face Detection")?;

    // TODO: make ops work on GPU
    // let mut ops: Vec<OperationTree> = Vec::new();

    // for face in detection.faces {
    //     let mouth = face.mouth;
    //     let l_eye = face.l_eye;
    //     let r_eye = face.r_eye;
    //     let nose = face.nose;

    //     // let copy: Operation = Copy::new(mouth.clone().into(), r_eye.into()).into();
    //     // ops.push(copy.into());
    //     // let scale: Operation = Scale::new(mouth.clone().into(), 3.).into();
    //     // ops.push(scale.into());
    //     // let swap: Operation = Swap::new(r_eye.clone().into(), l_eye.into()).into();
    //     // ops.push(swap.into());
    //     let rotate: Operation = Rotate::new(mouth.into(), 45.).into();
    //     ops.push(rotate.into());
    // }

    // for (idx, op) in ops.iter().enumerate() {
    //     // TODO: refactor op list execution to operate "chunkwise",
    //     // allowing time to be checked here before resuming
    //     check_time(within_ms, start, &format!("Image Manipulation {idx}"))?;
    //     info!("Running op {:?}", op);
    //     op.execute(gpu, &mut img)?;
    // }

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
