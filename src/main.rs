#![warn(unused_extern_crates)]

use anyhow::{Error, Result};
use clap::Parser;
use image::{ImageReader, RgbImage};
use imggpu::resize::GpuExecutor;
use num_cpus::get as get_cpu_count;
use pipeline::Detection;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, span, warn, Level};
use tracing_subscriber::fmt;
use tracing_subscriber::fmt::format::FmtSpan;
use video::create_input_stream;

use crate::manipulation::{Copy, Operation, OperationTree, Rotate, Scale, Swap, Tile};
use crate::pipeline::Pipeline;
use crate::video::process_frames;
mod imggpu;
mod manipulation;
mod pipeline;
mod shapes;
mod video;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    max_threads: Option<usize>,

    #[arg(short, long, default = 500)]
    max_frame_lag_ms: u32,
}

fn main() -> Result<()> {
    fmt::fmt()
        .with_span_events(FmtSpan::CLOSE)
        .with_target(false)
        .with_level(false)
        .init();

    let args = Args::parse();

    let total_threads = get_cpu_count();
    let total_threads = args.max_threads.unwrap_or(total_threads).min(total_threads);
    let mut pipeline = Pipeline::new(total_threads / 2)?;

    // TODO: allow arg to specify a video camera, fps
    let camera = create_input_stream(30);

    let resolution = camera.resolution()?;
    let mut output_stream = OutputVideoStream::new(resolution.width(), resolution.height())?;
    let mut gpu = GpuExecutor::new()?;

    while let result = camera.frame_texture(&gpu.device, &gpu.queue, Some("frame")) {
        let texture = match result {
            Ok(texture) => texture,
            Err(e) => {
                error!("Failed to pull frame from webcam: {e:?}");
                continue;
            }
        };
        let frame = process_frame(texture, &mut gpu, &mut pipeline, rgs.max_frame_lag_ms);

        output_stream.write_frame(&frame);
    }

    camera.stop_stream()?;
    output_stream.close()?;

    Ok(())
}

fn process_frame(
    tex: wgpu::Texture,
    gpu: &mut GpuExecutor,
    pipeline: &mut Pipeline,
    within_ms: u32,
) -> Result<()> {
    let span = span!(Level::INFO, "process_frame");
    let _guard = span.enter();
    let start = Instant::now();

    // TODO: pipeline take texture as input
    let detection = pipeline.run(&texture);

    let face_detection = match pipeline.run(&mut tex) {
        Ok(f) => f,
        Err(e) => return Err(Error::msg("Face detection failed: {e:?}")),
    };

    check_time(within_ms, start, "Face Detection")?;

    let mut ops: Vec<OperationTree> = Vec::new();

    for face in face_detection.faces {
        let mouth = face.mouth;
        let l_eye = face.l_eye;
        let r_eye = face.r_eye;
        let nose = face.nose;

        let copy: Operation = Copy::new(mouth.clone().into(), r_eye.into()).into();
        ops.push(copy.into());
        // let scale: Operation = Scale::new(mouth.clone().into(), 3.).into();
        // ops.push(scale.into());
        // let swap: Operation = Swap::new(r_eye.clone().into(), l_eye.into()).into();
        // ops.push(swap.into());
        let rotate: Operation = Rotate::new(mouth.into(), 45.).into();
        ops.push(rotate.into());
    }

    for (idx, op) in ops.iter().enumerate() {
        // TODO: refactor op list execution to operate "chunkwise",
        // allowing time to be checked here before resuming
        check_time(within_ms, start, &format!("Image Manipulation {idx}"))?;
        op.execute(gpu, img)?;
    }

    Ok(())
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
