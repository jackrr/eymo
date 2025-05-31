#![warn(unused_extern_crates)]

use anyhow::{Error, Result};
use clap::Parser;
use image::{ImageReader, RgbImage};
use log::{debug, info, warn};
use num_cpus::get as get_cpu_count;
use std::sync::Arc;
use std::time::Instant;

use crate::manipulation::{Copy, Executable, Operation, OperationTree, Swap};
use crate::pipeline::Pipeline;
use crate::video::process_frames;
mod manipulation;
mod pipeline;
mod shapes;
mod video;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    image_path: Option<String>,

    #[arg(short, long)]
    output_path: Option<String>,

    #[arg(short, long)]
    max_threads: Option<usize>,
}

#[show_image::main]
fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    let total_threads = get_cpu_count();
    let total_threads = args.max_threads.unwrap_or(total_threads).min(total_threads);
    let pipeline = Pipeline::new(total_threads / 2)?;

    match args.image_path {
        Some(p) => {
            let mut img = ImageReader::open(&p)?.decode()?.into_rgb8();
            let start = Instant::now();
            let result = pipeline.run_trace(&mut img);
            debug!("{result:?}");
            debug!("Took {:?}", start.elapsed());

            let output_path: &str = &args.output_path.unwrap_or("tmp/result.png".to_string());
            img.save(output_path)?;
            info!("Result at {:?}", output_path);
            return Ok(());
        }
        None => debug!("No image specified, running in webcam mode"),
    }

    // Default mode: Webcam stream

    // TODO: allow arg to specify a video camera
    process_frames(total_threads / 2 - 1, pipeline)?;

    Ok(())
}

fn check_time(within_ms: u32, start: Instant, waypoint: &str) -> Result<()> {
    let elapsed_ms = start.elapsed().as_millis();
    // if elapsed_ms >= within_ms.into() {
    //     return Err(Error::msg(format!(
    //         "{elapsed_ms}ms exceeds allowed time of {within_ms}ms at {waypoint}",
    //     )));
    // }

    debug!("{elapsed_ms}ms at {waypoint}");

    Ok(())
}

fn process_frame(within_ms: u32, img: &mut RgbImage, pipeline: Arc<Pipeline>) -> Result<()> {
    let start = Instant::now();
    let face_detection = pipeline.run(&img)?;
    debug!("Face detection took {:?}", start.elapsed());
    check_time(within_ms, start, "Face Detection")?;

    let mut ops: Vec<OperationTree> = Vec::new();

    for face in face_detection.faces {
        let mouth = face.mouth;
        let l_eye = face.l_eye;
        let r_eye = face.r_eye;

        let swap: Operation = Swap::new(mouth.clone().into(), l_eye.into()).into();
        let copy: Operation = Copy::new(mouth.into(), r_eye.into()).into();
        ops.push(swap.into());
        ops.push(copy.into());
    }

    for op in ops {
        // TODO: refactor op list execution to operate "chunkwise",
        // allowing time to be checked here before resuming

        // TODO: timeout management from here
        // if start_at.elapsed().as_millis() >= within_ms.into() {
        //         return Err(Error::msg(format!(
        //             "process_image exceeded allowed time of {}ms",
        //             within_ms
        //         )));
        //     }

        op.execute(img)?;
        check_time(within_ms, start, "Image Manipulation")?;
    }

    Ok(())
}
