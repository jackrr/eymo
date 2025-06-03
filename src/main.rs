#![warn(unused_extern_crates)]

use anyhow::{Error, Result};
use clap::Parser;
use image::{ImageReader, RgbImage};
use log::{debug, info, warn};
use num_cpus::get as get_cpu_count;
use pipeline::Detection;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::manipulation::{Copy, Executable, Operation, OperationTree, Swap, Tile};
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

    let latest_frame: Arc<Mutex<Option<RgbImage>>> = Arc::new(Mutex::new(None));
    let detection: Arc<RwLock<Option<Detection>>> = Arc::new(RwLock::new(None));

    // Dedicated thread for ML stuff
    let ml_frame = Arc::clone(&latest_frame);
    let detection_result = Arc::clone(&detection);
    let res = thread::spawn(move || loop {
        let img = {
            let mut guard = ml_frame.lock().unwrap();
            guard.take()
        };

        if let Some(img) = img {
            match pipeline.run(&img) {
                Ok(face) => {
                    let mut d = detection_result.write().unwrap();
                    *d = Some(face)
                }
                Err(e) => warn!("{e:?}"),
            }
        } else {
            thread::sleep(Duration::from_millis(1));
        }
    });

    // TODO: allow arg to specify a video camera
    process_frames(total_threads / 2 - 1, detection, latest_frame)?;

    let _ = res.join();

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

fn process_frame(
    within_ms: u32,
    img: &mut RgbImage,
    detection: Arc<RwLock<Option<Detection>>>,
) -> Result<()> {
    let start = Instant::now();
    let face_detection_lock = detection.read().unwrap();

    let face_detection = match *face_detection_lock {
        Some(ref fd) => fd.clone(),
        None => {
            return Err(Error::msg("No face detection, skipping frame processing"));
        }
    };
    drop(face_detection_lock);

    check_time(within_ms, start, "Face Detection")?;

    let mut ops: Vec<OperationTree> = Vec::new();

    // for face in face_detection.faces {
    //     let mouth = face.mouth;
    //     let l_eye = face.l_eye;
    //     let r_eye = face.r_eye;

    //     let swap: Operation = Swap::new(mouth.clone().into(), l_eye.into()).into();
    //     let copy: Operation = Copy::new(mouth.into(), r_eye.into()).into();
    //     ops.push(swap.into());
    //     ops.push(copy.into());

    //     ops
    // }

    if face_detection.faces.len() > 0 {
        let face = face_detection.faces[0].clone();
        let tile: Operation = Tile::new(face.mouth.into(), 1.).into();
        ops.push(tile.into());
    }

    for (idx, op) in ops.iter().enumerate() {
        // TODO: refactor op list execution to operate "chunkwise",
        // allowing time to be checked here before resuming
        check_time(within_ms, start, &format!("Image Manipulation {idx}"))?;
        op.execute(img)?;
    }

    Ok(())
}
