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

    let face_detection = Arc::new(RwLock::new(DetectionResult {
        detection: None,
        calced_at: Instant::now(),
    }));

    let latest_img: Arc<Mutex<DetectionInput>> = Arc::new(Mutex::new(None));

    let model_output = face_detection.clone();
    let model_input = latest_img.clone();
    // TODO: try doing model inference as part of image processing fn
    // and profile performance
    let model_res = thread::spawn(move || -> Result<()> {
        loop {
            let mut model_input = model_input.lock().unwrap();

            if let Some(image) = model_input.take() {
                drop(model_input);
                debug!("Re-running face detection model...");
                let detection = pipeline.run(&image)?;

                // write updated faces into shared state
                let mut model_output = model_output.write().unwrap();
                model_output.detection = detection.into();
                debug!(
                    "{:?} elapsed since last face detection run",
                    model_output.calced_at.elapsed()
                );
                model_output.calced_at = Instant::now();
                drop(model_output)
            } else {
                drop(model_input);
            }
            thread::sleep(Duration::from_millis(1));
        }
    });

    process_frames(
        total_threads / 2,
        face_detection.clone(),
        latest_img.clone(),
    )?;

    model_res.join();

    Ok(())
}

struct DetectionResult {
    detection: Option<Detection>,
    calced_at: Instant,
}

type DetectionInput = Option<RgbImage>;

fn process_frame(
    within_ms: u32,
    img: &mut RgbImage,
    face_detection: &Arc<RwLock<DetectionResult>>,
) -> Result<()> {
    let start = Instant::now();
    let face_detection = face_detection.read().unwrap();

    let faces = match &face_detection.detection {
        Some(d) => d.faces.clone(),
        None => {
            debug!("No faces detected.");
            return Ok(());
        }
    };

    drop(face_detection); // free lock early
    let mut ops: Vec<OperationTree> = Vec::new();

    for face in faces {
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
        if start.elapsed().as_millis() >= within_ms.into() {
            return Err(Error::msg(format!(
                "process_image exceeded allowed time of {}ms",
                within_ms
            )));
        }
    }

    Ok(())
}
