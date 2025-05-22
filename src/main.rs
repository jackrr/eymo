#![warn(unused_extern_crates)]

use anyhow::{Error, Result};
use clap::Parser;
use image::{DynamicImage, ImageReader};
use log::{debug, info, warn};
use num_cpus::get as get_cpu_count;
use ort::session::Session;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::cv::{detect_features, initialize_model, shapes};
use crate::video::process_frames;
mod cv;
mod video;

const MODEL_YOLO_V11_POSE_M: &str = "yolo11m-pose.onnx";
const MODEL_YOLO_V11_POSE_S: &str = "yolo11s-pose.onnx";
const MODEL_YOLO_V11_POSE_N: &str = "yolo11n-pose.onnx";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    model: Option<String>,

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

    let models = HashSet::from([
        MODEL_YOLO_V11_POSE_S,
        MODEL_YOLO_V11_POSE_N,
        MODEL_YOLO_V11_POSE_M,
    ]);

    let model_name: &str = &args.model.unwrap_or(MODEL_YOLO_V11_POSE_N.to_string());
    let output_path: &str = &args.output_path.unwrap_or("tmp/result.png".to_string());

    if models.contains(model_name) {
        debug!("Using model {model_name:?}");
    } else {
        return Err(Error::msg(format!("Unrecognized model {model_name:?}")));
    }

    let model = initialize_model(model_name)?;

    match args.image_path {
        Some(p) => {
            let mut img = ImageReader::open(&p)?.decode()?;
            let result = detect_features(&model, &mut img, true)?;
            debug!("{result:?}");
            img.save(output_path)?;
            info!("Result at {:?}", output_path);
            return Ok(());
        }
        None => debug!("No image specified, running in webcam mode"),
    }

    // Default mode: Webcam stream

    // TODO: allow arg to specify a video camera
    let max_threads = get_cpu_count() / 3;
    let face_state = Arc::new(Mutex::new(FaceState {
        faces: Vec::new(),
        calced_at: Instant::now(),
        latest_frame: None,
    }));

    let model_face_state = face_state.clone();
    thread::spawn(move || -> Result<()> {
        loop {
            let mut f_state = model_face_state.lock().unwrap();

            if let Some(mut image) = f_state.latest_frame.take() {
                drop(f_state);
                debug!("Re-running face detection model...");
                let faces = detect_features(&model, &mut image, false)?;

                // write updated faces into shared state
                let mut f_state = model_face_state.lock().unwrap();
                f_state.faces = faces;
                f_state.calced_at = Instant::now();
                drop(f_state)
            } else {
                drop(f_state);
            }
            thread::sleep(Duration::from_millis(1));
        }

        Ok(())
    });

    process_frames(
        args.max_threads.unwrap_or(max_threads).min(max_threads),
        &face_state.clone(),
    )?;

    Ok(())
}

struct FaceState {
    faces: Vec<shapes::Face>,
    calced_at: Instant,
    latest_frame: Option<DynamicImage>,
}

fn process_frame(image: &mut DynamicImage, shared_state: &Arc<Mutex<FaceState>>) -> Result<()> {
    let mut face_state = shared_state.lock().unwrap();
    face_state.latest_frame = image.clone().into();
    let faces = face_state.faces.clone();
    drop(face_state); // free lock early

    debug!("Have faces {:?}", faces);
    // do stuff with faces and image here!

    Ok(())
}
