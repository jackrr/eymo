#![warn(unused_extern_crates)]

use anyhow::{Error, Result};
use clap::Parser;
use image::{DynamicImage, GenericImage, GenericImageView, ImageReader};
use log::{debug, info, warn};
use num_cpus::get as get_cpu_count;
use std::collections::HashSet;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::cv::shapes::{Face, FaceFeatureKind, Rect};
use crate::cv::{detect_features, initialize_model};
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
            let start = Instant::now();
            let result = detect_features(&model, &mut img, true)?;
            debug!("{result:?}");
            debug!("Took {:?}", start.elapsed());
            img.save(output_path)?;
            info!("Result at {:?}", output_path);
            return Ok(());
        }
        None => debug!("No image specified, running in webcam mode"),
    }

    // Default mode: Webcam stream

    // TODO: allow arg to specify a video camera

    let face_detection = Arc::new(RwLock::new(FaceDetection {
        faces: Vec::new(),
        calced_at: Instant::now(),
    }));

    let latest_img: Arc<Mutex<DetectionInput>> = Arc::new(Mutex::new(None));

    let model_output = face_detection.clone();
    let model_input = latest_img.clone();
    thread::spawn(move || -> Result<()> {
        loop {
            let mut model_input = model_input.lock().unwrap();

            if let Some(mut image) = model_input.take() {
                drop(model_input);
                debug!("Re-running face detection model...");
                let faces = detect_features(&model, &mut image, false)?;

                // write updated faces into shared state
                let mut model_output = model_output.write().unwrap();
                model_output.faces = faces;
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

        Ok(())
    });

    // let max_threads = get_cpu_count() / 3;
    let max_threads = 2;
    process_frames(
        args.max_threads.unwrap_or(max_threads).min(max_threads),
        &face_detection.clone(),
        &latest_img.clone(),
    )?;

    Ok(())
}

struct FaceDetection {
    faces: Vec<Face>,
    calced_at: Instant,
}

type DetectionInput = Option<DynamicImage>;

fn process_frame(
    within_ms: u32,
    img: &mut DynamicImage,
    face_detection: &Arc<RwLock<FaceDetection>>,
) -> Result<()> {
    let start = Instant::now();
    let face_detection = face_detection.read().unwrap();
    let faces = face_detection.faces.clone();
    drop(face_detection); // free lock early

    // Do this before every time consuming operation
    if start.elapsed().as_millis() >= within_ms.into() {
        return Ok(());
    }

    // swap mouth and eyes
    for face in faces {
        let mut mouth: Option<Rect> = None;
        let mut l_eye: Option<Rect> = None;
        let mut r_eye: Option<Rect> = None;
        for f in face.features {
            match f.kind {
                FaceFeatureKind::Mouth => mouth = f.bounds.clone().into(),
                FaceFeatureKind::LeftEye => l_eye = f.bounds.clone().into(),
                FaceFeatureKind::RightEye => r_eye = f.bounds.clone().into(),
                default => {}
            }

            let mouth = mouth.unwrap();
            let l_eye = l_eye.unwrap();
            let r_eye = r_eye.unwrap();

            // mouth to leye
            let mouth_view = *img.view(
                mouth.left().try_into().unwrap(),
                mouth.top().try_into().unwrap(),
                mouth.width,
                mouth.height,
            );
            let mut m_img = image::RgbaImage::new(mouth.width, mouth.height);
            m_img.copy_from(&mouth_view, 0, 0)?;

            img.copy_within(
                l_eye.into(),
                mouth.left().try_into().unwrap(),
                mouth.top().try_into().unwrap(),
            );
            img.copy_from(&m_img, l_eye.left(), l_eye.top())?;
            img.copy_from(&m_img, r_eye.left(), r_eye.top())?;

            if start.elapsed().as_millis() >= within_ms.into() {
                return Err(Error::msg(format!(
                    "process_image exceeded allowed time of {}ms",
                    within_ms
                )));
            }
        }
    }

    Ok(())
}
