#![warn(unused_extern_crates)]

use anyhow::{Error, Result};
use clap::Parser;
use image::{GenericImage, GenericImageView, ImageReader, RgbImage};
use log::{debug, info, warn};
use num_cpus::get as get_cpu_count;
use pipeline::Detection;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::pipeline::Pipeline;
use crate::video::process_frames;
mod manipulation;
mod pipeline;
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

    for path in ["steve.png"] {
        // for path in ["selfie.png", "family.jpg", "steve.png"] {
        let mut img = ImageReader::open(&("tmp/".to_owned() + path))?
            .decode()?
            .into_rgb8();
        let start = Instant::now();
        let result = pipeline.run_trace(&mut img);
        debug!("{result:?}");
        debug!("Took {:?}", start.elapsed());

        img.save("tmp/out-".to_owned() + path)?;
        debug!("Result at {:?}", "tmp/out-".to_owned() + path);
    }
    return Ok(());

    // match args.image_path {
    //     Some(p) => {
    //         let mut img = ImageReader::open(&p)?.decode()?.into_rgb8();
    //         let start = Instant::now();
    //         let result = pipeline.run_trace(&mut img);
    //         debug!("{result:?}");
    //         debug!("Took {:?}", start.elapsed());

    //         let output_path: &str = &args.output_path.unwrap_or("tmp/result.png".to_string());
    //         img.save(output_path)?;
    //         info!("Result at {:?}", output_path);
    //         return Ok(());
    //     }
    //     None => debug!("No image specified, running in webcam mode"),
    // }

    // Default mode: Webcam stream

    // TODO: allow arg to specify a video camera

    let face_detection = Arc::new(RwLock::new(DetectionResult {
        detection: None,
        calced_at: Instant::now(),
    }));

    let latest_img: Arc<Mutex<DetectionInput>> = Arc::new(Mutex::new(None));

    let model_output = face_detection.clone();
    let model_input = latest_img.clone();
    // thread::spawn(move || -> Result<()> {
    //     loop {
    //         let mut model_input = model_input.lock().unwrap();

    //         if let Some(mut image) = model_input.take() {
    //             drop(model_input);
    //             debug!("Re-running face detection model...");
    //             let detection = pipeline.run(&image)?;

    //             // write updated faces into shared state
    //             let mut model_output = model_output.write().unwrap();
    //             model_output.detection = detection.into();
    //             debug!(
    //                 "{:?} elapsed since last face detection run",
    //                 model_output.calced_at.elapsed()
    //             );
    //             model_output.calced_at = Instant::now();
    //             drop(model_output)
    //         } else {
    //             drop(model_input);
    //         }
    //         thread::sleep(Duration::from_millis(1));
    //     }

    //     Ok(())
    // });

    process_frames(
        total_threads / 2,
        &face_detection.clone(),
        &latest_img.clone(),
    )?;

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
    // let face_detection = face_detection.read().unwrap();
    // // TDOO: use match instead of unwrap to avoid panic
    // let faces = face_detection.detection.unwrap().faces.clone();
    // drop(face_detection); // free lock early

    // Do this before every time consuming operation
    if start.elapsed().as_millis() >= within_ms.into() {
        return Ok(());
    }

    // // swap mouth and eyes
    // for face in faces {
    //     let mut mouth: Option<Rect> = None;
    //     let mut l_eye: Option<Rect> = None;
    //     let mut r_eye: Option<Rect> = None;
    //     for f in face.features {
    //         match f.kind {
    //             FaceFeatureKind::Mouth => mouth = f.bounds.clone().into(),
    //             FaceFeatureKind::LeftEye => l_eye = f.bounds.clone().into(),
    //             FaceFeatureKind::RightEye => r_eye = f.bounds.clone().into(),
    //             default => {}
    //         }
    //     }

    //     let mouth = mouth.ok_or(Error::msg("No mouth detected"))?;
    //     let l_eye = l_eye.ok_or(Error::msg("No left eye detected"))?;
    //     let r_eye = r_eye.ok_or(Error::msg("No right eye detected"))?;

    //     // mouth to leye
    //     let mouth_view = *img.view(
    //         mouth.left().try_into().unwrap(),
    //         mouth.top().try_into().unwrap(),
    //         mouth.width,
    //         mouth.height,
    //     );
    //     let mut m_img = image::RgbaImage::new(mouth.width, mouth.height);
    //     m_img.copy_from(&mouth_view, 0, 0)?;

    //     img.copy_within(
    //         l_eye.into(),
    //         mouth.left().try_into().unwrap(),
    //         mouth.top().try_into().unwrap(),
    //     );
    //     img.copy_from(&m_img, l_eye.left(), l_eye.top())?;
    //     img.copy_from(&m_img, r_eye.left(), r_eye.top())?;

    //     if start.elapsed().as_millis() >= within_ms.into() {
    //         return Err(Error::msg(format!(
    //             "process_image exceeded allowed time of {}ms",
    //             within_ms
    //         )));
    //     }
    // }

    Ok(())
}
