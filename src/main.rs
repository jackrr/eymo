use anyhow::{Error, Result};
use clap::Parser;
use image::ImageReader;
use log::debug;
use std::collections::HashSet;

use crate::cv::{detect_features, initialize_model};
mod cv;

use nokhwa::{
    nokhwa_initialize,
    pixel_format::{RgbAFormat, RgbFormat},
    query,
    utils::{ApiBackend, RequestedFormat, RequestedFormatType},
    CallbackCamera,
};

const MODEL_YOLO_V11_POSE_M: &str = "yolo11m-pose.onnx";
const MODEL_YOLO_V11_POSE_S: &str = "yolo11s-pose.onnx";
const MODEL_YOLO_V11_POSE_N: &str = "yolo11n-pose.onnx";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model: Option<String>,

    #[arg(short, long)]
    image_path: Option<String>,

    #[arg(short, long)]
    output_path: Option<String>,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    let models = HashSet::from([
        MODEL_YOLO_V11_POSE_S,
        MODEL_YOLO_V11_POSE_N,
        MODEL_YOLO_V11_POSE_M,
    ]);

    let model_name: &str = &args.model.unwrap_or(MODEL_YOLO_V11_POSE_N.to_string());
    let output_path: &str = &args.output_path.unwrap_or("result.png".to_string());

    if models.contains(model_name) {
        debug!("Using model {model_name:?}");
    } else {
        return Err(Error::msg(format!("Unrecognized model {model_name:?}")));
    }

    let model = initialize_model(model_name)?;

    match args.image_path {
        Some(p) => {
            let mut img = ImageReader::open(&p)?.decode()?;
            let result = detect_features(&model, &mut img)?;
            debug!("{result:?}");
            img.save(output_path)?;
            return Ok(());
        }
        None => debug!("No image specified, running in webcam mode"),
    }

    // Default mode: Webcam stream

    // only needs to be run on OSX
    nokhwa_initialize(|granted| {
        debug!("User said {}", granted);
    });
    // TODO: allow arg to specify device
    let cameras = query(ApiBackend::Auto).unwrap();
    cameras
        .iter()
        .for_each(|cam| debug!("Found camera: {:?}", cam));

    let mut threaded = CallbackCamera::new(
        cameras.last().unwrap().index().clone(),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
        |buffer| {
            // TODO: do the processing!
            let image = buffer.decode_image::<RgbFormat>().unwrap();
            println!("{}x{} {}", image.width(), image.height(), image.len());
        },
    )
    .unwrap();
    threaded.open_stream().unwrap();
    #[allow(clippy::empty_loop)] // keep it running
    loop {
        // prob use ggez to make a canvas to draw each image to
        // https://github.com/l1npengtul/nokhwa/blob/senpai/examples/capture/src/main.rs#L43-L74
        // defer to _external_ process for translating shown window into camera stream (OSP, v4loopback, ffmpeg, etc)
        let frame = threaded.poll_frame().unwrap();
        let image = frame.decode_image::<RgbFormat>().unwrap();
        println!(
            "{}x{} {} naripoggers",
            image.width(),
            image.height(),
            image.len()
        );
    }
}
