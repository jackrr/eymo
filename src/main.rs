use anyhow::{Error, Result};
use clap::Parser;
use image::ImageReader;
use log::debug;
use std::collections::HashSet;

use crate::cv::{detect_features, initialize_model};
mod cv;

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
            img.save(output_path);
            return Ok(());
        }
        None => debug!("No image specified, running in webcam mode"),
    }

    // TODO: Stream mode from webcam
    return Err(Error::msg("Stream mode not yet implemented"));

    // TODO: in debug mode, use a window to show output stream
    // Ok(())
}
