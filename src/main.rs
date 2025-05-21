#![warn(unused_extern_crates)]

use anyhow::{Error, Result};
use clap::Parser;
use image::{DynamicImage, ImageReader};
use log::{debug, info, warn};
use ort::session::Session;
use show_image::{create_window, ImageInfo, ImageView, WindowProxy};
use std::collections::HashSet;
use std::time;

use crate::cv::{detect_features, initialize_model};
mod cv;

use nokhwa::{
    nokhwa_initialize,
    pixel_format::{RgbAFormat, RgbFormat},
    query,
    utils::{ApiBackend, RequestedFormat, RequestedFormatType},
    Buffer, CallbackCamera,
};

struct Moment {
    label: String,
    at: time::Instant,
}

struct Run {
    events: Vec<Moment>,
}

#[derive(debug)]
struct Stats {
    min: u128,
    max: u128,
    avg: f32,
}

impl Stats {
    pub fn from_durations(ds: Vec<time::Duration>) -> Stats {
        let nanos = ds.iter().map(|d| d.as_nanos());

        Stats {
            min: nanos.clone().reduce(|min, d| d.min(min)).unwrap_or(0),
            max: nanos.clone().reduce(|max, d| d.max(max)).unwrap_or(0),
            avg: nanos.reduce(|s, d| s + d).unwrap_or(0) as f32 / ds.len() as f32,
        }
    }
}

impl Run {
    pub fn elapsed(
        &self,
        start_label: Option<String>,
        end_label: Option<String>,
    ) -> time::Duration {
        let start = match start_label {
            Some(s) => self.events.iter().find(|e| e.label == s).unwrap().at,
            None => self.events.first().unwrap().at,
        };

        let end = match end_label {
            Some(s) => self.events.iter().find(|e| e.label == s).unwrap().at,
            None => self.events.last().unwrap().at,
        };

        end - start
    }
}

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
            // TODO: uncomment me!
            // let result = detect_features(&model, &mut img)?;
            debug!("{result:?}");
            img.save(output_path)?;
            info!("Result at {:?}", output_path);
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

    let (sender, receiver) = flume::unbounded();

    let mut camera = CallbackCamera::new(
        cameras.last().unwrap().index().clone(),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
        move |buffer| {
            sender
                .send(buffer)
                .expect("Error sending frame to buffer stream");
        },
    )
    .unwrap();

    camera.open_stream().unwrap();

    let window = create_window("image", Default::default())?;

    let mut runs = Vec::new();

    #[allow(clippy::empty_loop)] // keep it running
    loop {
        // let pull_frame_at = time::Instant::now();
        let frame = receiver.recv()?;
        // ~1ms
        let mut run = Run { events: Vec::new() };
        run.events.push(Moment {
            label: "frame_received".to_string(),
            at: time::Instant::now(),
        });

        let mut image = DynamicImage::from(frame.decode_image::<RgbFormat>().unwrap());
        run.events.push(Moment {
            label: "frame_decoded".to_string(),
            at: time::Instant::now(),
        });

        // ~45ms
        let result = detect_features(&model, &mut image, &mut run)?;
        run.events.push(Moment {
            label: "features_built".to_string(),
            at: time::Instant::now(),
        });

        // ~8ms
        window.set_image("image", image)?;
        run.events.push(Moment {
            label: "frame_displayed".to_string(),
            at: time::Instant::now(),
        });

        runs.push(run);

        if runs.len() > 5 {
            break;
        }
    }

    let full_stats = Stats::from_durations(runs.iter().map(|r| r.elapsed(None, None)));
}
