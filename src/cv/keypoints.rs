use std::time::Instant;

use crate::cv::shapes::{overlap_pct, Detection, Keypoint, Point, Rect};
use anyhow::Result;
use image::{imageops::FilterType, DynamicImage, Pixel, Rgba};
use imageproc::drawing::{self, Canvas};
use log::{debug, info};
use ndarray::parallel::prelude::IntoParallelRefMutIterator;
use ndarray::{Array, Dim, IxDynImpl, ViewRepr};
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

pub use ort::session::Session;

pub fn initialize_model(model_file_path: &str) -> Result<Session> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(format!("./models/{model_file_path:}"))?;

    Ok(model)
}

const MIN_CONFIDENCE: f32 = 0.85;

pub fn process_image(
    model: &Session,
    img: &mut DynamicImage,
    trace: bool,
) -> Result<Vec<Detection>> {
    let height = 640;
    let width = 640;

    let start = Instant::now();
    // 175ms
    let resized = img.resize(width, height, FilterType::Nearest);
    debug!("Took {:?} to resize image", start.elapsed());

    let resized_width = resized.width();
    let resized_height = resized.height();

    let start = Instant::now();
    let model_input =
        Array::from_shape_fn((1, 3, height as usize, width as usize), |(_, c, y, x)| {
            let y: u32 = y as u32;
            let x: u32 = x as u32;
            if y >= resized_height {
                0.
            } else if x >= resized_width {
                0.
            } else {
                resized.get_pixel(x, y)[c] as f32 / 255.0
            }
        });
    debug!("Took {:?} to build ndarray", start.elapsed());

    let input = Tensor::from_array(model_input)?;

    let start = Instant::now();
    let outputs = model.run(ort::inputs!["images" => input]?)?;
    debug!("Took {:?} to run model", start.elapsed());

    let result = outputs["output0"].try_extract_tensor::<f32>()?;

    let detections = process_result(
        result,
        img.width() as f32 / resized_width as f32,
        img.height() as f32 / resized_height as f32,
    );
    debug!("Pose detection found {:?} faces", detections.len());
    if detections.len() == 0 {
        info!("No faces found.");
    }

    if trace {
        display_results(img, &detections);
    }

    Ok(detections)
}

fn process_result(
    result: ndarray::ArrayBase<ViewRepr<&f32>, Dim<IxDynImpl>>,
    x_scale: f32,
    y_scale: f32,
) -> Vec<Detection> {
    let mut detections: Vec<Detection> = Vec::new();

    for row in result.squeeze().columns() {
        let row: Vec<_> = row.iter().copied().collect();
        let c = row[4];
        if c < MIN_CONFIDENCE {
            continue;
        }

        let xc = (row[0] * x_scale).round() as i32; // centerpoint x
        let yc = (row[1] * y_scale).round() as i32; // centerpoint y
        let w = (row[2] * x_scale).round() as i32;
        let h = (row[3] * y_scale).round() as i32;

        let face_loc = Rect::at(xc - w / 2, yc - h / 2).of_size(w as u32, h as u32);

        let mut has_better_dup = false;
        for (i, d) in detections.iter().enumerate() {
            if overlap_pct(&face_loc, &d.face) > 20. {
                // pick the one with higher confidence
                if d.confidence > c {
                    has_better_dup = true;
                } else {
                    detections.swap_remove(i);
                }
                break;
            }
        }

        if has_better_dup {
            continue;
        }

        let mut detection = Detection {
            face: face_loc,
            keypoints: Vec::new(),
            confidence: c,
        };

        // 0 - nose, 1 - l eye, 2 r eye, 3 l ear, 4 r ear
        for k in 0..5 {
            let k_idx = 5 + (k * 3);
            let kc = row[k_idx + 2];

            if kc < MIN_CONFIDENCE {
                continue;
            }

            let kx = (row[k_idx] * x_scale).round() as i32;
            let ky = (row[k_idx + 1] * y_scale).round() as i32;

            detection.keypoints.push(Keypoint {
                feature_idx: k as u8,
                point: Point::new(kx, ky),
                confidence: kc,
            })
        }

        detections.push(detection);
    }

    detections
}

fn display_results(img: &mut DynamicImage, detections: &Vec<Detection>) {
    info!("Writing detections to image");

    for d in detections {
        drawing::draw_hollow_rect_mut(img, d.face, Rgba([255u8, 0u8, 0u8, 255u8]));

        for kp in &d.keypoints {
            drawing::draw_filled_circle_mut(
                img,
                (kp.point.x, kp.point.y),
                10,
                Rgba([0u8, 0u8, 255u8, 255u8]),
            );
        }
    }
}
