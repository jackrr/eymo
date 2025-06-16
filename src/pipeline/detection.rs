use super::model::{initialize_model, Session};
use crate::imggpu::resize::{resize, ResizeAlgo};
use crate::shapes::point::PointF32;
use crate::shapes::rect::{Rect, RectF32};
use anchors::gen_anchors;
use anyhow::Result;
use image::RgbImage;
use ndarray::Array;
use ort::value::Tensor;
use tracing::{debug, span, Level};

mod anchors;

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;

pub struct FaceDetector {
    model: Session,
    anchors: [RectF32; 896],
}

#[derive(Debug, Clone)]
pub struct Face {
    pub bounds: Rect,
    pub l_eye: PointF32,
    pub r_eye: PointF32,
    confidence: f32,
}

impl Face {
    pub fn with_eyes(confidence: f32, bounds: Rect, l_eye: PointF32, r_eye: PointF32) -> Face {
        Face {
            l_eye,
            r_eye,
            bounds,
            confidence,
        }
    }

    pub fn rot_theta(&self) -> f32 {
        let dx = self.r_eye.x - self.l_eye.x;
        let dy = self.r_eye.y - self.l_eye.y;
        -dy.atan2(dx)
    }
}

impl FaceDetector {
    /*
    BlazeFace model wrapper using ort to run the model, then manually
    process the results into one or more faces


    Model Input: 128x128 f32 image
    Model Output:
    - 896 length array of confidence scores (classificators)
    - 896 length 2D array of detection coords (regressors)

    The first 4 values in the detection coords are centroid, width,
    height offsets applied to a particular cell among 2 predetermined
    grids. The index in the 896 length array determines which square
    in the grids is referred to.  The remaning 12 values are points
    for key features (eyes, ears, etc).

    // scale gets interpolated

    */
    pub fn new(threads: usize) -> Result<FaceDetector> {
        Ok(FaceDetector {
            model: initialize_model("mediapipe_face_detection_short_range.onnx", threads)?,
            anchors: gen_anchors(),
        })
    }

    pub fn run(&self, img: &RgbImage) -> Result<Vec<Face>> {
        let span = span!(Level::INFO, "face_detector");
        let _guard = span.enter();

        // resize -- approx 60ms
        let span_res = span!(Level::INFO, "face_detector_resize");
        let res_span_guard = span_res.enter();
        let resized = resize(img, WIDTH, HEIGHT, ResizeAlgo::Nearest)?;
        drop(res_span_guard);

        // ndarr -- approx 7ms
        let span_ndarr = span!(Level::INFO, "face_detector_ndarr");
        let ndarr_span_guard = span_ndarr.enter();
        let resized_width = resized.width();
        let resized_height = resized.height();

        let input_arr =
            Array::from_shape_fn((1, HEIGHT as usize, WIDTH as usize, 3), |(_, y, x, c)| {
                let x: u32 = x as u32;
                let y: u32 = y as u32;

                if y >= resized_height {
                    0.
                } else if x >= resized_width {
                    0.
                } else {
                    (resized.get_pixel(x, y)[c] as f32 / 127.5) - 1. // -1.0 - 1.0 range
                }
            });
        drop(ndarr_span_guard);

        // tensor -- approx 23micros
        let span_tensor = span!(Level::INFO, "face_detector_tensor");
        let tensor_span_guard = span_tensor.enter();
        let input = Tensor::from_array(input_arr)?;
        drop(tensor_span_guard);

        // model -- 2-20ms (either 2ms or like 18-20ms... why?)
        let span_model = span!(Level::INFO, "face_detector_model");
        let model_span_guard = span_model.enter();
        let outputs = self.model.run(ort::inputs!["input" => input]?)?;
        let regressors = outputs["regressors"].try_extract_tensor::<f32>()?;
        let classificators = outputs["classificators"].try_extract_tensor::<f32>()?;
        drop(model_span_guard);

        let span_result = span!(Level::INFO, "face_detector_results");
        let result_span_guard = span_result.enter();
        let scores = classificators.as_slice().unwrap();

        let detections = regressors.squeeze();
        let mut row_idx = 0;
        let mut results: Vec<Face> = Vec::new();

        for res in detections.rows() {
            let score = sigmoid_stable(scores[row_idx]);
            if score > 0.5 {
                let x_scale = img.width() as f32 / resized_width as f32;
                let y_scale = img.height() as f32 / resized_height as f32;

                // TODO: gen_anchor needs work...
                // let mut anchor = gen_anchor(row_idx.try_into().unwrap())?;
                let mut anchor = self.anchors[row_idx].clone();
                let ax = anchor.x.clone();
                let ay = anchor.y.clone();

                let scaled: Rect = anchor
                    .adjust(res[0], res[1], res[2], res[3])
                    .scale(x_scale, y_scale)
                    .into();

                let mut better_found = false;
                for (i, d) in results.iter().enumerate() {
                    if d.bounds.overlap_pct(&scaled) > 30. {
                        if d.confidence > score {
                            better_found = true;
                        } else {
                            results.swap_remove(i);
                        }
                        break;
                    }
                }
                if !better_found {
                    let l_eye = PointF32 {
                        x: ((ax + res[4]) * x_scale),
                        y: ((ay + res[5]) * y_scale),
                    };
                    let r_eye = PointF32 {
                        x: ((ax + res[6]) * x_scale),
                        y: ((ay + res[7]) * y_scale),
                    };
                    results.push(Face::with_eyes(score, scaled, l_eye, r_eye));
                }
            }
            row_idx += 1;
        }

        drop(result_span_guard);

        debug!("Detected {} faces", results.len());

        Ok(results)
    }
}

fn sigmoid_stable(x: f32) -> f32 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        x.exp() / (1. + x.exp())
    }
}
