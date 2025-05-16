use anyhow::Result;
use ndarray::{Array, Dim, IxDynImpl, ViewRepr};
use opencv::{
    core::{self, Mat, MatTraitConst, Point, Rect, Scalar, Size},
    highgui, imgproc, objdetect,
    prelude::*,
    videoio,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::cmp::{max, min};

enum Model {
    YoloV11PoseM = "yolo11m-pose.onnx",
    YoloV11PoseS = "yolo11s-pose.onnx",
    YoloV11PoseN = "yolo11n-pose.onnx",
}

fn main() -> Result<()> {
    highgui::named_window("stream", highgui::WINDOW_AUTOSIZE)?;

    // These seemingly don't work on wayland
    // highgui::move_window("stream", 100, 100)?;
    // highgui::move_window("face_crop", 1000, 100)?;
    // highgui::move_window("face_resize", 1000, 800)?;

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    let mut frame = Mat::default();
    let mut frame_gray = Mat::default();
    let mut face_resized = Mat::default();

    let min_face_size = Size::new(50, 50);
    let max_face_size = Size::new(500, 500);

    // src: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    let mut face_detector =
        objdetect::CascadeClassifier::new("./models/haarcascade_frontalface_default.xml")?;

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("./models/yolo11m-pose.onnx")?;

    let mut has_result = false;

    loop {
        cam.read(&mut frame)?;
        // 1. use opencv to find faces quickly
        imgproc::cvt_color(&frame.clone(), &mut frame_gray, imgproc::COLOR_BGR2GRAY, 0)?;

        let mut faces = core::Vector::<core::Rect>::new();
        let _ = face_detector.detect_multi_scale(
            &frame_gray,
            &mut faces,
            1.1,
            3,
            0,
            min_face_size,
            max_face_size,
        );

        for face in faces {
            // 2. use face bounding box to crop + resize crop to input size
            // let face_region = expand_rect(face, 100, frame.cols(), frame.rows());

            // let face_crop = Mat::roi(&frame, face_region)?;

            // imgproc::rectangle(
            //     &mut frame,
            //     face_region,
            //     core::VecN([255.0, 255.0, 0., 0.]),
            //     1,
            //     imgproc::LINE_8,
            //     0,
            // )?;

            let height = 640;
            let width = 640;

            imgproc::resize(
                // &face_crop,
                &frame,
                &mut face_resized,
                Size { width, height },
                0.0,
                0.0,
                imgproc::INTER_AREA,
            )?;

            let mut face_f32 = Array::zeros((1, 3, height as usize, width as usize));
            let height = face_resized.rows() as usize;
            let width = face_resized.cols() as usize;
            let chans = face_resized.channels() as usize;

            // b r g -> r g b
            let chan_map = vec![2, 0, 1];

            unsafe {
                let mat_slice =
                    std::slice::from_raw_parts(face_resized.data(), height * width * chans);

                for y in 0..height {
                    for x in 0..width {
                        for ch in 0..chans {
                            let idx = (y * width * chans) + (x * chans) + ch;
                            face_f32[[0, chan_map[ch], y, x]] = (mat_slice[idx] as f32) / 255.;
                        }
                    }
                }
            }

            let input = Tensor::from_array(face_f32)?;
            let outputs = model.run(ort::inputs!["images" => input]?)?;
            let results = outputs["output0"].try_extract_tensor::<f32>()?;
            // show_results(&mut frame, results, face.x, face.y, )?;
            show_results(&mut face_resized, results, 0, 0)?;

            has_result = true;
        }

        let mut wait_time = 1;

        if has_result {
            wait_time = 0;
            highgui::imshow("stream", &face_resized)?;
        } else {
            highgui::imshow("stream", &frame)?;
        }

        let key = highgui::wait_key(wait_time)?;
        if key == 113 {
            // quit with q
            break;
        }

        has_result = false;
    }

    Ok(())
}

fn show_results(
    img: &mut Mat,
    result: ndarray::ArrayBase<ViewRepr<&f32>, Dim<IxDynImpl>>,
    x_offset: i32,
    y_offset: i32,
    model: Model,
) -> Result<()> {
    // yolov11n-pose: float32[1,56,2100]
    // yolov11m-pose: float32[1,56,8400]
    // yolov11m-pose: float32[1,56,8400]
    let detection_size = 8400;
    let detections = 56;

    let res = result.as_slice();
    let mut min_x = 10000000;
    let mut max_x = -1;
    let mut min_y = 10000000;
    let mut max_y = -1;

    match res {
        Some(s) => {
            for i in 0..detections {
                let d_idx = i * detection_size;

                let x = s[d_idx] as i32 + x_offset;
                let y = s[d_idx + 1] as i32 + y_offset;
                let w = s[d_idx + 2].round() as i32;
                let h = s[d_idx + 3].round() as i32;

                min_x = min(min_x, x);
                max_x = max(max_x, x);

                min_y = min(min_y, y);
                max_y = max(max_y, y);

                // confidence
                let c = s[d_idx + 4];
                if c < 70.0 {
                    continue;
                }

                imgproc::rectangle(
                    img,
                    Rect::new(x, y, w, h),
                    core::VecN([255., 0., 0., 0.]),
                    1,
                    imgproc::LINE_8,
                    0,
                )?;

                // 17 keypoints: x,y,confidence
                // Nose
                // Left Eye
                // Right Eye
                // Left Ear
                // Right Ear
                // Left Shoulder
                // Right Shoulder
                // Left Elbow
                // Right Elbow
                // Left Wrist
                // Right Wrist
                // Left Hip
                // Right Hip
                // Left Knee
                // Right Knee
                // Left Ankle
                // Right Ankle

                for k in 0..17 {
                    let k_idx = d_idx + (k * 3);
                    let kx = s[k_idx].round() as i32 + x_offset;
                    let ky = s[k_idx + 1].round() as i32 + y_offset;
                    let kc = s[k_idx + 2];

                    // if kc < 70.0 {
                    //     continue;
                    // }

                    imgproc::circle(
                        img,
                        Point::new(kx, ky),
                        5,
                        Scalar::new(0., 0., 255., 0.),
                        -1, // fill
                        imgproc::LINE_8,
                        0,
                    )?;

                    imgproc::put_text(
                        img,
                        &k.to_string(),
                        Point::new(kx + 10, ky + 5),
                        imgproc::FONT_HERSHEY_COMPLEX_SMALL,
                        0.5,
                        Scalar::new(255., 255., 255., 255.),
                        1, //thickness
                        imgproc::LINE_8,
                        false,
                    )?;
                }
                println!(
                    "Result {:?}: {:?},{:?} {:?}x{:?} conf: {:?}",
                    i, x, y, w, h, c
                );
            }
        }
        None => println!("Wheres the slice yo"),
    }

    println!("x: ({min_x:?},{max_x:?}), y: ({min_y:?},{max_y:?})");

    Ok(())
}

fn expand_rect(rect: Rect, pad: i32, max_x: i32, max_y: i32) -> Rect {
    let tl = Point::new(max(rect.x - pad, 0), max(rect.y - pad, 0));
    let br = Point::new(
        min(rect.x + rect.width + pad, max_x),
        min(rect.y + rect.height + pad, max_y),
    );

    Rect {
        x: tl.x,
        y: tl.y,
        width: br.x - tl.x,
        height: br.y - tl.y,
    }
}
