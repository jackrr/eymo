use anyhow::Result;
use ndarray::Array;
use opencv::{
    core::{self, Mat, MatTraitConst, Size},
    highgui, imgproc, objdetect,
    prelude::*,
    videoio,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

fn main() -> Result<()> {
    highgui::named_window("window", highgui::WINDOW_FULLSCREEN)?;

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    let mut frame = Mat::default();
    let mut frame_gray = Mat::default();

    let min_face_size = Size::new(50, 50);
    let max_face_size = Size::new(500, 500);

    // src: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    let mut face_detector =
        objdetect::CascadeClassifier::new("./models/haarcascade_frontalface_default.xml")?;

    // TODO: use yolov11
    // https://huggingface.co/qualcomm/Facial-Landmark-Detection/tree/main
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("./models/Facial-Landmark-Detection.onnx")?;

    loop {
        // TODO:
        // 1. render each stage (resized, cropped, model outputs in full) in window (for debugging)
        // 2. do not advance frame until user input is given on prev
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
            let _ = imgproc::rectangle(
                &mut frame,
                face,
                core::VecN([255., 0., 0., 0.]),
                1,
                imgproc::LINE_8,
                0,
            );
            println!("Found an face at {:?}", face);

            // 2. use face bounding box to crop + resize crop to 128 px

            // Riffing off of
            // https://github.com/quic/ai-hub-models/blob/v0.28.2/qai_hub_models/models/facemap_3dmm/app.py#L63-L72
            // and
            // https://github.com/quic/ai-hub-models/blob/v0.28.2/qai_hub_models/models/facemap_3dmm/model.py#L45-L58
            let face_crop = Mat::roi(&frame, face)?;
            let mut face_resized = Mat::default();
            let height = 128;
            let width = 128;
            imgproc::resize(
                &face_crop,
                &mut face_resized,
                Size { width, height },
                0.0,
                0.0,
                imgproc::INTER_AREA,
            )?;

            let mut face_f32 = Array::zeros((1, 3, 128, 128));
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
            let outputs = model.run(ort::inputs!["image" => input]?)?;
            let results = outputs["parameters_3dmm"].try_extract_tensor::<f32>()?;
            // uhhh wtf is this output
            // https://github.com/quic/ai-hub-models/blob/v0.28.2/qai_hub_models/models/facemap_3dmm/utils.py#L14
            println!("Results! {:?}", results);
        }

        highgui::imshow("window", &frame)?;
        let key = highgui::wait_key(1)?;
        if key == 113 {
            // quit with q
            break;
        }
    }
    Ok(())
}
