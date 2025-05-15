use anyhow::Result;
use candle_core::{safetensors, DType, Device, Tensor};
use candle_onnx;
use opencv::{
    core::{self, Size},
    highgui, imgproc, objdetect,
    prelude::*,
    videoio,
};
use std::collections::HashMap;

fn main() -> Result<()> {
    highgui::named_window("window", highgui::WINDOW_FULLSCREEN)?;

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    // Various stages of image processing
    let mut frame = Mat::default();
    let mut frame_gray = Mat::default();
    let mut face_128 = Mat::default();
    let mut face_rgb = Mat::default();
    let mut face_float = Mat::default();

    let min_face_size = Size::new(50, 50);
    let max_face_size = Size::new(500, 500);

    // https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    let mut face_detector =
        objdetect::CascadeClassifier::new("./models/haarcascade_frontalface_default.xml")?;

    // https://huggingface.co/qualcomm/Facial-Landmark-Detection/tree/main
    // For some reason candle shits itself on this model, can't handle
    // the pads on the MaxPool tensor
    let model = candle_onnx::read_file("./models/Facial-Landmark-Detection.onnx")?;

    // let face_model = safetensors::load(
    //     "./models/diff_control_sd15_landmarks_fp16.safetensors",
    //     &Device::Cpu,
    // );

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

            imgproc::resize(
                &face_crop,
                &mut face_128,
                Size {
                    width: 128,
                    height: 128,
                },
                0.0,
                0.0,
                imgproc::INTER_AREA,
            )?;
            imgproc::cvt_color(&face_128, &mut face_rgb, imgproc::COLOR_BGR2RGB, 0)?;
            face_rgb.convert_to(&mut face_float, core::CV_32F, 1.0, 0.0)?;
            println!("Got image ready for tensoring {:?}", face_float);

            let tensor = Tensor::from_raw_buffer(
                face_float.data_bytes()?,
                DType::F32,
                &[face_rgb.rows() as usize, face_rgb.cols() as usize, 3],
                &Device::Cpu,
            )?
            .permute((2, 0, 1))?
            .unsqueeze(0)?;
            println!("Made tensor!");

            let inputs: HashMap<String, Tensor> = HashMap::from([("image".to_string(), tensor)]);
            let results = candle_onnx::simple_eval(&model, inputs)?;
            println!("Made results!");

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
