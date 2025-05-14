use clap::{App, Arg};
use dlib_face_recognition::{FaceDetector, LandmarkPredictor};
use image::{ImageBuffer, Rgb};
use opencv::{
    core::{Mat, Point, Scalar, CV_8UC3},
    highgui::{imshow, wait_key},
    imgproc::{circle, FILLED},
    prelude::*,
    videoio::{VideoCapture, CAP_ANY},
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let matches = App::new("Facial Landmark Detection")
        .version("1.0")
        .author("Your Name")
        .about("Detects facial landmarks from webcam feed")
        .arg(
            Arg::with_name("camera")
                .short("c")
                .long("camera")
                .value_name("ID")
                .help("Camera device ID (default: 0)")
                .default_value("0"),
        )
        .arg(
            Arg::with_name("landmarks_model")
                .short("l")
                .long("landmarks")
                .value_name("FILE")
                .help("Path to facial landmarks model file")
                .default_value("shape_predictor_68_face_landmarks.dat")
                .required(true),
        )
        .get_matches();

    // Get camera ID and model path from arguments
    let camera_id = matches
        .value_of("camera")
        .unwrap()
        .parse::<i32>()
        .unwrap_or(0);
    let landmarks_model = matches.value_of("landmarks_model").unwrap();

    println!("Starting facial landmark detection...");
    println!("Press 'q' to exit");

    // Initialize webcam
    let mut cap = VideoCapture::new(camera_id, CAP_ANY)?;
    if !cap.is_opened()? {
        return Err(format!("Cannot open camera {}", camera_id).into());
    }

    // Initialize face detector and landmark predictor
    let face_detector = FaceDetector::new()?;
    let landmark_predictor = LandmarkPredictor::new(landmarks_model)?;

    // Main processing loop
    let mut frame = Mat::default();
    loop {
        // Capture frame from webcam
        cap.read(&mut frame)?;
        if frame.empty() {
            println!("No frame captured, exiting...");
            break;
        }

        // Convert OpenCV Mat to image format compatible with dlib
        let (height, width) = (frame.rows(), frame.cols());
        let mut rgb_data = vec![0u8; height as usize * width as usize * 3];
        let mut rgb_image = unsafe {
            Mat::new_rows_cols_with_data(
                height,
                width,
                CV_8UC3,
                rgb_data.as_mut_ptr() as *mut _,
                Mat::auto_step(),
            )?
        };
        opencv::imgproc::cvt_color(&frame, &mut rgb_image, opencv::imgproc::COLOR_BGR2RGB, 0)?;

        // Convert to image format for dlib
        let img_buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(
            width as u32,
            height as u32,
            rgb_image.data_bytes()?.to_vec(),
        )
        .ok_or("Failed to create image buffer")?;

        // Detect faces
        let faces = face_detector.face_locations(&img_buffer);
        println!("Detected {} faces", faces.len());

        // For each face, detect landmarks and draw them
        for face_rect in faces {
            // Draw face rectangle
            let top_left = Point::new(face_rect.left() as i32, face_rect.top() as i32);
            let bottom_right = Point::new(face_rect.right() as i32, face_rect.bottom() as i32);
            opencv::imgproc::rectangle(
                &mut frame,
                opencv::core::Rect::new(
                    top_left.x,
                    top_left.y,
                    bottom_right.x - top_left.x,
                    bottom_right.y - top_left.y,
                ),
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                opencv::imgproc::LINE_8,
                0,
            )?;

            // Detect and draw landmarks
            let landmarks = landmark_predictor.face_landmarks(&img_buffer, &face_rect)?;
            for point in landmarks.iter() {
                circle(
                    &mut frame,
                    Point::new(point.x as i32, point.y as i32),
                    2,
                    Scalar::new(0.0, 0.0, 255.0, 0.0),
                    FILLED,
                    opencv::imgproc::LINE_8,
                    0,
                )?;
            }
        }

        // Display the result
        imshow("Facial Landmarks", &frame)?;

        // Check for quit command
        if wait_key(30)? == 'q' as i32 {
            break;
        }
    }

    Ok(())
}
