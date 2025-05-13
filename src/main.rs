use anyhow::Result;
use opencv::{core, highgui, imgproc, objdetect, prelude::*, videoio};

fn main() -> Result<()> {
    highgui::named_window("window", highgui::WINDOW_FULLSCREEN)?;

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let mut frame = Mat::default();
    let mut detect = Mat::default();

    let mut eye_detector = objdetect::CascadeClassifier::new("./models/haarcascade_eye.xml")?;
    let min_eye = core::Size::new(10, 10);
    let max_eye = core::Size::new(1000, 1000);

    loop {
        cam.read(&mut frame)?;
        imgproc::cvt_color(&frame.clone(), &mut detect, imgproc::COLOR_BGR2GRAY, 0)?;
        let mut rects = core::Vector::<core::Rect>::new();

        let _ = eye_detector.detect_multi_scale(&detect, &mut rects, 1.1, 3, 0, min_eye, max_eye);

        for rect in rects {
            let _ = imgproc::rectangle(
                &mut frame,
                rect,
                core::VecN([255., 0., 0., 0.]),
                1,
                imgproc::LINE_8,
                0,
            );
            println!("Found an eye at {:?}", rect)
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
