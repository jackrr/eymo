use anyhow::Result;
use face::face_detection_to_features;
use image::DynamicImage;
use shapes::Face;
mod face;
mod keypoints;
mod shapes;

pub use crate::cv::keypoints::initialize_model;

pub fn detect_features(
    model: &keypoints::Session,
    img: &mut DynamicImage, // mut for debug mode
) -> Result<Vec<Face>> {
    let detections = keypoints::process_image(model, img)?;
    let mut faces: Vec<Face> = Vec::new();

    for d in detections {
        faces.push(face_detection_to_features(d, img));
    }

    Ok(faces)
}
