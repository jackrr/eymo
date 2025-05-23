pub use crate::cv::shapes::{Detection, Face, FaceFeature, FaceFeatureKind, Point};
use image::{DynamicImage, Rgba};
use imageproc::drawing;
use log::{debug, error, log_enabled, Level};

use super::shapes::make_rect;

pub fn face_detection_to_features(d: Detection, img: &mut DynamicImage) -> Face {
    let mut face = Face::new(d.face);
    let face_width = d.face.right() - d.face.left();
    let face_height = d.face.bottom() - d.face.top();

    for kp in d.keypoints {
        match kp.feature_idx {
            0 => {
                let nose = FaceFeature {
                    bounds: make_rect(
                        kp.point,
                        (face_width as f32 * 0.14).round() as i32,
                        (face_height as f32 * 0.16).round() as i32,
                    ),
                    kind: FaceFeatureKind::Nose,
                };
                face.features.push(nose);

                face.features.push(FaceFeature {
                    bounds: make_rect(
                        Point::new(
                            kp.point.x,
                            kp.point.y + (0.15 * face_height as f32).round() as i32,
                        ),
                        (face_width as f32 * 0.22).round() as i32,
                        (face_height as f32 * 0.10).round() as i32,
                    ),
                    kind: FaceFeatureKind::Mouth,
                });
            }
            1 => {
                face.features.push(FaceFeature {
                    bounds: make_rect(
                        kp.point,
                        (face_width as f32 * 0.16).round() as i32,
                        (face_height as f32 * 0.12).round() as i32,
                    ),
                    kind: FaceFeatureKind::LeftEye,
                });
            }
            2 => {
                face.features.push(FaceFeature {
                    bounds: make_rect(
                        kp.point,
                        (face_width as f32 * 0.16).round() as i32,
                        (face_height as f32 * 0.12).round() as i32,
                    ),
                    kind: FaceFeatureKind::RightEye,
                });
            }
            3 => {
                face.features.push(FaceFeature {
                    bounds: make_rect(
                        kp.point,
                        (face_width as f32 * 0.04).round() as i32,
                        (face_height as f32 * 0.10).round() as i32,
                    ),
                    kind: FaceFeatureKind::LeftEar,
                });
            }
            4 => {
                face.features.push(FaceFeature {
                    bounds: make_rect(
                        kp.point,
                        (face_width as f32 * 0.04).round() as i32,
                        (face_height as f32 * 0.10).round() as i32,
                    ),
                    kind: FaceFeatureKind::RightEar,
                });
            }
            _ => {
                error!("No FaceFeature known for keypoint idx {:?}", kp.feature_idx);
                continue;
            }
        }
    }

    if log_enabled!(Level::Debug) {
        debug!("Displaying face features");
        display_face(img, &face);
    }

    face
}

fn display_face(img: &mut DynamicImage, face: &Face) {
    drawing::draw_hollow_rect_mut(img, face.bounds, Rgba([0u8, 255u8, 0u8, 255u8]));

    for f in face.features.iter() {
        drawing::draw_hollow_rect_mut(img, f.bounds, Rgba([0u8, 255u8, 0u8, 255u8]));
    }
}
